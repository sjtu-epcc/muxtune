#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Basic modules for PEFT abstraction. """

import os
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import functools
from abc import ABC, abstractmethod

import torch
from torch import nn

from muxtune.core.modules.batched_ops import (
    batched_base_op_forward, batched_base_op_backward, batched_adapter_forward, batched_adapter_backward)
from muxtune.core.data.mixed_tensor import ChunkedTensor, MixedTensor
from muxtune.global_envs import global_configs, PeftType

__all__ = [
    "PeftModuleConfig", "PeftModuleGroup", "PeftModule", 
    "Adapter", "InputDispatcher", "OutputAggregator",
]


@dataclass
class PeftModuleConfig:
    """ General PEFT module configurations. """

    peft_type: PeftType = None
    """ PEFT module type (e.g., LoRA). """

    module_name: str = None
    """ Unique identifier of the PeftModule, in the format of:
    "[base_op_module_name]::[peft_module_name]" (e.g., "qkv_proj::peft_module_0"). """

    input_dispatcher: "InputDispatcher" = None
    """ Input dispatcher for the PEFT module. """

    output_aggregator: "OutputAggregator" = None
    """ Output aggregator for the PEFT module. """

    device: str = "cuda"
    """ Device string of the PEFT module parameters. """ 

    dtype: torch.dtype = torch.float16
    """ Data type of the PEFT module parameters. """


class PeftModuleGroup:
    """ A group of PEFT modules on a single base operator. """

    def __init__(self):
        self.base_op = None
        self.peft_modules = {}  # peft module name -> peft module

    def add_peft_module(self, peft_module: "PeftModule"):
        assert self.base_op is not None, \
            "Base operator should be hooked before adding PEFT modules."
        assert peft_module.config.module_name not in self.peft_modules, \
            f"PEFT module name {peft_module.config.module_name} already exists in the group."
        
        peft_module.register_base_op(self.base_op)
        self.peft_modules[peft_module.config.module_name] = peft_module

    def hook_to_base_op(
        self, base_op: nn.Module, attr_name: str = "peft_module_group", 
        prev_fw_func_name: str = "prev_fw_func",
    ) -> nn.Module:
        """ Hook to the base operator. 
        
        Args:
            base_op: The base operator module to be hooked.
            attr_name: The attribute name of the PEFT module group (default: "peft_module_group").
            prev_fw_func_name: The attribute name of the original forward function of the 
                base operator, after the PeftModuleGroup is hooked (default: "prev_fw_func").
        """
        assert self.base_op is None, "PEFT module group has already been hooked."
        setattr(base_op, attr_name, self)
        setattr(base_op, prev_fw_func_name, base_op.forward)
        self.base_op = base_op
        
        def __hooked_forward(module, *args, **kwargs) -> MixedTensor:
            peft_module_group = getattr(module, attr_name)
            input_tensors = args[0]
            forced_adapter_name = os.environ.get("FORCED_ADAPTER_NAME_DEBUG", None)
            output_tensors = MixedTensor()
            for peft_module_name, peft_module in peft_module_group.peft_modules.items():
                peft_group_index = int(peft_module_name.split("_")[-1])
                chunked_input_tensors: List[ChunkedTensor] = input_tensors[peft_group_index]
                chunked_output_tensors = []
                for chunk in chunked_input_tensors:
                    chunk_mask, layout = chunk.chunk_mask, chunk.layout
                    if forced_adapter_name is not None:
                        output_tensor = peft_module._single_forward(
                            forced_adapter_name, chunk.value, prev_fw_func_name)
                    else:
                        output_tensor = peft_module.batched_forward(
                            chunk.value, prev_fw_func_name)
                    chunked_output_tensors.append(ChunkedTensor(output_tensor, chunk_mask, layout))
                output_tensors[peft_group_index] = chunked_output_tensors

            return output_tensors

        # Overriding a GraphModuleImpl forward freezes the forward call and 
        # later modifications on the graph will fail.
        # Reference: https://pytorch.slack.com/archives/C3PDTEV8E/p1705929610405409
        if "GraphModuleImpl" in str(type(base_op)):
            base_op.__class__.forward = functools.update_wrapper(
                functools.partial(__hooked_forward, base_op), 
                getattr(base_op, prev_fw_func_name),
            )
        else:
            base_op.forward = functools.update_wrapper(
                functools.partial(__hooked_forward, base_op), 
                getattr(base_op, prev_fw_func_name),
            )
        return base_op


class PeftModule:
    """ PEFT module class, including adapter, dispatch (input) and aggregate (output) rules, 
    and base op (of the LLM backbone). 
    
    All adapters within the same PeftModule are spatially executed in a batched manner; 
    all PeftModules are temporally scheduled.

    Args:
        config: The configuration of the PEFT module.
    """

    def __init__(self, config: PeftModuleConfig):
        self.config = config
        self.base_op = None
        self.adapters = {}  # adapter name -> adapter
        self.input_dispatcher = config.input_dispatcher
        self.output_aggregator = config.output_aggregator

        self.microbatch_sizes = []
        self.ordered_adapter_names = []
        
    def register_one_adapter(self, adapter: "Adapter", microbatch_size: int):
        """ Register one adapter. """
        self.adapters[adapter.name] = adapter
        self.microbatch_sizes.append(microbatch_size)
        self.ordered_adapter_names.append(adapter.name)

        assert self.adapters[adapter.name].device == self.config.device, \
            f"Adapter ({adapter.name}) device {self.adapters[adapter.name].device} does not match " + \
            f"PEFT module device {self.config.device}."
        assert self.adapters[adapter.name].dtype == self.config.dtype, \
            f"Adapter ({adapter.name}) dtype {self.adapters[adapter.name].dtype} does not match " + \
            f"PEFT module dtype {self.config.dtype}."
    
    def _single_forward(
        self, adapter_name: str, peft_in: torch.Tensor, prev_fw_func_name: str = "forward",
    ) -> torch.Tensor:
        """ Forward a single adapter (and its base op). """

        assert self.base_op is not None, "PeftModule is not hooked to a base operator."
        base_op_func = getattr(self.base_op, prev_fw_func_name)
        a_in, b_in = self.input_dispatcher.dispatch(peft_in)
        a_out = self.adapters[adapter_name](a_in) if a_in is not None else None
        b_out =base_op_func(b_in) if b_in is not None else None
        a_out = self.adapters[adapter_name](b_out) if a_out is None else a_out  # maybe forward from other output
        b_out = base_op_func(a_out) if b_out is None else b_out
        return self.output_aggregator.aggregate(a_out, b_out)

    def batched_forward(
            self, batched_peft_in: torch.Tensor, prev_fw_func_name: str = "forward",
        ) -> torch.Tensor:
        """ Forward all adapters (and their base ops) in a batched manner. 
        
        Args:
            batched_peft_in: Multi-adapter input tensor batched along batch dimension.
        """
        
        assert self.base_op is not None, "PeftModule is not hooked to a base operator."
        if not batched_peft_in.requires_grad:
            # PyTorch only builds grad_fn for a custom Function's output when at least one
            # input has requires_grad=True. Otherwise backward is never called.
            batched_peft_in = batched_peft_in.requires_grad_(True)
        return _BatchedPeftModuleForwardWrapper.apply(
            batched_peft_in, self.microbatch_sizes, self.config.peft_type, 
            self.base_op, [self.adapters[name] for name in self.ordered_adapter_names], 
            self.input_dispatcher, self.output_aggregator, prev_fw_func_name,
        )
    
    def register_base_op(self, base_op: nn.Module):
        """ Register the base operator for this module. """
        assert self.base_op is None, "Base operator has already been registered."
        self.base_op = base_op


class _BatchedPeftModuleForwardWrapper(torch.autograd.Function):
    """ Batched forward wrapper for PeftModule. """

    @staticmethod
    def forward(
        ctx, batched_peft_in: torch.Tensor, split_sizes: List[int], peft_type: PeftType,
        base_op: nn.Module, adapters: List["Adapter"], input_dispatcher: "InputDispatcher", 
        output_aggregator: "OutputAggregator", prev_fw_func_name: str = "forward",
    ) -> torch.Tensor:
        ctx.split_sizes = split_sizes
        ctx.peft_type = peft_type
        ctx.base_op = base_op
        ctx.adapters = adapters
        ctx.input_dispatcher = input_dispatcher
        ctx.output_aggregator = output_aggregator
        ctx.prev_fw_func_name = prev_fw_func_name
        
        a_ins, b_ins = [], []
        peft_ins = torch.split(batched_peft_in, split_sizes, dim=global_configs.batch_dimension)
        for peft_in in peft_ins:
            a_in, b_in = input_dispatcher.dispatch(peft_in)
            a_ins.append(a_in)
            b_ins.append(b_in)

        a_outs = batched_adapter_forward(ctx, peft_type, a_ins, adapters) if a_in[0] is not None else None
        b_outs = batched_base_op_forward(base_op, b_ins, split_sizes, prev_fw_func_name) \
            if b_in[0] is not None else None
        # maybe forward from the other module's output
        a_outs = batched_adapter_forward(ctx, peft_type, b_outs, adapters) if a_outs is None else a_outs
        b_outs = batched_base_op_forward(base_op, a_outs, split_sizes, prev_fw_func_name) \
            if b_outs is None else b_outs

        outs = []
        for a_out, b_out in zip(a_outs, b_outs):
            out = output_aggregator.aggregate(a_out, b_out)
            outs.append(out)

        ctx.a_ins = a_ins   # for backward pass
        ctx.a_outs = a_outs

        return torch.cat(outs, dim=global_configs.batch_dimension)

    @staticmethod
    def backward(
        ctx, batched_grad_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None, None, None, None, None]:
        grad_outs = torch.split(batched_grad_out, ctx.split_sizes, dim=global_configs.batch_dimension)
        a_grad_outs, b_grad_outs = [], []
        for grad_out in grad_outs:
            a_grad_out, b_grad_out = ctx.output_aggregator.reversed_aggregate(grad_out)
            a_grad_outs.append(a_grad_out)
            b_grad_outs.append(b_grad_out)
        
        b_grad_ins = batched_base_op_backward(ctx.base_op, grad_outputs=b_grad_outs, split_sizes=ctx.split_sizes)
        a_grad_ins = batched_adapter_backward(ctx, ctx.peft_type, ctx.adapters, grad_outputs=a_grad_outs)

        grad_ins = []
        for a_grad_in, b_grad_in in zip(a_grad_ins, b_grad_ins):
            grad_in = ctx.input_dispatcher.reversed_dispatch(a_grad_in, b_grad_in)
            grad_ins.append(grad_in)

        return torch.cat(grad_ins, dim=global_configs.batch_dimension), None, None, None, None, None, None, None


class Adapter(nn.Module, ABC):
    """ Base class for adapter module. 
    
    Args:
        name: Unique identifier of the adapter: "[base_op_module_name]::[task_name]" ("qkv_proj::task_0").
        device: Device string of the adapter.
        dtype: Data type of adapter parameters.
    """

    def __init__(self, name: str, device: str, dtype: torch.dtype):
        super().__init__()
        self.name = name
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class InputDispatcher(ABC):
    """ Base class for input dispatcher. """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def dispatch(self, peft_in: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """ Dispatch the input tensor of PEFT module to adapter and LLM base operator. 

        Args:
            peft_in: The input tensor of PEFT module.
        
        Return:
            A tuple of input tensors for adapter and base operator. If one of them is None, it means the 
            corresponding input tensor comes from the output tensor of the other module.
        """
        pass

    @abstractmethod
    def reversed_dispatch(self, adapter_grad_in: Optional[torch.Tensor], base_grad_in: Optional[torch.Tensor]) -> torch.Tensor:
        """ Reversed dispatch by aggregating gradients from adapter and LLM base operator.

        Args:
            adapter_grad_in: The gradient tensor of adapter. If None, it means adapter input comes 
                from the output tensor of base operator.
            base_grad_in: The gradient tensor of LLM base operator. If None, it means base operator 
                input comes from the output tensor of adapter.
        
        Return:
            The gradient to be propagated to the previous operator.
        """
        pass


class OutputAggregator(ABC):
    """ Base class for output aggregator. """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def aggregate(self, adapter_out: Optional[torch.Tensor], base_out: Optional[torch.Tensor]) -> torch.Tensor:
        """ Aggregate the output tensors of adapter and LLM base operator. 
        
        Args:
            adapter_out: The output tensor of adapter. If None, it means the adapter output serves as 
                the input tensor of base operator.
            base_out: The output tensor of LLM base operator. If None, it means the base operator output 
                serves as the input tensor of adapter.
        """
        pass

    @abstractmethod
    def reversed_aggregate(self, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """ Reversed aggregate by splitting the gradient to adapter and LLM base operator. 

        Args:
            grad_out: The gradient tensor of the aggregated output.
        
        Return:
            A tuple of gradient tensors for adapter and base operator. If one of them is None, it means the 
            corresponding output serves as the input tensor of the other module.
        """
        pass
