#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Basic modules for PEFT abstraction. """

from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
from torch import nn

from muxtune.core.batched_ops import (
    batched_base_op_forward, batched_base_op_backward, batched_adapter_forward)
from muxtune.global_envs import PeftType

__all__ = [ "PeftModuleConfig", "PeftModule", "Adapter", "InputDispatcher", "OutputAggregator" ]


@dataclass
class PeftModuleConfig:
    """ General PEFT module configurations. """

    peft_type: PeftType = None
    """ PEFT module type (e.g., LoRA). """

    module_name: str = None
    """ Unique identifier of the PeftModule, in the format of:
    "[base_op_module_name]::[peft_module_index]" (e.g., "qkv_proj::peft_module_0"). """

    input_dispatcher: "InputDispatcher" = None
    """ Input dispatcher for the PEFT module. """

    output_aggregator: "OutputAggregator" = None
    """ Output aggregator for the PEFT module. """

    device: str = "cuda"
    """ Device string of the PEFT module parameters. """ 

    dtype: torch.dtype = torch.float16
    """ Data type of the PEFT module parameters. """


class PeftModule:
    """ PEFT module class, including adapter, dispatch (input) and aggregate (output) rules, 
    and base op (of the LLM backbone). 
    
    NOTE: All adapters within the same PeftModule are spatially executed in a batched manner; 
        all PeftModules are temporally scheduled.
    """

    def __init__(self, config: PeftModuleConfig, base_op: nn.Module):
        self.config = config
        self.base_op = base_op
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
    
    def _single_forward(self, adapter_name: str, peft_in: torch.Tensor) -> torch.Tensor:
        """ Forward a single adapter (and its base op). """

        a_in, b_in = self.input_dispatcher.dispatch(peft_in)
        a_out = self.adapters[adapter_name](a_in) if a_in is not None else None
        b_out = self.base_op(b_in) if b_in is not None else None
        a_out = self.adapters[adapter_name](b_out) if a_out is None else a_out  # maybe forward from base output
        b_out = self.base_op(a_out) if b_out is None else b_out                 # maybe forward from adapter output
        return self.output_aggregator.aggregate(a_out, b_out)

    def batched_forward(self, batched_peft_in: torch.Tensor) -> torch.Tensor:
        """ Forward all adapters (and their base ops) in a batched manner. 
        
        Args:
            batched_peft_in: Multi-adapter input tensor batched along batch dimension.
        """
        return _BatchedPeftModuleForwardWrapper.apply(
            batched_peft_in, self.microbatch_sizes, self.config.peft_type, 
            self.base_op, [self.adapters[name] for name in self.ordered_adapter_names], 
            self.input_dispatcher, self.output_aggregator,
        )

        
class _BatchedPeftModuleForwardWrapper(torch.autograd.Function):
    """ Batched forward wrapper for PeftModule. """

    @staticmethod
    def forward(
        ctx, batched_peft_in: torch.Tensor, split_sizes: List[int], peft_type: PeftType,
        base_op: nn.Module, adapters: List["Adapter"], input_dispatcher: "InputDispatcher", 
        output_aggregator: "OutputAggregator",
    ) -> torch.Tensor:
        """ Forward function for batched PeftModule. """

        ctx.save_for_backward(batched_peft_in)
        ctx.split_sizes = split_sizes
        ctx.peft_type = peft_type
        ctx.adapters = adapters
        ctx.input_dispatcher = input_dispatcher
        ctx.output_aggregator = output_aggregator
        
        a_ins, b_ins = [], []
        peft_ins = torch.split(batched_peft_in, split_sizes, dim=0)
        for peft_in in peft_ins:
            a_in, b_in = input_dispatcher.dispatch(peft_in)
            a_ins.append(a_in)
            b_ins.append(b_in)

        a_outs = batched_adapter_forward(ctx, peft_type, a_ins, adapters) if a_in[0] is not None else None
        b_outs = batched_base_op_forward(base_op, b_ins, split_sizes) if b_in[0] is not None else None
        # maybe forward from the other module's output
        a_outs = batched_adapter_forward(ctx, peft_type, b_outs, adapters) if a_outs is None else a_outs
        b_outs = batched_base_op_forward(base_op, a_outs, split_sizes) if b_outs is None else b_outs

        outs = []
        for a_out, b_out in zip(a_outs, b_outs):
            out = output_aggregator.aggregate(a_out, b_out)
            outs.append(out)

        ctx.a_ins = a_ins   # for backward pass
        ctx.b_ins = b_ins
        ctx.a_outs = a_outs
        ctx.b_outs = b_outs

        return torch.cat(outs, dim=0)

    @staticmethod
    def backward(
        ctx, batched_grad_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None, None, None, None]:
        """ Backward function for batched PeftModule. """
        
        batched_peft_in, = ctx.saved_tensors
        grad_outs = torch.split(batched_grad_out, ctx.split_sizes, dim=0)

        a_grad_outs, b_grad_outs = [], []
        for grad_out in grad_outs:
            a_grad_out, b_grad_out = ctx.output_aggregator.reversed_aggregate(grad_out)
            a_grad_outs.append(a_grad_out)
            b_grad_outs.append(b_grad_out)
        
        b_grad_ins = batched_base_op_backward(
            outputs=ctx.b_outs, inputs=ctx.b_ins, grad_outputs=b_grad_outs, split_sizes=ctx.split_sizes,
        )

        raise NotImplementedError


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
