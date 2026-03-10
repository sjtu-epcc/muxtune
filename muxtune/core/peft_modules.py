#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Basic modules for PEFT abstraction. """

from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
from torch import nn

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
        self.input_dispatchers = {}
        self.output_aggregators = {}

    def register_one_adapter(
        self, adapter: "Adapter", input_dispatcher: "InputDispatcher", output_aggregator: "OutputAggregator",
    ):
        """ Register one adapter. """
        self.adapters[adapter.name] = adapter
        self.input_dispatchers[adapter.name] = input_dispatcher
        self.output_aggregators[adapter.name] = output_aggregator

        assert self.adapters[adapter.name].device == self.config.device, \
            f"Adapter ({adapter.name}) device {self.adapters[adapter.name].device} does not match " + \
            f"PEFT module device {self.config.device}."
        assert self.adapters[adapter.name].dtype == self.config.dtype, \
            f"Adapter ({adapter.name}) dtype {self.adapters[adapter.name].dtype} does not match " + \
            f"PEFT module dtype {self.config.dtype}."
    
    def _single_forward(self, adapter_name: str, peft_in: torch.Tensor) -> torch.Tensor:
        """ Forward a single adapter. """

        a_in, b_in = self.input_dispatchers[adapter_name].dispatch(peft_in)
        a_out = self.adapters[adapter_name](a_in) if a_in is not None else None
        b_out = self.base_op(b_in) if b_in is not None else None
        a_out = self.adapters[adapter_name](b_out) if a_out is None else a_out  # maybe forward from base output
        b_out = self.base_op(a_out) if b_out is None else b_out                 # maybe forward from adapter output
        return self.output_aggregators[adapter_name].aggregate(a_out, b_out)

    def batched_forward(self, peft_in: torch.Tensor) -> torch.Tensor:
        """ Forward all adapters in a batched manner. 
        
        Args:
            peft_in (torch.Tensor): Multi-task input tensor batched along batch dimension.
        """
        raise NotImplementedError


class Adapter(nn.Module, ABC):
    """ Base class for adapter module. 
    
    Args:
        name: Unique identifier of the adapter, formatted as  
            "[base_op_module_name]::[task_name]" (e.g., "qkv_proj::task_0").
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
