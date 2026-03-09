#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Basic modules for PEFT abstraction. """

from typing import List
from abc import ABC, abstractmethod

import torch
from torch import nn

from muxtune.global_envs import PeftType

__all__ = [ "PeftModule", "Adapter", "InputDispatcher", "OutputAggregator" ]


class PeftModule(nn.Module, ABC):
    """ Base class for PEFT module, including adapter, dispatch (input) and aggregate (output) rules, 
    and base op (of the LLM backbone). 
    
    Args:
        peft_type: The type of PEFT module (e.g., LoRA).
        base_op: The targeted operator of the LLM backbone for adapter attachment.
    """

    def __init__(self, peft_type: PeftType, base_op: nn.Module):
        super(PeftModule, self).__init__()

        self.type = peft_type
        self.base_op = base_op
        self.adapter = None
        self.input_dispatcher = None
        self.output_aggregator = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Abstracted forward workflow. """
        pass

    @abstractmethod
    def define_adapter(self):
        pass

    @abstractmethod
    def define_input_dispatcher(self):
        pass
    

class Adapter(nn.Module, ABC):
    """ Base class for adapter module. """

    def __init__(self):
        super(Adapter, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class InputDispatcher(ABC):
    """ Base class for input dispatcher. """

    def __init__(self):
        super(InputDispatcher, self).__init__()

    @abstractmethod
    def dispatch(self, x: torch.Tensor) -> torch.Tensor:
        pass


class OutputAggregator(ABC):
    """ Base class for output aggregator. """

    def __init__(self):
        super(OutputAggregator, self).__init__()

    @abstractmethod
    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        pass
