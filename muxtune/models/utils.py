#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

from typing import Dict, List, Any
from abc import ABC, abstractmethod

import torch
from torch import nn

__all__ = [
    "SubModuleBase",
    "WrappedTensor",
]


class SubModuleBase(torch.nn.Module, ABC):
    """ Base class for sub-module. """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, intermediate_: Dict[str, Any], 
        input_keywords: List[str] = ["hidden_states", ],
    ):
        """ Forward the sub-module with any user-defined codes.
        Any child class must adhere to the input argument format.
        
        Args:
            intermediate_(Dict[str, Any]): Intermediate object that includes all required 
                input keyword arguments, and those required by the latter sub-modules.
            input_keywords (List[str]): List of keywords required in this sub-module.
        """
        pass


class WrappedTensor:
    """
    A wrapper for tensors that enables caller functions to pass an indirect reference
    to callee functions. By wrapping the tensor, the caller's direct reference is removed,
    allowing the tensor to be garbage collected once the callee unwraps and frees it.
    """

    def __init__(self, tensor: torch.Tensor):
        self._wrapper = [tensor]

    def unwrap(self):
        """
        Returns the wrapped tensor while deleting the internal reference.
        Can only be called once.
        """
        if len(self._wrapper) == 0:
            raise RuntimeError(f"WrappedTensor has already been unwrapped")
        return self._wrapper.pop(0)
