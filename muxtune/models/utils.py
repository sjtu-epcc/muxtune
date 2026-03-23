#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn

__all__ = [
    "WrappedTensor",
]


@dataclass
class SubModuleData:
    """ Data collection between sub-modules. 
    
    Each sub-module strictly takes a `SubModuleData` as its input, computes, then 
    produces another `SubModuleData` as its output (input to the next sub-module).
    """

    input_ids: nn.Tensor = None
    
    position_ids: nn.Tensor = None

    attention_mask: nn.Tensor = None

    decoder_input: nn.Tensor = None

    labels: nn.Tensor = None

    loss_mask: nn.Tensor = None

    runtime_gather_output: Optional[bool] = None    

    hidden_states: nn.Tensor = None

    bias: nn.Tensor = None


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
