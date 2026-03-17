#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import torch

__all__ = [
    "WrappedTensor",
]


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
