#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Utility functions. """

import torch
from torch import nn

__all__ = [ "BackwardThrottler" ]


class BackwardThrottler(nn.Module):
    """ Backward throttler to cutoff loss computation and batched backward pass 
    for multiple adapters. For each adapter, its loss is seperately computed, while 
    the remaining backward pass is concurrently executed with other batched adapters.

    This module should be inserted before the final output layer of LLM backbone.
    """

    def __init__(self):
        super().__init__()
        self.last_output_tensors = {}       # microbatch index -> tensor
        self.detached_output_tensors = {}   # used for loss computation, detached from graph

    def forward(self, x: torch.Tensor, microbatch_index: int) -> torch.Tensor:
        self.last_output_tensors[microbatch_index] = x
        # detach
        detached_output_tensor = x.detach()
        detached_output_tensor.requires_grad = True
        detached_output_tensor.retain_grad()
        detached_output_tensor.grad = None
        self.detached_output_tensors[microbatch_index] = detached_output_tensor
        return detached_output_tensor

    def batched_backward(self, microbatch_index: int) -> torch.Tensor:
        raise NotImplementedError        
    