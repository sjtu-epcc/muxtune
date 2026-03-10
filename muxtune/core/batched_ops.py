#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Batched operations for PEFT adapters. """

import torch


class _BatchedPeftForward(torch.autograd.Function):
    """ Autograd function for batched forward of multiple adapters. """

    @staticmethod
    def forward(ctx, batched_input: torch.Tensor) -> torch.Tensor:
        """ Forward function for batched adapters. """
        raise NotImplementedError

    @staticmethod
    def backward(ctx, batched_grad_out: torch.Tensor) -> torch.Tensor:
        """ Backward function for batched adapters. """
        raise NotImplementedError
