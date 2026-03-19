#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
from typing import Tuple, List, Dict
import time
import functools
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn

from muxtune.core.data.mixed_tensor import ChunkedTensor, MixedTensor
from muxtune.global_envs import global_configs

__all__ = [ "BackwardThrottler", "NonBaseOpModule" ]


class BackwardThrottler(nn.Module):
    """ Backward throttler to cutoff loss computation and batched backward pass 
    for multiple tasks. For each task, its loss is seperately computed, while 
    the remaining backward pass is concurrently executed with other batched tasks.

    This module should be hooked to the final output layer of LLM backbone. 
    """

    def __init__(self):
        super().__init__()
        self.last_output_tensors = {}     # microbatch index -> peft_group_index -> tensor
        self.detached_output_tensors = {}   # used for loss computation, detached from graph
        self.nonbase_op = None

    def forward(self, x: torch.Tensor, peft_group_index: int) -> torch.Tensor:
        forced_adapter_name = os.environ.get("FORCED_ADAPTER_NAME_DEBUG", None)
        if forced_adapter_name:
            return x    # single adapter for debug
        
        microbatch_index = global_configs.current_microbatch_index
        if microbatch_index not in self.last_output_tensors:
            self.last_output_tensors[microbatch_index] = {}
        self.last_output_tensors[microbatch_index][peft_group_index] = x
        # detach
        detached_output_tensor = x.detach()
        detached_output_tensor.requires_grad = True
        detached_output_tensor.retain_grad()
        detached_output_tensor.grad = None
        if microbatch_index not in self.detached_output_tensors:
            self.detached_output_tensors[microbatch_index] = {}
        self.detached_output_tensors[microbatch_index][peft_group_index] = detached_output_tensor
        return detached_output_tensor

    def backward(self, losses: MixedTensor[int, torch.tensor], peft_group_index: int) -> None:
        """ Backward losses for batched tasks, concatenate input gradients, and continune 
        backward pass in a batched manner.
        
        Args:
            losses: Loss scalars for each batched task.
            peft_group_index: Index of the hybrid task.
        """

        microbatch_index = global_configs.current_microbatch_index
        last_output_tensor = self.last_output_tensors[microbatch_index][peft_group_index]
        for (_, loss) in losses.items():
            # the graph is cutoff before detached_output_tensor, so `retain_graph` might not
            # cause too much memory footprint.
            torch.autograd.backward(loss, grad_tensors=None, retain_graph=True)
        
        detached_output_tensor = self.detached_output_tensors[microbatch_index][peft_group_index]
        assert detached_output_tensor.grad is not None, \
            "No gradient accumulated to the detached output tensor in BackwardThrottler."

        torch.autograd.backward(last_output_tensor, grad_tensors=detached_output_tensor.grad)
        del self.last_output_tensors[microbatch_index][peft_group_index]
        del self.detached_output_tensors[microbatch_index][peft_group_index]

    def hook_to_nonbase_op(self, nonbase_op: nn.Module, attr_name: str = "backward_throttler", 
        prev_fw_func_name: str = "prev_fw_func",
    ) -> nn.Module:
        """ Hook to the non-base operator (normally the last output layer), called at its post 
        backward hook.

        Args:
            attr_name: The attribute name of the backward throttler (default: "backward_throttler").
            prev_fw_func_name: The attribute name of the original forward function of the 
                target module after hooked (default: "prev_fw_func"). 
        """
        assert self.nonbase_op is None, "Backward throttler has already been hooked."
        setattr(nonbase_op, attr_name, self)
        setattr(nonbase_op, prev_fw_func_name, nonbase_op.forward)
        self.nonbase_op = nonbase_op

        def __hooked_forward(module, *args, **kwargs) -> MixedTensor:
            bw_throttler = getattr(module, attr_name)
            input_tensors = args[0]
            module_op_func = getattr(module, prev_fw_func_name)
            output_tensors = MixedTensor()
            for (peft_group_index, chunked_input_tensors) in input_tensors.items():
                chunked_output_tensors = []
                for chunk in chunked_input_tensors:
                    chunk_mask, layout = chunk.chunk_mask, chunk.layout
                    output_tensor = bw_throttler(chunk.value, peft_group_index)
                    output_tensor = module_op_func(output_tensor)
                    chunked_output_tensors.append(ChunkedTensor(output_tensor, chunk_mask, layout))
                output_tensors[peft_group_index] = chunked_output_tensors
            
            return output_tensors

        # Overriding a GraphModuleImpl forward freezes the forward call and 
        # later modifications on the graph will fail.
        # Reference: https://pytorch.slack.com/archives/C3PDTEV8E/p1705929610405409
        if "GraphModuleImpl" in str(type(nonbase_op)):
            nonbase_op.__class__.forward = functools.update_wrapper(
                functools.partial(__hooked_forward, nonbase_op), 
                getattr(nonbase_op, prev_fw_func_name),
            )
        else:
            nonbase_op.forward = functools.update_wrapper(
                functools.partial(__hooked_forward, nonbase_op), 
                getattr(nonbase_op, prev_fw_func_name),
            )
        return nonbase_op


class NonBaseOpModule:
    """ Non-base operator module to enable spatial-temporal execution of colocated PEFT tasks.
    
    This module should be hooked to every non-baseop nn.Module in the model, taking an 
    `MixedTensor` object as the input, and sequentially executing each batched tensor.
    """

    def __init__(self):
        self.nonbase_op = None
    
    def hook_to_nonbase_op(
        self, nonbase_op: nn.Module, attr_name: str = "nonbase_op_module", 
        prev_fw_func_name: str = "prev_fw_func",
    ) -> nn.Module:
        """ Hook to the non-base operator. 
        
        Args:
            nonbase_op: The non-base operator to be hooked.
            attr_name: The attribute name of the non-base operator module 
                (default: "nonbase_op_module").
            prev_fw_func_name: The attribute name of the original forward function of the 
                non-base operator after hooked (default: "prev_fw_func").
        """
        assert self.nonbase_op is None, "Non-base operator module has already been hooked."
        setattr(nonbase_op, attr_name, self)
        setattr(nonbase_op, prev_fw_func_name, nonbase_op.forward)
        self.nonbase_op = nonbase_op

        def __hooked_forward(module, *args, **kwargs) -> MixedTensor:
            input_tensors = args[0]
            nonbase_op_func = getattr(module, prev_fw_func_name)
            output_tensors = MixedTensor()
            for (peft_group_index, chunked_input_tensors) in input_tensors.items():
                chunked_output_tensors = []
                for chunk in chunked_input_tensors:
                    chunk_mask, layout = chunk.chunk_mask, chunk.layout
                    output_tensor = nonbase_op_func(chunk.value)
                    chunked_output_tensors.append(ChunkedTensor(output_tensor, chunk_mask, layout))
                output_tensors[peft_group_index] = chunked_output_tensors
            
            return output_tensors
        
        # Overriding a GraphModuleImpl forward freezes the forward call and 
        # later modifications on the graph will fail.
        # Reference: https://pytorch.slack.com/archives/C3PDTEV8E/p1705929610405409
        if "GraphModuleImpl" in str(type(nonbase_op)):
            nonbase_op.__class__.forward = functools.update_wrapper(
                functools.partial(__hooked_forward, nonbase_op), 
                getattr(nonbase_op, prev_fw_func_name),
            )
        else:
            nonbase_op.forward = functools.update_wrapper(
                functools.partial(__hooked_forward, nonbase_op), 
                getattr(nonbase_op, prev_fw_func_name),
            )

        return nonbase_op
