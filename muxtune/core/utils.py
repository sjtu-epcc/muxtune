#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Utility functions. """

import os
import functools
from collections import OrderedDict

import torch
from torch import nn

__all__ = [ "BackwardThrottler", "register_backward_throttler", "NonBaseOpModule", ]


class BackwardThrottler(nn.Module):
    """ Backward throttler to cutoff loss computation and batched backward pass 
    for multiple adapters. For each adapter, its loss is seperately computed, while 
    the remaining backward pass is concurrently executed with other batched adapters.

    This module should be inserted before the final output layer of LLM backbone, called
    at its post backward hook.
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

    def batched_backward(self, microbatch_index: int) -> None:
        last_output_tensor = self.last_output_tensors[microbatch_index]
        detached_output_tensor = self.detached_output_tensors[microbatch_index]
        assert detached_output_tensor.grad is not None, \
            "No gradient accumulated to the detached output tensor in BackwardThrottler."
        
        torch.autograd.backward(last_output_tensor, grad_tensors=detached_output_tensor.grad)
        del self.last_output_tensors[microbatch_index]
        del self.detached_output_tensors[microbatch_index]


def register_backward_throttler(
    module: nn.Module, attr_name: str = "backward_throttler", prev_fw_func_name: str = "prev_fw_func",
) -> nn.Module:
    """ Register backward throttler to the module, called at its post backward hook. Normally, 
    the target module should be the last output layer of the model.

    Args:
        attr_name: The attribute name of the backward throttler (default: "backward_throttler").
    """
    assert not hasattr(module, attr_name), "Backward throttler has already been hooked."
    bw_throttler = BackwardThrottler()
    setattr(module, attr_name, bw_throttler)
    setattr(module, prev_fw_func_name, module.forward)

    def __hooked_forward(module, *args, **kwargs) -> OrderedDict:
        bw_throttler = getattr(module, attr_name)
        input_tensors = args[0]
        module_op_func = getattr(module, prev_fw_func_name)
        microbatch_index = kwargs.get("microbatch_index", -1)
        output_tensors = OrderedDict()
        for (peft_module_index, input_tensor) in input_tensors.items():
            act = bw_throttler(input_tensor, microbatch_index)
            output_tensors[peft_module_index] = module_op_func(act)
        
        return output_tensors

    # Overriding a GraphModuleImpl forward freezes the forward call and 
    # later modifications on the graph will fail.
    # Reference: https://pytorch.slack.com/archives/C3PDTEV8E/p1705929610405409
    if "GraphModuleImpl" in str(type(module)):
        module.__class__.forward = functools.update_wrapper(
            functools.partial(__hooked_forward, module), 
            getattr(module, prev_fw_func_name),
        )
    else:
        module.forward = functools.update_wrapper(
            functools.partial(__hooked_forward, module), 
            getattr(module, prev_fw_func_name),
        )

    def __post_backward_hook(module, grad_input, grad_output) -> torch.Tensor:
        print(grad_input)
        print(grad_output)
        exit(0)

    _ = module.register_full_backward_hook(__post_backward_hook)


class NonBaseOpModule:
    """ Non-base operator module to enable spatial-temporal execution of colocated PEFT tasks.
    
    This module should be hooked to every non-baseop nn.Module in the model, taking an 
    `OrderedDict` object as the input, and sequentially executing each batched tensor.
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

        def __hooked_forward(module, *args, **kwargs) -> OrderedDict:
            input_tensors = args[0]
            nonbase_op_func = getattr(module, prev_fw_func_name)
            output_tensors = OrderedDict()
            for (peft_module_index, input_tensor) in input_tensors.items():
                output_tensors[peft_module_index] = nonbase_op_func(input_tensor)
            
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
        