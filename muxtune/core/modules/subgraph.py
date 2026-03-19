#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Sub-graph implementation. """

import enum

import torch
from torch import nn

__all__ = [ "SubGraphType", "SubGraph", ]


class SubGraphType(enum.Enum):
    """ Subgraph type class. """

    COMPUTE = enum.auto()
    COMMUNICATE = enum.auto()
    ADAPTER = enum.auto()


class SubGraph:
    """ Sub-graph class that contains consecutive computation operators, 
    a single communication operator, or a batch of adapters.

    Each hybrid task has a dedicated `SubGraph` for the same operators.
    
    Args:
        subgraph_type: The type of the subgraph.
        prev: The previous subgraph in the launch schedule.
        next: The next subgraph in the launch schedule.
        hybrid_task_index: The global index of the hybrid task.
    """

    def __init__(
        self, subgraph_type: SubGraphType, prev: "SubGraph", 
        next: "SubGraph", hybrid_task_index: int,
    ):  
        self.subgraph_type = subgraph_type
        self.prev = prev
        self.next = next
        self.hybrid_task_index = hybrid_task_index
        self._modules = []
        self._aux_input_buffer = {}     # module name -> (operation, tensor)
    
    def record(self, module: nn.Module):
        """ Record a single `nn.Module`. """
        self._modules.append(module)
    
    def set_auxiliary_input(
        self, target_module_name: str, tensor: torch.Tensor, 
        operation: str = "task_wise_add",
    ):
        """ Explicitly set one auxiliary input operation to the output tensor 
        of a single module in this subgraph. """
        self._aux_input_buffer[target_module_name] = (operation, tensor)

    def forward(self, batched_input: torch.Tensor):
        if self.subgraph_type == SubGraphType.COMPUTE:
            act = batched_input
            for module in self._modules:
                act = module(act)
                if module.name in self._aux_input_buffer:
                    (operation, aux_in) = self._aux_input_buffer[module.name]

                raise NotImplementedError

        elif self.subgraph_type == SubGraphType.COMMUNICATE:
            pass
        elif self.subgraph_type == SubGraphType.ADAPTER:
            pass
        else:
            raise ValueError(f"Invalid subgraph type: {self.subgraph_type}")
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
