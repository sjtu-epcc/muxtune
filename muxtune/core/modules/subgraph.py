#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Sub-graph implementation. """

from typing import Union, List
import enum

import torch
from torch import nn

from muxtune.core.data.tensors import ChunkedTensor
from muxtune.global_envs import stream_manager, global_configs

__all__ = [ "SubGraphType", "SubGraph", ]


class SubGraphType(enum.Enum):
    """ Subgraph type class. """

    NON_BASE_COMPUTE = enum.auto()
    COMMUNICATE = enum.auto()
    PEFT = enum.auto()


class SubGraph:
    """ Sub-graph class that contains consecutive non-base computation 
    operators, a single communication operator, or a base computation 
    operator hooked with a PeftModuleGroup.

    Each hybrid task has a dedicated `SubGraph` for the same operators.
    
    Args:
        subgraph_type: The type of the subgraph.
        prev: The previous subgraph in the launch schedule.
        next: The next subgraph in the launch schedule.
        hybrid_task_index: The global index of the hybrid task.
    """

    def __init__(
        self, subgraph_type: SubGraphType, prev: "SubGraph", 
        next: "SubGraph", hybrid_task_index: int, **kwargs,
    ):  
        self.subgraph_type = subgraph_type
        self.prev = prev
        self.next = next
        self.hybrid_task_index = hybrid_task_index
        self._modules = []
        self._kwargs = kwargs   # e.g., process group for communication operator
    
    def record(self, module: nn.Module):
        """ Record a single `nn.Module`. """
        self._modules.append(module)

    def forward(self, chunked_input_tensors: List[ChunkedTensor]) -> List[ChunkedTensor]:
        """ Forward chunked tensors of a hybrid task. """
    
        chunked_output_tensors = []
        if self.subgraph_type == SubGraphType.NON_BASE_COMPUTE:
            for chunk in chunked_input_tensors:
                tensor, chunk_mask, layout = chunk.value, chunk.chunk_mask, chunk.layout
                for module in self._modules:
                    tensor = module(tensor)
                chunked_output_tensors.append(ChunkedTensor(tensor, chunk_mask, layout))
        
        elif self.subgraph_type == SubGraphType.COMMUNICATE:
            assert len(self._modules) == 1, \
                f"A communication sub-graph should only consist of one primitive."
            seq_dim = global_configs.seq_dim
            comm_stream = stream_manager.get_communicate_stream()
            chunk_sizes = [chunk.value.shape[seq_dim] for chunk in chunked_input_tensors]
            chunk_cfgs = [(chunk.chunk_mask, chunk.layout) for chunk in chunked_input_tensors]
            merged_tensor = torch.cat(      # merge to communicate once
                [chunk.value for chunk in chunked_input_tensors], dim=seq_dim)
            with torch.cuda.stream(comm_stream):
                self._modules[0](merged_tensor, **self._kwargs)
            tensors = torch.split(merged_tensor, chunk_sizes, dim=seq_dim)
            chunked_output_tensors = [
                ChunkedTensor(tensor, chunk_cfgs[i][0], chunk_cfgs[i][1]) 
                    for i, tensor in enumerate(tensors)
            ]

        elif self.subgraph_type == SubGraphType.PEFT:
            assert len(self._modules) == 1, \
                f"A peft-enabled sub-graph should only consist of one PeftModule."
            for chunk in chunked_input_tensors:
                tensor, chunk_mask, layout = chunk.value, chunk.chunk_mask, chunk.layout
                tensor = self._modules[0].batched_forward(tensor, "prev_fw_func")
                chunked_output_tensors.append(ChunkedTensor(tensor, chunk_mask, layout))
        else:
            raise ValueError(f"Invalid subgraph type: {self.subgraph_type}")
        
        return chunked_output_tensors
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
