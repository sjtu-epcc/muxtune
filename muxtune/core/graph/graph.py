#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Model graph implementation. """

from typing import Union, List, Any
import enum

import torch
from torch import nn

from muxtune.core.graph.partition import PartitionPlan
from muxtune.core.data.tensors import ChunkedTensor
from muxtune.global_envs import stream_manager, global_configs

__all__ = [ "ModelGraphManager", "SubGraphType", "SubGraph", ]


class ModelGraphManager:
    """ Model graph manager. 
    
    Each hybrid task maintains different sub-graphs for the same set 
    of model operators. All hybrid tasks share the same partitioning 
    plan of the PEFT model.
    """

    def __init__(self, hybrid_task_indices: List[int]):
        self.hybrid_task_indices = hybrid_task_indices
        self._subgraphs = {}    # hybrid task index -> List[SubGraph]
    
    def construct_subgraphs(self, model: nn.Module, partition_plan: PartitionPlan):
        """ Construct subgraphs from hybrid tasks and partitioning plan. 
        
        Args:
            model (nn.Module): The backbone model.
            partition_plan (PartitionPlan): Model graph partition plan for all hybrid tasks.
        """
        raise NotImplementedError
    
    def get_subgraph(self, subgraph_index: int, hybrid_task_index: int) -> "SubGraph":
        assert hybrid_task_index in self._subgraphs, \
            f"Invalid hybrid task index: {hybrid_task_index}"
        assert -1 < subgraph_index < len(self._subgraphs[0]), \
            f"Invalid subgraph index: {subgraph_index}"
        return self._subgraphs[hybrid_task_index][subgraph_index]


class SubGraphType(enum.Enum):
    """ Subgraph type class. """

    NON_PEFT_COMPUTE = enum.auto()
    COMMUNICATE = enum.auto()
    PEFT_COMPUTE = enum.auto()


class SubGraph:
    """ Sub-graph class that contains consecutive non-base computation 
    operators, a single communication operator, or a base computation 
    operator hooked with a PeftModuleGroup.

    Each hybrid task has a dedicated `SubGraph` for the same operators.
    
    Args:
        type (SubGraphType): The type of the subgraph.
        subgraph_index: The global index of a single subgraph in the model 
            graph, same across hybrid tasks.
        hybrid_task_index: The global index of the hybrid task.
    
    Examples:
        >>> # pesudocode for how sub-graphs of the same index are executed
        >>> # input: subgraph_index (int), inputs (MixedTensor)
        >>> outputs = MixedTensor()
        >>> for (hybrid_task_index, chunked_inputs) in inputs.items():
        >>>     subgraph = get_subgraph(subgraph_index, hybrid_task_index)
        >>>     outputs[hybrid_task_index] = subgraph(chunked_inputs)
        >>> return outputs
    """

    def __init__(
        self, type: SubGraphType, subgraph_index: int, hybrid_task_index: int, **kwargs,
    ):  
        self.type = type
        self.subgraph_index = subgraph_index
        self.hybrid_task_index = hybrid_task_index
        self.prev = None
        self.next = None
        self._modules = []
        self._kwargs = kwargs   # e.g., process group for communication operator
    
    def record(self, module: nn.Module):
        """ Record a single `nn.Module` into the sub-graph. """
        self._modules.append(module)

    def forward(self, chunked_input_tensors: List[ChunkedTensor]) -> List[ChunkedTensor]:
        chunked_output_tensors = []
        if self.type == SubGraphType.NON_PEFT_COMPUTE:
            if self.prev.type == SubGraphType.COMMUNICATE:
                # wait for communication complete
                compute_stream = stream_manager.get_compute_stream()
                prev_subgraph_index = self.prev.subgraph_index
                event_name = f"subgraph_{prev_subgraph_index}::hybrid_task_{self.hybrid_task_index}"
                stream_manager.sync_wait_event(event_name, compute_stream)

            for chunk in chunked_input_tensors:
                tensor, chunk_mask, layout = chunk.value, chunk.chunk_mask, chunk.layout
                for module in self._modules:
                    tensor = module(tensor)
                chunked_output_tensors.append(ChunkedTensor(tensor, chunk_mask, layout))
        
        elif self.type == SubGraphType.COMMUNICATE:
            assert len(self._modules) == 1, \
                f"A communication sub-graph should only consist of one primitive."
            seq_dim = global_configs.seq_dim
            comm_stream = stream_manager.get_communicate_stream()
            chunk_sizes = [chunk.value.shape[seq_dim] for chunk in chunked_input_tensors]
            chunk_cfgs = [(chunk.chunk_mask, chunk.layout) for chunk in chunked_input_tensors]
            # merge to communicate once
            merged_tensor = torch.cat(
                [chunk.value for chunk in chunked_input_tensors], dim=seq_dim)
            
            with torch.cuda.stream(comm_stream):
                self._modules[0](merged_tensor, **self._kwargs)
            event_name = f"subgraph_{self.subgraph_index}::hybrid_task_{self.hybrid_task_index}"
            stream_manager.record_wait_event(event_name, comm_stream)
            
            tensors = [
                t.contiguous() for t in torch.split(merged_tensor, chunk_sizes, dim=seq_dim)
            ]   # split(..., dim=1) returns non-contiguous views
            chunked_output_tensors = [
                ChunkedTensor(tensor, chunk_cfgs[i][0], chunk_cfgs[i][1]) 
                    for i, tensor in enumerate(tensors)
            ]

        elif self.type == SubGraphType.PEFT_COMPUTE:
            assert len(self._modules) == 1, \
                f"A peft-enabled sub-graph should only consist of one base operator."
            chunked_output_tensors = self._modules[0](
                chunked_input_tensors, hybrid_task_index=self.hybrid_task_index)
        else:
            raise ValueError(f"Invalid subgraph type: {self.type}")
        
        return chunked_output_tensors
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
