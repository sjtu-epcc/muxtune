#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Model compiling and graph partitioning implementation. """

from typing import Tuple, Dict, List, Any

import torch
from torch import nn
import torch._dynamo as dynamo

from muxtune.core.graph.ir import IRNode, IRGraph

__all__ = [ "PartitionPlan", "compile_model_and_generate_partition_plan", ]


class PartitionPlan:
    """ Model graph partition plan. """

    def __init__(self):
        self._map = {}  # subgraph index -> List[module name]


def compile_model_and_generate_partition_plan(
    backbone: nn.Module,
    base_input_args: Tuple[Any],
    base_input_kwargs: Dict[str, Any],
    peft_base_op_names: List[str],
) -> PartitionPlan:
    """ Compile the model into IR graph and generate partition plan. 

    Workflow: compile backbone -> mark PEFT modules -> insert communication
        -> partition graph -> construct partition plan.
    
    Args:
        backbone: The backbone model to be compiled.
        base_input_args: Input arguments of model `.forward()`.
        base_input_kwargs: Input keyword arguments of model `.forward()`.
        peft_base_op_names: Module names of PEFT base operators.
    """

    graph = _transform_to_ir_graph(backbone, *base_input_args, **base_input_kwargs)
    graph = _mark_peft_base_operators(graph, peft_base_op_names)
    raise NotImplementedError


def _mark_peft_base_operators(graph: IRGraph, peft_base_op_names: List[str]) -> IRGraph:
    """ Mark PEFT base operators in the model graph. """
    raise NotImplementedError


def _transform_to_ir_graph(model: nn.Module, *input_args, **input_kwargs) -> IRGraph:
    """ Transform model into IR graph, tracing with Python-level JIT compiler. """
   
    fx_module = dynamo.export(model)(*input_args, **input_kwargs)[0]
    fx_graph = fx_module.graph
    # construct ir graph
    graph = IRGraph(fx_graph.owning_module)
    output_vals = graph.graph_copy(fx_graph, val_map={}, return_output_node=True)
    output_val, old_output_node = output_vals
    graph.output(output_val, type_expr=getattr(old_output_node, "type", None))   # Insert output node
    return graph
