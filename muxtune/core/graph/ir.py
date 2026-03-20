#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Graph intermediate representation. """

from typing import (
    Union, Callable, Any, Tuple, List, Optional, Dict, Type)
import copy

import torch
import torch.fx as fx
from torch.fx.node import (Target, Argument, map_arg)
from torch.fx import GraphModule

__all__ = [
    "IRNode",
    "IRGraph",
]


class IRNode(fx.Node):
    """ IR node object to support automatic graph transformation.
    
    Args:
        graph: The `Graph` to which this `Node` should belong.
        name: The name of this `Node`.
        op: The opcode for this `Node`. Can be one of `placeholder`, `call_method`, `call_module`, 
            `call_function`, `get_attr`, `output`.
        target: The target this op should call (e.g., function object if `op` is `call_function`).
        args: The args to be passed to `target`.
        kwargs: The kwargs to be passed to `target`.
        return_type: The python type expression representing the type of the output of this node. 
                     This field can be used for annotation of values in the generated code or for 
                     other types of analyses.
    """

    def __init__(
        self,
        graph: Union["IRGraph", fx.Graph],
        name: str,
        op: str,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        return_type: Optional[Any] = None,
    ) -> None:
        super().__init__(graph, name, op, target, args, kwargs, return_type)
        
        self.layer_type = None
        self.input_shapes, self.output_shapes = None, None
        self.dtype = None
        self.node_pass, self.node_parallelism = None, None
        self.layout = None
        self.latency: Optional[float] = None
        self.stage_index: int = None    # which pipeline stage this node belongs to
        self.flops = None
        self.memory_access = None
        self.fw_node = None             # corresponded forward IR node for a backward one


class IRGraph(fx.Graph):
    """ IR graph object to support automatic graph transformation. 
    
    Args:
        owning_module: Torch module which this graph belongs to.
    """

    def __init__(
        self,
        owning_module: Optional[GraphModule] = None,
    ) -> None:
        """ Initialize a IR graph object. """
        super().__init__(owning_module)
        self._root: IRNode = IRNode(self, "", "root", "", (), {})
        self._insert = self._root.prepend   # iterating direction is `_prev` in `g.nodes`

    def node_copy(
        self, node: fx.Node, arg_transform: Callable[[IRNode], Argument] = ...
    ) -> IRNode:
        """ Construct an `IRNode` from the given `fx.Node`. """
        if isinstance(node, IRNode):
            args = map_arg(node.args, arg_transform)
            kwargs = map_arg(node.kwargs, arg_transform)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            result_node = self.create_node(
                node.op, node.target, args, kwargs, node.name, node.type,
            )
            result_node.meta = copy.copy(node.meta)
            return result_node
        else:
            assert isinstance(node, fx.Node)
            return super().node_copy(node, arg_transform)

    def create_node(
        self,
        op: str,
        target: Target,
        args: Optional[Tuple[Argument, ...]] = None,
        kwargs: Optional[Dict[str, Argument]] = None,
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> IRNode:
        assert op in (
            "call_function",
            "call_method",
            "get_attr",
            "call_module",
            "placeholder",
            "output",
        )
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        
        candidate = name if name is not None else self._target_to_str(target)
        name = self._graph_namespace.create_name(candidate, None)
        n = IRNode(self, name, op, target, args, kwargs, type_expr)
        self._graph_namespace.associate_name_with_obj(name, n)
        self._insert(n)
        self._len += 1
        return n
    
    def __deepcopy__(self, memo: Dict[int, Any]) -> "IRGraph":
        """Deepcopy the IRGraph object, including all nodes and custom attributes.
        
        Args:
            memo: A dictionary to keep track of already copied objects, 
                  preventing infinite recursion from circular references.
        Returns:
            A deepcopied instance of IRGraph.
        """
        
        # create a new IRGraph instance
        new_graph = IRGraph(owning_module=self.owning_module)
        # map original nodes to new nodes (to resolve node references)
        node_map: Dict[fx.Node, IRNode] = {}
        # copy all nodes (including custom attributes)
        for node in self.nodes:
            # copy the node itself using node_copy (ensures IRNode type and handles arg/kwarg references)
            # arg_transform replaces original nodes with their new copies in the new graph
            new_node = new_graph.node_copy(
                node,
                arg_transform=lambda n: node_map[n] if isinstance(n, fx.Node) else n
            )
            # explicitly copy custom attributes of IRNode (default deepcopy skips these)
            new_node.layer_type = copy.deepcopy(node.layer_type, memo)
            new_node.input_shapes = copy.deepcopy(node.input_shapes, memo)
            new_node.output_shapes = copy.deepcopy(node.output_shapes, memo)
            new_node.node_pass = copy.deepcopy(node.node_pass, memo)
            new_node.node_parallelism = copy.deepcopy(node.node_parallelism, memo)
            new_node.dtype = copy.deepcopy(node.dtype, memo)
            new_node.layout = copy.deepcopy(node.layout, memo)
            new_node.latency = copy.deepcopy(node.latency, memo)
            new_node.stage_index = copy.deepcopy(node.stage_index, memo)
            new_node.flops = copy.deepcopy(node.flops, memo)
            new_node.memory_access = copy.deepcopy(node.memory_access, memo)
            # handle fw_node reference (map to the corresponding node in the new graph)
            if node.fw_node is not None:
                new_node.fw_node = node_map[node.fw_node]
            else:
                new_node.fw_node = None
            
            # record the mapping from original node to new node
            node_map[node] = new_node
        
        # copy the _root node of IRGraph (ensure correct root reference)
        new_graph._root = node_map[self._root] if self._root in node_map else new_graph._root
        new_graph._insert = new_graph._root.prepend  # Maintain insertion logic
        return new_graph
