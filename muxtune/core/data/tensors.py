#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

from typing import List, Dict, Optional, Any
from collections import OrderedDict

import torch

__all__ = [ "ChunkedTensor", "MixedTensor" ]


class ChunkedTensor:
    """ Chunked tensor class to shard partial sequences along sequence dimension. 

    Args:
        value: The underlying tensor from the physical view.
        chunk_mask: List of booleans with the length of `b`. If a boolean is `True`, the 
            corresponded sequence is a sharded one; if `False`, is a complete one.
        layout: The layout of the tensor (Default: "s:b:h").
    """

    def __init__(self, value: torch.Tensor, chunk_mask: List[bool], layout: str = "s:b:h"):
        self.value = value
        self.chunk_mask = chunk_mask
        self.layout = layout


class MixedTensor(OrderedDict):
    """ Mixed tensor class across spatial-temporal (intra-stage) fused tasks.

    The mapping is peft group index (`int`) -> a list of data chunks (`List[ChunkedTensor]`) 
    or a `torch.Tensor`. When using `MixedTensor({ key: value })` with `chunk_tensors` as None,
    it falls back to standard `OrderedDict` from `int` to `torch.Tensor`.
    
    Args:
        chunked_tensors: Dict from peft group index to chunked tensors per group.
        chunk_config: Dict from peft group index to {'chunk_mask': [False, False], 
            'layout': 's:b:h'}).
    """

    def __init__(
        self,
        *args,
        chunked_tensors: Optional[Dict[int, List[torch.Tensor]]] = None,
        chunk_config: Optional[Dict[int, Dict[str, Any]]] = None,
        **kwargs,
    ):
        if chunked_tensors is not None:     # auto transform
            chunk_config = chunk_config or {}
            od = OrderedDict()
            for hybrid_task_index, tensors in chunked_tensors.items():
                cfg = chunk_config.get(hybrid_task_index, {})
                layout = cfg.get("layout", "s:b:h")
                batch_dim = layout.split(":").index("b")
                chunk_mask = cfg.get("chunk_mask", [False] * tensors[0].shape[batch_dim])
                od[hybrid_task_index] = [
                    ChunkedTensor(tensor, chunk_mask, layout) for tensor in tensors
                ]
            super().__init__(od, **kwargs)
        else:
            super().__init__(*args, **kwargs)
