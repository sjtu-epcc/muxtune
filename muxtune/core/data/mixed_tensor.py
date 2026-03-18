#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

from typing import List
from collections import OrderedDict

import torch
from torch import nn

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
    """ Mixed tensor class across spatial-temporal fused tasks.
    
    The mapping is peft group index (`int`) -> a list of data chunks (`List[ChunkedTensor]`). 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _create_from_ordered_dict(self):
        # Add here
        raise NotImplementedError
