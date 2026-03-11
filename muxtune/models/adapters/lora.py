#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" LoRA implementations. """

from typing import Union
import math

import torch
from torch import nn

from muxtune.core.peft_modules import Adapter, InputDispatcher, OutputAggregator
from muxtune.global_envs import PeftType

__all__ = [ "LoraAdapter", "LoraInputDispatcher", "LoraOutputAggregator" ]


class LoraAdapter(Adapter):
    """ LoRA adapter. """

    def __init__(
        self, name: str, device: str, dtype: torch.dtype = torch.float16, 
        lora_r: int = 8, lora_alpha: int = 16, in_features: int = None, 
        out_features: int = None, lora_dropout: float = 0.05, init_weights: bool = True,
        init_lora_B_normal: bool = False,
    ):
        super().__init__(name, device, dtype)
        self.type = PeftType.LoRA
        self.r = lora_r
        self.alpha = lora_alpha
        self.scaling = lora_alpha / lora_r
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else lambda x: x

        # Parameters
        self.lora_A = nn.Linear(in_features, lora_r, bias=False, device=torch.device(device), dtype=dtype)
        self.lora_B = nn.Linear(lora_r, out_features, bias=False, device=torch.device(device), dtype=dtype)
        if init_weights:
            # Initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            if not init_lora_B_normal:
                nn.init.zeros_(self.lora_B.weight)
            else:
                nn.init.normal_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == self.dtype, f"Input dtype {x.dtype} does not match adapter dtype {self.dtype}."
        # return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return self.lora_B(self.lora_A(x)) * self.scaling


class LoraInputDispatcher(InputDispatcher):
    """ LoRA input dispatcher. """

    def __init__(self):
        super().__init__()

    def dispatch(self, peft_in: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        return peft_in, peft_in     # adapter and base op share the same input
    
    def reversed_dispatch(self, adapter_grad_in, base_grad_in):
        return adapter_grad_in + base_grad_in   # the input gradient is the sum of adapter and base op input gradients


class LoraOutputAggregator(OutputAggregator):
    """ LoRA output aggregator. """

    def __init__(self):
        super().__init__()

    def aggregate(self, adapter_out: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        return adapter_out + base_out   # adapter output is added to the base op output

    def reversed_aggregate(self, grad_out: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        return grad_out, grad_out   # the output gradient is propagated to both adapter and base op
