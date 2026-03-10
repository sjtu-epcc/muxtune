#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Batched operations. """

from typing import List, Tuple
import torch
from torch import nn

from muxtune.triton.grouped_gemm import triton_grouped_gemm
from muxtune.global_envs import PeftType

__all__ = [
    "batched_base_op_forward", "batched_adapter_forward",
]


@torch.no_grad()
def batched_base_op_forward(
    base_op: nn.Module, inputs: List[torch.Tensor], split_sizes: List[int],
) -> List[torch.Tensor]:
    """ Batched forward for base op with multi-adapter inputs. """

    batched_input = torch.cat(inputs, dim=0)
    batched_output = base_op(batched_input)
    return torch.split(batched_output, split_sizes, dim=0)


@torch.no_grad()
def batched_adapter_forward(
    peft_type: PeftType, inputs: List[torch.Tensor], adapters: List[nn.Module], 
) -> List[torch.Tensor]:
    """ Batched forward for adapters of the same PEFT type. """

    if peft_type == PeftType.LoRA:
        return _batched_lora_forward(inputs, adapters)
    else:
        raise NotImplementedError(f"Unsuported PEFT type {peft_type} for batched forward.")


@torch.no_grad()
def _batched_lora_forward(
    inputs: List[torch.Tensor], adapters: List[nn.Module],
) -> List[torch.Tensor]:
    """ Batched forward for LoRA adapters. """

    assert all([inputs[i].dtype == adapters[i].dtype for i in range(len(inputs))]), \
        "Input dtypes do not match adapter dtypes."
    
    # Dropout
    inputs = [adapter.dropout(input_) for input_, adapter in zip(inputs, adapters)]
    # Lora A
    lora_a_outs = triton_grouped_gemm(inputs, [adapter.lora_A.weight.T.contiguous() for adapter in adapters])
    # Lora B
    lora_b_outs = triton_grouped_gemm(lora_a_outs, [adapter.lora_B.weight.T.contiguous() for adapter in adapters])
    # Scaling
    scaled_lora_outs = [lora_b_out * adapter.scaling for lora_b_out, adapter in zip(lora_b_outs, adapters)]

    return scaled_lora_outs
