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
    "batched_base_op_forward", "batched_base_op_backward", 
    "batched_adapter_forward", "batched_adapter_backward",
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
def batched_base_op_backward(
    outputs: List[torch.Tensor], inputs: List[torch.Tensor], grad_outputs: List[torch.Tensor], split_sizes: List[int],
) -> List[torch.Tensor]:
    """ Batched backward for base op with multi-adapter grad outputs. """
    
    batched_output = torch.cat(outputs, dim=0)
    batched_input = torch.cat(inputs, dim=0)
    batched_grad_out = torch.cat(grad_outputs, dim=0)
    batched_grad_in = torch.autograd.grad([batched_output], inputs=[batched_input], grad_outputs=[batched_grad_out])[0]
    return torch.split(batched_grad_in, split_sizes, dim=0)


@torch.no_grad()
def batched_adapter_forward(
    ctx, peft_type: PeftType, inputs: List[torch.Tensor], adapters: List[nn.Module], 
) -> List[torch.Tensor]:
    """ Batched forward for adapters of the same PEFT type. """

    if peft_type == PeftType.LoRA:
        return _batched_lora_forward(ctx, inputs, adapters)
    else:
        raise NotImplementedError(f"Unsuported PEFT type {peft_type} for batched forward.")


@torch.no_grad()
def batched_adapter_backward(
    ctx, peft_type: PeftType, outputs: List[torch.Tensor], inputs: List[torch.Tensor], grad_outputs: List[torch.Tensor], 
    adapters: List[nn.Module], 
) -> List[torch.Tensor]:
    """ Batched backward for adapters of the same PEFT type. """

    if peft_type == PeftType.LoRA:
        return _batched_lora_backward(ctx, outputs, inputs, grad_outputs, adapters)
    else:
        raise NotImplementedError(f"Unsuported PEFT type {peft_type} for batched backward.")


@torch.no_grad()
def _batched_lora_forward(
    ctx, inputs: List[torch.Tensor], adapters: List[nn.Module],
) -> List[torch.Tensor]:
    """ Batched forward for LoRA adapters. """

    assert all([inputs[i].dtype == adapters[i].dtype for i in range(len(inputs))]), \
        "Input dtypes do not match adapter dtypes."
    inputs = [adapter.dropout(input_) for input_, adapter in zip(inputs, adapters)]
    lora_a_outs = triton_grouped_gemm(inputs, [adapter.lora_A.weight.T.contiguous() for adapter in adapters])
    lora_b_outs = triton_grouped_gemm(lora_a_outs, [adapter.lora_B.weight.T.contiguous() for adapter in adapters])
    scaled_lora_outs = [lora_b_out * adapter.scaling for lora_b_out, adapter in zip(lora_b_outs, adapters)]
    ctx.lora_a_outs = lora_a_outs   # for backward pass
    return scaled_lora_outs


@torch.no_grad()
def _batched_lora_backward(
    ctx, outputs: List[torch.Tensor], inputs: List[torch.Tensor], grad_outputs: List[torch.Tensor], 
    adapters: List[nn.Module], 
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """ Batched backward for LoRA adapters. """

    lora_b_grad_outs = [grad_out / adapter.scaling for grad_out, adapter in zip(grad_outputs, adapters)]
    lora_b_grad_ins = triton_grouped_gemm(lora_b_grad_outs, [adapter.lora_B.weight.contiguous() for adapter in adapters])
    lora_b_grad_weights = triton_grouped_gemm(lora_b_grad_outs, [adapter.lora_A.weight.contiguous() for adapter in adapters])

