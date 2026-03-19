#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Batched operations. """

from typing import List, Tuple
import torch
from torch import nn

from muxtune.triton.grouped_gemm import triton_grouped_gemm
from muxtune.global_envs import global_configs, PeftType

__all__ = [
    "batched_base_op_forward", "batched_base_op_backward", 
    "batched_adapter_forward", "batched_adapter_backward",
]


@torch.no_grad()
def batched_base_op_forward(
    base_op: nn.Module, inputs: List[torch.Tensor], split_sizes: List[int], 
    prev_fw_func_name: str = "forward",
) -> List[torch.Tensor]:
    """ Batched forward for base op with multi-adapter inputs. """

    base_op_func = getattr(base_op, prev_fw_func_name)
    batched_input = torch.cat(inputs, dim=global_configs.batch_dimension)
    batched_output = base_op_func(batched_input)
    return torch.split(batched_output, split_sizes, dim=global_configs.batch_dimension)


@torch.no_grad()
def batched_base_op_backward(
    base_op: nn.Module, grad_outputs: List[torch.Tensor], split_sizes: List[int],
) -> List[torch.Tensor]:
    """ Batched backward for base op with multi-adapter grad outputs. """
    
    batched_grad_out = torch.cat(grad_outputs, dim=global_configs.batch_dimension)
    batched_grad_in = torch.matmul(batched_grad_out, base_op.weight.contiguous())
    return torch.split(batched_grad_in, split_sizes, dim=global_configs.batch_dimension)


@torch.no_grad()
def batched_adapter_forward(
    ctx, peft_type: PeftType, inputs: List[torch.Tensor], adapters: List[nn.Module], 
) -> List[torch.Tensor]:
    """ Batched forward for adapters of the same PEFT type. """

    input_shapes = []
    for inp in inputs:
        assert inp.dim() == 3   # [s, b, h]
        input_shapes.append(inp.shape)
        inp = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])

    if peft_type == PeftType.LoRA:
        outputs = _batched_lora_forward(ctx, inputs, adapters)
    else:
        raise NotImplementedError(f"Unsuported PEFT type {peft_type} for batched forward.")
    
    for i, out in enumerate(outputs):
        out = out.view(input_shapes[i])
    return outputs


@torch.no_grad()
def batched_adapter_backward(
    ctx, peft_type: PeftType, adapters: List[nn.Module], grad_outputs: List[torch.Tensor], 
) -> List[torch.Tensor]:
    """ Batched backward for adapters of the same PEFT type. """

    grad_output_shapes = []
    for grad_out in grad_outputs:
        assert grad_out.dim() == 3
        grad_output_shapes.append(grad_out.shape)
        grad_out = grad_out.view(grad_out.shape[0] * grad_out.shape[1], grad_out.shape[2])

    if peft_type == PeftType.LoRA:
        grad_inputs = _batched_lora_backward(ctx, adapters, grad_outputs)
    else:
        raise NotImplementedError(f"Unsuported PEFT type {peft_type} for batched backward.")

    for i, grad_in in enumerate(grad_inputs):
        grad_in = grad_in.view(grad_output_shapes[i])
    return grad_inputs

@torch.no_grad()
def _batched_lora_forward(
    ctx, inputs: List[torch.Tensor], adapters: List[nn.Module],
) -> List[torch.Tensor]:
    """ Batched forward for LoRA adapters. """

    assert all([inputs[i].dtype == adapters[i].dtype for i in range(len(inputs))]), \
        "Input dtypes do not match adapter dtypes."
    
    # dropout_keep_probs = [1 - adapter.dropout.p for adapter in adapters]
    # dropout_masks = [
    #     torch.bernoulli(torch.full_like(input_, keep_prob))     
    #         for input_, keep_prob in zip(inputs, dropout_keep_probs)
    # ]
    # inputs = [(input_ * mask) / keep_prob for input_, mask, keep_prob in zip(inputs, dropout_masks, dropout_keep_probs)]
    # FIXME(chunyu): Currently we skip dropout. 
    lora_a_outs = triton_grouped_gemm(
        [inp.contiguous() for inp in inputs], [adapter.lora_A.weight.T.contiguous() for adapter in adapters]
    )
    lora_b_outs = triton_grouped_gemm(lora_a_outs, [adapter.lora_B.weight.T.contiguous() for adapter in adapters])
    scaled_lora_outs = [lora_b_out * adapter.scaling for lora_b_out, adapter in zip(lora_b_outs, adapters)]
    # ctx.lora_dropout_masks = dropout_masks   # for backward pass
    # ctx.lora_keep_probs = dropout_keep_probs
    ctx.lora_a_ins = inputs
    ctx.lora_a_outs = lora_a_outs
    return scaled_lora_outs


@torch.no_grad()
def _batched_lora_backward(
    ctx, adapters: List[nn.Module], grad_outputs: List[torch.Tensor], 
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """ Batched backward for LoRA adapters. """

    lora_b_grad_outs = [grad_out * adapter.scaling for grad_out, adapter in zip(grad_outputs, adapters)]
    lora_b_grad_ins = triton_grouped_gemm(lora_b_grad_outs, [adapter.lora_B.weight.contiguous() for adapter in adapters])
    lora_b_grad_weights = triton_grouped_gemm([out.T.contiguous() for out in ctx.lora_a_outs], lora_b_grad_outs)
    lora_a_grad_ins = triton_grouped_gemm(lora_b_grad_ins, [adapter.lora_A.weight.contiguous() for adapter in adapters])
    lora_a_grad_weights = triton_grouped_gemm([inp.T.contiguous() for inp in ctx.lora_a_ins], lora_b_grad_ins)
    
    # accumulate gradient buffers
    for adapter, lora_a_grad_weight, lora_b_grad_weight in zip(adapters, lora_a_grad_weights, lora_b_grad_weights):
        lora_a_grad_weight_ = lora_a_grad_weight.T.contiguous()
        lora_b_grad_weight_ = lora_b_grad_weight.T.contiguous()
        if adapter.lora_A.weight.grad is None:
            adapter.lora_A.weight.grad = lora_a_grad_weight_.to(
                device=adapter.lora_A.weight.device, dtype=adapter.lora_A.weight.dtype)
        else:
            adapter.lora_A.weight.grad += lora_a_grad_weight_
        if adapter.lora_B.weight.grad is None:
            adapter.lora_B.weight.grad = lora_b_grad_weight_.to(
                device=adapter.lora_B.weight.device, dtype=adapter.lora_B.weight.dtype)
        else:
            adapter.lora_B.weight.grad += lora_b_grad_weight_

    return lora_a_grad_ins
