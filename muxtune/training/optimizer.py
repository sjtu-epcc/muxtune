#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Optimizer for PEFT adapters. """

from typing import Dict, List, Optional, Tuple

import torch
from megatron.core.transformer import MegatronModule
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.optimizer.grad_scaler import (
    ConstantGradScaler, DynamicGradScaler)

from muxtune.global_envs import PeftType

__all__ = [ "get_optimizers_and_schedulers", ]


def get_optimizers_and_schedulers(
    model: List[MegatronModule],
    global_batch_size: int,
    num_iters: int,
    optimizer_str: str = "adam",
    lr: Optional[float] = 1e-3,
    max_lr: Optional[float] = 1e-3,
    min_lr: Optional[float] = 1e-4,
    weight_decay: float = 0.01,
    fp16: bool = False,
    bf16: bool = False,
    params_dtype: torch.dtype = torch.float32,
    loss_scale: float = 16,
    use_distributed_optimizer: bool = False,
    num_lr_warmup_iters: int = 0,
    lr_decay_style: str = "cosine",
) -> Tuple[Dict[str, torch.optim.Optimizer], Dict[str, OptimizerParamScheduler]]:
    """ Get optimizers and learning rate schedulers per PEFT task. """

    # Optimizers
    optimizers = {}
    param_groups = _get_adapter_param_groups(model)
    for (task_name, adapter_param_groups) in param_groups.items():
        if optimizer_str == "adam":
            optimizer = torch.optim.Adam(
                params=adapter_param_groups, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_str}")

        # Grad scaler
        if loss_scale is not None:
            grad_scaler = ConstantGradScaler(loss_scale)
        else:
            raise NotImplementedError
        
        def __scale_loss(loss: torch.Tensor) -> torch.Tensor:
            return grad_scaler.scale * loss
        
        setattr(optimizer, "grad_scaler", grad_scaler)
        setattr(optimizer, "scale_loss", __scale_loss)
        optimizers[task_name] = optimizer    

    # Learning rate schedulers
    opt_param_schedulers = {}
    lr_warmup_steps = num_lr_warmup_iters * global_batch_size
    lr_decay_steps = weight_decay_steps = num_iters * global_batch_size
    for (task_name, optimizer) in optimizers:
        opt_param_schedulers[task_name] = OptimizerParamScheduler(
            optimizer=optimizer, init_lr=lr, max_lr=max_lr,
            min_lr=min_lr, lr_warmup_steps=lr_warmup_steps, 
            lr_decay_steps=lr_decay_steps, lr_decay_style=lr_decay_style,
            start_wd=0, end_wd=0,
            wd_incr_steps=weight_decay_steps, wd_incr_style="constant",
        )

    return optimizers, opt_param_schedulers


def _get_adapter_param_groups(
    model: List[MegatronModule],
    peft_module_group_attr_name: str = "peft_module_group",
) -> Dict[str, List[Dict]]:
    """ Get parameter groups of each adapter on the model partition. """

    param_groups = {}
    for model_chunk in model:
        for (_, module) in model_chunk.named_modules():
            if not hasattr(module, peft_module_group_attr_name):
                continue    # not hooked with peft module group

            peft_module_group = getattr(module, peft_module_group_attr_name, None)
            for (_, peft_module) in peft_module_group.peft_modules.items():
                for (adapter_name, adapter) in peft_module.adapters.items():
                    task_name = adapter_name.split("::")[-1]
                    if task_name not in param_groups:
                        param_groups[task_name] = []
                    
                    if peft_module.config.peft_type == PeftType.LoRA:
                        param_groups[task_name].append({
                            "params": adapter.lora_A.parameters(), 
                            "differentiable": True,
                            "capturable": True,
                        })
                        param_groups[task_name].append({
                            "params": adapter.lora_B.parameters(), 
                            "differentiable": True,
                            "capturable": True,
                        })
    
    return param_groups
