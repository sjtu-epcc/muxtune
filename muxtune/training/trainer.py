#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Trainer implementation for Megatron. """

import os
from typing import (
    TYPE_CHECKING, Callable, Dict, Generator, List, Any,  
    Optional, Tuple, Type, Union, cast, Iterator)
import enum
from contextlib import nullcontext
import logging
from collections import OrderedDict
import time
import numpy as np

import torch
import torch.distributed

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.pipeline_parallel.schedules import (
    set_current_microbatch)
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.transformer.module import Float16Module
from megatron.core.distributed import (
    DistributedDataParallel, DistributedDataParallelConfig)
from megatron.core.utils import get_attr_wrapped_model

from muxtune.global_envs import logger, global_timer
from muxtune.training.utils import log_on_rank, ModelType
from muxtune.training.pipeline_parallel.schedules import forward_backward_pipelining_without_interleaving

__all__ = [ "Trainer", ]


class Trainer:
    """
    Trainer class to wrap Megatron training procedure.

    Args:
        trainer_configs: The transformer config object in Megatron, with monkey patched attributes.
        forward_step_func: The forward step function for the model that takes the
            data iterator as the first argument, and model as the second. Detailed in 
            the docstring of `megatron.core.pipeline_parallel.schedules.forward_step()`.
        model: The model object defined in the format of MegatronModule.
        model_type: Type of the model to be trained.
        train_data_iterator: The iterator of training data.
        optimizers: The optimizers per task.
        opt_param_schedulers: The learning rate schedulers per task.
        pipeline_strategy: Strategy of the pipeline execution.
        forward_only: Only perform forward pass.
        use_pytorch_profiler: Use PyTorch's native profiler for in-depth performance analysis.
    """

    def __init__(
        self,
        megatron_transformer_config: TransformerConfig,
        forward_step_func: Callable,
        model: Union[MegatronModule, List[MegatronModule]],
        model_type: ModelType,
        train_data_iterator: Iterator,
        optimizers: Dict[str, torch.optim.Optimizer],
        opt_param_schedulers: Dict[str, OptimizerParamScheduler],
        pipeline_strategy: str = None,
        forward_only: bool = False,
        use_pytorch_profiler: bool = False,
        pytorch_profiler_dp_ranks: List[int] = [],
        pytorch_profiler_tp_ranks: List[int] = [],
    ) -> None:
        """ Initialize a Trainer object. """

        self.megatron_config = megatron_transformer_config
        self.model = model if isinstance(model, list) else [model]
        self.model_type = model_type

        self.train_data_iterator = train_data_iterator
        self.optimizers = optimizers
        self.opt_param_schedulers = opt_param_schedulers
        self.pipeline_strategy = pipeline_strategy
        self.forward_only = forward_only
        self.use_pytorch_profiler = use_pytorch_profiler
        if use_pytorch_profiler:
            dp_rank = parallel_state.get_data_parallel_rank()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            if (
                dp_rank not in pytorch_profiler_dp_ranks or 
                tp_rank not in pytorch_profiler_tp_ranks
            ):
                self.use_pytorch_profiler = False

        self.forward_step_func = forward_step_func
        self.forward_backward_func = None

        self._init_forward_backward_func()
        self._freeze_all_backbone_modules()

    def _init_forward_backward_func(self) -> None:
        """ Initialize forward-backward pipeline function. """
        self.forward_backward_func = forward_backward_pipelining_without_interleaving

    def _freeze_all_backbone_modules(self) -> None:
        """ Freeze all backbone modules of the model. """
        for model_chunk in self.model:
            for (_, param) in model_chunk.named_parameters():
                param.requires_grad = False
                param.grad = None   # Clear grad
    
    def train_step(self) -> Tuple[List[torch.Tensor]]:
        """ Training step for one iteration of a mini-batch. """

        # This model partition belongs to the last stage of the pipeline
        last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=True)

        # Zero grad of model and optimizer
        if self.optimizers is not None:
            for _, optimizer in self.optimizers.items():
                optimizer.zero_grad()
        
        global_timer.start("train_step", use_cuda_event=True)

        # Forward-backward pass
        assert self.forward_backward_func is not None, \
            "Forward-backward function of the trainer is not set."
        tr_loss = self.forward_backward_func(
            forward_step_func=self.forward_step_func,
            data_iterator=self.train_data_iterator,
            model=self.model,
            num_microbatches=self.megatron_config.num_micro_batches,
            seq_length=self.megatron_config.sequence_length,
            micro_batch_size=(
                self.megatron_config.global_batch_size // 
                self.megatron_config.num_micro_batches
            ),
            forward_only=self.forward_only,
        )
        
        # FIXME(chunyu): Currently we only consider forward-backward computation without
        # other processes such as optimizer step.
        global_timer.stop("train_step", use_cuda_event=True)

        # Unscale grad
        self._maybe_unscale_grad()

        if self.forward_only:
            return (tr_loss, ) if last_stage else (None, )
        
        # Update parameters
        for (_, optimizer) in self.optimizers.items():
            with torch.no_grad():
                optimizer.step()

        # Update learning rate
        increment = (
            self.megatron_config.global_batch_size * 
            parallel_state.get_data_parallel_world_size()
        )
        for (_, opt_param_scheduler) in self.opt_param_schedulers.items():
            opt_param_scheduler.step(increment=increment)
        
        torch.cuda.synchronize()
        torch.distributed.barrier()
        return (tr_loss, ) if last_stage else (None, )

    def train(self, metric_types: List[str] = []) -> List[float]:
        """ Training entrypoint. """
        
        # Turn on training mode which enables dropout
        for model_chunk in self.model:
            model_chunk.train()

        # Setup training configurations
        megatron_config = get_attr_wrapped_model(self.model[0], attr="config")
        megatron_config.grad_scale_func = next(iter(self.optimizers.values())).scale_loss \
            if self.optimizers is not None else None
        megatron_config.finalize_model_grads_func = finalize_model_grads \
            if self.optimizers is not None else None

        prof = None
        if self.use_pytorch_profiler:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA, ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            prof.start()

        # Training loop
        tr_losses = []
        for iter_index in range(self.megatron_config.num_iters):
            if self.use_pytorch_profiler:
                prof.step()
            
            (tr_loss, ) = self.train_step()
            tr_losses.append(tr_loss)
            log_on_rank(
                msg=f"Losses of the training iteration ({iter_index}): {tr_loss}",
                rank=[0, parallel_state.get_pipeline_model_parallel_last_rank()],
                group_type=["tensor", "pipeline"],
                logger=logger,
                log_level="info",
            )
        
        if self.use_pytorch_profiler:
            prof.stop()
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            prof.export_chrome_trace(f"/app/torch_exec_trace_pp_rank_{pp_rank}_tp_rank_{tp_rank}.json")
            log_on_rank(
                msg=f"Save PyTorch execution trace to /app/torch_exec_trace_pp_rank_{pp_rank}_tp_rank_{tp_rank}.json",
                rank=[0, parallel_state.get_pipeline_model_parallel_first_rank()],
                group_type=["tensor", "pipeline"],
                logger=logger,
                log_level="info",
            )

        if len(metric_types) > 0:
            raw_record = global_timer._records.pop("train_step")
            e2e_iter_times = [round(_r.elapsed, 3) for _r in raw_record]
            e2e_iter_times = e2e_iter_times[1:]     # Warmup iter excluded

            if "e2e_iter_time" in metric_types:
                log_on_rank(
                    msg=f"Average end-to-end iteration time (ms): {np.mean(e2e_iter_times)}",
                    rank=[0, 0],
                    group_type=["tensor", "pipeline"],
                    logger=logger,
                    log_level="info",
                )

        return tr_losses

    def _maybe_unscale_grad(self) -> None:
        """ Unscale gradient if gradient scaler is enabled. """

        for (_, optimizer) in self.optimizers:
            if (not hasattr(optimizer, "grad_scaler") or optimizer.grad_scaler is None):
                continue

            main_grads = []
            for model_chunk in self.model:
                for param in model_chunk.parameters():
                    if param.grad is not None:
                        main_grads.append(param.grad.data)

            found_inf = torch.tensor([0.0], dtype=torch.float, device='cuda')
            torch._amp_foreach_non_finite_check_and_unscale_(
                main_grads, found_inf, optimizer.grad_scaler.inv_scale,
            )
