#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
from typing import Dict, Set, Any
import unittest
import logging
from functools import partial
import numpy as np


class BasicFuncTest(unittest.TestCase):
    """ Basic functionality test cases.
    
    This unit testcase should be executed with `torchrun`, i.e., `torchrun --nproc-per-node N script.py`.
    """

    # @unittest.skip("Pass.")
    def test_correctness_of_spatial_temporal_task_colocation(self):
        import torch

        from muxtune.models.adapters.lora import LoraAdapter, LoraInputDispatcher, LoraOutputAggregator
        from muxtune.core.modules.peft_modules import PeftModuleConfig, PeftModule, PeftModuleGroup
        from muxtune.core.modules.utils import BackwardThrottler, NonBaseOpModule
        from muxtune.core.data.mixed_tensor import MixedTensor, ChunkedTensor
        from muxtune.global_envs import PeftType, global_configs, logger

        class DummyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layer = torch.nn.Linear(4, 4, device="cuda", dtype=torch.float16)
                self.base_op = torch.nn.Linear(4, 4, device="cuda", dtype=torch.float16)
                self.output_layer = torch.nn.Linear(4, 4, device="cuda", dtype=torch.float16)

            def forward(self, x):
                act = self.input_layer(x)
                act = self.base_op(act)
                return self.output_layer(act)

        backbone = DummyBackbone()
        for (_, param) in backbone.named_parameters():
                param.requires_grad = False
                param.grad = None   # Clear grad

        config = PeftModuleConfig(
            PeftType.LoRA, 0, LoraInputDispatcher(), LoraOutputAggregator(), "cuda", torch.float16)
        peft_module_group = PeftModuleGroup()
        backbone.base_op = peft_module_group.hook_to_base_op(backbone.base_op)
        peft_module = PeftModule(config)
        peft_module_group.add_peft_module(peft_module)

        # Non-base operator layers
        nonbase_op_module = NonBaseOpModule()
        backbone.input_layer = nonbase_op_module.hook_to_nonbase_op(backbone.input_layer)

        # Last layer
        bw_throttler = BackwardThrottler()
        backbone.output_layer = bw_throttler.hook_to_nonbase_op(backbone.output_layer)

        task_names = ["task_0", "task_1", ]
        task_inputs = [
            torch.randn((2, 2, 4), device="cuda", dtype=torch.float16), # [s, b, h]
            torch.randn((2, 4, 4), device="cuda", dtype=torch.float16),
        ]
        task_labels = [
            torch.randn((2, 2, 4), device="cuda", dtype=torch.float16),
            torch.randn((2, 4, 4), device="cuda", dtype=torch.float16),
        ]
        microbatch_sizes = [2, 4]
        for i, task_name in enumerate(task_names):
            adapter = LoraAdapter(
                f"peft_module_0::{task_name}", peft_module.config.device, peft_module.config.dtype, 
                lora_r=2, lora_alpha=4, in_features=4, out_features=4,
            )
            peft_module.register_one_adapter(adapter, task_inputs[i].shape[1])

        optimizer_0 = torch.optim.Adam(peft_module.adapters["peft_module_0::task_0"].parameters(), lr=1e-3)
        optimizer_1 = torch.optim.Adam(peft_module.adapters["peft_module_0::task_1"].parameters(), lr=1e-3)

        global_configs.current_microbatch_index = 0    

        os.environ["FORCED_ADAPTER_NAME_DEBUG"] = "peft_module_0::task_0"
        input_task_0 = MixedTensor(chunked_tensors={ 0: [task_inputs[0]] })
        peft_out_0 = backbone(input_task_0)
        logger.info(f"Single-task forward output of task_0: {peft_out_0}\n\n")

        os.environ["FORCED_ADAPTER_NAME_DEBUG"] = "peft_module_0::task_1"
        input_task_1 = MixedTensor(chunked_tensors={ 0: [task_inputs[1]] })
        peft_out_1 = backbone(input_task_1)
        logger.info(f"Single-task forward output of task_1: {peft_out_1}\n\n")

        del os.environ["FORCED_ADAPTER_NAME_DEBUG"]
        optimizer_0.zero_grad()
        optimizer_1.zero_grad()

        loss_0 = torch.nn.functional.mse_loss(peft_out_0[0][0].value, task_labels[0])
        loss_1 = torch.nn.functional.mse_loss(peft_out_1[0][0].value, task_labels[1])
        loss_0.backward()
        loss_1.backward()

        logger.info(f"Single-task backward task_0 adapter grad: LoRA A: " + 
              f"{peft_module.adapters['peft_module_0::task_0'].lora_A.weight.grad} " + 
              f"| LoRA B: {peft_module.adapters['peft_module_0::task_0'].lora_B.weight.grad}\n\n")
        logger.info(f"Single-task backward task_1 adapter grad: LoRA A: " + 
              f"{peft_module.adapters['peft_module_0::task_1'].lora_A.weight.grad} " + 
              f" | LoRA B: {peft_module.adapters['peft_module_0::task_1'].lora_B.weight.grad}\n\n")

        single_a0 = peft_module.adapters["peft_module_0::task_0"].lora_A.weight.grad.clone()
        single_b0 = peft_module.adapters["peft_module_0::task_0"].lora_B.weight.grad.clone()
        single_a1 = peft_module.adapters["peft_module_0::task_1"].lora_A.weight.grad.clone()
        single_b1 = peft_module.adapters["peft_module_0::task_1"].lora_B.weight.grad.clone()

        batched_in = MixedTensor(chunked_tensors={ 0: [torch.cat(task_inputs, dim=1)] })
        batched_out = backbone(batched_in)
        logger.info(f"Multi-task batched forward output: {batched_out}")

        optimizer_0.zero_grad()
        optimizer_1.zero_grad()

        for (peft_group_index, out) in batched_out.items():
            peft_out_0, peft_out_1 = torch.split(out[0].value, microbatch_sizes, dim=1)
            loss_0 = torch.nn.functional.mse_loss(peft_out_0, task_labels[0])
            loss_1 = torch.nn.functional.mse_loss(peft_out_1, task_labels[1])
            losses = MixedTensor({ 0: loss_0, 1: loss_1 })
            bw_throttler.backward(losses, peft_group_index)

        logger.info(f"Multi-task backward task_0 adapter grad: LoRA A: " + 
              f"{peft_module.adapters['peft_module_0::task_0'].lora_A.weight.grad} " + 
              f"| LoRA B: {peft_module.adapters['peft_module_0::task_0'].lora_B.weight.grad}\n\n")
        logger.info(f"Multi-task backward task_1 adapter grad: LoRA A: " + 
              f"{peft_module.adapters['peft_module_0::task_1'].lora_A.weight.grad} " + 
              f" | LoRA B: {peft_module.adapters['peft_module_0::task_1'].lora_B.weight.grad}\n\n")
        
        self.assertTrue(torch.allclose(single_a0, peft_module.adapters["peft_module_0::task_0"].lora_A.weight.grad, rtol=1e-2, atol=1e-2), "task_0 lora_A: single vs batched")
        self.assertTrue(torch.allclose(single_b0, peft_module.adapters["peft_module_0::task_0"].lora_B.weight.grad, rtol=1e-2, atol=1e-2), "task_0 lora_B: single vs batched")
        self.assertTrue(torch.allclose(single_a1, peft_module.adapters["peft_module_0::task_1"].lora_A.weight.grad, rtol=1e-2, atol=1e-2), "task_1 lora_A: single vs batched")
        self.assertTrue(torch.allclose(single_b1, peft_module.adapters["peft_module_0::task_1"].lora_B.weight.grad, rtol=1e-2, atol=1e-2), "task_1 lora_B: single vs batched")

        logger.info("Passed correctness test for spatial-temporal multi-task cololocation.")

    @unittest.skip("Pass.")
    def test_peft_enabled_meagtron_gpt(self):
        """ Run the parallelized Megatron model with PEFT adapters hooked. """

        # import os
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        import torch
        from torch import nn

        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core import mpu
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.utils import get_attr_wrapped_model

        from muxtune.training.utils import initialize_distributed
        from muxtune.models.backbones.gpt import GPTModel
        from muxtune.training.trainer import Trainer
        from muxtune.training.optimizer import get_optimizers_and_schedulers
        from muxtune.training.utils import (setup_model, get_train_data_iterator, 
                                            get_input_batch, ModelType)
        from muxtune.global_envs import global_configs

        def model_provider(
            megatron_config: Any, max_seq_len: int, vocab_size: int, device: str,
        ) -> nn.Module:
            """ Model provider function. """
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            return GPTModel(megatron_config, vocab_size, max_seq_len, pre_process, post_process, device=device)

        # TODO(chunyu): implement chunk-level aligner, and correctness verification.

        model_type = ModelType.GPT
        global_bs = 8
        seq_len = 1024
        num_micro_batches = 2
        hidden_size = 4096
        num_attention_heads = 32
        num_layers = 16
        dtype = torch.float16
        device = "cuda"
        vocab_size = 50000
        num_iters = 5
        
        global_configs.tensor_model_parallel_size = 1
        global_configs.pipeline_model_parallel_size = 2
        global_configs.data_parallel_size = 1

        global_configs.num_nodes = 1
        global_configs.num_devices_per_node = 2
        
        # Get number of microbatches
        # In Megatron, each DP replica constantly processes `micro_batch_size` samples per micro-batch, 
        # only changing the number of micro-batches to maintain constant global batch size.
        micro_batch_size =  global_bs // num_micro_batches
        assert global_bs % (micro_batch_size * global_configs.data_parallel_size) == 0, \
            f"Global batch size {global_bs} must be divisible by micro-batch-size {micro_batch_size} " + \
            f"x data parallel size {global_configs.data_parallel_size}."
        num_micro_batches = global_bs // (micro_batch_size * global_configs.data_parallel_size)
        global_bs = micro_batch_size * num_micro_batches    # of this DP rank

        # Init distributed
        initialize_distributed(
            tensor_model_parallel_size=global_configs.tensor_model_parallel_size,
            pipeline_model_parallel_size=global_configs.pipeline_model_parallel_size,
        )
        model_parallel_cuda_manual_seed(1234)

        # Create TransformerConfig for Megatron   
        megatron_config = TransformerConfig(
            tensor_model_parallel_size=global_configs.tensor_model_parallel_size,
            pipeline_model_parallel_size=global_configs.pipeline_model_parallel_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            params_dtype=dtype,
            pipeline_dtype=dtype,
            fp16=(dtype == torch.float16),
        )

        # NOTE: We monkey patch all input configuratons into `megatron_config`, 
        # instead of explicitly pass-in into the arguments of `Trainer()`.
        setattr(megatron_config, "global_batch_size", global_bs)
        setattr(megatron_config, "vocab_size", vocab_size)
        setattr(megatron_config, "num_micro_batches", num_micro_batches)
        setattr(megatron_config, "sequence_length", seq_len)
        setattr(megatron_config, "num_iters", num_iters)
        setattr(megatron_config, "dtype", dtype)

        # Model
        model_kwargs={
            "megatron_config": megatron_config,
            "max_seq_len": seq_len,
            "vocab_size": vocab_size,
            "device": device,
        }
        model = model_provider(**model_kwargs) if model_provider else None
        model = setup_model(model)

        train_data_iterator = get_train_data_iterator(
            vocab_size, global_bs, num_micro_batches, seq_len, num_iters=num_iters,
        )
        (optimizers, opt_param_schedulers) = get_optimizers_and_schedulers(
            model, global_bs, num_iters, fp16=(dtype == torch.float16), params_dtype=dtype, 
        )

        # Definition of forward step function
        def __forward_step_func(data_iterator, model):
            """ Vanilla forward step function. """

            def __loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
                """ Loss function definition. """
                losses = output_tensor.float()
                loss_mask = loss_mask.view(-1).float()
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
                # If you have data parallel reduce loss across data parallel groups.
                # If pipeline parallel, loss computation is done only in last stage.
                return loss, {"lm loss": loss}

            megatron_config = get_attr_wrapped_model(model, attr="config")
            # Input batch
            (tokens, labels, loss_mask, 
             attention_mask, position_ids) = get_input_batch(
                 data_iterator=data_iterator,
                 micro_batch_size=(
                     megatron_config.global_batch_size // megatron_config.num_micro_batches
                 ),
                 seq_len=megatron_config.sequence_length,
                 pipeline_model_parallel_size=megatron_config.pipeline_model_parallel_size,
            )
            # Forward
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

            return output_tensor,  partial(__loss_func, loss_mask)

        # Trainer
        trainer = Trainer(
            megatron_transformer_config=megatron_config,
            forward_step_func=__forward_step_func,
            model=model,
            model_type=model_type,
            train_data_iterator=train_data_iterator,
            optimizers=optimizers,
            opt_param_schedulers=opt_param_schedulers,
            pipeline_strategy="1f1b",
            forward_only=False,
            use_pytorch_profiler=False,
            pytorch_profiler_dp_ranks=[0, ], 
            pytorch_profiler_tp_ranks=[0, ],
        )
        trainer.train(metric_types=["e2e_iter_time", ])


if __name__ == '__main__':
    unittest.main()
