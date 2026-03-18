#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
from typing import Dict, Set, Any
import unittest
import logging
from functools import partial
import time
import numpy as np


class BasicFuncTest(unittest.TestCase):
    """ Basic functionality test cases.
    
    This unit testcase should be executed with `torchrun`, i.e., `torchrun --nproc-per-node N script.py`.
    """

    @unittest.skip("Pass.")
    def test_correctness_of_spatial_temporal_task_colocation(self):
        import torch

        from muxtune.models.adapters.lora import LoraAdapter, LoraInputDispatcher, LoraOutputAggregator
        from muxtune.core.peft_modules import PeftModuleConfig, PeftModule, PeftModuleGroup
        from muxtune.core.utils import MixedTensor, BackwardThrottler
        from muxtune.global_envs import PeftType, global_configs, logger

        class DummyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base_op = torch.nn.Linear(4, 4, device="cuda", dtype=torch.float16)
                self.output_layer = torch.nn.Linear(4, 4, device="cuda", dtype=torch.float16)

            def forward(self, x):
                act = self.base_op(x)
                return self.output_layer(act)

        backbone = DummyBackbone()
        for (_, param) in backbone.named_parameters():
                param.requires_grad = False
                param.grad = None   # Clear grad

        config = PeftModuleConfig(
            PeftType.LoRA, "peft_module_0", LoraInputDispatcher(), LoraOutputAggregator(), "cuda", torch.float16,
        )
        peft_module_group = PeftModuleGroup()
        backbone.base_op = peft_module_group.hook_to_base_op(backbone.base_op)
        peft_module = PeftModule(config)
        peft_module_group.add_peft_module(peft_module)

        bw_throttler = BackwardThrottler()
        backbone.output_layer = bw_throttler.hook_to_nonbase_op(backbone.output_layer)

        task_names = ["task_0", "task_1", ]
        task_inputs = [
            torch.randn(2, 4, device="cuda", dtype=torch.float16),
            torch.randn(4, 4, device="cuda", dtype=torch.float16),
        ]
        task_labels = [
            torch.randn(2, 4, device="cuda", dtype=torch.float16),
            torch.randn(4, 4, device="cuda", dtype=torch.float16),
        ]
        microbatch_sizes = [2, 4]
        for i, task_name in enumerate(task_names):
            adapter = LoraAdapter(
                f"peft_module_0::{task_name}", peft_module.config.device, peft_module.config.dtype, 
                lora_r=2, lora_alpha=4, in_features=4, out_features=4,
            )
            peft_module.register_one_adapter(adapter, task_inputs[i].shape[0])

        optimizer_0 = torch.optim.Adam(peft_module.adapters["peft_module_0::task_0"].parameters(), lr=1e-3)
        optimizer_1 = torch.optim.Adam(peft_module.adapters["peft_module_0::task_1"].parameters(), lr=1e-3)

        global_configs.current_microbatch_index = 0    

        os.environ["FORCED_ADAPTER_NAME_DEBUG"] = "peft_module_0::task_0"
        input_task_0 = MixedTensor({ 0: task_inputs[0] })
        peft_out_0 = backbone(input_task_0)
        logger.info(f"Single-task forward output of task_0: {peft_out_0}\n\n")

        os.environ["FORCED_ADAPTER_NAME_DEBUG"] = "peft_module_0::task_1"
        input_task_1 = MixedTensor({ 0: task_inputs[1] })
        peft_out_1 = backbone(input_task_1)
        logger.info(f"Single-task forward output of task_1: {peft_out_1}\n\n")

        del os.environ["FORCED_ADAPTER_NAME_DEBUG"]
        optimizer_0.zero_grad()
        optimizer_1.zero_grad()

        loss_0 = torch.nn.functional.mse_loss(peft_out_0[0], task_labels[0])
        loss_1 = torch.nn.functional.mse_loss(peft_out_1[0], task_labels[1])
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

        batched_in = MixedTensor({ 0: torch.cat(task_inputs, dim=0) })
        batched_out = backbone(batched_in)
        logger.info(f"Multi-task batched forward output: {batched_out}")

        optimizer_0.zero_grad()
        optimizer_1.zero_grad()

        for (peft_group_index, out) in batched_out.items():
            peft_out_0, peft_out_1 = torch.split(out, microbatch_sizes, dim=0)
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

if __name__ == '__main__':
    unittest.main()
