#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
from typing import Dict, Set, Any
import unittest
import logging
from functools import partial
from collections import OrderedDict
import time
import numpy as np


class BasicFuncTest(unittest.TestCase):
    """ Basic functionality test cases.
    
    This unit testcase should be executed with `torchrun`, i.e., `torchrun --nproc-per-node N script.py`.
    """

    # @unittest.skip("Pass.")
    def test_lora_impl(self):
        import torch

        from muxtune.models.adapters.lora import LoraAdapter, LoraInputDispatcher, LoraOutputAggregator
        from muxtune.core.peft_modules import PeftModuleConfig, PeftModule
        from muxtune.global_envs import PeftType

        base_op = torch.nn.Linear(4, 4, device="cuda", dtype=torch.float16)
        config = PeftModuleConfig(PeftType.LoRA, "test_module", "cuda", torch.float16)
        peft_module = PeftModule(config, base_op)

        task_names = ["task_0", "task_1", ]
        task_inputs = [
            torch.randn(2, 4, device="cuda", dtype=torch.float16),
            torch.randn(4, 4, device="cuda", dtype=torch.float16),
        ]
        for task_name in task_names:
            adapter = LoraAdapter(
                f"test_module::{task_name}", peft_module.config.device, peft_module.config.dtype, 
                lora_r=2, lora_alpha=4, in_features=4, out_features=4,
            )
            input_dispatcher = LoraInputDispatcher()
            output_aggregator = LoraOutputAggregator()

            peft_module.register_one_adapter(adapter, input_dispatcher, output_aggregator)

        peft_out = peft_module._single_forward("test_module::task_0", task_inputs[0])
        print("Single-task forward output", peft_out)


if __name__ == '__main__':
    unittest.main()
