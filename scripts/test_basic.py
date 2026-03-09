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
        adapter = LoraAdapter("test_module::task_0", "cuda", torch.float16, lora_r=2, lora_alpha=4, in_features=4, out_features=4)
        input_dispatcher = LoraInputDispatcher()
        output_aggregator = LoraOutputAggregator()

        config = PeftModuleConfig(peft_type=PeftType.LoRA, module_name="test_module", device="cuda", dtype=torch.float16)
        peft_module = PeftModule(config, base_op)
        peft_module.register_one_adapter(adapter, input_dispatcher, output_aggregator)

        peft_in = torch.randn(2, 4, device="cuda", dtype=torch.float16)
        peft_out = peft_module.single_forward("test_module::task_0", peft_in)

        print("PEFT output:", peft_out)


if __name__ == '__main__':
    unittest.main()
