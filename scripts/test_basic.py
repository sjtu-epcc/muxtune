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
    def test_muxtune_import(self):
        import muxtune
        logging.info(f"Successfully imported muxtune: {muxtune.__version__}")


if __name__ == '__main__':
    unittest.main()
