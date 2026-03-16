#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import enum
from dataclasses import dataclass, field
import logging

__all__ = [
    "PeftType",
    "global_configs",
    "logger",
]


# Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG)


class PeftType(enum.Enum):
    LoRA = enum.auto()


@dataclass
class GlobalEnvConfigs:
    """ Global environmental configurations. 
    
    In distributed training, each parallel rank (i.e., each process) maintains its own global environmental 
    configurations. Thus, runtime modified configurations should be boardcasted among all ranks.
    """

    ###########
    # Runtime #
    ###########
    current_microbatch_index: int = -1
    """ Current microbatch index on this rank. """


global_configs = GlobalEnvConfigs()
""" Object of global configurations. """
