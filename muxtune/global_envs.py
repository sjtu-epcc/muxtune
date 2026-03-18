#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import enum
from typing import Tuple, List, Dict
from dataclasses import dataclass, field
import logging
import time

import torch

__all__ = [
    "logger", "PeftType", "global_configs",
    "TimeRecord", "Timer", "global_timer",
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

    #####################
    # Model Parallelism #
    #####################
    tensor_model_parallel_size: int = 1
    """ Intra-layer model parallelism. Splits tensors across GPU ranks. """

    data_parallel_size: int = 1 
    """ Data parallelism (Torch DDP). Perform gradient sync after backward. """

    pipeline_model_parallel_size: int = 1
    """ Inter-layer model parallelism. Splits transformer layers across GPU ranks. """

    remove_p2p_comm_redundancy_in_hybrid_parallel: bool = False
    """ Perform inter-stage P2P communication only between intra-stage's rank 0, rather than
    between ranks of all PP groups. Then, in the destination stage, perform an additional 
    broadcast across all intra-stage ranks. """
    
    ###########################
    # Resource Configurations #
    ###########################
    num_nodes: int = 1
    """ Number of nodes. """

    num_devices_per_node: int = 1
    """ Number of devices per node. """

    ###########
    # Runtime #
    ###########
    current_microbatch_index: int = -1
    """ Current microbatch index on this rank. """


global_configs = GlobalEnvConfigs()
""" Object of global configurations. """


@dataclass
class TimeRecord:
    """ Record of one timing operation. """

    opcode: str = None
    """ Opcode to identify the operation to be timed. """

    start_t: float = None
    """ Start timestamp obtained from `timer.time()`. """

    stop_t: float = None
    """ Stop timestamp obtained from `timer.time()`. """

    elapsed: float = None
    """ Elapsed time of the operation. """

    start_event: torch.cuda.Event = None
    """ CUDA event to start the timing. """

    stop_event: torch.cuda.Event = None
    """ CUDA event to stop the timing. """

    def __post_init__(self):
        """ Post initialization. """
        assert self.opcode, "The opcode of this time record must be set for identification."


class Timer:
    """ Class of global timer of each process. """

    def __init__(self) -> None:
        """ Initialize a timer object. """

        self._records: Dict[str, List[TimeRecord]] = {}
        self._barrier_group = None

    def set_barrier_group(self, barrier_group: torch.distributed.ProcessGroup):
        """ Sets the barrier group used to synchronize ranks. """
        self._barrier_group = barrier_group
    
    def _maybe_sync(self, barrier: bool = False):
        """ Synchronize ranks and local CUDA events. """

        if barrier:
            # Synchronize all ranks
            torch.distributed.barrier(group=self._barrier_group)
        # Synchronize local cuda events
        torch.cuda.synchronize()

    def start(self, opcode: str, use_cuda_event: bool = False, barrier: bool = False):
        """ Start a timer operation. 

        This can introduce additional overhead as some synchronization and barrier operations 
        are called to measure time cost precisely. 
        
        Args:
            opcode: The opcode as the identification of the operation to be timed.
            use_cuda_event: Use `torch.cuda.Event()` to record the elapsed time of a GPU computation.
                             Compared to `barrier` option, this can measure the time cost of a GPU computation
                             more precisely.
            barrier: Use `torch.distributed.barrier()` to synchronize all ranks in the barrier group.
                      When pipeline is used, it should be carefully used (by setting barrier group) to
                      avoid deadlock between blocking of the communication primitives and the barrier
                      option in the timer.
        """

        assert (
            opcode not in self._records or 
            sum([int(_r.elapsed is None) for _r in self._records[opcode]]) == 0
        ), \
            f"There exists one non-stopped time record of opcode: {opcode}"

        self._maybe_sync(barrier)

        if use_cuda_event:
            # Use CUDA event to measure time cost
            # This directly measures the execution time of all kernels from the target operation
            time_record = TimeRecord(
                opcode=opcode,
                start_event=torch.cuda.Event(enable_timing=True),
                stop_event=torch.cuda.Event(enable_timing=True),
            )
            time_record.start_event.record()
        else:
            # Use Python built-in timer to measure time cost
            # This is more suitable for measuring the end-to-end time including API call, kernel 
            # lanuch and kernel execution.
            start_t=time.time()
            time_record = TimeRecord(opcode=opcode, start_t=start_t)

        if opcode in self._records:
            self._records[opcode].append(time_record)
        else:
            self._records[opcode] = [time_record]
    
    def stop(self, opcode: str, use_cuda_event: bool = False, barrier: bool = False):
        """ Stop a timer operation. """

        assert opcode in self._records, f"Undefined opcode in the timer: {opcode}"
        assert sum([int(_r.elapsed is None) for _r in self._records[opcode]]) == 1, \
            f"Must be exactly only one non-stopped time record of opcode: {opcode}"
        assert self._records[opcode][-1].elapsed is None, \
            "The last started record should be not stopped."

        self._maybe_sync(barrier)

        if use_cuda_event:
            # Use CUDA event to measure time cost
            assert isinstance(self._records[opcode][-1].stop_event, torch.cuda.Event), \
                f"The CUDA event to stop timing is not set on the last started record: " + \
                f"{self._records[opcode][-1].stop_event.__class__}"
            
            self._records[opcode][-1].stop_event.record()
            # Synchronize
            # Otherwise, the following error may occur:
            # `RuntimeError: CUDA error: device not ready. CUDA kernel errors might be asynchronously 
            # reported at some other API call, so the stacktrace below might be incorrect`.
            torch.cuda.synchronize()
            # Get elapsed time
            self._records[opcode][-1].elapsed = self._records[opcode][-1].start_event.elapsed_time(
                end_event=self._records[opcode][-1].stop_event,
            ) # Record in milsecond
            # Clear events
            self._records[opcode][-1].start_event = None
            self._records[opcode][-1].stop_event = None
        else:
            # Use Python built-in timer to measure time cost
            stop_t = time.time()
            self._records[opcode][-1].stop_t = stop_t
            self._records[opcode][-1].elapsed = (stop_t - self._records[opcode][-1].start_t)


global_timer = Timer()
""" Object of global timer. """
