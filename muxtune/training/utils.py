#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
from typing import Callable, Iterator, Optional, List, Tuple, Union, Dict
import enum
import random
from collections import OrderedDict
import numpy as np
import time
from dataclasses import dataclass
import logging

import torch
from torch.utils.data import DataLoader, Dataset
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import Float16Module
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDatasetConfig
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.optimizer import OptimizerConfig, MegatronOptimizer, get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.distributed import (
    DistributedDataParallel, DistributedDataParallelConfig)
from megatron.core.utils import get_attr_wrapped_model

__all__ = [
    "ModelType",
    "initialize_distributed",
    "setup_model", 
    "get_train_data_iterator", 
    "get_input_batch",
    "log_on_rank",
]

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


class ModelType(enum.Enum):
    """ Type class of the model type. """

    GPT = enum.auto()
    LLAMA = enum.auto()
    MoE = enum.auto()


def initialize_distributed(
    tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1, 
    virtual_pipeline_model_parallel_size: int = None, tp_comm_overlap: bool = False,
):
    """ Initialize the Megatron backend. """
    torchrun_world_size = int(os.environ['WORLD_SIZE'])
    # Torch setup for distributed training
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(world_size=torchrun_world_size, rank=global_rank, backend="nccl")

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size,
    )    


def setup_model(model: MegatronModule, wrap_with_ddp: bool = True) -> List[MegatronModule]:
    """ Prepare model setups. """
    if not isinstance(model, list):
        model = [model]

    megatron_config = get_attr_wrapped_model(model[0], attr="config")
    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
    
    # GPU allocation
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # FP16 conversion
    if (
        getattr(megatron_config, "dtype", torch.float16) == torch.float16 or
        getattr(megatron_config, "dtype", torch.float16) == torch.bfloat16
    ):
        model = [Float16Module(megatron_config, model_module) for model_module in model]
        # Empty CUDA cache released by FP16 conversion
        torch.cuda.empty_cache()

    if wrap_with_ddp:
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=getattr(megatron_config, "grad_reduce_in_fp32", False),
            overlap_grad_reduce=getattr(megatron_config, "overlap_grad_reduce", False),
            use_distributed_optimizer=getattr(megatron_config, "use_distributed_optimizer", False),
            check_for_nan_in_grad=getattr(megatron_config, "check_for_nan_in_grad", False),
        )
        # Wrap model
        model = [
            DistributedDataParallel(
                config=megatron_config,
                ddp_config=ddp_config,
                module=_model_chunk,
                # data_parallel_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                # expert_data_parallel_group=parallel_state.get_data_modulo_expert_parallel_group(),
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(_chunk_idx > 0)
            ) for (_chunk_idx, _model_chunk) in enumerate(model)
        ]
        # Broadcast params from data parallel src rank to other data parallel ranks
        if getattr(megatron_config, "data_parallel_random_init", False):
            for model_chunk in model:
                model_chunk.broadcast_params()

    return model


def get_train_data_iterator(
    vocab_size: int, global_batch_size, num_micro_batches, sequence_length: int, 
    random_seed: int = 1234, path_to_cache: str = None, num_iters: int = 10,
) -> Iterator:
    """ Get training data iterator for a single task. """

    # Dataset split
    num_train_valid_test_samples = [
        num_iters * global_batch_size,num_iters * global_batch_size, num_iters * global_batch_size,
    ]
    split_weights = [str(_w) for _w in num_train_valid_test_samples]
    dataset_split = ",".join(split_weights)
    
    # Dataset
    dc_kwargs = {
        "tokenizer": _build_tokenizer(),
        "sequence_length": sequence_length,
        "blend": None,
        "split": dataset_split,
        "random_seed": random_seed,
        "reset_position_ids": False,
        "reset_attention_mask": False,
        "eod_mask_loss": False,
        "path_to_cache": path_to_cache,
    }
    dataset_config = GPTDatasetConfig(**dc_kwargs)
    
    def __is_dataset_built_on_rank():
        """ 
        Returns True if the dataset should be built on the current rank and False 
        otherwise. It should be Megatron Core parallelism aware i.e. global rank, 
        local group rank, and virtual rank may inform its return value. 

        When interleaved pipeline is enabled, the dataset should be built on the
        first and last pipeline stage, instead of the virtual pipeline stage.
        """
        return (
            parallel_state.is_pipeline_first_stage(ignore_virtual=True) or 
            parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        ) and parallel_state.get_tensor_model_parallel_rank() == 0

    (train_dataset, _, _) = BlendedMegatronDatasetBuilder(
        cls=MockGPTDataset, sizes=num_train_valid_test_samples,
        is_built_on_rank=__is_dataset_built_on_rank,
        config=dataset_config,
    ).build()

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=(global_batch_size // num_micro_batches),
        shuffle=False,
    )
    return iter(train_dataloader)


def _build_tokenizer(
    tokenizer_type: str = "SentencePieceTokenizer",
    tokenizer_model_file_path: str = f"{CUR_PATH}/../datasets/tokenizers/tokenizer.model",
):
    """ Build the tokenizer for transformer model. """
    if tokenizer_type == "SentencePieceTokenizer":
        from muxtune.datasets.tokenizer import GPTSentencePieceTokenizer
        tokenizer = GPTSentencePieceTokenizer(tokenizer_model_file_path)
    else:
        raise TypeError(f"Invalid tokenizer type: {tokenizer_type}")

    return tokenizer


def get_input_batch(
    data_iterator: Iterator,
    micro_batch_size: int,
    seq_len: int,
    pipeline_model_parallel_size: int = 1,
    create_attention_mask_in_dataloader: bool = True,
) -> Dict[str, Optional[torch.Tensor]]:
    """ Get one batch of tokens, labels, loss_mask, attention_mask and position_ids. """

    if (
        (not parallel_state.is_pipeline_first_stage()) and
        (not parallel_state.is_pipeline_last_stage())
    ):
        # Intermediate stages with intermediate generated tensors as input
        return (None, None, None, None, None)
    
    # Get batch based on the rank of tensor parallel of this model partition
    batch = _get_batch_on_this_tp_rank(
        data_iterator, micro_batch_size, seq_len, pipeline_model_parallel_size, 
        create_attention_mask_in_dataloader,
    )

    return (
        batch["tokens"], batch["labels"], batch["loss_mask"], batch["attention_mask"], batch["position_ids"],
    )


def _get_batch_on_this_tp_rank(
    data_iterator: Iterator,
    micro_batch_size: int,
    seq_len: int,
    pipeline_model_parallel_size: int = 1,
    create_attention_mask_in_dataloader: bool = True,
) -> Dict[str, Optional[torch.Tensor]]:
    """ Get the input batch of this tensor parallel rank. """

    def __broadcast(item):
        """ Broadcast item across tensor parallel group. """
        if item is not None:
            torch.distributed.broadcast(
                tensor=item, 
                src=parallel_state.get_tensor_model_parallel_src_rank(), 
                group=parallel_state.get_tensor_model_parallel_group(),             
            )

    if parallel_state.get_tensor_model_parallel_rank() == 0:
        # Rank 0 of all tensor parallel ranks
        # Data item
        data = next(data_iterator) if data_iterator is not None else None

        # Construct batch
        batch = {
            "tokens": data["tokens"].cuda(non_blocking = True),
            "labels": data["labels"].cuda(non_blocking = True),
            "loss_mask": data["loss_mask"].cuda(non_blocking = True),
            "attention_mask": None if "attention_mask" not in data \
                else data["attention_mask"].cuda(non_blocking = True),
            "position_ids": data["position_ids"].cuda(non_blocking = True)
        }

        # Broadcast data to other ranks
        if pipeline_model_parallel_size == 1:
            # No pipeline
            __broadcast(batch["tokens"])
            __broadcast(batch["labels"])
            __broadcast(batch["loss_mask"])
            __broadcast(batch["attention_mask"])
            __broadcast(batch["position_ids"])

        elif parallel_state.is_pipeline_first_stage():
            # First stage with pipeline
            __broadcast(batch["tokens"])
            __broadcast(batch["attention_mask"])
            __broadcast(batch["position_ids"])

        elif parallel_state.is_pipeline_last_stage():
            # Last stage with pipeline
            __broadcast(batch["labels"])
            __broadcast(batch["loss_mask"])
            __broadcast(batch["attention_mask"])

    else:
        # Other ranks within tensor parallel
        # Placeholders
        tokens = torch.empty(
            (micro_batch_size, seq_len), dtype=torch.int64, device=torch.cuda.current_device(),
        )
        labels = torch.empty(
            (micro_batch_size, seq_len), dtype=torch.int64, device=torch.cuda.current_device(),
        )
        loss_mask = torch.empty(
            (micro_batch_size, seq_len), dtype=torch.float32, device=torch.cuda.current_device(),
        )
        position_ids = torch.empty(
            (micro_batch_size, seq_len), dtype=torch.int64, device=torch.cuda.current_device(),
        )
        attention_mask=torch.empty(
            (micro_batch_size, 1, seq_len, seq_len), dtype=torch.bool, device=torch.cuda.current_device(),
        ) if create_attention_mask_in_dataloader else None

        # Receive data from rank 0
        if pipeline_model_parallel_size == 1:
            # No pipeline
            __broadcast(tokens)
            __broadcast(labels)
            __broadcast(loss_mask)
            __broadcast(attention_mask)
            __broadcast(position_ids)

        elif parallel_state.is_pipeline_first_stage():
            # First stage with pipeline
            labels=None
            loss_mask=None
            __broadcast(tokens)
            __broadcast(attention_mask)
            __broadcast(position_ids)

        elif parallel_state.is_pipeline_last_stage():
            # Last stage with pipeline
            tokens=None
            position_ids=None
            __broadcast(labels)
            __broadcast(loss_mask)
            __broadcast(attention_mask)

        # Construct batch
        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }

    return batch


def log_on_rank(
    msg: str, 
    rank: Union[int, List[int]], 
    group_type: Union[str, List[str]], 
    logger: logging.Logger,
    log_level: str = "info",
): 
    """ Log message on the target rank. """
    
    if isinstance(rank, int):
        rank = [rank]
    if isinstance(group_type, str):
        group_type = [group_type]
    
    if log_level == "info":
        log_func = logger.info
    
    if torch.distributed.is_initialized():
        # Get rank based on parallel group type
        this_ranks = []
        for t in group_type:
            if t == "tensor":
                this_ranks.append(parallel_state.get_tensor_model_parallel_rank())
            elif t == "pipeline":
                this_ranks.append(parallel_state.get_pipeline_model_parallel_rank())
            else:
                raise TypeError(f"Invalid parallel group type: {t}")

        if all([this_ranks[_i] == rank[_i] for _i in range(len(rank))]):
            log_func(msg)
    else:
        log_func(msg)
