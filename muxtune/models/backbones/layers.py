#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Megatron parallel linear layers. """

import os
from typing import Literal, Optional

import torch
from torch.nn.parameter import Parameter
from torch._dynamo import allow_in_graph
from megatron.core import mpu
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

__all__ = [
    "LanguageModelEmbedding",
    "ColumnParallelLinear",
    "RowParallelLinear",
]


class LanguageModelEmbedding(MegatronModule):
    """ Megatron LanguageModelEmbedding. """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
    ):
        super().__init__(config)

        self.config = config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.add_position_embedding: bool = position_embedding_type == 'learned_absolute'

        # TODO(chunyu): Support Megatron's VocabParallelEmbedding.
        self.word_embeddings = torch.nn.Embedding(
            self.vocab_size,
            self.config.hidden_size,
        )
        if self.add_position_embedding:
            self.position_embeddings = torch.nn.Embedding(
                self.max_sequence_length,
                self.config.hidden_size,
            )
        
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(self.config.hidden_dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """ Forward of LanguageModelEmbedding. """

        # Word embeddings
        words_embeddings = self.word_embeddings(input_ids)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings
        
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()
        
        embeddings = self.embedding_dropout(embeddings)

        return embeddings


class ColumnParallelLinear(torch.nn.Module):
    """ Megatron ColumnParallelLinear. """

    def __init__(
        self, input_size, output_size, tp_size, bias, device, dtype, skip_bias_add=False, 
        is_expert=False, sequence_parallel=False, disable_grad_reduce=False, **kwargs,
    ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.output_size_per_partition = output_size // tp_size
        self.device = device
        self.dtype = dtype
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert

        self.weight = Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                device=self.device,
                dtype=self.dtype,
            )
        )

        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        else:
            self.register_parameter("bias", None)

        self.sequence_parallel = sequence_parallel
        self.disable_grad_reduce = disable_grad_reduce
        self.allreduce_dgrad = (
            tp_size > 1 and not self.sequence_parallel and not self.disable_grad_reduce
        )

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ):
        """ Forward of ColumnParallelLinear. 
        
        Args:
            input_:
                3D tensor whose order of dimension is [sequence, batch, hidden]
            weight (optional):
                weight tensor to use, compulsory when skip_weight_param_allocation is True.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `gather_output` arg in the constructor will be used.

        Returns:
            - output
            - bias
        """

        bias = self.bias if not self.skip_bias_add else None

        # NOTE(chunyu): We explicitly conduct TP backward allreduce here, instead of in 
        #   `linear_with_grad_accumulation_and_async_allreduce` as in Megatron.
        input_parallel = _CopyToModelParallelRegion.apply(input_)

        if not self.weight.requires_grad:
            raise NotImplementedError
        else:
            output_parallel = linear_with_grad_accumulation_and_async_allreduce(input_parallel, self.weight, bias)
        
        # TODO(chunyu): Add the autograd function for gather_from_tensor_model_parallel_region here.
        output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """ Megatron RowParallelLinear. """

    def __init__(
        self, input_size, output_size, tp_size, bias, device, dtype, input_is_parallel=False, 
        skip_bias_add=False, is_expert=False, sequence_parallel=False, **kwargs
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.input_size_per_partition = input_size // tp_size
        self.device = device
        self.dtype = dtype
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert

        self.weight = Parameter(
            torch.empty(
                self.output_size,
                self.input_size_per_partition,
                device=self.device,
                dtype=self.dtype,
            )
        )

        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        else:
            self.register_parameter("bias", None)

        self.sequence_parallel = sequence_parallel
        self.explicit_expert_comm = False
    
    def forward(self, input_: torch.Tensor):
        """ Forward of RowParallelLinear. """

        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            # TODO(chunyu): Add the autograd function for scatter_to_tensor_model_parallel_region() here.
            # input_parallel = scatter_to_tensor_model_parallel_region(input_, group=self.tp_group)
            raise NotImplementedError
        
        # Matrix multiply.
        allreduce_dgrad = False

        if not self.weight.requires_grad:
            raise NotImplementedError
        else:
            output_parallel = linear_with_grad_accumulation_and_async_allreduce(input_parallel, self.weight, bias=None)

        # All-reduce across all the partitions.
        if self.explicit_expert_comm:
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:
            raise NotImplementedError
        else:
            output_ = _ReduceFromModelParallelRegion.apply(output_parallel)
            # torch.distributed.all_reduce(output_parallel.contiguous(), group=mpu.get_tensor_model_parallel_group())
            # output_ = output_parallel

        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias


def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
):
    # TODO(chunyu): Currently we don't use customized autograd function.
    output = torch.matmul(input, weight.t())
    if bias is not None:
        output = output + bias
    return output


@allow_in_graph
class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return input_

    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _reduce(grad_output)


@allow_in_graph
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return grad_output


def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""
    group = mpu.get_tensor_model_parallel_group()
    assert group is not None, "group should not be None"

    # Bypass the function if we are using only 1 GPU.
    if group.size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_.contiguous(), group=group)
    return input_
