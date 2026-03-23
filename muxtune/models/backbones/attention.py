#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Megatron attention modules. """

import os
from typing import Any, Dict, Optional, Union, Tuple, Callable, List
import math

import torch
from torch import Tensor
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
# from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.utils import make_viewless_tensor
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer.utils import attention_mask_func
from megatron.core import parallel_state, tensor_parallel

from muxtune.models.backbones.layers import ColumnParallelLinear, RowParallelLinear
from muxtune.models.utils import SubModuleBase

__all__ = [ "SelfAttention", ]


class SelfAttention(MegatronModule):
    """ Self-attention module.

    Args:
        config: Transformer configuration in Megatron.
    """

    def __init__(
        self, 
        config: TransformerConfig, 
        layer_number: Optional[int] = None,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        attention_type: str = 'self',
    ) -> None:
        super().__init__(config=config)

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        # Per attention head and per partition values
        world_size = self.config.tensor_model_parallel_size
        assert self.config.num_attention_heads % world_size == 0
        self.hidden_size_per_attention_head = self.query_projection_size // self.config.num_attention_heads
        assert self.config.num_attention_heads % world_size == 0
        self.num_attention_heads_per_partition = self.config.num_attention_heads // world_size
        assert self.config.num_query_groups % world_size == 0
        self.num_query_groups_per_partition = self.config.num_query_groups // world_size
        
        self.linear_qkv = ColumnParallelLinear(
            input_size=self.config.hidden_size, 
            output_size=(self.query_projection_size + 2 * self.kv_projection_size),
            tp_size=self.config.tensor_model_parallel_size,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
            skip_bias_add=False,
        )
        self.q_layernorm = None
        self.k_layernorm = None

        self.core_attention = DotProductAttention(
            self.config, layer_number, attn_mask_type, attention_type,
        )

        self.linear_proj = RowParallelLinear(
            input_size=self.query_projection_size,
            output_size=self.config.hidden_size,
            tp_size=self.config.tensor_model_parallel_size,
            bias=self.config.add_bias_linear,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
            input_is_parallel=True,
            skip_bias_add=False,
        )

        self.attn_qkv_split_module = AttnQKVSplitModule(
            ["hidden_states", ], ["query", "key", "value", ],
            self.num_query_groups_per_partition, self.num_attention_heads_per_partition, 
            self.hidden_size_per_attention_head, True, self.q_layernorm, self.k_layernorm,
        )
        self.core_attn_module = CoreAttnModule(
            ["query", "key", "value", "attention_mask", ], ["hidden_states", "attention_mask", ], 
            self.core_attention, self.attn_mask_type,
        )
    
    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, split_qkv=True):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`. If `split_qkv=False`, then
        the unsplit mixed_qkv tensor is returned.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        # Return unsplit mixed_qkv and split_arg_list
        if not split_qkv:
            return mixed_qkv, split_arg_list
        
        # [sq, b, ng, (np/ng + 2) * hn]
        # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)
        
        return query, key, value

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """ Forward method.
        
        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Optional[Tensor]): Attention bias.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.
        """
        split_qkv = True

        # Query, key, and value
        qkv_output = self.get_query_key_value_tensors(
            hidden_states, key_value_states, split_qkv=True,    # TODO(chunyu): Current split qkv for self-attn
        )

        if split_qkv:
            query, key, value = qkv_output
        else:
            raise NotImplementedError("Currently only split_qkv=True is supported.")
        
        # Core attention computation
        core_attn_out = self.core_attention(
            query, key, value, attention_mask, attn_mask_type=self.attn_mask_type,
        )

        # Output. [s, b, h]
        output, bias = self.linear_proj(core_attn_out)

        return output, bias
    
    def get_submodules(self):
        """ Get sub-modules after graph partitioning. """
        submodules = []
        submodules.extend(self.linear_qkv.get_submodules())
        submodules.append(self.attn_qkv_split_module)
        submodules.append(self.core_attn_module)
        submodules.extend(self.linear_proj.get_submodules())
        return submodules


class DotProductAttention(MegatronModule):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models:
    https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
    ):
        super().__init__(config)
        
        self.config: TransformerConfig = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        world_size = self.config.tensor_model_parallel_size
        assert projection_size % world_size == 0
        self.hidden_size_per_partition = projection_size // world_size
        assert projection_size % self.config.num_attention_heads == 0
        self.hidden_size_per_attention_head = projection_size // self.config.num_attention_heads
        assert self.config.num_attention_heads % world_size == 0
        self.num_attention_heads_per_partition = self.config.num_attention_heads // world_size
        assert self.config.num_query_groups % world_size == 0
        self.num_query_groups_per_partition = self.config.num_query_groups // world_size

        coeff = None
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
        )
        
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward method.

        Args:
            query (Tensor): Query tensor of shape [sq, b, np, hn].
            key (Tensor): Key tensor of shape [sk, b, np, hn].
            value (Tensor): Value tensor of shape [sk, b, np, hn].
            attention_mask (Optional[Tensor]): Attention mask.
            attn_mask_type (Optional[AttnMaskType]): Attention mask type.
        """
        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        # [b, np, sq, sk]
        output_size = (query.size(1), query.size(2), query.size(0), key.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use
        # simple strides to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        # Megatron's global memory buffer is located on cuda device by default
        # We need to explicitly overwrite this buffer ouside model definition
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu"
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),  # [b * np, sq, hn]
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=self.softmax_scale,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:
            # with tensor_parallel.random.get_cuda_rng_tracker().fork():
            # NOTE: We give up rng tracker (necessary for tensor correctness) for compilation
            attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value.size(1), value.size(2), query.size(0), value.size(3))

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context


class AttnQKVSplitModule(SubModuleBase):
    
    module_type = "compute"

    def __init__(
        self, 
        input_keywords: List[str], 
        output_keywords: List[str], 
        num_query_groups_per_partition: int, 
        num_attention_heads_per_partition: int,
        hidden_size_per_attention_head: int,
        split_qkv: bool = True,
        q_layernorm: Callable = None,
        k_layernorm: Callable = None,
    ):
        super().__init__(input_keywords, output_keywords)
        self.num_query_groups_per_partition = num_query_groups_per_partition
        self.num_attention_heads_per_partition = num_attention_heads_per_partition
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.split_qkv = split_qkv
        self.q_layernorm = q_layernorm
        self.k_layernorm = k_layernorm
    
    def forward(self, intermediate_: Dict[str, Any]):
        (mixed_qkv, ) = self.preprocess(intermediate_)
        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        # Return unsplit mixed_qkv and split_arg_list
        if not self.split_qkv:
            intermediate_["mixed_qkv"] = mixed_qkv
            intermediate_["split_arg_list"] = split_arg_list
            return intermediate_
        
        # [sq, b, ng, (np/ng + 2) * hn]
        # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        return self.postprocess(intermediate_, [query, key, value, ])


class CoreAttnModule(SubModuleBase):

    module_type = "compute"

    def __init__(
        self, 
        input_keywords: List[str], 
        output_keywords: List[str], 
        core_attention: DotProductAttention, 
        attn_mask_type: str,
    ):
        super().__init__(input_keywords, output_keywords)
        self.core_attention = core_attention
        self.attn_mask_type = attn_mask_type
    
    def forward(self, intermediate_: Dict[str, Any]):
        (query, key, value, attention_mask, ) = self.preprocess(intermediate_)
        output_ = self.core_attention(query, key, value, attention_mask, self.attn_mask_type)
        return self.postprocess(intermediate_, [output_, attention_mask])
