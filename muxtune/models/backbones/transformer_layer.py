#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Megatron transformer layer. """

from typing import Any, Dict, Optional, Union, Callable, List

import torch
from torch import Tensor
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import BaseTransformerLayer
from megatron.core.transformer.transformer_config import TransformerConfig
# from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm
from megatron.core.utils import make_viewless_tensor
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

from muxtune.models.backbones.attention import SelfAttention
from muxtune.models.backbones.mlp import MLP
from muxtune.models.utils import SubModuleBase

__all__ = [ "TransformerLayer", ]


class TransformerLayer(MegatronModule, BaseTransformerLayer):
    """A single transformer layer with static submodules.

    Args:
        config: Transformer configuration in Megatron.
    """

    def __init__(
        self, 
        config: TransformerConfig, 
        layer_number: Optional[int] = None,
        hidden_dropout: Optional[float] = None,
    ) -> None:
        super().__init__(config=config)

        self.layer_number = layer_number
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # [Module 1: Input Layernorm] Optional Layernorm on the input data
        # self.input_layernorm = WrappedTorchLayerNorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        self.input_layernorm = torch.nn.LayerNorm(
            config.hidden_size, 
            eps=config.layernorm_epsilon, 
            device=torch.cuda.current_device(), 
            dtype=config.params_dtype,
        )

        # [Module 2: SelfAttention]
        self.self_attention = SelfAttention(config, layer_number)

        # [Module 3: BiasDropoutFusion]
        self.self_attn_bda = get_bias_dropout_add

        # [Module 7: Pre MLP] Optional Layernorm before MLP
        # self.pre_mlp_layernorm = WrappedTorchLayerNorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        self.pre_mlp_layernorm = torch.nn.LayerNorm(
            config.hidden_size, 
            eps=config.layernorm_epsilon, 
            device=torch.cuda.current_device(), 
            dtype=config.params_dtype,
        )

        # [Module 8: MLP block]
        self.mlp = MLP(config) if config.num_moe_experts is None else None

        # [Module 9: BiasDropoutFusion]
        self.mlp_bda = get_bias_dropout_add

        # self.is_moe_layer = isinstance(self.mlp, MoELayer)
        self.is_moe_layer = False
        
        self.bias_dropout_add_exec_handler = torch.enable_grad

        self.input_layernorm_module = InputLayerNormModule(
            ["hidden_states"], ["hidden_states", "residual", ], self.input_layernorm)
        self.attn_bias_dropout_add_module = BiasDropoutAddModule(
            ["hidden_states", "bias", "residual", ], ["hidden_states", ],
            self.bias_dropout_add_exec_handler, self.self_attn_bda, self.training, 
            self.config.bias_dropout_fusion, self.hidden_dropout,
        )
        self.pre_mlp_layernorm_module = PreMlpLayerNormModule(
            ["hidden_states"], ["hidden_states", "residual", ], self.pre_mlp_layernorm)
        self.mlp_bias_dropout_add_module = BiasDropoutAddModule(
            ["hidden_states", "bias", "residual", ], ["hidden_states", ],
            self.bias_dropout_add_exec_handler, self.mlp_bda, self.training, 
            self.config.bias_dropout_fusion, self.hidden_dropout,
        )

    def forward(self, *args, **kwargs):
        """ Forward method. 
        
        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        hidden_states, context = self._forward_attention(*args, **kwargs)
        output = self._forward_mlp(hidden_states)
        return output, context

    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                hidden_states (Tensor): Transformed hidden states before the MLP layernorm.
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """

        # Residual connection.
        residual = hidden_states

        # TODO(chunyu): Add custom backward function for TP all-reduce in backward pass.

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        return hidden_states, None

    def _forward_mlp(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        """
        Perform a forward pass through the MLP layer and the layernorms before and after
        the MLP operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.

        Returns:
            Tensor: Transformed hidden states after the MLP layernorm.
        """

        # Residual connection.
        residual = hidden_states

        # Optional Pre MLP Layer norm
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP block
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
    
        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output

    def get_submodules(self):
        """ Get sub-modules after graph partitioning. """
        submodules = []
        submodules.append(self.input_layernorm_module)
        submodules.extend(self.self_attention.get_submodules())
        submodules.append(self.attn_bias_dropout_add_module)
        submodules.append(self.pre_mlp_layernorm_module)
        submodules.extend(self.mlp.get_submodules())
        submodules.append(self.mlp_bias_dropout_add_module)
        return submodules


class InputLayerNormModule(SubModuleBase):

    module_type = "compute"

    def __init__(self, input_keywords: List[str], output_keywords: List[str], 
                 input_layernorm: torch.nn.LayerNorm):
        super().__init__(input_keywords, output_keywords)
        self.input_layernorm = input_layernorm

    def forward(self, intermediate_: Dict[str, Any]):
        (input_, ) = self.preprocess(intermediate_)
        residual = input_
        output_ = self.input_layernorm(input_)
        return self.postprocess(intermediate_, [output_, residual, ])


class PreMlpLayerNormModule(SubModuleBase):

    module_type = "compute"

    def __init__(self, input_keywords: List[str], output_keywords: List[str], 
                 pre_mlp_layernorm: torch.nn.LayerNorm):
        super().__init__(input_keywords, output_keywords)
        self.pre_mlp_layernorm = pre_mlp_layernorm

    def forward(self, intermediate_: Dict[str, Any]):
        (input_, ) = self.preprocess(intermediate_)
        residual = input_
        output_ = self.pre_mlp_layernorm(input_)
        return self.postprocess(intermediate_, [output_, residual, ])


class BiasDropoutAddModule(SubModuleBase):

    module_type = "compute"

    def __init__(
        self, input_keywords: List[str], output_keywords: List[str], 
        bias_dropout_add_exec_handler: Any, bda: Callable, 
        training: bool, bias_dropout_fusion: bool, hidden_dropout: float,
    ):
        super().__init__(input_keywords, output_keywords)
        self.bias_dropout_add_exec_handler = bias_dropout_add_exec_handler
        self.bda = bda
        self.training = training
        self.bias_dropout_fusion = bias_dropout_fusion
        self.hidden_dropout = hidden_dropout
    
    def forward(self, intermediate_: Dict[str, Any]):
        (input_, bias, residual, ) = self.preprocess(intermediate_)
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.bda(self.training, self.bias_dropout_fusion)(
                (input_, bias), residual, self.hidden_dropout,
            )
        
        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output_ = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        return self.postprocess(intermediate_, [output_, ])
