#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Megatron-native GPT model. """

from collections import OrderedDict
from typing import Dict, Literal, Optional, Union

import torch
from torch import Tensor

from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import mpu
from megatron.core.transformer.enums import ModelType

from muxtune.models.backbones.layers import ColumnParallelLinear, LanguageModelEmbedding
from muxtune.models.backbones.transformer_block import TransformerBlock

__all__ = [ "GPTModel", ]


class GPTModel(LanguageModule):
    """ GPT model with static submodules. 
    
    Args:
        config (TransformerConfig):
            Transformer config.
        vocab_size (int):
            Vocabulary size.
        max_sequence_length (int):
            maximum size of sequence. This is used for positional embedding.
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        parallel_output (bool, optional):
            Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional):
            When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type.. Defaults to 'learned_absolute'.
        device (Union[str, torch.device], optional):
            Device to use. Defaults to gloal_configs.current_device.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            'learned_absolute', 'rope', 'mrope', 'yarn', 'none'
        ] = 'learned_absolute',
        # scatter_embedding_sequence_parallel: bool = True,
        device: Union[str, torch.device] = torch.cuda.current_device(),
    ) -> None:
        super().__init__(config)

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.max_position_embeddings = max_sequence_length
        if hasattr(self.config, 'position_embedding_type'):
            self.position_embedding_type = self.config.position_embedding_type
        else:
            self.position_embedding_type = position_embedding_type
        
        # megatron core pipelining currently depends on model type
        self.model_type = ModelType.encoder_or_decoder

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

        # TODO(chunyu): Support rope and mrope position embeddings.

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # Output
        if self.post_process:
            self.output_layer = ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config.tensor_model_parallel_size,
                bias=False,
                device=device,
                dtype=self.config.params_dtype,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
            )
        
        embedding = self.embedding if self.pre_process else None
        output_layer_weight = self.output_layer.weight if post_process else None
        self.pre_process_module = GPTModelPreProcessModule(
            embedding, self.config.params_dtype)
        self.post_process_module = GPTModelPostProcessModule(
            embedding, output_layer_weight, self.share_embeddings_and_output_weights, 
            self.pre_process, self.post_process)
        self.compute_loss_module = ComputeLossModule() if self.post_process else None
    
    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def _preprocess(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        decoder_input: Tensor = None,
    ):
        """Preprocesses inputs for the transformer decoder.

        Applies embeddings to input tokens, or uses `decoder_input` from a previous
        pipeline stage. Also sets up rotary positional embeddings.
        """
        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None
        
        # Convert to params_dtype
        if decoder_input is not None and decoder_input.dtype != self.config.params_dtype:
            decoder_input = decoder_input.to(self.config.params_dtype)

        return decoder_input

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        loss_mask: Optional[Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
        decoder_input = self._preprocess(input_ids, position_ids, decoder_input)
        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
        )
        return self._postprocess(hidden_states, labels, runtime_gather_output)

    def _postprocess(
        self,
        hidden_states,
        labels,
        runtime_gather_output=None,
    ):
        """Postprocesses decoder hidden states to generate logits or compute loss.

        Applies Multi-Token Prediction if enabled, generates output logits through
        the output layer, and computes language model loss when labels are provided.
        """
        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if not self.post_process:
            return hidden_states
        
        logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)
        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)
        return loss
    
    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the embedding weight or output logit weights when share input embedding and
        output weights set to True or when use Multi-Token Prediction (MTP) feature.

        Returns:
            Tensor: During pre processing or MTP process it returns the input embeddings weight.
            Otherwise, during post processing it returns the final output layers weight.
        """
        if self.pre_process:
            # Multi-Token Prediction (MTP) need both embedding layer and output layer.
            # So there will be both embedding layer and output layer in the mtp process stage.
            # In this case, if share_embeddings_and_output_weights is True, the shared weights
            # will be stored in embedding layer, and output layer will not have any weight.
            assert hasattr(
                self, 'embedding'
            ), f"embedding is needed in this pipeline stage, but it is not initialized."
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    def compute_language_model_loss(self, labels: Tensor, logits: Tensor) -> Tensor:
        """ Computes the language model loss (Cross entropy across vocabulary)

        Args:
            labels (Tensor): The labels of dimension [batch size, seq length]
            logits (Tensor): The final logits returned by the output layer of the transformer model

        Returns:
            Tensor: Loss tensor of dimensions [batch size, sequence_length]

        TODO(chunyu): Support cross_entropy_loss_fusion (currently incompatible).
        """
        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        # [s, b, vocab] -> [s, vocab, b]
        logits = logits.transpose(1, 2).contiguous()    
        torch_loss_func = torch.nn.CrossEntropyLoss()
        loss = torch_loss_func(logits, labels)

        # [s b] => [b, s]
        # loss = loss.transpose(0, 1).contiguous()
        return loss   

    def get_submodules(self):
        """ Get sub-modules after graph partitioning. """
        submodules = []
        submodules.append(self.pre_process_module)
        submodules.extend(self.decoder.get_submodules())
        submodules.append(self.post_process_module)
        if self.post_process:
            submodules.extend(self.output_layer.get_submodules())
            submodules.append(self.compute_loss_module)
        return submodules


class GPTModelPreProcessModule(torch.nn.Module):

    module_type = "compute"

    def __init__(self, embedding: LanguageModelEmbedding, params_dtype: torch.dtype):
        super().__init__()
        self.embedding = embedding
        self.params_dtype = params_dtype
    
    def forward(
        self, 
        input_ids: Tensor,
        position_ids: Tensor,
        decoder_input: Tensor = None,
        *args, **kwargs,
    ):
        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None
        
        # Convert to params_dtype
        if decoder_input is not None and decoder_input.dtype != self.params_dtype:
            decoder_input = decoder_input.to(self.params_dtype)

        return decoder_input, args, kwargs


class GPTModelPostProcessModule(torch.nn.Module):

    module_type = "compute"

    def __init__(
        self, 
        embedding: LanguageModelEmbedding,
        output_layer_weight: torch.Tensor,
        share_embeddings_and_output_weights: bool,
        pre_process: bool,
        post_process: bool,
    ):
        super().__init__()
        self.embedding = embedding
        self.output_layer_weight = output_layer_weight
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.pre_process =pre_process
        self.post_process = post_process
    
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            if self.pre_process:
                # Multi-Token Prediction (MTP) need both embedding layer and output layer.
                # So there will be both embedding layer and output layer in the mtp process stage.
                # In this case, if share_embeddings_and_output_weights is True, the shared weights
                # will be stored in embedding layer, and output layer will not have any weight.
                assert hasattr(
                    self, 'embedding'
                ), f"embedding is needed in this pipeline stage, but it is not initialized."
                output_weight = self.embedding.word_embeddings.weight
            elif self.post_process:
                output_weight = self.output_layer_weight
            else:
                output_weight = None

        return hidden_states, output_weight, args, kwargs,


class ComputeLossModule(torch.nn.Module):

    module_type = "compute"

    def __init__(self, ):
        super().__init__()
    
    def forward(self, logits: torch.Tensor, *args, **kwargs):
        labels = kwargs.get("labels", None)
        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()
        
        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        # [s, b, vocab] -> [s, vocab, b]
        logits = logits.transpose(1, 2).contiguous()    
        torch_loss_func = torch.nn.CrossEntropyLoss()
        loss = torch_loss_func(logits, labels)

        # [s b] => [b, s]
        # loss = loss.transpose(0, 1).contiguous()
        return loss
