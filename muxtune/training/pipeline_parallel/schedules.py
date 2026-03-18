#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" Optimized implementations of Megatron pipeline schedules. """


from typing import Iterator, List, Union
import contextlib

import torch
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.schedules import (
    clear_embedding_activation_buffer, 
    get_tensor_shapes,
    forward_step, backward_step, 
    deallocate_output_tensor, 
    check_first_val_step,
    finish_embedding_wgrad_compute
)
from megatron.core.utils import (
    drain_embedding_wgrad_compute,
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    get_model_xattn,
)

from muxtune.training.pipeline_parallel import p2p_communication
from muxtune.global_envs import global_configs


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)
    encoder_decoder_xattn = get_model_xattn(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = recv_forward(recv_tensor_shapes, config)
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        send_forward(output_tensor, send_tensor_shapes, config)
        total_num_tokens += num_tokens.item()

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, config)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        total_num_tokens += num_tokens.item()

        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, config)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, config)

        else:
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, config)
            else:
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            send_backward(input_tensor_grad, recv_tensor_shapes, config)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store


def _broadcast(item):
    """ Broadcast item across tensor parallel group. """
    if not isinstance(item, list):
        item = [item]

    for item_ in item:
        if item_ is not None:
            torch.distributed.broadcast(
                tensor=item_, 
                src=parallel_state.get_tensor_model_parallel_src_rank(), 
                group=parallel_state.get_tensor_model_parallel_group(),             
            )


def recv_forward(tensor_shapes, config):
    input_tensors = []
    if (
        not global_configs.remove_p2p_comm_redundancy_in_hybrid_parallel or
        parallel_state.is_pipeline_first_stage() or
        parallel_state.get_tensor_model_parallel_rank() == 0
    ):
        for tensor_shape in tensor_shapes:
            if tensor_shape is None:
                input_tensors.append(None)
            else:
                input_tensors.append(p2p_communication.recv_forward(tensor_shape, config))

    if parallel_state.is_pipeline_first_stage():
        # No need to broadcast
        return input_tensors

    if (
        global_configs.remove_p2p_comm_redundancy_in_hybrid_parallel and 
        parallel_state.get_tensor_model_parallel_rank() == 0
    ):
        # Broadcast to other TP ranks
        _broadcast(input_tensors)

    elif (
        global_configs.remove_p2p_comm_redundancy_in_hybrid_parallel and 
        parallel_state.get_tensor_model_parallel_rank() > 0
    ):
        # Receive from TP rank 0
        for tensor_shape in tensor_shapes:
            if tensor_shape is None:
                input_tensors.append(None)
            else:
                input_tensor_ = torch.empty(
                    tensor_shape,
                    requires_grad=True,
                    device=torch.cuda.current_device(),
                    dtype=config.pipeline_dtype,
                )
                input_tensors.append(input_tensor_)

        _broadcast(input_tensors)

    return input_tensors


def recv_backward(tensor_shapes, config):
    output_tensor_grads = []
    if (
        not global_configs.remove_p2p_comm_redundancy_in_hybrid_parallel or
        parallel_state.is_pipeline_last_stage() or
        parallel_state.get_tensor_model_parallel_rank() == 0
    ):
        for tensor_shape in tensor_shapes:
            if tensor_shape is None:
                output_tensor_grads.append(None)
            else:
                output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape, config))
    
    if parallel_state.is_pipeline_last_stage():
        # No need to broadcast
        return output_tensor_grads

    if (
        global_configs.remove_p2p_comm_redundancy_in_hybrid_parallel and 
        parallel_state.get_tensor_model_parallel_rank() == 0
    ):
        # Broadcast to other TP ranks
        _broadcast(output_tensor_grads)

    elif (
        global_configs.remove_p2p_comm_redundancy_in_hybrid_parallel and 
        parallel_state.get_tensor_model_parallel_rank() > 0
    ):
        # Receive from TP rank 0
        for tensor_shape in tensor_shapes:
            if tensor_shape is None:
                output_tensor_grads.append(None)
            else:
                input_tensor_ = torch.empty(
                    tensor_shape,
                    requires_grad=True,
                    device=torch.cuda.current_device(),
                    dtype=config.pipeline_dtype,
                )
                output_tensor_grads.append(input_tensor_)

        _broadcast(output_tensor_grads)
    
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, config):
    if (
        not global_configs.remove_p2p_comm_redundancy_in_hybrid_parallel or 
        parallel_state.get_tensor_model_parallel_rank()  == 0   # Only TP rank 0 sends forward
    ):
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]
        for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
            if tensor_shape is None:
                continue
            p2p_communication.send_forward(output_tensor, config)
    

def send_backward(input_tensor_grads, tensor_shapes, config):
    if (
        not global_configs.remove_p2p_comm_redundancy_in_hybrid_parallel or 
        parallel_state.get_tensor_model_parallel_rank()  == 0   # Only TP rank 0 sends backward
    ):    
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]
        for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
            if tensor_shape is None:
                continue
            p2p_communication.send_backward(input_tensor_grad, config)


def send_forward_recv_backward(output_tensors, tensor_shapes, config):
    output_tensor_grads = []
    if (
        not global_configs.remove_p2p_comm_redundancy_in_hybrid_parallel or 
        parallel_state.is_pipeline_last_stage() or
        parallel_state.get_tensor_model_parallel_rank()  == 0   # Only TP rank 0 sends-recv
    ):
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]
        for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
            if tensor_shape is None:
                output_tensor_grads.append(None)
                continue
            output_tensor_grad = p2p_communication.send_forward_recv_backward(
                output_tensor, tensor_shape, config
            )
            output_tensor_grads.append(output_tensor_grad)

    if parallel_state.is_pipeline_last_stage():
        # No need to broadcast
        return output_tensor_grads
        
    if (
        global_configs.remove_p2p_comm_redundancy_in_hybrid_parallel and 
        parallel_state.get_tensor_model_parallel_rank() == 0
    ):
        # Broadcast to other TP ranks
        _broadcast(output_tensor_grads)

    elif (
        global_configs.remove_p2p_comm_redundancy_in_hybrid_parallel and 
        parallel_state.get_tensor_model_parallel_rank() > 0
    ):
        # Receive from TP rank 0
        for tensor_shape in tensor_shapes:
            if tensor_shape is None:
                output_tensor_grads.append(None)
            else:
                input_tensor_ = torch.empty(
                    tensor_shape,
                    requires_grad=True,
                    device=torch.cuda.current_device(),
                    dtype=config.pipeline_dtype,
                )
                output_tensor_grads.append(input_tensor_)

        _broadcast(output_tensor_grads)
    
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, config):
    input_tensors = []
    if (
        not global_configs.remove_p2p_comm_redundancy_in_hybrid_parallel or
        parallel_state.is_pipeline_first_stage() or
        parallel_state.get_tensor_model_parallel_rank() == 0
    ):
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]
        for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
            if tensor_shape is None:
                input_tensors.append(None)
                continue
            input_tensor = p2p_communication.send_backward_recv_forward(
                input_tensor_grad, tensor_shape, config
            )
            input_tensors.append(input_tensor)
    
    if parallel_state.is_pipeline_first_stage():
        # No need to broadcast
        return input_tensors

    if (
        global_configs.remove_p2p_comm_redundancy_in_hybrid_parallel and 
        parallel_state.get_tensor_model_parallel_rank() == 0
    ):
        # Broadcast to other TP ranks
        _broadcast(input_tensors)

    elif (
        global_configs.remove_p2p_comm_redundancy_in_hybrid_parallel and 
        parallel_state.get_tensor_model_parallel_rank() > 0
    ):
        # Receive from TP rank 0
        for tensor_shape in tensor_shapes:
            if tensor_shape is None:
                input_tensors.append(None)
            else:
                input_tensor_ = torch.empty(
                    tensor_shape,
                    requires_grad=True,
                    device=torch.cuda.current_device(),
                    dtype=config.pipeline_dtype,
                )
                input_tensors.append(input_tensor_)

        _broadcast(input_tensors)

    return input_tensors
