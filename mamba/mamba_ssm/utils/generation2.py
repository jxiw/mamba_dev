# Copyright (c) 2023, Albert Gu, Tri Dao.
import gc
import time
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Sequence, Union, List, Tuple, Dict

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.profiler import ProfilerActivity, profile, record_function
from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None
    # previous state memory
    draft_model: bool = True
    prev_memory_list: List[Dict[int, Tuple[Tensor, Tensor]]] = field(default_factory=list)
    # for verify state memory
    prev_key_value_memory_dict: dict = field(default_factory=dict)
    is_backtrack: bool = False
    is_save: bool = True

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L231
def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for none top-k values to -inf. Done in-place."""
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits.masked_fill_(indices_to_remove, float("-Inf"))


# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L170
def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf. Done in-place."""
    if top_p <= 0.0 or top_p >= 1.0:
        return
    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits.masked_fill_(indices_to_remove, float("-inf"))


def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, "top-p should be in (0, 1]."
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            if temperature != 1.0:
                logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1),
            ]
        else:
            # Clone so that when we modify for top_p we don't change the original logits
            logits_top = logits / temperature if temperature != 1.0 else logits.clone()
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(
                dim=-1
            )


@torch.inference_mode()
def decode(
    input_ids,
    model,
    max_length,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    eos_token_id=None,
    teacher_outputs=None,
    vocab_size=None,
    tensor_parallel=1,
    cg=False,
    enable_timing=False,
    verify_block=10,
    verifier=None,
    text_to_draft_tokens=None,
    draft_tokens_to_text=None,
    text_to_verifier_tokens=None,
    verifier_tokens_to_text=None,
    verifier_tokens_draft_tokens_dict=None,
    prompt_start_idx=None,
    is_boundary=None,
):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    batch_size, seqlen_og = input_ids.shape
    teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
    if cg:
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
            tensor_parallel=tensor_parallel,
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)
    else:
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

    def get_logits(input_ids, inference_params):
        decoding = inference_params.seqlen_offset > 0
        if decoding:
            position_ids = torch.full(
                (batch_size, 1),
                inference_params.seqlen_offset,
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            position_ids = None
        if not cg or not decoding:
            logits = model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
            ).logits.squeeze(dim=1)
        else:
            logits = model._decoding_cache.run(
                input_ids, position_ids, inference_params.seqlen_offset
            ).squeeze(dim=1)
        return logits[..., :vocab_size] if vocab_size is not None else logits

    def sample_tokens(logits, inference_params):
        if teacher_outputs is None or teacher_output_len <= inference_params.seqlen_offset:
            token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        else:
            token = teacher_outputs[:, inference_params.seqlen_offset]
        # return rearrange(token, "b -> b 1")
        return token.unsqueeze(1)

    def should_stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).all():
            return True
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        return False

    def get_mismatch_position(verify_model_logits, verifier_tokens, start_verify_idx, topK=2):
        # get the ranks of label output logits from draft model
        # within top K candidate of verify model
        verifier_tokens_logits = torch.gather(verify_model_logits, 2, verifier_tokens.unsqueeze(-1))
        ranks = ((verify_model_logits > verifier_tokens_logits).sum(dim=2))[:, start_verify_idx:]
        mismatch_position = (ranks >= topK).int()
        first_mismatch_indices = torch.argmax(mismatch_position, dim=1)
        print("ranks:", ranks)
        return ranks[0, first_mismatch_indices] >= topK, first_mismatch_indices

    start = torch.cuda.Event(enable_timing=enable_timing)
    end = torch.cuda.Event(enable_timing=enable_timing)

    if enable_timing:
        if tensor_parallel > 1:
            torch.distributed.barrier()
        start.record()
    
    scores, sequences = [], [input_ids]
    # output_sequences = []
    verify_current_step = 0
    print("====state====")
    
    verify_inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size, draft_model=False)
    
    while not should_stop(sequences[-1], inference_params):
        # keep decoding
        scores.append(get_logits(sequences[-1], inference_params))
        inference_params.seqlen_offset += sequences[-1].shape[1]
        sequences.append(sample_tokens(scores[-1], inference_params))
        
        if verify_current_step % verify_block == verify_block - 1:
            print("sequences:", sequences)
            
            # print("reach to block!")
            # print("length:", len(inference_params.prev_memory_list))
            # reach to the end of block
            # concat sequences
            if verify_current_step == verify_block - 1:
                # for first verification
                current_sequences = torch.cat(sequences, dim=1)
                print("verify sequence:", current_sequences)
            else:
                # for non first verification
                current_sequences = torch.cat(sequences, dim=1)[:, -verify_block:]
                print("verify sequence:", current_sequences)
            # 1. convert sequences tokens to text
            text = draft_tokens_to_text(current_sequences)
            print("draft model text:", text)
            # 2. convert text to byte tokens
            verifier_tokens = text_to_verifier_tokens(text)
            # print("verifier_tokens model text:", verifier_tokens)
            # 3. verify output using byte level model
            # we run our mambabyte model and see whether it matches the word level
            verify_model_logits = verifier(verifier_tokens[:, :-1], inference_params=verify_inference_params).logits
            # 4. get the mismatch position
            if verify_current_step == verify_block - 1:
                is_mismatch, mismatch_position = get_mismatch_position(verify_model_logits, verifier_tokens[:, 1:], start_verify_idx=prompt_start_idx - 1)
                mismatch_position = prompt_start_idx - 1 + mismatch_position
            else:
                is_mismatch, mismatch_position = get_mismatch_position(verify_model_logits, verifier_tokens[:, 1:], start_verify_idx=0)
            print("is_mismatch, mismatch_position:", is_mismatch, mismatch_position)
            if is_mismatch:
                verifier_matched_tokens = verifier_tokens[0, :mismatch_position + 1].tolist()
                print("verifier_matched_tokens:", verifier_matched_tokens)
                # update the verifier state
                verify_inference_params.is_backtrack = True
                print(verifier_tokens[:, :mismatch_position + 1])
                restart_token = torch.argmax(verifier(verifier_tokens[:, :mismatch_position + 1], inference_params=verify_inference_params).logits, dim=-1)
                # for unprocess tokens
                unprocess_tokens = []
                # tokens up to current position to text
                mismatch_token = torch.argmax(verify_model_logits[0, mismatch_position], dim=-1)
                # breakpoint()
                print("restart_token:", restart_token, "mismatch_token:", mismatch_token)
                unprocess_tokens.append(mismatch_token.item())
                print("unprocess_tokens:", unprocess_tokens)
                # decoding until reach to a empty space
                verify_inference_params.is_backtrack = False
                # verify_inference_params.is_save = False
                while not is_boundary(mismatch_token):
                    # print("match_token shape 1:", mismatch_token.shape)
                    verify_model_logits = verifier(mismatch_token[None, :], inference_params=verify_inference_params).logits
                    # print("verify_model_logits shape:", verify_model_logits.shape)
                    mismatch_token = torch.argmax(verify_model_logits, dim=-1).squeeze(dim=0)
                    # print("match_token shape 2:", mismatch_token.shape)
                    # breakpoint()
                    unprocess_tokens.append(mismatch_token.item())
                # verify_inference_params.is_save = True
                print("unprocess_tokens until space:", unprocess_tokens)
                token_dict = verifier_tokens_draft_tokens_dict(current_sequences)
                
                draft_last_match_position = len(token_dict) - 1
                for i, (start_verify_token, end_verify_token) in enumerate(token_dict):
                    if start_verify_token > mismatch_position:
                        draft_last_match_position = i - 1
                        break
                
                print("verifier_matched_tokens + unprocess_tokens:", verifier_matched_tokens, unprocess_tokens)
                
                corrected_text = verifier_tokens_to_text(verifier_matched_tokens + unprocess_tokens)
                print("corrected_text:", corrected_text)
                
                correct_draft_tokens = text_to_draft_tokens(corrected_text)
                print("correct_draft_tokens:", correct_draft_tokens)
                
                # match_text = verifier_tokens_to_text(verifier_matched_tokens.tolist())
                # print("match_text:", match_text)
                # match_draft_tokens = text_to_draft_tokens(match_text)
                # print("match_draft_tokens:", match_draft_tokens)
                # mismatch_draft_position = len(match_draft_tokens)
                # unprocess_text = verifier_tokens_to_text(unprocess_tokens)
                # print("unprocess_text:", unprocess_text)
                # breakpoint()
                # unprocess_draft_tokens = text_to_draft_tokens(unprocess_text)
                # breakpoint()
                
                unprocess_draft_tokens = correct_draft_tokens[draft_last_match_position + 1:]
                inference_params.key_value_memory_dict = inference_params.prev_memory_list[draft_last_match_position]
                # breakpoint()
                # sequences_to_add = []
                for unprocess_draft_token in unprocess_draft_tokens:
                    # for update hidden state
                    # sequences_to_add.append(unprocess_draft_token[None, :])
                    get_logits(unprocess_draft_token[None, :], inference_params)
                
                inference_params.prev_memory_list.clear()
                
                correct_draft_tokens = [correct_draft_tokens.reshape(1, 1) for correct_draft_tokens in correct_draft_tokens.unbind(dim=1)]
                sequences = sequences[:-verify_block] + correct_draft_tokens
                
                print("sequences after verify:", sequences)
                # output_sequences.append(torch.cat([match_draft_tokens, unprocess_draft_tokens], dim=-1))
                # print(output_sequences)
                # breakpoint()
                
                # element_to_pop = verify_block - mismatch_draft_position
                # sequences = sequences[:-element_to_pop] + sequences_to_add
                # print("sequences:", sequences)
                
                inference_params.seqlen_offset += (-verify_block + len(correct_draft_tokens))
            else:
                # clear cache and move to the next step and keep decoding
                inference_params.prev_memory_list.clear()
                # output_sequences.append(current_sequences)
                # print(output_sequences)
                # breakpoint()
        
        # go to next step
        verify_current_step += 1
        
    if enable_timing:
        end.record()
        if tensor_parallel > 1:
            torch.distributed.barrier()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")
    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    return output_cls(sequences=torch.cat(sequences, dim=1), scores=tuple(scores))


class GenerationMixin:
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError

    def generate(
        self,
        input_ids,
        max_length,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,
    ):
        output = decode(
            input_ids, self, max_length, top_k=top_k, top_p=top_p, temperature=temperature, **kwargs
        )
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences


def allocate_inference_cache(
    max_batch_size,
    max_seqlen,
    nheads,
    headdim,
    layers: Union[int, Sequence],
    device,
    dtype=torch.float16,
):
    assert dtype in [torch.float16, torch.bfloat16, torch.float32]
    kv_cache_shape = (max_batch_size, max_seqlen, 2, nheads, headdim)
    if isinstance(layers, int):
        layers = range(layers)
    return {i: torch.empty(kv_cache_shape, device=device, dtype=dtype) for i in layers}


@dataclass
class DecodingCGCache:
    max_batch_size: int = 0
    max_seqlen: int = 0
    device = None
    dtype = None
    callables: dict = field(default_factory=dict)
    mempool = None
    inference_params: Optional[InferenceParams] = None
    run: Optional[Callable] = None


@torch.inference_mode()
def update_graph_cache(
    model,
    cache,
    batch_size,
    seqlen_og,
    max_seqlen,
    decoding_seqlens=(1,),
    tensor_parallel=1,
    dtype=None,
    n_warmups=2,
):
    if cache is None:
        cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if dtype is None:
        dtype = param_example.dtype
    if (
        (device, dtype) != (cache.device, cache.dtype)
        or batch_size > cache.max_batch_size
        or max_seqlen > cache.max_seqlen
    ):  # Invalidate the cache
        cache.callables = {}
        cache.mempool = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        if hasattr(model, "allocate_inference_cache"):
            inf_cache = model.allocate_inference_cache(batch_size, max_seqlen, dtype)
        else:
            headdim = getattr(
                model.config,
                "head_dim",
                model.config.hidden_size // model.config.num_attention_heads,
            )
            inf_cache = allocate_inference_cache(
                batch_size,
                max_seqlen,
                model.config.num_attention_heads // tensor_parallel,
                headdim,
                model.config.num_hidden_layers,
                device,
                dtype,
            )
        lengths_per_sample = torch.full((batch_size,), seqlen_og, dtype=torch.int32, device=device)
        cache.inference_params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            seqlen_offset=seqlen_og,
            key_value_memory_dict=inf_cache,
            lengths_per_sample=lengths_per_sample,
        )
        cache.mempool = torch.cuda.graphs.graph_pool_handle()
    for decoding_seqlen in decoding_seqlens:
        if (batch_size, decoding_seqlen) not in cache.callables:
            cache.callables[batch_size, decoding_seqlen] = capture_graph(
                model,
                cache.inference_params,
                batch_size,
                max_seqlen,
                decoding_seqlen=decoding_seqlen,
                mempool=cache.mempool,
                n_warmups=n_warmups,
            )

    def dispatch(input_ids, position_ids, seqlen):
        batch_size, decoding_seqlen = input_ids.shape[:2]
        return cache.callables[batch_size, decoding_seqlen](input_ids, position_ids, seqlen)

    cache.run = dispatch
    cache.inference_params.seqlen_offset = 0  # Reset so it's not confusing
    return cache


def capture_graph(
    model, inference_params, batch_size, max_seqlen, decoding_seqlen=1, mempool=None, n_warmups=2
):
    device = next(iter(model.parameters())).device
    input_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    position_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    seqlen_offset_og = inference_params.seqlen_offset
    inference_params.seqlen_offset = max_seqlen - decoding_seqlen
    inference_params.lengths_per_sample[:] = inference_params.seqlen_offset

    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            logits = model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=decoding_seqlen,
            ).logits
        s.synchronize()
        # This might be needed for correctness if we run with NCCL_GRAPH_MIXING_SUPPORT=0,
        # which requires that graph launch and non-captured launch to not overlap (I think,
        # that's how I interpret the documentation). I'm not sure if this is required.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)
    # Captures the graph
    # To allow capture, automatically sets a side stream as the current stream in the context
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        logits = model(
            input_ids,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=decoding_seqlen,
        ).logits

    def run(new_input_ids, new_position_ids, seqlen):
        inference_params.lengths_per_sample[:] = seqlen
        input_ids.copy_(new_input_ids)
        position_ids.copy_(new_position_ids)
        graph.replay()
        return logits.clone()

    inference_params.seqlen_offset = seqlen_offset_og
    return run
