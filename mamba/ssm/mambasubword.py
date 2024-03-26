import math
import os
import json
from functools import partial

from collections import namedtuple
import torch.nn.functional as F

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel

from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.models.mixer_seq_simple import MixerModel
from mamba_ssm.utils.generation2 import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class SubWordMambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}
            
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - \
                (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=False, **factory_kwargs)

        if hasattr(config, 'gen'):
            self.gen = config.gen
        else:
            self.gen = False
            
        self.pad_id = self.config.pad_id
        
        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, return_loss=False, reduce="mean", is_first=None, eval_type=0, slider=2, num_last_tokens=0):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(
            input_ids, inference_params=inference_params)
        
        # for generation
        if self.gen and num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]

        lm_logits = self.lm_head(hidden_states)
        
        if self.gen:
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
            return CausalLMOutput(logits=lm_logits)

        if not return_loss:
            return lm_logits
        else:
            preds = rearrange(lm_logits, 'b n c -> b c n')
            labels = rearrange(input_ids, 'b ... -> b (...)')
            if is_first is not None:
                if eval_type == 0:
                    cross_entropy_loss = F.cross_entropy(
                        preds[..., :-1],
                        labels[..., 1:],
                        ignore_index=self.pad_id,
                        reduction="none",
                    )
                    avg_cross_entropy_loss_full = cross_entropy_loss.mean(
                        dim=1)

                    count_half = torch.count_nonzero(
                        labels[:, -cross_entropy_loss.shape[1]//slider:], dim=1)
                    count_half = torch.where(count_half == 0, torch.tensor(
                        1, device=count_half.device), count_half)  # to prevent nan

                    avg_cross_entropy_loss_half = cross_entropy_loss[:, -
                                                                     cross_entropy_loss.shape[1]//slider:].sum(dim=1) / count_half
                    loss = is_first * avg_cross_entropy_loss_full + \
                        (1 - is_first) * avg_cross_entropy_loss_half

                    return torch.mean(loss)

                else:

                    cross_entropy_loss = F.cross_entropy(
                        preds[..., :-1],
                        labels[..., 1:],
                        ignore_index=self.pad_id,
                        reduction="none",
                    )

                    half_length = cross_entropy_loss.shape[1]//slider
                    avg_cross_entropy_loss_full = cross_entropy_loss.sum(dim=1)
                    avg_cross_entropy_loss_half = cross_entropy_loss[:, -half_length:].sum(
                        dim=1)
                    loss = is_first * avg_cross_entropy_loss_full + \
                        (1 - is_first) * avg_cross_entropy_loss_half

                    count_full = torch.count_nonzero(
                        labels * is_first[:, None], dim=1)
                    count_half = torch.count_nonzero(
                        labels[:, -half_length:] * (1 - is_first[:, None]), dim=1)

                    total_byte = torch.sum(count_full + count_half)
                    return torch.sum(loss), total_byte

            if reduce == "mean":
                cross_entropy_loss = F.cross_entropy(
                    preds[..., :-1],
                    labels[..., 1:],
                    ignore_index=self.pad_id
                )
                return cross_entropy_loss
            else:
                cross_entropy_loss = F.cross_entropy(
                    preds[..., :-1],
                    labels[..., 1:],
                    reduction="sum",
                    ignore_index=self.pad_id
                )
                non_zero_count = torch.count_nonzero(labels[..., 1:])
                return cross_entropy_loss, non_zero_count

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config = load_config_hf(pretrained_model_name)
        model = cls(**config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(
            pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)
