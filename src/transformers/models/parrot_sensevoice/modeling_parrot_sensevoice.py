# coding=utf-8
# Copyright 2025 the Bairong Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ParrotSenseVoice model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_parrot_sensevoice import ParrotSenseVoiceConfig
from .sensevoice_encoder import SenseVoiceEncoderSmall


PARROTSENSEVOICE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ParrotAudioConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare ParrotSenseVoice Model outputting raw hidden-states without any specific head on top.",
    PARROTSENSEVOICE_START_DOCSTRING,
)
class ParrotSenseVoicePreTrainedModel(PreTrainedModel):
    config_class = ParrotSenseVoiceConfig
    # base_model_prefix = "model"
    base_model_prefix = "sense_voice_small"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_attention_backend = False

    def _init_weights(self, module):
        # important: this ported version of ParrotSenseVoice isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed
        std = self.config.init_std if hasattr(self.config, "init_std") else self.config.init_std

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()



PARROTSENSEVOICEENCODER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ParrotSenseVoiceConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    """The audio model from ParrotSenseVoice without any head or projection on top.""",
    PARROTSENSEVOICEENCODER_START_DOCSTRING,
)
class ParrotSenseVoiceEncoder(ParrotSenseVoicePreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`ParrotSenseVoiceEncoderLayer`].

    Args:
        config: ParrotSenseVoiceConfig
    """

    # Ignore copy
    config_class = ParrotSenseVoiceConfig
    main_input_name = "input_features"
    _no_split_modules = ["EncoderLayerSANM"]

    def __init__(self, config: ParrotSenseVoiceConfig):
        super().__init__(config)
        self.sense_voice_small = SenseVoiceEncoderSmall(
            input_size=config.input_size,
            output_size=config.output_size,
            attention_heads=config.attention_heads,
            linear_units=config.linear_units,
            num_blocks=config.num_blocks,
            tp_blocks=config.tp_blocks,
            dropout_rate=config.dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            normalize_before=config.normalize_before,
            kernel_size=config.kernel_size,
            sanm_shfit=config.sanm_shfit
        )

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    # TODO: refine output type to BaseModelOutput
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs_pad, olens, _ = self.sense_voice_small(
            input_features,
            ilens = audio_feature_lengths
        )
        return xs_pad, olens

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Computes the output length of the parrot sensevoice encoder
        """
        return input_lengths, input_lengths


__all__ = ["ParrotSenseVoiceEncoder", "ParrotSenseVoicePreTrainedModel"]
