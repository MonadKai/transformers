# coding=utf-8
# Copyright 2025 the Bairong Inc. team. All rights reserved.
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
"""Parrot2AudioMoe model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.parrot_sensevoice.configuration_parrot_sensevoice import ParrotSenseVoiceConfig
from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


Parrot2AudioMoeEncoderConfig = ParrotSenseVoiceConfig


class Parrot2AudioMoeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Parrot2AudioMoeForConditionalGeneration`]. It is used to instantiate an
    Parrot2AudioMoe model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Parrot2AudioMoe.

    e.g. [bairong-inc/Parrot2AudioMoe-32B-A3B](https://huggingface.co/bairong-inc/Parrot2AudioMoe-32B-A3B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `Parrot2AudioMoeEncoderConfig`):
            The config object or dictionary of the audio backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen3Config`):
            The config object or dictionary of the text backbone.
        audio_token_index (`int`, *optional*, defaults to 151646):
            The audio token index to encode the audio prompt.

    Example:

    ```python
    >>> from transformers import Parrot2AudioMoeForConditionalGeneration, Parrot2AudioMoeConfig, Parrot2AudioMoeEncoderConfig, Qwen3MoeConfig

    >>> # Initializing a Parrot2AudioMoeEncoder config
    >>> audio_config = Parrot2AudioMoeEncoderConfig()

    >>> # Initializing a Qwen3Moe config
    >>> text_config = Qwen3MoeConfig()

    >>> # Initializing a Parrot2AudioMoe configuration
    >>> configuration = Parrot2AudioMoeConfig(audio_config, text_config)

    >>> # Initializing a model from the parrot2_audio style configuration
    >>> model = Parrot2AudioMoeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "parrot2_audio_moe"
    sub_configs = {"text_config": Qwen3MoeConfig, "audio_config": Parrot2AudioMoeEncoderConfig}

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_index=151669,
        adaptor_ffn_dim=4096,
        **kwargs,
    ):
        self.audio_token_index = audio_token_index
        self.adaptor_ffn_dim = adaptor_ffn_dim

        if isinstance(audio_config, dict):
            audio_config = Parrot2AudioMoeEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Parrot2AudioMoeEncoderConfig(
                attention_dropout_rate=0.1,
                attention_heads=4,
                dropout_rate=0.1,
                input_layer="pe",
                input_size=560,
                kernel_size=11,
                linear_units=2048,
                normalize_before=True,
                num_blocks=50,
                output_size=512,
                pos_enc_class="SinusoidalPositionEncoder",
                positional_dropout_rate=0.1,
                sanm_shfit=0,
                selfattention_layer_type="sanm",
                tp_blocks=20,
                init_std=0.02,
            )

        self.audio_config = audio_config

        if isinstance(text_config, dict):
            # text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "qwen3_moe"
            # text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            text_config = Qwen3MoeConfig(**text_config)
        elif text_config is None:
            # text_config = CONFIG_MAPPING["qwen3_moe"]()
            text_config = Qwen3MoeConfig(
                vocab_size=151936,
                hidden_size=2048,
                intermediate_size=6144,
                num_hidden_layers=48,
                num_attention_heads=32,
                num_key_value_heads=4,
                hidden_act="silu",
                max_position_embeddings=40960,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=True,
                tie_word_embeddings=False,
                rope_theta=10000.0,
                rope_scaling=None,
                attention_bias=False,
                use_sliding_window=False,
                sliding_window=None,
                attention_dropout=0.0,
                decoder_sparse_step=1,
                moe_intermediate_size=768,
                num_experts_per_tok=8,
                num_experts=128,
                norm_topk_prob=True,
                output_router_logits=False,
                router_aux_loss_coef=0.001,
                mlp_only_layers=[],
                bos_token_id=151643,
                eos_token_id=151645,
                head_dim=128,
                max_window_layers=48,
            )
        self.text_config = text_config

        super().__init__(**kwargs)


__all__ = ["Parrot2AudioMoeConfig", "Parrot2AudioMoeEncoderConfig"]
