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
"""Parrot2Audio model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.parrot_sensevoice.configuration_parrot_sensevoice import ParrotSenseVoiceConfig
from transformers.models.qwen3 import Qwen3Config
from transformers.utils import logging


logger = logging.get_logger(__name__)


# class Parrot2AudioEncoderConfig(PretrainedConfig):
#     r"""
#     This is the configuration class to store the configuration of a [`Parrot2AudioEncoder`]. It is used to instantiate a
#     Parrot2Audio audio encoder according to the specified arguments, defining the model architecture. Instantiating a
#     configuration with the defaults will yield a similar configuration to that of the audio encoder of the Parrot2Audio
#     architecture.

#     e.g. [bairong-inc/Parrot2Audio-14B-QA](https://huggingface.co/bairong-inc/Parrot2Audio-14B-QA)

#     Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
#     documentation from [`PretrainedConfig`] for more information.

#     Args:
#         attention_dropout_rate (`float`, *optional*, defaults to 0.1):
#             The dropout ratio for the attention probabilities.
#         attention_heads (`int`, *optional*, defaults to 4):
#             Number of attention heads for each attention layer in the Transformer encoder.
#         dropout_rate (`float`, *optional*, defaults to 0.1):
#             The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
#         input_layer (`str`, *optional*, defaults to 'pe'):
#             The type of input layer to use.
#         input_size (`int`, *optional*, defaults to 560):
#             The size of the input features.
#         kernel_size (`int`, *optional*, defaults to 11):
#             The kernel size for the convolutional layers.
#         linear_units (`int`, *optional*, defaults to 2048):
#             The number of units in the linear layers.
#         normalize_before (`bool`, *optional*, defaults to True):
#             Whether to normalize the input before the first layer.
#         num_blocks (`int`, *optional*, defaults to 50):
#             The number of blocks in the encoder.
#         output_size (`int`, *optional*, defaults to 512):
#             The size of the output features.
#         pos_enc_class (`str`, *optional*, defaults to 'SinusoidalPositionEncoder'):
#             The class of the position encoder.
#         positional_dropout_rate (`float`, *optional*, defaults to 0.1):
#             The dropout ratio for the positional encodings.
#         sanm_shfit (`int`, *optional*, defaults to 0):
#             The shift size for the SANM layers.
#         selfattention_layer_type (`str`, *optional*, defaults to 'sanm'):
#             The type of self-attention layer to use.
#         tp_blocks (`int`, *optional*, defaults to 20):
#             The number of blocks in the Transformer encoder.
#         adaptor_ffn_dim (`int`, *optional*, defaults to 8192):
#             The number of units in the feed-forward network.

#     Example:

#     ```python
#     >>> from transformers import Parrot2AudioEncoderConfig, Parrot2AudioEncoder

#     >>> # Initializing a Parrot2AudioEncoderConfig
#     >>> configuration = Parrot2AudioEncoderConfig()

#     >>> # Initializing a ParrotAudioEncoder (with random weights)
#     >>> model = Parrot2AudioEncoder(configuration)

#     >>> # Accessing the model configuration
#     >>> configuration = model.config
#     ```"""

#     model_type = "parrot2_audio_encoder"

#     def __init__(
#         self,
#         # occurs in config.json `audio_config`
#         attention_dropout_rate=0.1,
#         attention_heads=4,
#         dropout_rate=0.1,
#         input_layer='pe',
#         input_size=560,
#         kernel_size=11,
#         linear_units=2048,
#         normalize_before=True,
#         num_blocks=50,
#         output_size=512,
#         pos_enc_class='SinusoidalPositionEncoder',
#         positional_dropout_rate=0.1,
#         sanm_shfit=0,
#         selfattention_layer_type='sanm',
#         tp_blocks=20,
#         init_std=0.02,
#         # not in config.json `audio_config`
#         **kwargs,
#     ):
#         super().__init__(**kwargs)

#         self.attention_dropout_rate = attention_dropout_rate
#         self.attention_heads = attention_heads
#         self.dropout_rate = dropout_rate
#         self.input_layer = input_layer
#         self.input_size = input_size
#         self.kernel_size = kernel_size
#         self.linear_units = linear_units
#         self.normalize_before = normalize_before
#         self.num_blocks = num_blocks
#         self.output_size = output_size
#         self.pos_enc_class = pos_enc_class
#         self.positional_dropout_rate = positional_dropout_rate
#         self.sanm_shfit = sanm_shfit
#         self.selfattention_layer_type = selfattention_layer_type
#         self.tp_blocks = tp_blocks
#         self.init_std = init_std


Parrot2AudioEncoderConfig = ParrotSenseVoiceConfig


class Parrot2AudioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Parrot2AudioForConditionalGeneration`]. It is used to instantiate an
    Parrot2Audio model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Parrot2Audio.

    e.g. [bairong-inc/Parrot2Audio-14B-QA](https://huggingface.co/bairong-inc/Parrot2Audio-14B-QA)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `Parrot2AudioEncoderConfig`):
            The config object or dictionary of the audio backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen3Config`):
            The config object or dictionary of the text backbone.
        audio_token_index (`int`, *optional*, defaults to 151646):
            The audio token index to encode the audio prompt.

    Example:

    ```python
    >>> from transformers import Parrot2AudioForConditionalGeneration, Parrot2AudioConfig, Parrot2AudioEncoderConfig, Qwen3Config

    >>> # Initializing a Parrot2AudioEncoder config
    >>> audio_config = Parrot2AudioEncoderConfig()

    >>> # Initializing a Qwen3 config
    >>> text_config = Qwen3Config()

    >>> # Initializing a Parrot2Audio configuration
    >>> configuration = Parrot2AudioConfig(audio_config, text_config)

    >>> # Initializing a model from the parrot2_audio style configuration
    >>> model = Parrot2AudioForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "parrot2_audio"
    sub_configs = {"text_config": Qwen3Config, "audio_config": Parrot2AudioEncoderConfig}

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_index=151665,
        adaptor_ffn_dim=8192,
        **kwargs,
    ):
        self.audio_token_index = audio_token_index
        self.adaptor_ffn_dim = adaptor_ffn_dim

        if isinstance(audio_config, dict):
            audio_config = Parrot2AudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Parrot2AudioEncoderConfig(
                attention_dropout_rate=0.1,
                attention_heads=4,
                dropout_rate=0.1,
                input_layer='pe',
                input_size=560,
                kernel_size=11,
                linear_units=2048,
                normalize_before=True,
                num_blocks=50,
                output_size=512,
                pos_enc_class='SinusoidalPositionEncoder',
                positional_dropout_rate=0.1,
                sanm_shfit=0,
                selfattention_layer_type='sanm',
                tp_blocks=20,
            )

        self.audio_config = audio_config

        if isinstance(text_config, dict):
            # text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "qwen3"
            # text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            text_config = Qwen3Config(**text_config)
        elif text_config is None:
            # text_config = CONFIG_MAPPING["qwen3"]()
            text_config = Qwen3Config(
                vocab_size=151936,
                hidden_size=5120,
                intermediate_size=17408,
                num_hidden_layers=40,
                num_attention_heads=40,
                num_key_value_heads=8,
                hidden_act="silu",
                max_position_embeddings=45000,
                initializer_range=0.02,
                rms_norm_eps=1e-06,
                use_cache=True,
                tie_word_embeddings=False,
                rope_theta=1000000.0,
                rope_scaling=None,
                use_sliding_window=False,
                sliding_window=131072,
                max_window_layers=40,
                attention_dropout=0.0,
                torch_dtype="bfloat16",
                attention_bias=False,
                bos_token_id=151643,
                eos_token_id=151645,
                head_dim=128,
            )
        self.text_config = text_config

        super().__init__(**kwargs)


__all__ = ["Parrot2AudioConfig", "Parrot2AudioEncoderConfig"]
