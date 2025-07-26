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
"""ParrotAudio model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.parrot_sensevoice.configuration_parrot_sensevoice import ParrotSenseVoiceConfig
from transformers.models.qwen2 import Qwen2Config
from transformers.utils import logging


logger = logging.get_logger(__name__)


ParrotAudioEncoderConfig = ParrotSenseVoiceConfig


class ParrotAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ParrotAudioForConditionalGeneration`]. It is used to instantiate an
    ParrotAudio model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ParrotAudio.

    e.g. [bairong-inc/ParrotAudio-14B-QA](https://huggingface.co/bairong-inc/ParrotAudio-14B-QA)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `ParrotAudioEncoderConfig`):
            The config object or dictionary of the audio backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        audio_token_index (`int`, *optional*, defaults to 151646):
            The audio token index to encode the audio prompt.

    Example:

    ```python
    >>> from transformers import ParrotAudioForConditionalGeneration, ParrotAudioConfig, ParrotAudioEncoderConfig, Qwen2Config

    >>> # Initializing a ParrotAudioEncoder config
    >>> audio_config = ParrotAudioEncoderConfig()

    >>> # Initializing a Qwen2 config
    >>> text_config = Qwen2Config()

    >>> # Initializing a ParrotAudio configuration
    >>> configuration = ParrotAudioConfig(audio_config, text_config)

    >>> # Initializing a model from the parrot_audio style configuration
    >>> model = ParrotAudioForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "parrot_audio"
    sub_configs = {"text_config": Qwen2Config, "audio_config": ParrotAudioEncoderConfig}

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
            audio_config = ParrotAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = ParrotAudioEncoderConfig(
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
            )

        self.audio_config = audio_config

        if isinstance(text_config, dict):
            # text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "qwen2"
            # text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            text_config = Qwen2Config(**text_config)
        elif text_config is None:
            # text_config = CONFIG_MAPPING["qwen2"]()
            text_config = Qwen2Config(
                vocab_size=152064,
                hidden_size=5120,
                intermediate_size=13824,
                num_hidden_layers=32,
                num_attention_heads=40,
                num_key_value_heads=8,
                hidden_act="silu",
                max_position_embeddings=32768,
                initializer_range=0.02,
                rms_norm_eps=1e-06,
                use_cache=True,
                tie_word_embeddings=False,
                rope_theta=1000000.0,
                rope_scaling=None,
                use_sliding_window=False,
                sliding_window=131072,
                max_window_layers=70,
                attention_dropout=0.0,
                torch_dtype="bfloat16",
                bos_token_id=151643,
                eos_token_id=151645,
            )
        self.text_config = text_config

        super().__init__(**kwargs)


__all__ = ["ParrotAudioConfig", "ParrotAudioEncoderConfig"]
