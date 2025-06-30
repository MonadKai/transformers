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
from transformers.models.qwen2 import Qwen2Config
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ParrotAudioEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ParrotAudioEncoder`]. It is used to instantiate a
    ParrotAudio audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the ParrotAudio
    architecture.

    e.g. [bairong-inc/ParrotAudio-14B-QA](https://huggingface.co/bairong-inc/ParrotAudio-14B-QA)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `Qwen2AudioProcessor` class.
        encoder_layers (`int`, *optional*, defaults to 32):
            Number of encoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        d_model (`int`, *optional*, defaults to 1280):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.

    Example:

    ```python
    >>> from transformers import ParrotAudioEncoderConfig, ParrotAudioEncoder

    >>> # Initializing a ParrotAudioEncoderConfig
    >>> configuration = ParrotAudioEncoderConfig()

    >>> # Initializing a ParrotAudioEncoder (with random weights)
    >>> model = ParrotAudioEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "parrot_audio_encoder"

    def __init__(
        self,
        # occurs in config.json `audio_config`
        attention_dropout_rate= 0.1,
        attention_heads= 4,
        dropout_rate= 0.1,
        input_layer= 'pe',
        input_size= 560,
        kernel_size= 11,
        linear_units= 2048,
        normalize_before= True,
        num_blocks= 50,
        output_size= 512,
        pos_enc_class= 'SinusoidalPositionEncoder',
        positional_dropout_rate= 0.1,
        sanm_shfit= 0,
        selfattention_layer_type= 'sanm',
        tp_blocks= 20,
        adaptor_ffn_dim=2048,
        adaptor_downsample_rate=1,
        # not in config.json `audio_config`
        init_std=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.attention_dropout_rate = attention_dropout_rate
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.input_layer = input_layer
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.linear_units = linear_units
        self.normalize_before = normalize_before
        self.num_blocks = num_blocks
        self.output_size = output_size
        self.pos_enc_class = pos_enc_class
        self.positional_dropout_rate = positional_dropout_rate
        self.sanm_shfit = sanm_shfit
        self.selfattention_layer_type = selfattention_layer_type
        self.tp_blocks = tp_blocks
        self.adaptor_ffn_dim = adaptor_ffn_dim
        self.adaptor_downsample_rate = adaptor_downsample_rate
        self.init_std = init_std


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
        audio_token_index=151646,
        **kwargs,
    ):
        self.audio_token_index = audio_token_index

        if isinstance(audio_config, dict):
            audio_config = ParrotAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = ParrotAudioEncoderConfig(
                attention_dropout_rate= 0.1,
                attention_heads= 4,
                dropout_rate= 0.1,
                input_layer= 'pe',
                input_size= 560,
                kernel_size= 11,
                linear_units= 2048,
                normalize_before= True,
                num_blocks= 50,
                output_size= 512,
                pos_enc_class= 'SinusoidalPositionEncoder',
                positional_dropout_rate= 0.1,
                sanm_shfit= 0,
                selfattention_layer_type= 'sanm',
                tp_blocks= 20,
            )

        self.audio_config = audio_config

        if isinstance(text_config, dict):
            # text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "qwen2"
            # text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            text_config = Qwen2Config(**text_config)
        elif text_config is None:
            # text_config = CONFIG_MAPPING["qwen2"]()
            text_config = Qwen2Config()
        self.text_config = text_config

        super().__init__(**kwargs)


__all__ = ["ParrotAudioConfig", "ParrotAudioEncoderConfig"]
