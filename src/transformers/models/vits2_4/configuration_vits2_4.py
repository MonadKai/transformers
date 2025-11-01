# coding=utf-8
# Copyright 2023 The Kakao Enterprise Authors and the HuggingFace Inc. team. All rights reserved.
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
"""VITS2_4 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Vits2_4Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VitsModel`]. It is used to instantiate a VITS
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the VITS
    [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 38):
            Vocabulary size of the VITS model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method of [`VitsModel`].
        hidden_size (`int`, *optional*, defaults to 192):
            Dimensionality of the text encoder layers.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer encoder.
        window_size (`int`, *optional*, defaults to 4):
            Window size for the relative positional embeddings in the attention layers of the Transformer encoder.
        use_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the key, query, value projection layers in the Transformer encoder.
        ffn_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556)
            for more details.
        ffn_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the 1D convolution layers used by the feed-forward network in the Transformer encoder.
        flow_size (`int`, *optional*, defaults to 192):
            Dimensionality of the flow layers.
        spectrogram_bins (`int`, *optional*, defaults to 513):
            Number of frequency bins in the target spectrogram.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        use_stochastic_duration_prediction (`bool`, *optional*, defaults to `True`):
            Whether to use the stochastic duration prediction module or the regular duration predictor.
        num_speakers (`int`, *optional*, defaults to 1):
            Number of speakers if this is a multi-speaker model.
        speaker_embedding_size (`int`, *optional*, defaults to 0):
            Number of channels used by the speaker embeddings. Is zero for single-speaker models.
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the HiFi-GAN upsampling network.
        upsample_rates (`tuple[int]` or `list[int]`, *optional*, defaults to `[8, 8, 2, 2]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the HiFi-GAN upsampling network.
            The length of `upsample_rates` defines the number of convolutional layers and has to match the length of
            `upsample_kernel_sizes`.
        upsample_kernel_sizes (`tuple[int]` or `list[int]`, *optional*, defaults to `[16, 16, 4, 4]`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the HiFi-GAN upsampling
            network. The length of `upsample_kernel_sizes` defines the number of convolutional layers and has to match
            the length of `upsample_rates`.
        resblock_kernel_sizes (`tuple[int]` or `list[int]`, *optional*, defaults to `[3, 7, 11]`):
            A tuple of integers defining the kernel sizes of the 1D convolutional layers in the HiFi-GAN
            multi-receptive field fusion (MRF) module.
        resblock_dilation_sizes (`tuple[tuple[int]]` or `list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
            HiFi-GAN multi-receptive field fusion (MRF) module.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation.
        depth_separable_channels (`int`, *optional*, defaults to 2):
            Number of channels to use in each depth-separable block.
        depth_separable_num_layers (`int`, *optional*, defaults to 3):
            Number of convolutional layers to use in each depth-separable block.
        duration_predictor_flow_bins (`int`, *optional*, defaults to 10):
            Number of channels to map using the unonstrained rational spline in the duration predictor model.
        duration_predictor_tail_bound (`float`, *optional*, defaults to 5.0):
            Value of the tail bin boundary when computing the unconstrained rational spline in the duration predictor
            model.
        duration_predictor_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the 1D convolution layers used in the duration predictor model.
        duration_predictor_dropout (`float`, *optional*, defaults to 0.5):
            The dropout ratio for the duration predictor model.
        duration_predictor_num_flows (`int`, *optional*, defaults to 4):
            Number of flow stages used by the duration predictor model.
        duration_predictor_filter_channels (`int`, *optional*, defaults to 256):
            Number of channels for the convolution layers used in the duration predictor model.
        prior_encoder_num_flows (`int`, *optional*, defaults to 4):
            Number of flow stages used by the prior encoder flow model.
        prior_encoder_num_wavenet_layers (`int`, *optional*, defaults to 4):
            Number of WaveNet layers used by the prior encoder flow model.
        posterior_encoder_num_wavenet_layers (`int`, *optional*, defaults to 16):
            Number of WaveNet layers used by the posterior encoder model.
        wavenet_kernel_size (`int`, *optional*, defaults to 5):
            Kernel size of the 1D convolution layers used in the WaveNet model.
        wavenet_dilation_rate (`int`, *optional*, defaults to 1):
            Dilation rates of the dilated 1D convolutional layers used in the WaveNet model.
        wavenet_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the WaveNet layers.
        speaking_rate (`float`, *optional*, defaults to 1.0):
            Speaking rate. Larger values give faster synthesised speech.
        noise_scale (`float`, *optional*, defaults to 0.667):
            How random the speech prediction is. Larger values create more variation in the predicted speech.
        noise_scale_duration (`float`, *optional*, defaults to 0.8):
            How random the duration prediction is. Larger values create more variation in the predicted durations.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the output audio waveform is digitalized expressed in hertz (Hz).

    Example:

    ```python
    >>> from transformers import Vits2_4Model, Vits2_4Config

    >>> # Initializing a "fishaudio/BertVits2" style configuration
    >>> configuration = Vits2_4Config()

    >>> # Initializing a model (with random weights) from the "fishaudio/BertVits2" style configuration
    >>> model = Vits2_4Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vits2_4"

    def __init__(
        self,
        n_vocab: int = 112,
        spec_channels: int = 1025,
        segment_size: int = 16384,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        resblock: str = "1",
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates: list[int] = [8, 8, 2, 2, 2],
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: list[int] = [16, 16, 8, 2, 2],
        n_layers_q: int = 3,
        use_spectral_norm: bool = False,
        n_speakers: int = 7,
        gin_channels: int = 256,
        # SynnthesizerTrn init arguments
        use_sdp: bool = True,
        n_flow_layer: int = 4,
        n_layers_trans_flow: int = 6,
        flow_share_parameter: bool = False,
        use_transformer_flow: bool = True,
        # additional kwargs
        use_spk_conditioned_encoder: bool = True,
        use_noise_scaled_mas: bool = True,
        mas_noise_scale_initial: float = 0.01,
        noise_scale_delta: float = 2e-6,
        current_mas_noise_scale: float = 0.01,
        use_mel_posterior_encoder: bool = False,
        use_duration_discriminator: bool = True,
        use_wavlm_discriminator: bool = True,
        **kwargs,
    ):
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.n_layers_q = n_layers_q
        self.use_spectral_norm = use_spectral_norm
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        # SynnthesizerTrn init arguments
        self.use_sdp = use_sdp
        self.n_flow_layer = n_flow_layer
        self.n_layers_trans_flow = n_layers_trans_flow
        self.flow_share_parameter = flow_share_parameter
        self.use_transformer_flow = use_transformer_flow
        # additional kwargs
        self.use_spk_conditioned_encoder = use_spk_conditioned_encoder
        self.use_noise_scaled_mas = use_noise_scaled_mas
        self.mas_noise_scale_initial = mas_noise_scale_initial
        self.noise_scale_delta = noise_scale_delta
        self.current_mas_noise_scale = current_mas_noise_scale
        self.use_mel_posterior_encoder = use_mel_posterior_encoder
        self.use_duration_discriminator = use_duration_discriminator
        self.use_wavlm_discriminator = use_wavlm_discriminator

        super().__init__(**kwargs)

        if len(upsample_kernel_sizes) != len(upsample_rates):
            raise ValueError(
                f"The length of `upsample_kernel_sizes` ({len(upsample_kernel_sizes)}) must match the length of "
                f"`upsample_rates` ({len(upsample_rates)})"
            )

        if len(resblock_dilation_sizes) != len(resblock_kernel_sizes):
            raise ValueError(
                f"The length of `resblock_dilation_sizes` ({len(resblock_dilation_sizes)}) must match the length of "
                f"`resblock_kernel_sizes` ({len(resblock_kernel_sizes)})"
            )



__all__ = ["Vits2_4Config"]
