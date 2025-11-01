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
"""PyTorch VITS2_4 model."""

import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch

# import torch.utils.checkpoint
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from ...activations import ACT2FN
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .attentions_vits2_4 import FFN, Decoder, Encoder, MultiHeadAttention
from .commons_vits2_4 import fused_add_tanh_sigmoid_multiply, generate_path, get_padding, init_weights, sequence_mask
from .configuration_vits2_4 import Vits2_4Config
from .modular_vits2_4 import (
    LRELU_SLOPE,
    WN,
    ConvFlow,
    ConvReluNorm,
    DDSConv,
    ElementwiseAffine,
    Flip,
    LayerNorm,
    Log,
    ResBlock1,
    ResBlock2,
    ResidualCouplingLayer,
    TransformerCouplingLayer,
)
from .symbols_vits2_4 import num_languages, num_tones, symbols
from .vector_quantize_pytorch import VectorQuantize


logger = logging.get_logger(__name__)


@dataclass
class Vits2_4ModelOutput(ModelOutput):
    r"""
    waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
        The final audio waveform predicted by the model.
    sequence_lengths (`torch.FloatTensor` of shape `(batch_size,)`):
        The length in samples of each element in the `waveform` batch.
    spectrogram (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`):
        The log-mel spectrogram predicted at the output of the flow model. This spectrogram is passed to the Hi-Fi
        GAN decoder model to obtain the final audio waveform.
    """

    waveform: Optional[torch.FloatTensor] = None
    sequence_lengths: Optional[torch.FloatTensor] = None
    spectrogram: Optional[tuple[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class DurationDiscriminator(nn.Module):  # vits2
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.LSTM = nn.LSTM(2 * filter_channels, filter_channels, batch_first=True, bidirectional=True)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(nn.Linear(2 * filter_channels, 1), nn.Sigmoid())

    def forward_probability(self, x, dur):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = x.transpose(1, 2)
        x, _ = self.LSTM(x)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, dur)
            output_probs.append(output_prob)

        return output_probs


class TransformerCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
        share_parameter=False,
    ):
        super().__init__()
        assert share_parameter is False
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = None

        for i in range(n_flows):
            self.flows.append(
                TransformerCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    n_layers,
                    n_heads,
                    p_dropout,
                    filter_channels,
                    mean_only=True,
                    wn_sharing_parameter=self.wn,
                    gin_channels=self.gin_channels,
                )
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=True):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = Log()
        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, z, g=None):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        flows = list(reversed(self.flows))
        flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
        for flow in flows:
            z = flow(z, x_mask, g=x, reverse=True)
        z0, z1 = torch.split(z, [1, 1], 1)
        logw = z0
        return logw

    def forward_torch(self, x, x_mask, w=None, g=None, reverse=True, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class Bottleneck(nn.Sequential):
    def __init__(self, in_dim, hidden_dim):
        c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        super().__init__(*[c_fc1, c_fc2])


class Block(nn.Module):
    def __init__(self, in_dim, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp(self.norm(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor):
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=0,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.emb = nn.Embedding(len(symbols), hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)
        self.language_emb = nn.Embedding(num_languages, hidden_channels)
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1)
        self.bert_pre_proj = nn.Conv1d(2048, 1024, 1)
        self.in_feature_net = nn.Sequential(
            # input is assumed to an already normalized embedding
            nn.Linear(512, 1028, bias=False),
            nn.GELU(),
            nn.LayerNorm(1028),
            *[Block(1028, 512) for _ in range(1)],
            nn.Linear(1028, 512, bias=False),
            # normalize before passing to VQ?
            # nn.GELU(),
            # nn.LayerNorm(512),
        )
        self.emo_vq = VectorQuantize(
            dim=512,
            codebook_size=64,
            codebook_dim=32,
            commitment_weight=0.1,
            decay=0.85,
            heads=32,
            kmeans_iters=20,
            separate_codebook_per_head=True,
            stochastic_sample_codes=True,
            threshold_ema_dead_code=2,
        )
        self.out_feature_net = nn.Linear(512, hidden_channels)

        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    # 转onnx 时候使用，生成emo_192,因为emo 模块会爆显存 可能是训练的模型问题，不过提取出来也可以加速~
    def generate_emo_192(self, emo):
        emo_emb = self.in_feature_net(emo.transpose(0, 1))
        emo_emb, _, _ = self.emo_vq(emo_emb.unsqueeze(1))
        emo_emb = self.out_feature_net(emo_emb)
        emo_emb = emo_emb.detach().numpy()

        np.save("onnx/emo.npy", emo_emb)
        print("success save emo_192 to: onnx/emo.npy!!!")

        return emo_emb

    # 跑torch 推理代码使用
    def generate_emo_192_torch(self, emo):
        emo_emb = self.in_feature_net(emo.transpose(0, 1))
        emo_emb, _, _ = self.emo_vq(emo_emb.unsqueeze(1))
        emo_emb = self.out_feature_net(emo_emb)

        return emo_emb

    def forward(self, x, x_lengths, tone, language, bert, emo, g=None):
        # x_mask = torch.ones_like(x).unsqueeze(0)
        bert_emb = self.bert_proj(self.bert_pre_proj(bert.transpose(0, 1).unsqueeze(0))).transpose(1, 2)
        # emo_emb = self.in_feature_net(emo.transpose(0, 1))
        # emo_emb, _, _ = self.emo_vq(emo_emb.unsqueeze(1))
        # emo_emb = self.out_feature_net(emo_emb)

        # emo_emb 直接替换了emo
        x = (self.emb(x) + self.tone_emb(tone) + self.language_emb(language) + bert_emb + emo) * math.sqrt(
            self.hidden_channels
        )  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        # x_mask = x_mask.to(x.dtype)

        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask, g=g)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=True):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class WavLMDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self, slm_hidden=768, slm_layers=13, initial_channel=64, use_spectral_norm=False):
        super(WavLMDiscriminator, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.pre = norm_f(Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0))

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(initial_channel, initial_channel * 2, kernel_size=5, padding=2)),
                norm_f(
                    nn.Conv1d(
                        initial_channel * 2,
                        initial_channel * 4,
                        kernel_size=5,
                        padding=2,
                    )
                ),
                norm_f(nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)),
            ]
        )

        self.conv_post = norm_f(Conv1d(initial_channel * 4, 1, 3, 1, padding=1))

    def forward(self, x):
        x = self.pre(x)

        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)]) # noqa: E501

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)

    def forward(self, inputs, mask=None):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


VITS2_4_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VitsConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VITS2_4_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        speaker_id (`int`, *optional*):
            Which speaker embedding to use. Only used for multispeaker models.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The complete VITS2_4 model, for text-to-speech synthesis.",
    VITS2_4_START_DOCSTRING,
)
class Vits2_4Model(nn.Module):
    def __init__(self, config: Vits2_4Config):
        super().__init__()
        self.n_vocab = config.n_vocab
        self.spec_channels = config.spec_channels
        self.segment_size = config.segment_size
        self.inter_channels = config.inter_channels
        self.hidden_channels = config.hidden_channels
        self.filter_channels = config.filter_channels
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.kernel_size = config.kernel_size
        self.p_dropout = config.p_dropout
        self.resblock = config.resblock
        self.resblock_kernel_sizes = config.resblock_kernel_sizes
        self.resblock_dilation_sizes = config.resblock_dilation_sizes
        self.upsample_rates = config.upsample_rates
        self.upsample_initial_channel = config.upsample_initial_channel
        self.upsample_kernel_sizes = config.upsample_kernel_sizes
        self.n_layers_q = config.n_layers_q
        self.use_spectral_norm = config.use_spectral_norm
        self.n_speakers = config.n_speakers
        self.gin_channels = config.gin_channels
        # SynnthesizerTrn init arguments
        self.use_sdp = config.use_sdp
        self.n_flow_layer = config.n_flow_layer
        self.n_layers_trans_flow = config.n_layers_trans_flow
        self.flow_share_parameter = config.flow_share_parameter
        self.use_transformer_flow = config.use_transformer_flow
        # additional kwargs
        self.use_spk_conditioned_encoder = config.use_spk_conditioned_encoder
        self.use_noise_scaled_mas = config.use_noise_scaled_mas
        self.mas_noise_scale_initial = config.mas_noise_scale_initial
        self.noise_scale_delta = config.noise_scale_delta
        self.current_mas_noise_scale = config.current_mas_noise_scale
        self.use_mel_posterior_encoder = config.use_mel_posterior_encoder
        self.use_duration_discriminator = config.use_duration_discriminator
        self.use_wavlm_discriminator = config.use_wavlm_discriminator

        assert self.use_spk_conditioned_encoder and self.gin_channels > 0
        if self.n_speakers >= 1:
            self.emb_g = nn.Embedding(self.n_speakers, self.gin_channels)
        else:
            self.ref_enc = ReferenceEncoder(self.spec_channels, self.gin_channels)


        self.enc_p = TextEncoder(
            self.n_vocab,
            self.inter_channels,
            self.hidden_channels,
            self.filter_channels,
            self.n_heads,
            self.n_layers,
            self.kernel_size,
            self.p_dropout,
            gin_channels=self.gin_channels,
        )

        # HINT: not used in production
        self.enc_q = PosteriorEncoder(
            self.spec_channels,
            self.inter_channels,
            self.hidden_channels,
            5,
            1,
            16,
            gin_channels=self.gin_channels,
        )

        self.sdp = StochasticDurationPredictor(self.hidden_channels, 192, 3, 0.5, 4, gin_channels=self.gin_channels)
        self.dp = DurationPredictor(self.hidden_channels, 256, 3, 0.5, gin_channels=self.gin_channels)

        if self.use_transformer_flow:
            self.flow = TransformerCouplingBlock(
                self.inter_channels,
                self.hidden_channels,
                self.filter_channels,
                self.n_heads,
                self.n_layers_trans_flow,
                5,
                self.p_dropout,
                self.n_flow_layer,
                gin_channels=self.gin_channels,
                share_parameter=self.flow_share_parameter,
            )
        else:
            self.flow = ResidualCouplingBlock(
                self.inter_channels,
                self.hidden_channels,
                5,
                1,
                self.n_flow_layer,
                gin_channels=self.gin_channels,
            )

        self.dec = Generator(
            self.inter_channels,
            self.resblock,
            self.resblock_kernel_sizes,
            self.resblock_dilation_sizes,
            self.upsample_rates,
            self.upsample_initial_channel,
            self.upsample_kernel_sizes,
            gin_channels=self.gin_channels,
        )

    @add_start_docstrings_to_model_forward(VITS2_4_INPUTS_DOCSTRING)
    def forward_streaming(
        self,
        x,
        x_lengths,
        sid,
        tone,
        language,
        bert,
        emo,
        noise_scale=0.667,
        length_scale=1,
        noise_scale_w=0.8,
        max_len=None,
        sdp_ratio=0,
        y=None,
    ):
        # print("x.size:",x.size())
        # print("x_lengths.size:", x_lengths.size())
        # print("sid.size:", sid.size())
        # print("tone.size:",tone.size())
        # print("language.size:", language.size())
        # print("bert.size:", bert.size())
        # print("emo.size:", emo.size())

        begin = time.time()
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        print(f"vits2 speaker embeding cost: {int((time.time() - begin) * 1000)}ms")

        begin = time.time()
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tone, language, bert, emo, g=g)
        print(f"vits2 text encoder cost: {int((time.time() - begin) * 1000)}ms")

        begin = time.time()

        logw = self.sdp.forward_torch(x, x_mask, g=g, noise_scale=noise_scale_w) * (sdp_ratio) + self.dp(
            x, x_mask, g=g
        ) * (1 - sdp_ratio)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        print(f"vits2 dp cost: {int((time.time() - begin) * 1000)}ms")

        begin = time.time()
        z = self.flow(z_p, y_mask, g=g, reverse=True) * y_mask
        print(f"vits2 flow cost: {int((time.time() - begin) * 1000)}ms")
        # o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        # yield o[0, 0].data.cpu().float().numpy()

        chunk_size = 40
        padding = 19
        upsample_rates = 512
        z_len = z.shape[-1]
        for start in range(0, z_len, chunk_size):
            begin = time.time()
            end = start + chunk_size
            if start < padding:
                # 历史数据长度小于 padding，pad 上所有历史数据
                l_pad = start * upsample_rates
                start = 0
            else:
                l_pad = padding * upsample_rates
                start -= padding
            if end > z_len:
                r_pad = 0
                end = z_len
            elif end + padding > z_len:
                # 未来数据长度小于 padding，pad 上所有未来数据
                r_pad = (z_len - end) * upsample_rates
                end = z_len
            else:
                r_pad = padding * upsample_rates
                end += padding

            o = self.dec(z[:, :, start:end], g=g)
            o = o[:, :, l_pad:-r_pad] if r_pad != 0 else o[:, :, l_pad:]
            o = o[0, 0]
            # clip heading sil
            if start == 0:
                # process add blank
                index = w_ceil[0, 0][0].item() + w_ceil[0, 0][1].item() + w_ceil[0, 0][2].item()
                index = int(index * upsample_rates * 0.75)
                o = o[index:]
                print(f"vits2 dec cost: {int((time.time() - begin) * 1000)}ms")
            yield o.data.cpu().float().numpy()


__all__ = ["Vits2_4Model"]
