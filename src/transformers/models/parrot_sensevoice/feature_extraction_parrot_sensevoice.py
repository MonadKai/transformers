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
"""Feature extraction class for ParrotSenseVoice."""

from functools import cached_property
from typing import Optional

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence
from transformers import FeatureExtractionMixin


def apply_cmvn(inputs: torch.Tensor, means: torch.Tensor, vars: torch.Tensor) -> torch.Tensor:
    device = inputs.device
    means = means.to(device)
    vars = vars.to(device)
    return (inputs + means) * vars


def apply_lfr(inputs: torch.Tensor, lfr_m: int, lfr_n: int) -> torch.Tensor:
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, inputs))
    T = T + (lfr_m - 1) // 2
    feat_dim = inputs.shape[-1]
    strides = (lfr_n * feat_dim, 1)
    sizes = (T_lfr, lfr_m * feat_dim)
    last_idx = (T - lfr_m) // lfr_n + 1
    num_padding = lfr_m - (T - last_idx * lfr_n)
    if num_padding > 0:
        num_padding = (2 * lfr_m - 2 * T + (T_lfr - 1 + last_idx) * lfr_n) / 2 * (T_lfr - last_idx)
        inputs = torch.vstack([inputs] + [inputs[-1:]] * int(num_padding))
    LFR_outputs = inputs.as_strided(sizes, strides)
    return LFR_outputs.clone().type(torch.float32)


class ParrotSenseVoiceFeatureExtractor(FeatureExtractionMixin):
    def __init__(self, 
        means: list[float],
        vars: list[float],
        cmvn_file: Optional[str] = None,
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        filter_length_min: int = -1,
        filter_length_max: int = -1,
        lfr_m: int = 1,
        lfr_n: int = 1,
        dither: float = 1.0,
        snip_edges: bool = True,
        upsacle_samples: bool = True,
        **kwargs,
    ):
        super().__init__(
            cmvn_file=cmvn_file,
            means=means,
            vars=vars,
            fs=fs,
            window=window,
            n_mels=n_mels,
            frame_length=frame_length,
            frame_shift=frame_shift,
            filter_length_min=filter_length_min,
            filter_length_max=filter_length_max,
            lfr_m=lfr_m,
            lfr_n=lfr_n,
            dither=dither,
            snip_edges=snip_edges,
            upsacle_samples=upsacle_samples,
            **kwargs
        )

    @cached_property
    def _means(self) -> torch.Tensor:
        return torch.as_tensor(self.means, dtype=torch.float32)

    @cached_property
    def _vars(self) -> torch.Tensor:
        return torch.as_tensor(self.vars, dtype=torch.float32)

    @torch.no_grad()
    def __call__(self,
        inputs: list[torch.Tensor],
        input_lengths: list[int],
        **kwargs,
    ):
        # input_lengths = [i.shape[0] for i in inputs]
        batch_size = len(inputs)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = inputs[i][:waveform_length]
            if self.upsacle_samples:
                waveform = waveform * (1 << 15)
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform)
            waveform = waveform.unsqueeze(0)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=self.n_mels,
                frame_length=min(self.frame_length,waveform_length/self.fs*1000),
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
                window_type=self.window,
                sample_frequency=self.fs,
                snip_edges=self.snip_edges,
            )  # [T, n_mels]

            if self.lfr_m != 1 or self.lfr_n != 1:
                mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
            mat = apply_cmvn(mat, self._means, self._vars)  # [feature_length, n_mels * 7]
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        max_len = feats_lens.max().item()
        idxs = torch.arange(max_len).expand(feats_lens.size(0), max_len)  # [batch_size, max_len]
        feature_attention_masks = idxs < feats_lens.unsqueeze(1)  # [batch_size, max_len]
        if batch_size == 1:
            feats_pad = feats[0][None, :, :]  # [1, feature_length, n_mels * 7]
        else:
            feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)  # [batch_size, feature_length, n_mels * 7]
        return {'input_features': feats_pad, 'attention_mask': feature_attention_masks}


__all__ = ["ParrotSenseVoiceFeatureExtractor"]
