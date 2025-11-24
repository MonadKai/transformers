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
"""
Processor class for ParrotSenseVoice.
"""

from typing import List, Optional, Union

import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput


class ParrotSenseVoiceProcessor(ProcessorMixin):
    r"""
    Constructs a ParrotSenseVoice processor which wraps a ParrotSenseVoice feature extractor and a Qwen2TokenizerFast tokenizer into a single processor.

    [`ParrotSenseVoiceProcessor`] offers all the functionalities of [`ParrotSenseVoiceFeatureExtractor`] and [`Qwen2TokenizerFast`]. See the
    [`~ParrotSenseVoiceProcessor.__call__`] and [`~ParrotSenseVoiceProcessor.decode`] for more information.

    Args:
        feature_extractor ([`ParrotSenseVoiceFeatureExtractor`], *optional*):
            The feature extractor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the default chat template
                is used.
        audio_token (`str`, *optional*, defaults to `"[FAKE_AUDIO]"`):
            The token to use for audio tokens.
        audio_bos_token (`str`, *optional*, defaults to `"<|vision_start|>"`):
            The token to use for audio bos tokens.
        audio_eos_token (`str`, *optional*, defaults to `"<|vision_end|>"`):
            The token to use for audio eos tokens.
    """

    attributes = ["feature_extractor", "tokenizer"]
    valid_kwargs = ["chat_template", "audio_token", "audio_bos_token", "audio_eos_token"]
    feature_extractor_class = "ParrotSenseVoiceFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_token="[FAKE_AUDIO]",
        audio_bos_token="<|vision_start|>",
        audio_eos_token="<|vision_end|>",
        placeholder_token="<placeholder>",
    ):
        if chat_template is None:
            chat_template = self.default_chat_template
        self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        self.audio_bos_token = tokenizer.audio_bos_token if hasattr(tokenizer, "audio_bos_token") else audio_bos_token
        self.audio_eos_token = tokenizer.audio_eos_token if hasattr(tokenizer, "audio_eos_token") else audio_eos_token
        self.placeholder_token = placeholder_token
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audios: Union[np.ndarray, List[np.ndarray]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to ParrotSenseVoiceTokenizerFast's [`~ParrotSenseVoiceTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to
        ParrotSenseVoiceFeatureExtractor's [`~ParrotSenseVoiceFeatureExtractor.__call__`] if `audios` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audios (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audios to be prepared. Each audio can be a NumPy array.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            sampling_rate (`int`, defaults to 16000):
                The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        """

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")
        elif isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        # ensure we have as much audios as audio tokens
        num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)
        num_audios = 1 if isinstance(audios, np.ndarray) else len(audios)
        if num_audio_tokens != num_audios:
            raise ValueError(
                f"Found {num_audio_tokens} {self.audio_token} token{'s' if num_audio_tokens > 1 else ''} in provided text but received {num_audios} audio{'s' if num_audios > 1 else ''}"
            )

        if audios is not None:
            audio_lens = [i.shape[0] for i in audios]
            audio_inputs = self.feature_extractor(audios, audio_lens, **kwargs)
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename attention_mask to prevent conflicts later on

            expanded_text = []
            audio_lengths = audio_inputs["feature_attention_mask"].sum(-1).tolist()

            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    audio_length = audio_lengths.pop(0)
                    num_audio_tokens = audio_length

                    expanded_audio_token = self.audio_token * num_audio_tokens

                    audio_token_start_idx = sample.find(self.audio_token)
                    audio_token_end_idx = audio_token_start_idx + len(self.audio_token)

                    has_bos = (
                        sample[audio_token_start_idx - len(self.audio_bos_token) : audio_token_start_idx]
                        == self.audio_bos_token
                    )
                    has_eos = (
                        sample[audio_token_end_idx : audio_token_end_idx + len(self.audio_eos_token)]
                        == self.audio_eos_token
                    )

                    # Check if this audio token is surrounded by bos/eos tokens
                    if not has_bos and not has_eos:
                        expanded_audio_token = self.audio_bos_token + expanded_audio_token + self.audio_eos_token

                    replace_str.append(expanded_audio_token)
                    sample = sample.replace(self.audio_token, self.placeholder_token, 1)

                while self.placeholder_token in sample:
                    sample = sample.replace(self.placeholder_token, replace_str.pop(0), 1)
                expanded_text.append(sample)
            text = expanded_text

        # HINT: avoid passing audio parameter to tokenizer
        # HINT: different image should use different logics
        # e.g. for image `docker pull image.sourcefind.cn:5000/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk25.04.1-rc5-rocblas101839-0811-das1.6-py3.10-20250812-beta`
        # we need to pop the audio parameter
        # but for image `docker pull image.sourcefind.cn:5000/dcu/admin/base/vllm:0.8.5-ubuntu22.04-dtk25.04.1-rc5-das1.6-py3.10-20250724`
        # we need to keep the audio parameter
        kwargs.pop("audio")
        inputs = self.tokenizer(text, padding=padding, **kwargs)

        if audios is not None:
            inputs.update(audio_inputs)

        return BatchFeature(data={**inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["feature_attention_mask"]))

    @property
    def default_chat_template(self):
        """
        This default vicuna template formats inputs in the form of a chat history. For each message in the chat history:
        * the template will output the role of the speaker followed by the content of the message.
        * content is a list of strings and audios.
        * If the content element is an audio, the template will output a sequence of [FAKE_AUDIO] tokens

        Example:

        ```python
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
                {"type": "text", "text": "What's that sound?"},
            ]},
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"},
                {"type": "text", "text": "How about this one?"},
            ]},
        ]

        result = template.render(messages=messages, add_generation_prompt=True)
        ```
        """
        # fmt: off
        return (
            "{% set audio_count = namespace(value=0) %}"
            "{% for message in messages %}"
                "{% if loop.first and message['role'] != 'system' %}"
                    "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
                "{% endif %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{% if message['content'] is string %}"
                    "{{ message['content'] }}<|im_end|>\n"
                "{% else %}"
                    "{% for content in message['content'] %}"
                        "{% if 'audio' in content or 'audio_url' in content or content['type'] == 'audio' %}"
                            "{% set audio_count.value = audio_count.value + 1 %}"
                            "<|vision_start|>[FAKE_AUDIO]<|vision_end|>"
                        "{% elif 'text' in content %}"
                            "{{ content['text'] }}"
                        "{% endif %}"
                    "{% endfor %}"
                    "<|im_end|>\n"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "<|im_start|>assistant\n"
            "{% endif %}"
        )
        # fmt: on


__all__ = ["ParrotSenseVoiceProcessor"]
