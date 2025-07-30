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
"""Feature extraction class for Parrot2AudioMoe."""

from transformers.models.parrot_sensevoice.feature_extraction_parrot_sensevoice import ParrotSenseVoiceFeatureExtractor


Parrot2AudioMoeFeatureExtractor = ParrotSenseVoiceFeatureExtractor

__all__ = ["Parrot2AudioMoeFeatureExtractor"]
