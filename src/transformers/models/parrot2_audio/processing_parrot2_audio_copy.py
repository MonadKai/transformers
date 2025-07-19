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
Processor class for Parrot2Audio.
"""

from typing import List, Optional, Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (PaddingStrategy,
                                                  PreTokenizedInput, TextInput)
from transformers.models.parrot_audio.processing_parrot_audio import ParrotAudioProcessor as Parrot2AudioProcessor

__all__ = ["Parrot2AudioProcessor"]
