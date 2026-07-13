# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO_ROOT / "examples/retrieval/bi_encoder/ministral3_3b_instruct.yaml"


def test_ministral3_embedding_recipe_preserves_distillation_tokenizer_policy():
    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["tokenizer"]["_target_"] == "nemo_automodel.NeMoAutoTokenizer.from_pretrained"
    assert config["tokenizer"]["force_tokenizers_backend"] is True
    assert config["tokenizer"]["add_bos_token"] is False
    assert config["tokenizer"]["add_eos_token"] is False
    assert config["dataloader"]["collate_fn"]["add_special_tokens"] is False
