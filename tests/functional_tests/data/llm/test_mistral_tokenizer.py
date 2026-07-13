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

"""Functional tests for NeMoAutoTokenizer with Mistral models.

Verifies that NeMoAutoTokenizer correctly dispatches to MistralCommonBackend
for Mistral model types and that basic tokenization operations work.
"""

import os
import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest
from transformers import TokenizersBackend
from transformers.tokenization_mistral_common import MistralCommonBackend as TransformersMistralCommonBackend

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import NeMoAutoTokenizerWithBosEosEnforced
from nemo_automodel._transformers.tokenization.tokenization_mistral_common import MistralCommonBackend
from nemo_automodel.components.datasets.llm import BiEncoderCollator

_TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "/home/TestData/automodel")
_TOKENIZER_BASE = Path(_TEST_DATA_DIR) / "tokenizers"
MISTRAL_7B_INSTRUCT_PATH = _TOKENIZER_BASE / "Mistral-7B-Instruct-v0.1"
MINISTRAL3_3B_INSTRUCT_PATH = Path(
    os.environ.get(
        "MINISTRAL3_TOKENIZER_PATH",
        _TOKENIZER_BASE / "Ministral-3-3B-Instruct-2512",
    )
)


@pytest.fixture
def mistral_tokenizer_path():
    if MISTRAL_7B_INSTRUCT_PATH.exists():
        return str(MISTRAL_7B_INSTRUCT_PATH)
    if MINISTRAL3_3B_INSTRUCT_PATH.exists():
        return str(MINISTRAL3_3B_INSTRUCT_PATH)
    pytest.fail("No Mistral tokenizer fixture is available")


@pytest.fixture
def ministral3_tokenizer_path():
    if not MINISTRAL3_3B_INSTRUCT_PATH.exists():
        pytest.skip("Ministral-3-3B-Instruct-2512 tokenizer fixture is unavailable")
    return str(MINISTRAL3_3B_INSTRUCT_PATH)


@pytest.fixture
def simple_conversation():
    return [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And of Germany?"},
    ]


class TestMistralTokenizerDispatch:
    """Verify NeMoAutoTokenizer dispatches to MistralCommonBackend for Mistral models."""

    def test_from_pretrained_returns_mistral_common_backend(self, mistral_tokenizer_path):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path)
        assert isinstance(tokenizer, MistralCommonBackend)

    def test_force_default_returns_hf_tokenizer(self, mistral_tokenizer_path):
        from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import NeMoAutoTokenizerWithBosEosEnforced

        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path, force_default=True)
        assert isinstance(tokenizer, NeMoAutoTokenizerWithBosEosEnforced)
        assert not isinstance(tokenizer, MistralCommonBackend)

    def test_force_hf_returns_raw_hf_tokenizer(self, mistral_tokenizer_path):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path, force_hf=True)
        assert not isinstance(tokenizer, MistralCommonBackend)

    def test_force_tokenizers_backend_preserves_source_assets(self, ministral3_tokenizer_path, tmp_path):
        tokenizer = NeMoAutoTokenizer.from_pretrained(
            ministral3_tokenizer_path,
            force_tokenizers_backend=True,
            add_bos_token=False,
            add_eos_token=False,
            padding_side="left",
        )

        assert isinstance(tokenizer, NeMoAutoTokenizerWithBosEosEnforced)
        assert isinstance(tokenizer, TokenizersBackend)
        tokenizer.save_pretrained(tmp_path)

        source = Path(ministral3_tokenizer_path)
        for filename in ("tokenizer.json", "tokenizer_config.json"):
            assert (tmp_path / filename).read_bytes() == (source / filename).read_bytes()

        expected = TokenizersBackend.from_pretrained(source, padding_side="left")
        texts = ["query: example", "literal <s> token"]
        assert tokenizer(texts, add_special_tokens=False) == expected(texts, add_special_tokens=False)


class TestMistralCommonBackendTokenization:
    """Verify basic encode/decode round-trip and special token handling."""

    def test_encode_decode_roundtrip(self, mistral_tokenizer_path):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path)
        text = "Hello, how are you doing today?"
        token_ids = tokenizer.encode(text)
        assert isinstance(token_ids, list)
        assert all(isinstance(t, int) for t in token_ids)
        assert len(token_ids) > 0

        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert text in decoded

    def test_vocab_size_positive(self, mistral_tokenizer_path):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path)
        assert tokenizer.vocab_size > 0

    def test_special_tokens_defined(self, mistral_tokenizer_path):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path)
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None

    def test_apply_chat_template(self, mistral_tokenizer_path, simple_conversation):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path)
        result = tokenizer.apply_chat_template(
            simple_conversation,
            tokenize=True,
            add_generation_prompt=True,
        )
        token_ids = result["input_ids"] if isinstance(result, Mapping) else result
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0

    def test_apply_chat_template_no_tokenize(self, mistral_tokenizer_path, simple_conversation):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path)
        result = tokenizer.apply_chat_template(
            simple_conversation,
            tokenize=False,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_batch_encode(self, mistral_tokenizer_path):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        texts = ["Hello world", "How are you?"]
        result = tokenizer(texts, padding=True)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert len(result["input_ids"]) == 2

    def test_retrieval_ids_match_tokenizer_json_without_special_tokens(self, ministral3_tokenizer_path):
        native = NeMoAutoTokenizer.from_pretrained(ministral3_tokenizer_path, padding_side="left")
        reference = TokenizersBackend.from_pretrained(ministral3_tokenizer_path, padding_side="left")
        texts = [
            "query: example",
            "passage: a longer example",
            "line one\nline two",
            "café 東京 😀",
        ]

        for max_length in (2, 4, 8, 16):
            expected = reference(
                texts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_token_type_ids=False,
            )
            actual = native(
                texts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_token_type_ids=False,
            )
            assert actual["input_ids"] == expected["input_ids"]
            assert actual["attention_mask"] == expected["attention_mask"]

        # mistral-common intentionally treats literal control-token strings as
        # ordinary text; tokenizer.json interprets them as control-token IDs.
        assert native.encode("literal <s> token", add_special_tokens=False) != reference.encode(
            "literal <s> token", add_special_tokens=False
        )

    def test_retrieval_collator_without_special_tokens(self, ministral3_tokenizer_path):
        native = NeMoAutoTokenizer.from_pretrained(ministral3_tokenizer_path, padding_side="left")
        reference = TokenizersBackend.from_pretrained(ministral3_tokenizer_path, padding_side="left")
        collator_kwargs = {
            "q_max_len": 6,
            "p_max_len": 7,
            "padding": True,
            "pad_to_multiple_of": 4,
            "add_special_tokens": False,
        }
        native_collator = BiEncoderCollator(
            tokenizer=native,
            **collator_kwargs,
        )
        reference_collator = BiEncoderCollator(
            tokenizer=reference,
            **collator_kwargs,
        )
        batch = [
            {"question": "short query", "doc_text": ["short document", "a much longer document to truncate"]},
            {"question": "a longer query to truncate", "doc_text": ["document", "another document"]},
        ]

        actual = native_collator(batch)
        expected = reference_collator(batch)

        assert actual.keys() == expected.keys()
        for key in actual:
            assert actual[key].equal(expected[key]), key

    def test_save_reload_preserves_token_ids(self, ministral3_tokenizer_path, tmp_path):
        tokenizer = NeMoAutoTokenizer.from_pretrained(ministral3_tokenizer_path, padding_side="left")
        tokenizer.save_pretrained(tmp_path)
        reloaded = MistralCommonBackend.from_pretrained(tmp_path, padding_side="left")

        texts = ["query: example", "passage: a longer example"]
        for add_special_tokens in (False, True):
            expected = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=8,
                add_special_tokens=add_special_tokens,
            )
            actual = reloaded(
                texts,
                padding=True,
                truncation=True,
                max_length=8,
                add_special_tokens=add_special_tokens,
            )
            assert actual["input_ids"] == expected["input_ids"]
            assert actual["attention_mask"] == expected["attention_mask"]

        upstream = TransformersMistralCommonBackend.from_pretrained(tmp_path, padding_side="left")
        expected = tokenizer(texts, add_special_tokens=False)
        actual = upstream(texts, add_special_tokens=False)
        assert actual["input_ids"] == expected["input_ids"]
        assert actual["attention_mask"] == expected["attention_mask"]

    def test_tokenizer_json_only_checkpoint_falls_back(self, ministral3_tokenizer_path, tmp_path):
        source_dir = tmp_path / "source"
        save_dir = tmp_path / "saved"
        source_dir.mkdir()
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copy(Path(ministral3_tokenizer_path) / filename, source_dir / filename)

        tokenizer = NeMoAutoTokenizer.from_pretrained(source_dir, padding_side="left")
        reference = TokenizersBackend.from_pretrained(source_dir, padding_side="left")
        texts = ["query: example", "passage: a longer example"]

        expected = reference(texts, add_special_tokens=False, padding=True)
        actual = tokenizer(texts, add_special_tokens=False, padding=True)
        assert actual["input_ids"] == expected["input_ids"]
        assert actual["attention_mask"] == expected["attention_mask"]

        tokenizer.save_pretrained(save_dir)
        for filename in ("tokenizer.json", "tokenizer_config.json"):
            assert (save_dir / filename).read_bytes() == (source_dir / filename).read_bytes()

        shutil.copy(source_dir / "config.json", save_dir / "config.json")
        reloaded = NeMoAutoTokenizer.from_pretrained(save_dir, padding_side="left")
        actual = reloaded(texts, add_special_tokens=False, padding=True)
        assert actual["input_ids"] == expected["input_ids"]
        assert actual["attention_mask"] == expected["attention_mask"]
