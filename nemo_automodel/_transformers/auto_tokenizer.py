# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from pathlib import Path
from typing import Callable, Optional, Type, Union

logger = logging.getLogger(__name__)


def _get_model_type(pretrained_model_name_or_path: str, trust_remote_code: bool = False) -> Optional[str]:
    """
    Determine the model type from the config.

    Args:
        pretrained_model_name_or_path: Model identifier or path
        trust_remote_code: Whether to trust remote code

    Returns:
        The model_type string, or None if it cannot be determined
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
        return getattr(config, "model_type", None)
    except Exception as e:
        logger.debug(f"Could not load config to determine model type: {e}")
        return None


def _get_tokenizer_registry():
    # Import lazily to avoid pulling in optional/custom backends (and transformers)
    # when users only do `from nemo_automodel import NeMoAutoTokenizer`.
    from nemo_automodel._transformers.tokenization.registry import TokenizerRegistry

    return TokenizerRegistry


class NeMoAutoTokenizer:
    """
    Auto tokenizer class that dispatches to appropriate tokenizer implementations.

    Similar to HuggingFace's AutoTokenizer, but with a custom registry for specialized
    tokenizer implementations.

    The dispatch logic is:
    1. If a custom tokenizer is registered for the model type, use it
    2. Otherwise, fall back to NeMoAutoTokenizerWithBosEosEnforced

    Example:
        >>> # Will use MistralCommonBackend if available for Mistral models
        >>> tokenizer = NeMoAutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        >>> # Force using HF AutoTokenizer with BOS/EOS enforcement
        >>> tokenizer = NeMoAutoTokenizer.from_pretrained("gpt2", force_default=True)
    """

    # Make registry accessible at class level
    _registry = None

    @classmethod
    def register(cls, model_type: str, tokenizer_cls: Union[Type, Callable]) -> None:
        """
        Register a custom tokenizer for a specific model type.

        Args:
            model_type: The model type string (e.g., "mistral", "llama")
            tokenizer_cls: The tokenizer class or factory function
        """
        _get_tokenizer_registry().register(model_type, tokenizer_cls)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *args,
        force_default: bool = False,
        force_hf: bool = False,
        force_tokenizers_backend: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Load a tokenizer from a pretrained model.

        Args:
            pretrained_model_name_or_path: Model identifier or path
            force_default: If True, always use NeMoAutoTokenizerWithBosEosEnforced
            force_hf: If True, return the raw HF AutoTokenizer without any wrapping
            force_tokenizers_backend: If True, load tokenizer.json through Transformers TokenizersBackend
            trust_remote_code: Whether to trust remote code when loading config
            **kwargs: Additional arguments passed to the tokenizer's from_pretrained

        Returns:
            A tokenizer instance appropriate for the model type
        """
        # If force_hf, just use the base HF AutoTokenizer
        if force_hf:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, *args, trust_remote_code=trust_remote_code, **kwargs
            )

        if force_tokenizers_backend:
            from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import (
                NeMoAutoTokenizerWithBosEosEnforced,
            )

            return NeMoAutoTokenizerWithBosEosEnforced.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                force_tokenizers_backend=True,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        # Try to determine model type from config
        model_type = _get_model_type(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)

        registry = _get_tokenizer_registry()

        if not force_default and model_type:
            tokenizer_cls = registry.get_custom_tokenizer_cls(model_type)
            if tokenizer_cls is not None:
                logger.info(f"Using custom tokenizer {tokenizer_cls.__name__} for model type '{model_type}'")
                try:
                    tokenizer = tokenizer_cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
                except ValueError as error:
                    error_message = str(error)
                    tokenizer_path = Path(pretrained_model_name_or_path)
                    local_tokenizer_json_only = (
                        tokenizer_path.is_dir()
                        and (tokenizer_path / "tokenizer.json").is_file()
                        and "No tokenizer file found in directory" in error_message
                    )
                    hub_tokenizer_json_only = "No valid tokenizer file found in the repo" in error_message
                    can_fall_back_to_tokenizer_json = tokenizer_cls.__name__ == "MistralCommonBackend" and (
                        local_tokenizer_json_only or hub_tokenizer_json_only
                    )
                    if not can_fall_back_to_tokenizer_json:
                        raise

                    from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import (
                        NeMoAutoTokenizerWithBosEosEnforced,
                    )

                    logger.warning(
                        "No native Mistral tokenizer artifact found in %s; loading tokenizer.json instead.",
                        pretrained_model_name_or_path,
                    )
                    tokenizer = NeMoAutoTokenizerWithBosEosEnforced.from_pretrained(
                        pretrained_model_name_or_path,
                        *args,
                        add_bos_token=False,
                        add_eos_token=False,
                        force_tokenizers_backend=True,
                        trust_remote_code=trust_remote_code,
                        **kwargs,
                    )
                from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import _ensure_pad_token_id

                _ensure_pad_token_id(tokenizer, pretrained_model_name_or_path)
                return tokenizer

        # Fall back to default BOS/EOS enforced tokenizer
        from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import NeMoAutoTokenizerWithBosEosEnforced

        return NeMoAutoTokenizerWithBosEosEnforced.from_pretrained(
            pretrained_model_name_or_path, *args, trust_remote_code=trust_remote_code, **kwargs
        )


__all__ = [
    "NeMoAutoTokenizer",
    "NeMoAutoTokenizerWithBosEosEnforced",
    "TokenizerRegistry",
]


def __getattr__(name: str):
    if name == "TokenizerRegistry":
        return _get_tokenizer_registry()
    if name == "NeMoAutoTokenizerWithBosEosEnforced":
        from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import NeMoAutoTokenizerWithBosEosEnforced

        return NeMoAutoTokenizerWithBosEosEnforced
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
