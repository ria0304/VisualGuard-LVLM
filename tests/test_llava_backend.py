# -*- coding: utf-8 -*-
"""
Tests for the LVLM backend's image-token handling and prompt construction.

These do not download a checkpoint: they exercise the pure logic around
placeholder detection and prompt rendering using stub objects, and they pin the
two failure modes that silently produce empty answers in practice.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.model.llava_backend import (
    LEGACY_IMAGE_TOKEN_INDEX,
    LLaVAHFBackend,
    LVLMBackendError,
    LVLMConfig,
    StepOutput,
    decode_tokens,
)


class _Cfg:
    def __init__(self, image_seq_length=576):
        self.image_seq_length = image_seq_length


class _Model:
    def __init__(self, cfg):
        self.config = cfg


def _backend(image_seq_length=576, image_token_index=None):
    backend = LLaVAHFBackend(LVLMConfig(model_name="dummy/model"))
    backend._image_token_index = (
        image_token_index if image_token_index is not None else 32000
    )
    backend.model = _Model(_Cfg(image_seq_length))
    return backend


# ---------------------------------------------------------------------------
# image placeholder detection
# ---------------------------------------------------------------------------


def test_span_covers_the_full_expanded_placeholder_run():
    """LLaVA expands <image> into one placeholder per vision patch."""
    backend = _backend()
    ids = torch.tensor([[1, 2, 32000, 32000, 32000, 32000, 3]])
    span = backend.image_token_span(ids, pixel_values=torch.zeros(1, 3, 4, 4))
    assert span == (2, 6)


def test_span_of_single_placeholder():
    backend = _backend()
    ids = torch.tensor([[1, 32000, 2]])
    assert backend.image_token_span(ids, torch.zeros(1, 3, 4, 4)) == (1, 2)


def test_span_is_none_without_pixel_values():
    backend = _backend()
    assert backend.image_token_span(torch.tensor([[1, 32000]]), None) is None


def test_span_is_unknown_without_a_placeholder():
    """No placeholder means the block's position is unknown -- do not guess.

    The old fallback assumed the image block started at index 0. LLaVA prompts
    begin ``"USER: "``, so it actually starts at 3: the guess returned a span
    covering three text tokens plus n_image-3 image tokens, quietly corrupting
    the attention signal with no warning. Reporting "unknown" is the honest
    answer, and it makes AttentionEvidence say the signal was unmeasured rather
    than measuring the wrong tokens.
    """
    backend = _backend(image_seq_length=576)
    ids = torch.tensor([[7, 8, 9, 10]])
    assert backend.image_token_span(ids, torch.zeros(1, 3, 4, 4)) is None


def test_span_of_a_real_llava_prompt_starts_after_the_role_prefix():
    """A LLaVA-1.5 prompt's image block starts at 3, not 0."""
    backend = _backend(image_seq_length=576)
    # "USER: " -> 3 text tokens, then the expanded image block, then "ASSISTANT:".
    ids = torch.tensor([[450, 525, 29941] + [32000] * 576 + [2, 29941]])
    span = backend.image_token_span(ids, torch.zeros(1, 3, 4, 4))
    assert span == (3, 3 + 576)


def test_unknown_span_disables_the_state_attention_signal():
    """An unknown span must yield "unmeasured", not a number from the wrong span."""
    from src.model.visual_evidence import AttentionEvidence, EvidenceConfig

    backend = _backend(image_seq_length=576)
    ids = torch.tensor([[450, 525, 29941] + [32000] * 576 + [2, 29941]])
    span = backend.image_token_span(ids, torch.zeros(1, 3, 4, 4))
    attentions = (torch.rand(1, 2, ids.shape[1], ids.shape[1]),)
    ev = AttentionEvidence(EvidenceConfig())
    # With the correct span, image attention is a real fraction of the mass.
    assert 0.0 < ev.state_image_attention(attentions, span) <= 1.0
    # Without it, the channel reports nothing rather than guessing.
    assert ev.state_image_attention(attentions, None) == 0.0


def test_span_with_legacy_sentinel():
    """The original LLaVA repo uses a negative out-of-vocab sentinel."""
    backend = _backend(image_token_index=LEGACY_IMAGE_TOKEN_INDEX)
    ids = torch.tensor([[1, -200, -200, 2]])
    assert backend.image_token_span(ids, torch.zeros(1, 3, 4, 4)) == (1, 3)


def test_placeholder_detection_is_not_position_one_only():
    """Regression guard: a naive index-of-only implementation missed runs."""
    backend = _backend()
    ids = torch.tensor([[5, 6, 32000, 32000, 7, 8, 9]])
    span = backend.image_token_span(ids, torch.zeros(1, 3, 4, 4))
    assert span == (2, 4)


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------


def test_llava_fallback_template_used_without_processor():
    backend = LLaVAHFBackend(LVLMConfig(model_name="dummy/model"))
    prompt = backend.build_prompt("Is there a dog?")
    assert prompt == "USER: <image>\nIs there a dog? ASSISTANT:"


def test_chat_template_is_preferred_when_available():
    """Qwen/Llama-3 based LVLMs need ChatML, not the LLaVA-1.1 format."""
    backend = LLaVAHFBackend(LVLMConfig(model_name="dummy/model"))
    calls = {}

    class _Processor:
        def apply_chat_template(self, messages, add_generation_prompt=True):
            calls["messages"] = messages
            calls["add_generation_prompt"] = add_generation_prompt
            return "<|im_start|>user\n<image>\nDescribe<|im_end|><|im_start|>assistant\n"

    backend._processor = _Processor()
    prompt = backend.build_prompt("Describe")
    assert prompt.startswith("<|im_start|>user")
    assert calls["add_generation_prompt"] is True
    content = calls["messages"][0]["content"]
    assert {"type": "image"} in content
    assert {"type": "text", "text": "Describe"} in content


def test_chat_template_failure_falls_back_to_template():
    """A broken template must not take the whole run down."""
    backend = LLaVAHFBackend(LVLMConfig(model_name="dummy/model"))

    class _Processor:
        def apply_chat_template(self, messages, add_generation_prompt=True):
            raise ValueError("no multimodal template")

    backend._processor = _Processor()
    assert backend.build_prompt("hi") == "USER: <image>\nhi ASSISTANT:"


def test_empty_chat_template_output_falls_back():
    backend = LLaVAHFBackend(LVLMConfig(model_name="dummy/model"))

    class _Processor:
        def apply_chat_template(self, messages, add_generation_prompt=True):
            return "   "

    backend._processor = _Processor()
    assert "USER:" in backend.build_prompt("hi")


# ---------------------------------------------------------------------------
# guards before load()
# ---------------------------------------------------------------------------


def test_operations_fail_clearly_before_load():
    backend = LLaVAHFBackend(LVLMConfig(model_name="dummy/model"))
    with pytest.raises(LVLMBackendError, match="not loaded"):
        backend.prepare_inputs("q", None)
    with pytest.raises(LVLMBackendError, match="Tokenizer unavailable"):
        _ = backend.tokenizer
    with pytest.raises(LVLMBackendError, match="Image processor unavailable"):
        _ = backend.image_processor


def test_raw_tensor_image_is_rejected_with_guidance():
    """Feeding a preprocessed tensor where a PIL image is expected is a bug."""
    backend = LLaVAHFBackend(LVLMConfig(model_name="dummy/model"))
    backend._processor = object()
    with pytest.raises(TypeError, match="not a preprocessed tensor"):
        backend._to_pil(torch.zeros(3, 4, 4))


def test_unsupported_image_type_rejected():
    backend = LLaVAHFBackend(LVLMConfig(model_name="dummy/model"))
    with pytest.raises(TypeError, match="Unsupported image input type"):
        backend._to_pil(12345)


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def test_quantisation_flags_are_mutually_exclusive():
    with pytest.raises(ValueError):
        LVLMConfig(load_in_4bit=True, load_in_8bit=True)


def test_attention_defaults_to_eager():
    """Attention evidence needs eager attention."""
    assert LVLMConfig().attn_implementation == "eager"


def test_decode_tokens_helper():
    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(str(i) for i in ids)

    assert decode_tokens(_Tok(), [1, 2, 3]) == "123"


def test_step_output_defaults():
    out = StepOutput(logits=torch.zeros(1, 5), attentions=None, past_key_values=None,
                     image_token_span=None)
    assert out.image_token_span is None
    assert out.attentions is None
