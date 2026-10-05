# -*- coding: utf-8 -*-
"""
Unit tests for the evidence mathematics.

These use small synthetic tensors and mocks deliberately: they verify the
*arithmetic*, not model behaviour. Nothing here touches a real dataset or a
pretrained checkpoint, and no result from these tests is a research result.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.model.visual_evidence import (
    AttentionEvidence,
    Candidate,
    EvidenceConfig,
    combine_evidence,
    hallucination_penalty,
    is_content_bearing,
    minmax_normalise,
    normalise_values,
    softmax_normalise,
    zscore_normalise,
)


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def test_minmax_maps_to_unit_interval():
    assert minmax_normalise([0.0, 0.5, 1.0]) == pytest.approx([0.0, 0.5, 1.0])


def test_minmax_degenerate_range_returns_ones():
    # An all-equal candidate set has no discriminative signal. Returning 0.0
    # would penalise every candidate and destroy the ranking.
    assert minmax_normalise([0.7, 0.7, 0.7]) == [1.0, 1.0, 1.0]


def test_minmax_empty():
    assert minmax_normalise([]) == []


def test_zscore_is_bounded_and_centred():
    values = [1.0, 2.0, 3.0, 4.0]
    out = zscore_normalise(values)
    assert all(0.0 <= v <= 1.0 for v in out)
    # Logistic squash is monotone, so ordering is preserved.
    assert out == sorted(out)


def test_zscore_single_value_is_midpoint():
    assert zscore_normalise([5.0]) == [0.5]


def test_softmax_is_normalised_and_stable_for_large_logits():
    out = softmax_normalise([1000.0, 1001.0, 999.0])
    assert math.isclose(sum(out), 1.0, rel_tol=1e-9)
    assert all(math.isfinite(v) for v in out)


def test_softmax_survives_extreme_values():
    out = softmax_normalise([1e4, -1e4, 0.0])
    assert all(math.isfinite(v) for v in out)
    assert math.isclose(sum(out), 1.0, rel_tol=1e-9)


def test_normalise_values_none_is_identity():
    assert normalise_values([0.3, 0.9], "none") == [0.3, 0.9]


# ---------------------------------------------------------------------------
# evidence combination
# ---------------------------------------------------------------------------


def test_combine_weighted_average():
    cfg = EvidenceConfig(alpha=1.0, beta=1.0, gamma=2.0)
    out = combine_evidence([1.0, 0.0], [0.0, 1.0], [0.0, 0.0], cfg)
    # total weight = 4, so each candidate contributes 1/4
    assert out == pytest.approx([0.25, 0.25])


def test_combine_single_channel_is_identity():
    cfg = EvidenceConfig(alpha=1.0, beta=0.0, gamma=0.0)
    out = combine_evidence([0.42, 0.9], [0.0, 0.0], [0.0, 0.0], cfg)
    assert out == pytest.approx([0.42, 0.9])


def test_combine_respects_channel_weights():
    cfg = EvidenceConfig(alpha=1.0, beta=0.0, gamma=0.0)
    out = combine_evidence([0.2], [1.0], [1.0], cfg)
    assert out == pytest.approx([0.2])


def test_combine_stays_in_unit_interval():
    cfg = EvidenceConfig(alpha=1.0, beta=1.0, gamma=1.0)
    out = combine_evidence([1.0, 1.0], [1.0, 1.0], [1.0, 1.0], cfg)
    assert all(0.0 <= v <= 1.0 for v in out)


def test_combine_rejects_mismatched_lengths():
    cfg = EvidenceConfig()
    with pytest.raises(ValueError):
        combine_evidence([1.0, 0.0], [1.0], [1.0, 0.0], cfg)


def test_combine_rejects_all_zero_weights():
    cfg = EvidenceConfig(alpha=0.0, beta=0.0, gamma=0.0)
    with pytest.raises(ValueError, match="alpha/beta/gamma"):
        combine_evidence([1.0], [1.0], [1.0], cfg)


def test_combine_empty_candidates():
    cfg = EvidenceConfig()
    assert combine_evidence([], [], [], cfg) == []


# ---------------------------------------------------------------------------
# hallucination penalty
# ---------------------------------------------------------------------------


def test_penalty_zero_above_threshold():
    cfg = EvidenceConfig(lam=1.0, threshold=0.5)
    assert hallucination_penalty(0.9, cfg) == 0.0


def test_penalty_positive_below_threshold():
    cfg = EvidenceConfig(lam=1.0, threshold=0.5)
    assert hallucination_penalty(0.0, cfg) == pytest.approx(1.0)
    assert hallucination_penalty(0.25, cfg) == pytest.approx(0.5)


def test_penalty_is_monotone_decreasing_in_evidence():
    cfg = EvidenceConfig(lam=1.0, threshold=0.8)
    values = [hallucination_penalty(v / 10, cfg) for v in range(11)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_penalty_scaled_by_lambda():
    a = hallucination_penalty(0.0, EvidenceConfig(lam=0.5, threshold=1.0))
    b = hallucination_penalty(0.0, EvidenceConfig(lam=1.0, threshold=1.0))
    assert b == pytest.approx(2 * a)


def test_penalty_zero_when_lambda_zero():
    cfg = EvidenceConfig(lam=0.0, threshold=0.5)
    assert hallucination_penalty(0.0, cfg) == 0.0


def test_penalty_guarded_against_zero_threshold():
    cfg = EvidenceConfig(lam=1.0, threshold=0.0)
    assert hallucination_penalty(0.0, cfg) == 0.0


# ---------------------------------------------------------------------------
# content-word detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("word", ["dog", "cat", "truck", "helmet", "tennis"])
def test_content_words_are_detected(word):
    assert is_content_bearing(word)


@pytest.mark.parametrize("word", ["the", "a", "an", "is", "and", "of", "there", "it"])
def test_function_words_are_rejected(word):
    assert not is_content_bearing(word)


@pytest.mark.parametrize("text", ["", "   ", ".", ",", "!?", "--", "\n"])
def test_punctuation_and_empty_are_rejected(text):
    assert not is_content_bearing(text)


def test_content_detection_is_case_insensitive():
    assert not is_content_bearing("THE")
    assert not is_content_bearing("Is")
    assert is_content_bearing("Dog")


# ---------------------------------------------------------------------------
# attention evidence
# ---------------------------------------------------------------------------


def _synthetic_attentions(
    num_layers: int = 4, heads: int = 2, q_len: int = 1, kv_len: int = 10, span=(2, 6)
) -> tuple:
    """Build normalised attention rows where the last query attends to `span`."""
    attns = []
    for _ in range(num_layers):
        layer = torch.zeros(1, heads, q_len, kv_len)
        start, end = span
        layer[:, :, :, start:end] = 1.0 / (end - start)
        attns.append(layer)
    return tuple(attns)


def test_state_image_attention_is_one_when_all_mass_on_image():
    ev = AttentionEvidence(EvidenceConfig(layer_fraction=1.0))
    attns = _synthetic_attentions()
    value = ev.state_image_attention(attns, image_token_span=(2, 6))
    assert value == pytest.approx(1.0)


def test_state_image_attention_is_zero_when_all_mass_on_text():
    ev = AttentionEvidence(EvidenceConfig(layer_fraction=1.0))
    # Attention is concentrated on positions 0-2 while the image span is 8-10.
    attns = _synthetic_attentions(span=(0, 2))
    value = ev.state_image_attention(attns, image_token_span=(8, 10))
    assert value == pytest.approx(0.0)


def test_state_image_attention_reflects_partial_mass():
    ev = AttentionEvidence(EvidenceConfig(layer_fraction=1.0))
    # Half the total mass sits inside the image span (0:2).
    attn = torch.tensor([[[[0.25, 0.25, 0.25, 0.25]]]])
    value = ev.state_image_attention((attn,), image_token_span=(0, 2))
    assert value == pytest.approx(0.5)


def test_state_image_attention_handles_missing_inputs():
    ev = AttentionEvidence(EvidenceConfig())
    assert ev.state_image_attention(None, (0, 4)) == 0.0
    assert ev.state_image_attention(_synthetic_attentions(), None) == 0.0
    assert ev.state_image_attention(_synthetic_attentions(), (4, 4)) == 0.0


def test_state_image_attention_uses_last_query_row():
    """For prefill, the row that predicts the next token is the last prompt row."""
    ev = AttentionEvidence(EvidenceConfig(layer_fraction=1.0))
    attn = torch.zeros(1, 1, 3, 4)
    attn[0, 0, 0, :] = 0.25                      # uniform
    attn[0, 0, 2, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # all on image
    value = ev.state_image_attention((attn,), image_token_span=(0, 2))
    assert value == pytest.approx(1.0)


def test_state_image_attention_aggregates_across_layers():
    """Layer aggregation is the mean of the per-layer image-attention ratios."""
    cfg = EvidenceConfig(layer_fraction=1.0, layer_aggregation="mean")
    ev = AttentionEvidence(cfg)
    # Row 1: half the mass on the image span. Row 2: none.
    one = torch.tensor([[[[0.5, 0.5, 0.0, 0.0]]]])   # image mass 1.0 / total 1.0
    zero = torch.tensor([[[[0.0, 0.0, 0.5, 0.5]]]])  # image mass 0.0
    mean = ev.state_image_attention((one, zero), image_token_span=(0, 2))
    assert mean == pytest.approx(0.5)


def test_state_image_attention_layer_max_aggregation():
    cfg = EvidenceConfig(layer_fraction=1.0, layer_aggregation="max")
    ev = AttentionEvidence(cfg)
    one = torch.tensor([[[[0.5, 0.5, 0.0, 0.0]]]])
    zero = torch.tensor([[[[0.0, 0.0, 0.5, 0.5]]]])
    assert ev.state_image_attention((one, zero), image_token_span=(0, 2)) == pytest.approx(1.0)


def test_select_layers_takes_top_fraction():
    cfg = EvidenceConfig(layer_fraction=0.5)
    assert AttentionEvidence.select_layers(8, cfg) == [4, 5, 6, 7]
    assert AttentionEvidence.select_layers(0, cfg) == []
    assert AttentionEvidence.select_layers(1, cfg) == [0]


def test_candidate_embed_cos_is_bounded_and_ranked():
    ev = AttentionEvidence(EvidenceConfig())
    image_emb = torch.tensor([[1.0, 0.0], [1.0, 0.0]])   # image points along +x
    token_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    out = ev.candidate_embed_cos(image_emb, token_emb, [0, 1, 2])
    assert all(0.0 <= v <= 1.0 for v in out)
    assert out[0] > out[1] == pytest.approx(out[2])


def test_candidate_embed_cos_neutral_without_embeddings():
    ev = AttentionEvidence(EvidenceConfig())
    assert ev.candidate_embed_cos(None, None, [1, 2, 3]) == [0.5, 0.5, 0.5]


# ---------------------------------------------------------------------------
# candidate / logit adjustment
# ---------------------------------------------------------------------------


def test_adjusted_logit_subtracts_penalty():
    cand = Candidate(token_id=1, text="dog", logit=10.0, penalty=2.5)
    assert cand.adjusted_logit == pytest.approx(7.5)


def test_candidate_ranking_changes_with_evidence():
    """A strongly penalised candidate must lose to a better-supported one."""
    supported = Candidate(
        token_id=1, text="dog", logit=9.0, ves=0.9,
    )
    unsupported = Candidate(
        token_id=2, text="cat", logit=10.0, ves=0.0,
    )
    # lam large enough that the 0.1 evidence deficit on "dog" outweighs the
    # 1.0 logit advantage of "cat".
    cfg = EvidenceConfig(lam=5.0, threshold=1.0)
    supported.penalty = hallucination_penalty(supported.ves, cfg)
    unsupported.penalty = hallucination_penalty(unsupported.ves, cfg)

    best = max([supported, unsupported], key=lambda c: c.adjusted_logit)
    assert best.token_id == 1


def test_zero_penalty_preserves_logit_order():
    a = Candidate(token_id=1, text="a", logit=1.0, penalty=0.0)
    b = Candidate(token_id=2, text="b", logit=2.0, penalty=0.0)
    assert max([a, b], key=lambda c: c.adjusted_logit).token_id == 2


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def test_invalid_aggregation_rejected():
    with pytest.raises(ValueError, match="head_aggregation"):
        EvidenceConfig(head_aggregation="nonsense")


def test_invalid_normalisation_rejected():
    with pytest.raises(ValueError, match="normalize"):
        EvidenceConfig(normalize="nonsense")


def test_invalid_mix_rejected():
    with pytest.raises(ValueError, match="attention_candidate_mix"):
        EvidenceConfig(attention_candidate_mix=1.5)


def test_negative_weights_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        EvidenceConfig(alpha=-1.0)


def test_threshold_must_be_probability():
    with pytest.raises(ValueError, match="threshold"):
        EvidenceConfig(threshold=1.4)


def test_total_weight_property():
    assert EvidenceConfig(alpha=1.0, beta=2.0, gamma=3.0).total_weight == pytest.approx(6.0)
