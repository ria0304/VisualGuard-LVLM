# -*- coding: utf-8 -*-
"""
Unit tests for the evidence mathematics.

These use small synthetic tensors and mocks deliberately: they verify the
*arithmetic*, not model behaviour. Nothing here touches a real dataset or a
pretrained checkpoint, and no result from these tests is a research result.
"""

from __future__ import annotations

import math
from collections import OrderedDict

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
    NEUTRAL_EVIDENCE,
    RegionEvidence,
    content_words,
    normalise_values,
    zscore_normalise,
)
from src.model.grounding import (
    Detection,
    GroundingConfig,
    GroundingDINOBackend,
    match_detection,
)


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def test_minmax_maps_to_unit_interval():
    assert minmax_normalise([0.0, 0.5, 1.0]) == pytest.approx([0.0, 0.5, 1.0])


def test_minmax_degenerate_range_is_neutral():
    """A constant set carries no information and must not read as full support.

    Mapping it to 1.0 (maximum evidence) meant any channel that failed to
    measure disabled the hallucination penalty for its own candidates, because
    VES then sat above the threshold for all of them.
    """
    assert minmax_normalise([0.7, 0.7, 0.7]) == [NEUTRAL_EVIDENCE] * 3
    assert minmax_normalise([0.0, 0.0]) == [NEUTRAL_EVIDENCE] * 2


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


def test_candidate_embed_cos_reports_unavailable_rather_than_a_constant():
    """``None`` means "cannot rank"; a constant here would read as full support.

    Returning ``[0.5] * n`` looked neutral but was not: the caller min-maxes the
    result, a constant is a degenerate range, and that mapped to ``1.0`` --
    maximum evidence -- so an unmeasurable channel disabled the penalty.
    """
    ev = AttentionEvidence(EvidenceConfig())
    assert ev.candidate_embed_cos(None, None, [1, 2, 3]) is None
    assert ev.candidate_embed_cos(torch.zeros(2, 4), None, [1, 2, 3]) is None
    assert ev.candidate_embed_cos(None, torch.zeros(4, 4), [1, 2, 3]) is None
    assert ev.candidate_embed_cos(torch.zeros(0, 4), torch.zeros(4, 4), [1]) is None
    assert ev.candidate_embed_cos(None, None, []) is None


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


# ---------------------------------------------------------------------------
# region evidence: cache correctness and match precision
# ---------------------------------------------------------------------------


class _CountingGrounding(GroundingDINOBackend):
    """Grounding DINO backend that records every real detector invocation."""

    def __init__(self, cache=True, cache_size=8):
        # Bypass the real __init__: no weights, no network.
        self.config = GroundingConfig(backend="hf_grounding_dino", cache=cache,
                                      cache_size=cache_size)
        self._cache = OrderedDict()
        self._anchors = {}
        self.calls = []
        self.model = object()  # marks the backend as available
        self.processor = None

    def _detect_uncached(self, image, phrases):
        self.calls.append(list(phrases))
        return [Detection(label=p, score=0.9, box=(0.0, 0.0, 1.0, 1.0))
                for p in phrases]


def test_detector_cache_is_keyed_on_phrases_not_just_image():
    """A new phrase set must reach the detector instead of returning stale results.

    Keying the cache on the image alone meant every decoding step after the
    first was scored against the *first* step's vocabulary, so region evidence
    was meaningless.
    """
    backend = _CountingGrounding()

    backend.detect("img.jpg", ["dog"])
    backend.detect("img.jpg", ["cat"])

    assert len(backend.calls) == 2, (
        f"distinct phrase sets reused one detection: {backend.calls}"
    )
    assert backend.calls[1] == ["cat"]

    # ...while an identical phrase set still hits the cache.
    backend.detect("img.jpg", ["cat"])
    assert len(backend.calls) == 2


def test_detector_cache_is_bounded():
    backend = _CountingGrounding(cache_size=2)
    for phrase in ["dog", "cat", "bird", "car"]:
        backend.detect("img.jpg", [phrase])
    assert len(backend._cache) <= 2, "detection cache grew without bound"


def test_detection_cache_uses_value_keys_for_paths():
    """Two equal path strings must share a cache entry.

    ``id(image)`` was used as the key, which CPython recycles after garbage
    collection, so a freed image could alias a different one.
    """
    backend = _CountingGrounding()
    backend.detect("a/img.jpg", ["dog"])
    backend.detect("a/img.jpg", ["dog"])
    assert len(backend.calls) == 1


@pytest.mark.parametrize(
    "label,phrase,expected",
    [
        ("dog", "dog", 0.9),                       # exact
        ("brown dog on grass", "dog", 0.9),        # token subsequence
        ("cattle", "cat", 0.0),                   # NOT a substring match
        ("business", "bus", 0.0),                 # NOT a substring match
        ("cattle", "cow", 0.0),                   # related but unsupported
        ("a dog", "dog", 0.9),                    # article stripped
        ("cat", "dog", 0.0),                      # unrelated
    ],
)
def test_match_detection_is_token_precise(label, phrase, expected):
    dets = [Detection(label=label, score=0.9, box=(0.0, 0.0, 1.0, 1.0))]
    score, _ = match_detection(dets, phrase)
    assert score == pytest.approx(expected)


def test_region_evidence_uses_fixed_vocabulary_from_question():
    """The detector runs once per image, on the question's content words.

    Running it per decoding step is what made region evidence impractical: the
    phrase set changes every step, so nothing would ever hit the cache.
    """
    cfg = EvidenceConfig(alpha=0.0, beta=0.0, gamma=1.0)
    backend = _CountingGrounding()
    evidence = RegionEvidence(cfg, backend)
    assert evidence.available

    evidence.prepare_for_image("img.jpg", "Is there a dog in the image?")
    assert len(backend.calls) == 1
    assert "dog" in evidence.vocabulary
    assert "image" not in evidence.vocabulary, evidence.vocabulary

    # Later scoring of candidates must not trigger further detector calls.
    evidence.score("dog")
    evidence.score("cat")
    assert len(backend.calls) == 1, backend.calls


def test_content_words_drops_question_scaffolding():
    assert content_words("Is there a dog in the image?") == ["dog"]
    # The POPE prompt template appends "Please answer this question with one
    # word." -- none of that may reach the detector as if it named an object.
    pope = content_words("Is there a dog in the image?\nPlease answer this question with one word.")
    assert pope == ["dog"], pope
    assert content_words("") == []
    assert content_words("a red bus and two cats") == ["red", "bus", "cats"]


# ---------------------------------------------------------------------------
# regression: an unmeasurable channel must not read as full support
# ---------------------------------------------------------------------------


class _NullEmbeddings(AttentionEvidence):
    """Channel whose inputs never yield a usable candidate ranking."""


def _attention_only_config(**kwargs):
    defaults = dict(alpha=1.0, beta=0.0, gamma=0.0, lam=1.0, threshold=0.35)
    defaults.update(kwargs)
    return EvidenceConfig(**defaults)


def _scorer(config):
    from src.model.visual_evidence import VisualEvidenceScorer

    return VisualEvidenceScorer(config=config, backend=None)


def test_unavailable_attention_channel_does_not_max_out_evidence():
    """The core regression for ``--method attention`` doing nothing.

    With no image features the candidate-level component is unavailable, so every
    candidate tied. Tying used to normalise to 1.0 -- maximum evidence -- which
    put VES above the threshold and made the penalty a no-op, so the attention
    ablation reported zero interventions while looking like a working run.
    """
    scorer = _scorer(_attention_only_config())
    candidates = [
        Candidate(token_id=i, logit=5.0 - i, text=f"w{i}", is_content=True)
        for i in range(5)
    ]
    bundle = scorer.score_candidates(None, candidates, state_image_attention=0.3)

    assert len({round(c.ves, 9) for c in candidates}) == 1, "candidates diverged"
    assert candidates[0].ves == pytest.approx(NEUTRAL_EVIDENCE)
    assert all(c.penalty == 0.0 for c in candidates)
    assert any("unavailable" in n or "state-level only" in n for n in bundle.notes), (
        f"the run did not record that the channel was unmeasurable: {bundle.notes}"
    )


def test_available_attention_channel_actually_ranks_candidates():
    """The positive control: with embeddings bound, the channel must discriminate."""
    vocab, dim = 8, 2
    scorer = _scorer(_attention_only_config())
    token_embeddings = torch.zeros(vocab, dim)
    token_embeddings[0] = torch.tensor([1.0, 0.0])   # aligned with the image
    token_embeddings[1] = torch.tensor([0.0, 1.0])   # orthogonal
    scorer.bind_embeddings(
        input_embeddings=token_embeddings,
        image_token_embeddings=torch.tensor([[1.0, 0.0]]),
    )
    candidates = [
        Candidate(token_id=0, logit=5.0, text="w0", is_content=True),
        Candidate(token_id=1, logit=4.0, text="w1", is_content=True),
    ]
    scorer.score_candidates(None, candidates, state_image_attention=0.3)

    assert candidates[0].ves > candidates[1].ves, (
        "a measurable channel did not rank the aligned candidate higher"
    )
    assert candidates[1].penalty > 0.0, "the weak candidate was not penalised"


def test_disabled_region_channel_is_neutral_not_zero():
    """An unavailable channel must not deflate VES for every candidate."""
    scorer = _scorer(_attention_only_config(gamma=1.0, alpha=0.0, beta=0.0))
    candidates = [
        Candidate(token_id=i, logit=5.0, text=f"w{i}", is_content=True)
        for i in range(3)
    ]
    bundle = scorer.score_candidates(None, candidates, state_image_attention=0.5)
    assert candidates[0].ves == pytest.approx(NEUTRAL_EVIDENCE)
    assert any("region_evidence_disabled" in n for n in bundle.notes)


def test_ves_denominator_ignores_zero_weight_channels():
    """A zero-weight channel must not shrink VES for every candidate.

    Dividing by *all configured* weights rather than the active ones deflated VES
    by a constant factor whenever a channel was weighted zero, which inflated the
    penalty for every token and made the reported ``ves`` meaningless.
    """
    gamma_off = EvidenceConfig(alpha=1.0, beta=1.0, gamma=0.0, lam=0.5)
    gamma_on = EvidenceConfig(alpha=1.0, beta=1.0, gamma=0.5, lam=0.5)

    assert combine_evidence([0.8], [0.8], [0.0], gamma_off)[0] == pytest.approx(0.8)
    # gamma=0 contributes to neither numerator nor denominator.
    assert combine_evidence([0.8], [0.8], [0.0], gamma_off)[0] == \
        combine_evidence([0.8], [0.8], [1.0], gamma_off)[0]

    # A weighted-but-unavailable channel contributes its neutral value and *is*
    # counted in the denominator, so VES sits between the weighted mean of the
    # measured channels and that mean shifted by the neutral third.
    measured_only = combine_evidence([0.8], [0.8], [NEUTRAL_EVIDENCE], gamma_on)[0]
    assert measured_only < 0.8
    assert measured_only > 0.6


def test_ves_requires_at_least_one_active_channel():
    zero = EvidenceConfig(alpha=0.0, beta=0.0, gamma=0.0)
    with pytest.raises(ValueError, match="at least one"):
        combine_evidence([0.5], [0.5], [0.5], zero)


def test_content_words_excludes_prompt_scaffolding():
    """Detector vocabulary must contain objects, not prompt boilerplate."""
    pope = content_words(
        "Is there a dog in the image?\nPlease answer this question with one word."
    )
    assert pope == ["dog"]

    mme = content_words(
        "What number of people are there?\n"
        "Answer the question using a single word or phrase."
    )
    assert "people" in mme
    for noise in ("using", "single", "phrase", "answer", "question", "what", "there"):
        assert noise not in mme, f"{noise!r} leaked into the detector vocabulary"


def test_region_evidence_requires_the_phrase_to_be_in_the_vocabulary():
    """A phrase the detector was never asked about carries no evidence."""
    region = RegionEvidence(EvidenceConfig(), backend=_StubGrounding(["dog"]))
    region.prepare("image.jpg", ["dog"])
    assert region.score("dog") == pytest.approx(0.9)
    # Never asked about this one, so it must not be matched against the labels.
    assert region.score("couch") == pytest.approx(0.0)


def test_region_prepared_requires_a_non_empty_vocabulary():
    region = RegionEvidence(EvidenceConfig(), backend=_StubGrounding(["dog"]))
    region.prepare("image.jpg", [])
    # An empty phrase list yields an empty (not None) detection list, which used
    # to read as "prepared" and suppress the diagnostic while contributing nothing.
    assert not region.prepared
    assert not region.has_detections


def test_region_reset_clears_detections_and_vocabulary():
    region = RegionEvidence(EvidenceConfig(), backend=_StubGrounding(["dog"]))
    region.prepare("image.jpg", ["dog"])
    assert region.prepared
    region.reset()
    assert not region.prepared
    assert region.vocabulary == ()
    assert region.score("dog") == pytest.approx(0.0)


class _StubGrounding:
    """Minimal grounding backend returning one box per phrase."""

    available = True
    reason = None

    def __init__(self, labels):
        self.labels = list(labels)

    def detect(self, image, phrases):
        from src.model.grounding import Detection

        return [
            Detection(label=label, score=0.9, box=(0.0, 0.0, 1.0, 1.0))
            for label in self.labels
            if label in phrases
        ]

    def reset(self):
        return None

    def close(self):
        return None
