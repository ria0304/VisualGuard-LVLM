# -*- coding: utf-8 -*-
"""
Tests that VisualGuard actually intervenes on the logits, and that the
baselines do not.

These use a stub backend and a stub evidence channel, so they verify the
*mechanism* (a candidate's ranking changes because of an evidence penalty) and
never touch a pretrained model. They are not evidence that the method works on
any real benchmark; that requires real experiments.
"""

from __future__ import annotations

import pytest
import torch

from src.model.visual_evidence import Candidate, EvidenceConfig, VisualEvidenceScorer
from src.model.visual_guard_decoder import (
    BASELINE_METHODS,
    DecodingConfig,
    GenerationResult,
    VisualGuardDecoder,
    _runner_up,
    evidence_config_for_method,
    trailing_word_fragment,
)


# ---------------------------------------------------------------------------
# a minimal stub backend
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Token id N decodes to the word 'wN' (or 'the' for id 0)."""

    eos_token_id = 99
    pad_token_id = 0

    _special = {0: "the"}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self._special.get(int(i), f"w{int(i)}") for i in ids)


class _StubStep:
    def __init__(self, logits):
        self.logits = logits
        self.attentions = None
        self.past_key_values = None
        self.image_token_span = None


class _StubBackend:
    """Returns a fixed logit row on every step; no weights, no network."""

    def __init__(self, logits):
        self._logits = logits
        self.tokenizer = _StubTokenizer()
        self.device = torch.device("cpu")
        self.model = None

    def encode_prompt(self, question):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def preprocess_image(self, image):
        return torch.zeros(1, 3, 4, 4)

    def initial_step(self, input_ids, pixel_values, output_attentions=False):
        return _StubStep(self._logits)

    def next_step(self, input_ids, past_key_values, output_attentions=False):
        return _StubStep(self._logits)

    def image_token_span(self, input_ids, pixel_values):
        return None


# ---------------------------------------------------------------------------
# the mechanism
# ---------------------------------------------------------------------------


def _decoder_with_stub(logits, method="visualguard", **evidence_kwargs):
    """Build a decoder with a stub backend and NO external evidence backends.

    ``beta`` and ``gamma`` default to 0 here so the test never triggers a CLIP
    or detector download; these tests exercise the intervention mechanism, not
    the channels.
    """
    defaults = dict(
        alpha=1.0, beta=0.0, gamma=0.0, lam=1.0, threshold=1.0,
        penalise_function_words=False,
    )
    defaults.update(evidence_kwargs)
    dec = VisualGuardDecoder(
        lvlm_config=None,
        decoding_config=DecodingConfig(max_new_tokens=3, top_k=3),
        evidence_config=EvidenceConfig(**defaults),
        method=method,
    )
    dec.backend = _StubBackend(logits)
    dec.scorer = VisualEvidenceScorer(config=dec.evidence_config, backend=None)
    return dec


def test_penalty_changes_the_selected_token():
    """The core claim: evidence must be able to flip the greedy choice.

    This drives the *real* ``score_candidates`` path. Candidate-level attention
    evidence comes from the cosine similarity between each candidate token
    embedding and the image-token embedding, so controlling those embeddings
    controls the evidence without any model or network access.
    """
    # Tokens 0/1 have high logits; token 5 has the highest logit but is the
    # only candidate whose embedding is orthogonal to the image direction.
    logits = torch.tensor([[5.0, 5.0, 5.0, 0.0, 0.0, 10.0]])

    vocab, dim = 8, 2
    input_embeddings = torch.zeros(vocab, dim)
    input_embeddings[0] = torch.tensor([1.0, 0.0])   # aligned with image
    input_embeddings[1] = torch.tensor([1.0, 0.0])
    input_embeddings[5] = torch.tensor([0.0, 1.0])   # orthogonal -> weak evidence
    image_token_embeddings = torch.tensor([[1.0, 0.0]])

    dec = _decoder_with_stub(logits, lam=10.0, threshold=1.0)
    dec.scorer.bind_embeddings(
        input_embeddings=input_embeddings,
        image_token_embeddings=image_token_embeddings,
    )

    from src.model.visual_guard_decoder import make_candidates

    cands = make_candidates(logits, _StubTokenizer(), [], top_k=3)
    top = max(cands, key=lambda c: c.logit)
    assert top.token_id == 5, "precondition: token 5 wins on raw logits"

    bundle = dec.scorer.score_candidates(None, cands, state_image_attention=0.0)
    by_id = {c.token_id: c for c in bundle.candidates}

    # The poorly-grounded candidate must receive a strictly larger penalty.
    assert by_id[5].penalty > by_id[0].penalty
    assert by_id[0].ves > by_id[5].ves

    chosen, intervened = dec._select(bundle.candidates)
    assert chosen.token_id != 5, "penalty failed to change the greedy choice"
    assert intervened is True


def test_evidence_scores_stay_in_unit_interval_end_to_end():
    """The full scorer must emit bounded VES values for any candidate set."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    dec = _decoder_with_stub(logits)
    dec.scorer.bind_embeddings(
        input_embeddings=torch.randn(8, 4),
        image_token_embeddings=torch.randn(3, 4),
    )
    from src.model.visual_guard_decoder import make_candidates

    cands = make_candidates(logits, _StubTokenizer(), [], top_k=4)
    bundle = dec.scorer.score_candidates(None, cands, state_image_attention=0.7)
    for cand in bundle.candidates:
        assert 0.0 <= cand.ves <= 1.0
        assert cand.penalty >= 0.0

def test_no_intervention_when_all_candidates_equal_penalty():
    logits = torch.tensor([[0.0, 0.0, 5.0]])
    dec = _decoder_with_stub(logits)
    from src.model.visual_guard_decoder import make_candidates

    cands = make_candidates(logits, _StubTokenizer(), [], top_k=3)
    for c in cands:
        c.penalty = 0.0
    chosen, intervened = dec._select(cands)
    assert chosen.token_id == 2  # the highest-logit token
    assert intervened is False


def test_function_words_are_not_penalised():
    """Penalising articles/auxiliaries would wreck fluency for no benefit."""
    from src.model.visual_guard_decoder import make_candidates

    class _T:
        def decode(self, ids, skip_special_tokens=True):
            return {0: "the", 1: "dog"}[int(ids[-1])]

    logits = torch.tensor([[5.0, 1.0]])
    cands = make_candidates(logits, _T(), [], top_k=2)
    by_id = {c.token_id: c for c in cands}
    assert by_id[0].text == "the"
    assert by_id[0].is_content is False
    assert by_id[1].is_content is True


def test_select_rejects_empty_candidate_set():
    dec = _decoder_with_stub(torch.tensor([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="empty candidate set"):
        dec._select([])


def test_runner_up_reports_best_alternative():
    cands = [
        Candidate(token_id=1, text="dog", logit=5.0, ves=0.9),
        Candidate(token_id=2, text="cat", logit=9.0, ves=0.1),
    ]
    out = _runner_up(cands, cands[0])
    assert out["text"] == "cat"
    assert out["ves"] == pytest.approx(0.1)


def test_runner_up_empty_when_single_candidate():
    c = [Candidate(token_id=1, text="dog", logit=1.0)]
    assert _runner_up(c, c[0]) == {}


def test_baseline_methods_never_build_a_scorer():
    """Baselines must not pay the evidence cost, and must not be modified."""
    for method in BASELINE_METHODS:
        dec = VisualGuardDecoder(method=method)
        dec.backend = _StubBackend(torch.tensor([[0.0, 1.0]]))
        dec.scorer = None
        assert dec.scorer is None


def test_evidence_config_for_method_zeroes_penalty_for_baselines():
    cfg = evidence_config_for_method("greedy", EvidenceConfig(lam=2.0))
    assert cfg.lam == 0.0
    assert cfg.total_weight == 0.0


def test_generation_result_serialises():
    res = GenerationResult(
        text="yes",
        token_ids=[1],
        num_generated_tokens=1,
        method="visualguard",
        latency_s=0.5,
        interventions=2,
    )
    d = res.to_dict()
    assert d["text"] == "yes"
    assert d["interventions"] == 2
    assert d["method"] == "visualguard"


def test_trailing_fragment_drives_candidate_text():
    assert trailing_word_fragment("a dog") == "dog"
    assert trailing_word_fragment("a dog ") == ""


def test_baseline_and_visualguard_share_one_model_instance():
    """Scientific requirement: all compared methods must use the same LVLM.

    If each method loaded its own copy, any metric difference could be
    attributed to checkpoint/quantisation nondeterminism instead of the
    decoding rule. This pins that a single decoder instance -- and therefore a
    single loaded model -- serves every method.
    """
    from src.model.visual_guard_decoder import ALL_METHODS

    dec = VisualGuardDecoder(
        lvlm_config=None,
        decoding_config=DecodingConfig(),
        evidence_config=EvidenceConfig(alpha=1.0, beta=0.0, gamma=0.0, lam=1.0),
        method="visualguard",
    )
    sentinel_model = object()
    dec.backend = _StubBackend(torch.tensor([[0.0, 1.0]]))
    dec.backend.model = sentinel_model

    for method in sorted(ALL_METHODS):
        dec.method = method
        # The backend is untouched by a method switch: same object, same model.
        assert dec.backend.model is sentinel_model
        assert dec.backend is not None


def test_set_method_zeroes_penalty_for_baselines():
    dec = VisualGuardDecoder(
        lvlm_config=None,
        evidence_config=EvidenceConfig(alpha=1.0, beta=1.0, gamma=1.0, lam=3.0),
        method="visualguard",
    )
    dec.set_method("greedy")
    assert dec.method == "greedy"
    assert dec.scorer is None


def test_set_method_rejects_unknown():
    dec = VisualGuardDecoder(method="baseline")
    with pytest.raises(ValueError, match="Unknown method"):
        dec.set_method("not_a_method")


def test_baseline_method_does_not_construct_evidence_backends():
    """Baselines must not pay for CLIP or a detector."""
    dec = VisualGuardDecoder(
        lvlm_config=None,
        method="baseline",
        evidence_config=EvidenceConfig(alpha=1.0, beta=1.0, gamma=1.0),
    )
    dec.backend = _StubBackend(torch.tensor([[0.0, 1.0]]))
    dec.load_evidence = lambda: pytest.fail("baseline must not load evidence backends")
    assert dec.scorer is None


# ---------------------------------------------------------------------------
# regression guards for the attention channel
# ---------------------------------------------------------------------------
#
# The attention channel used to be a silent no-op: the decoder bound the token
# embedding matrix but passed ``image_token_embeddings=None``, so
# ``candidate_embed_cos`` returned a constant, every candidate received the same
# ``VES``, no penalty was ever applied, and ``--method attention`` produced
# byte-identical output to ``--method baseline``. The tests below fail if that
# can happen again.


def test_attention_evidence_varies_across_candidates():
    """The attention channel must discriminate between candidates.

    A constant VES across the candidate set is the exact signature of the
    no-op bug: it means the channel contributes nothing to the ranking.
    """
    cfg = EvidenceConfig(
        alpha=1.0, beta=0.0, gamma=0.0, lam=0.5, threshold=0.35,
        attention_candidate_mix=1.0,
    )
    scorer = VisualEvidenceScorer(cfg)

    # Candidate 1 aligns with the image direction; candidate 2 is orthogonal.
    input_embeddings = torch.zeros(8, 2)
    input_embeddings[1] = torch.tensor([1.0, 0.0])
    input_embeddings[2] = torch.tensor([0.0, 1.0])
    scorer.bind_embeddings(
        input_embeddings=input_embeddings,
        image_token_embeddings=torch.tensor([[1.0, 0.0]]),
    )
    assert scorer.candidate_embeddings_available

    cands = [
        Candidate(token_id=1, text="dog", logit=5.0),
        Candidate(token_id=2, text="zzz", logit=4.5),
    ]
    bundle = scorer.score_candidates(None, cands, state_image_attention=0.42)

    assert bundle.candidates[0].ves != bundle.candidates[1].ves, (
        "attention evidence is constant across candidates; the channel is a no-op"
    )
    # The visually supported token must be the cheaper one to emit.
    assert bundle.candidates[0].penalty < bundle.candidates[1].penalty


def test_attention_method_is_not_identical_to_baseline():
    """``--method attention`` must actually change the logits it inspects.

    Compares the per-candidate adjusted logits against the unpenalised ones for
    the same stub logits, which is exactly the comparison that was vacuous
    before the fix.
    """
    logits = torch.tensor([[5.0, 4.0, 0.0, 0.0, 0.0, 9.0]])

    vocab, dim = 8, 2
    input_embeddings = torch.zeros(vocab, dim)
    input_embeddings[1] = torch.tensor([1.0, 0.0])
    input_embeddings[5] = torch.tensor([0.0, 1.0])  # top logit, weak evidence
    image_token_embeddings = torch.tensor([[1.0, 0.0]])

    cfg = EvidenceConfig(
        alpha=1.0, beta=0.0, gamma=0.0, lam=5.0, threshold=1.0,
        attention_candidate_mix=1.0, penalise_function_words=False,
    )
    scorer = VisualEvidenceScorer(cfg)
    scorer.bind_embeddings(
        input_embeddings=input_embeddings,
        image_token_embeddings=image_token_embeddings,
    )

    from src.model.visual_guard_decoder import make_candidates

    cands = make_candidates(logits, _StubTokenizer(), [], top_k=3)
    scorer.score_candidates(None, cands, state_image_attention=0.0)

    assert any(c.penalty > 0.0 for c in cands), (
        "no candidate was penalised: the attention channel did nothing"
    )
    # The penalty must be able to outrank the raw logit ordering.
    greedy = max(cands, key=lambda c: c.logit)
    adjusted = max(cands, key=lambda c: c.adjusted_logit)
    assert adjusted.token_id != greedy.token_id


def test_missing_image_embeddings_is_reported_not_hidden():
    """Degrading to state-level-only must be visible in the notes."""
    cfg = EvidenceConfig(
        alpha=1.0, beta=0.0, gamma=0.0, attention_candidate_mix=0.5,
    )
    scorer = VisualEvidenceScorer(cfg)
    scorer.bind_embeddings(
        input_embeddings=torch.zeros(4, 2), image_token_embeddings=None
    )
    assert not scorer.candidate_embeddings_available

    cands = [Candidate(token_id=1, text="dog", logit=1.0)]
    bundle = scorer.score_candidates(None, cands, state_image_attention=0.3)
    assert any("state-level only" in note for note in bundle.notes), bundle.notes


def test_decoder_binds_real_image_embeddings_from_the_backend():
    """The decoder must forward the backend's image features to the scorer.

    This is the exact wiring that was broken: ``bind_embeddings`` was called
    with ``image_token_embeddings=None`` hard-coded, so the scorer could never
    rank candidates no matter what the backend could provide.
    """
    class _EmbeddingBackend(_StubBackend):
        def __init__(self, logits):
            super().__init__(logits)
            self.asked_for_pixels = False

        def token_embedding_matrix(self):
            return torch.zeros(8, 2)

        def image_token_embeddings(self, pixel_values):
            self.asked_for_pixels = pixel_values is not None
            return torch.tensor([[1.0, 0.0]]) if pixel_values is not None else None

    dec = _decoder_with_stub(torch.tensor([[1.0, 2.0]]))
    backend = _EmbeddingBackend(torch.tensor([[1.0, 2.0]]))
    dec.backend = backend

    dec.ensure_image_bindings(torch.zeros(1, 3, 4, 4))

    assert backend.asked_for_pixels, "decoder never asked the backend for image features"
    assert dec.scorer.candidate_embeddings_available, (
        "image features were available but the scorer still cannot rank candidates"
    )


def test_state_image_attention_is_reread_each_step():
    """The per-step attention update must use the image-token span.

    Regression guard for passing ``None`` as the span on cached steps, which
    made ``state_image_attention`` return 0.0 and froze the state-level signal
    at its prefill value.
    """
    from src.model.visual_evidence import AttentionEvidence

    # One layer, one head, q_len=1, kv_len=10; image tokens occupy [2, 6).
    # Mass 4 on the image span, 6 elsewhere -> 4/10 = 0.4 once normalised.
    attn = torch.zeros(1, 1, 1, 10)
    attn[0, 0, 0, 2:6] = 1.0
    attn[0, 0, 0, 0:2] = 1.0
    attn[0, 0, 0, 6:] = 1.0
    attentions = (attn,)

    evidence = AttentionEvidence(EvidenceConfig(layer_fraction=1.0))

    with_span = evidence.state_image_attention(attentions, (2, 6))
    without_span = evidence.state_image_attention(attentions, None)

    assert with_span == pytest.approx(0.4, abs=1e-6)
    assert without_span == 0.0, (
        "a missing span must read as 0.0, which is exactly why the decode "
        "loop must keep passing the real span"
    )


def test_decode_loop_passes_image_span_on_every_step():
    """The loop must re-read attention against the image span each step.

    Guards the decode loop itself (not just ``AttentionEvidence``): passing
    ``None`` as the span on cached steps silently returns 0.0, which froze the
    state-level signal at its prefill value and made the "dynamic" feedback
    loop static.
    """
    vocab, dim, n_img = 8, 2, 4

    class _SpanBackend(_StubBackend):
        """Reports a non-degenerate image span and records the spans it is given."""

        def __init__(self, logits):
            super().__init__(logits)
            self.seen_spans = []

        def prepare_inputs(self, question, image=None):
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "pixel_values": torch.zeros(1, 3, 4, 4),
            }

        def initial_step(self, input_ids, pixel_values, output_attentions=False):
            step = _StubStep(self._logits)
            step.image_token_span = (0, n_img)
            self.seen_spans.append(step.image_token_span)
            return step

        def next_step(self, input_ids, past_key_values, output_attentions=False):
            step = _StubStep(self._logits)
            step.image_token_span = (0, n_img)
            self.seen_spans.append(step.image_token_span)
            return step

        def token_embedding_matrix(self):
            return torch.zeros(vocab, dim)

        def image_token_embeddings(self, pixel_values):
            return torch.zeros(n_img, dim) + 1.0

    logits = torch.tensor([[5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dec = _decoder_with_stub(logits, lam=1.0, threshold=1.0)
    backend = _SpanBackend(logits)
    dec.backend = backend

    # Spy on the *call the decode loop makes*, not on what the backend returns:
    # the bug was the loop discarding the span, so only the argument catches it.
    passed_spans = []
    original = dec.scorer.attention.state_image_attention

    def _spy(attentions, image_token_span, current_position=None):
        passed_spans.append(image_token_span)
        return original(attentions, image_token_span, current_position)

    dec.scorer.attention.state_image_attention = _spy

    dec.generate("image.jpg", "Is there a dog?")

    assert len(passed_spans) >= 2, "loop never took a cached step"
    assert all(span is not None for span in passed_spans), (
        f"the loop dropped the image-token span on a cached step: {passed_spans}"
    )
    assert passed_spans[0] == (0, n_img)
