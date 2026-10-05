# -*- coding: utf-8 -*-
"""
Visual evidence estimation for VisualGuard.

The module estimates how well a *candidate* token (or the decoding state that
would emit it) is supported by the image, along three complementary channels:

``AttentionEvidence``
    Is the decoding state actually looking at the image? Two components:

    * ``image_attention`` — state-level: how much attention mass the current
      query position places on the image-token span, aggregated over a
      configurable set of layers and heads.
    * ``embed_cos`` — candidate-level: cosine similarity between the
      candidate token's embedding and the image-token embeddings in the LM
      hidden space.

    The split matters. Self-attention rows belong to *positions*, not to
    candidate tokens, so attention alone cannot rank candidates at a step. We
    therefore expose both and mix them with
    :attr:`EvidenceConfig.attention_candidate_mix`, and say so in the README
    rather than overclaiming that attention is a faithful per-candidate
    explanation.

``SemanticEvidence``
    Does the candidate phrase look like this image? CLIP image-text similarity,
    normalised *within the candidate set* so the value is scale-free.

``RegionEvidence``
    Is there a detected region supporting the candidate phrase? Open-vocabulary
    detector (optional; neutral when disabled).

Combined score (the visual evidence score for candidate ``t``)::

    VES(t) = alpha * AttentionEvidence(t)
           + beta  * SemanticEvidence(t)
           + gamma * RegionEvidence(t)

All components are mapped into ``[0, 1]`` before combination, so alpha/beta/
gamma are directly interpretable as weights. Hallucination pressure follows
from insufficient support::

    penalty(t) = lambda * (1 - VES(t))_+   applied as  logit_t -= penalty(t)

Every public entry point is pure w.r.t. its inputs and testable with small
synthetic tensors; no benchmark code depends on this module's randomness.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .grounding import (
    GroundingBackend,
    NullGroundingBackend,
    build_grounding_backend,
    match_detection,
)

EPS = 1e-8


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class EvidenceConfig:
    """Configuration for evidence estimation and combination.

    Weights
    -------
    alpha, beta, gamma:
        Weights for attention / semantic / region evidence.
    lam:
        Strength of the hallucination logit penalty.

    Attention aggregation
    ---------------------
    layer_fraction:
        Fraction of transformer layers (measured from the top, where
        cross-modal grounding tends to be strongest) used for aggregation.
    head_aggregation:
        ``"mean"`` | ``"max"`` | ``"median"`` over heads.
    layer_aggregation:
        ``"mean"`` | ``"max"`` | ``"median"`` over layers.
    attention_candidate_mix:
        ``w`` in ``(1-w)*image_attention + w*embed_cos``. ``w=0`` gives the
        purely state-level signal; ``w=1`` gives a purely candidate-level one.

    Normalisation
    -------------
    normalize:
        ``"minmax"`` (default, per candidate set), ``"zscore"``, or ``"none"``.
    """

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5
    lam: float = 0.5
    threshold: float = 0.35

    layer_fraction: float = 0.5
    head_aggregation: str = "mean"
    layer_aggregation: str = "mean"
    attention_candidate_mix: float = 0.5

    normalize: str = "minmax"

    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_device: Optional[str] = None
    clip_max_batch: int = 64

    penalise_function_words: bool = True

    def __post_init__(self) -> None:
        if self.head_aggregation not in {"mean", "max", "median"}:
            raise ValueError(f"bad head_aggregation={self.head_aggregation!r}")
        if self.layer_aggregation not in {"mean", "max", "median"}:
            raise ValueError(f"bad layer_aggregation={self.layer_aggregation!r}")
        if self.normalize not in {"minmax", "zscore", "none"}:
            raise ValueError(f"bad normalize={self.normalize!r}")
        if not 0.0 <= self.attention_candidate_mix <= 1.0:
            raise ValueError("attention_candidate_mix must lie in [0, 1]")
        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError("alpha, beta, gamma must be non-negative")
        if self.lam < 0:
            raise ValueError("lam must be non-negative")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")

    @property
    def total_weight(self) -> float:
        return self.alpha + self.beta + self.gamma


# ---------------------------------------------------------------------------
# normalisation helpers (pure, unit-tested)
# ---------------------------------------------------------------------------


def minmax_normalise(values: Sequence[float]) -> List[float]:
    """Map ``values`` into ``[0, 1]``; degenerate range maps to all-ones.

    An all-equal candidate set carries no discriminative information, so we
    return 1.0 rather than 0.0: returning 0.0 would penalise every candidate
    equally and destroy the ranking, which is the opposite of the intent.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(hi, lo, rel_tol=1e-9, abs_tol=1e-12):
        return [1.0 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


def zscore_normalise(values: Sequence[float]) -> List[float]:
    """Map ``values`` to zero-mean/unit-variance, then squash to ``[0, 1]``.

    Uses a logistic squash so the output stays bounded, which matters because
    these values are consumed as probabilities in the logit penalty.
    """
    if not values:
        return []
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return [0.5]
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(max(var, EPS))
    return [1.0 / (1.0 + math.exp(-(v - mean) / (std + EPS))) for v in values]


def softmax_normalise(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    """Numerically stable softmax over a candidate set.

    Subtracting the max guarantees at least one ``exp`` term equals 1.0, so no
    epsilon is needed in the denominator and the outputs sum to exactly 1.0.
    """
    if not logits:
        return []
    t = max(temperature, EPS)
    m = max(logits)
    exps = [math.exp((v - m) / t) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def normalise_values(values: Sequence[float], how: str) -> List[float]:
    """Dispatch to the requested normalisation."""
    if how == "minmax":
        return minmax_normalise(values)
    if how == "zscore":
        return zscore_normalise(values)
    return list(values)


def _aggregate(tensor_values: Sequence[float], how: str) -> float:
    if not tensor_values:
        return 0.0
    if how == "max":
        return max(tensor_values)
    if how == "median":
        ordered = sorted(tensor_values)
        n = len(ordered)
        mid = n // 2
        return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return sum(tensor_values) / len(tensor_values)


def combine_evidence(
    attention: Sequence[float],
    semantic: Sequence[float],
    region: Sequence[float],
    config: EvidenceConfig,
) -> List[float]:
    """Compute ``VES(t)`` for each candidate.

    Channels with zero weight are ignored, and the result is divided by the
    sum of the active weights so ``VES`` stays in ``[0, 1]`` regardless of how
    many channels are switched on.
    """
    n = len(attention)
    if n == 0:
        return []
    if not (len(semantic) == n and len(region) == n):
        raise ValueError("evidence channels must have equal length")

    total = config.total_weight
    if total <= 0:
        raise ValueError(
            "at least one of alpha/beta/gamma must be > 0; "
            "otherwise VES is undefined"
        )

    out = []
    for a, s, r in zip(attention, semantic, region):
        ves = config.alpha * a + config.beta * s + config.gamma * r
        out.append(min(max(ves / total, 0.0), 1.0))
    return out


def hallucination_penalty(ves: float, config: EvidenceConfig) -> float:
    """Logit penalty for insufficient visual support.

    ``penalty(t) = lam * max(0, threshold - VES(t)) / threshold``

    Rescaling by ``threshold`` keeps the penalty comparable across settings:
    with ``threshold = 1`` the penalty saturates at ``lam`` for zero evidence.
    A hard threshold (rather than penalising every token) keeps the
    intervention targeted at weakly-grounded content, which is what protects
    the fluency / capability trade-off.
    """
    if config.threshold <= 0:
        return 0.0
    deficit = max(0.0, config.threshold - float(ves))
    return config.lam * (deficit / config.threshold)


# ---------------------------------------------------------------------------
# content-word detection
# ---------------------------------------------------------------------------

FUNCTION_WORDS = frozenset(
    """
    a an the this that these those and or but if then than so because as of in on at
    to for with by from into over under near between across about after before
    is are was were be been being am do does did doing have has had having
    it its it's they them their there here he she his her him hers we us our you
    your i me my mine myself yourself themselves
    not no nor only just also very too more most less least much many few some any
    as well while during through above below off out up down again further once
    can could should would may might must shall will
    """.split()
)

_PUNCT_RE = re.compile(r"^[^\w]+$")


def is_content_bearing(text: str) -> bool:
    """Whether ``text`` is a plausible visually-groundable unit.

    Rejects empty strings, pure punctuation, and function words. This is a
    deliberately conservative lexical filter: over-penalising articles and
    auxiliaries wrecks fluency without reducing object hallucination.
    """
    stripped = text.strip()
    if not stripped or _PUNCT_RE.match(stripped):
        return False
    return stripped.lower() not in FUNCTION_WORDS


# ---------------------------------------------------------------------------
# evidence records
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """A candidate continuation at one decoding step."""

    token_id: int
    text: str
    logit: float
    is_content: bool = True
    # filled in by the scorer
    attention_evidence: float = 0.0
    semantic_evidence: float = 0.0
    region_evidence: float = 0.0
    ves: float = 0.0
    penalty: float = 0.0

    @property
    def adjusted_logit(self) -> float:
        return self.logit - self.penalty


@dataclass
class EvidenceBundle:
    """Evidence for one candidate set, plus provenance metadata."""

    candidates: List[Candidate]
    state_image_attention: float = 0.0
    grounding_reason: Optional[str] = None
    attention_source: str = "image_attention"
    notes: List[str] = field(default_factory=list)

    @property
    def ves(self) -> List[float]:
        return [c.ves for c in self.candidates]


# ---------------------------------------------------------------------------
# channels
# ---------------------------------------------------------------------------


class AttentionEvidence:
    """Attention-based visual grounding signal.

    ``state_image_attention``
        Aggregation over layers/heads of the attention mass the current query
        position assigns to the image-token span.
    ``candidate_embed_cos``
        Cosine similarity between each candidate token embedding and the mean
        image-token embedding.
    """

    def __init__(self, config: EvidenceConfig) -> None:
        self.config = config

    @staticmethod
    def select_layers(num_layers: int, config: EvidenceConfig) -> List[int]:
        """Choose layer indices to aggregate over (top ``layer_fraction``)."""
        if num_layers <= 0:
            return []
        k = max(1, int(round(num_layers * config.layer_fraction)))
        return list(range(num_layers - k, num_layers))

    def state_image_attention(
        self,
        attentions: Optional[Tuple[torch.Tensor, ...]],
        image_token_span: Optional[Tuple[int, int]],
        current_position: Optional[int] = None,
    ) -> float:
        """Aggregate image-token attention for the current query position.

        Args:
            attentions: Per-layer attention tensors, each ``(B, H, q, kv)``.
            image_token_span: ``(start, end)`` of image tokens in the sequence.
            current_position: Query row to read. Defaults to the last row,
                which is the position that predicts the next token — for
                prefill this is the final prompt token, for cached decode steps
                the single newly appended token.

        Returns
        -------
        float
            Value in ``[0, 1]``: image attention mass divided by total mass.
        """
        if not attentions or image_token_span is None:
            return 0.0
        start, end = image_token_span
        if start >= end:
            return 0.0

        per_layer: List[float] = []
        n_layers = len(attentions)
        for layer_idx in self.select_layers(n_layers, self.config):
            attn = attentions[layer_idx]
            if attn is None:
                continue
            if attn.dim() != 4:
                # Some layers return additive attention maps; skip rather than
                # guess a layout.
                continue
            q_len, kv_len = attn.shape[-2], attn.shape[-1]
            if end > kv_len:
                continue
            row = current_position if current_position is not None else q_len - 1
            if row < 0 or row >= q_len:
                continue
            head_scores: List[float] = []
            for h in range(attn.shape[1]):
                head = attn[0, h, row, :].detach().float()
                head = head.clamp_min(0.0)
                total = head.sum()
                if total <= EPS:
                    continue
                head_scores.append(float(head[start:end].sum() / total))
            if head_scores:
                per_layer.append(_aggregate(head_scores, self.config.head_aggregation))

        if not per_layer:
            return 0.0
        return min(max(_aggregate(per_layer, self.config.layer_aggregation), 0.0), 1.0)

    def candidate_embed_cos(
        self,
        image_token_embeddings: Optional[torch.Tensor],
        token_embeddings: Optional[torch.Tensor],
        candidate_token_ids: Sequence[int],
    ) -> List[float]:
        """Cosine similarity of candidate embeddings vs the image embedding.

        Args:
            image_token_embeddings: ``(n_image, d)`` LM-space image embeddings.
            token_embeddings: ``(vocab, d)`` input-embedding matrix.
            candidate_token_ids: Candidate token ids.
        """
        n = len(candidate_token_ids)
        if (
            n == 0
            or image_token_embeddings is None
            or token_embeddings is None
            or image_token_embeddings.numel() == 0
        ):
            return [0.5] * n
        img_vec = F.normalize(image_token_embeddings.float().mean(dim=0), dim=-1)
        cand = F.normalize(token_embeddings.float()[list(candidate_token_ids)], dim=-1)
        sims = cand @ img_vec
        # Cosine lives in [-1, 1]; map to [0, 1] before later min-maxing.
        return [float((s + 1.0) / 2.0) for s in sims]


def _as_tensor(features: Any) -> torch.Tensor:
    """Extract a tensor from CLIP feature output.

    Depending on the ``transformers`` version, ``get_{image,text}_features``
    returns either a tensor or a model output object. Handle both so the
    semantic channel does not break across library versions.
    """
    if torch.is_tensor(features):
        return features
    for attr in ("pooler_output", "text_embeds", "image_embeds", "last_hidden_state"):
        value = getattr(features, attr, None)
        if torch.is_tensor(value):
            return value
    if isinstance(features, (tuple, list)) and features:
        first = features[0]
        if torch.is_tensor(first):
            return first
    raise TypeError(
        f"Could not extract a tensor from CLIP features of type {type(features)!r}"
    )


class SemanticEvidence:
    """CLIP image-text similarity channel."""

    def __init__(self, config: EvidenceConfig, device: Optional[torch.device] = None) -> None:
        self.config = config
        self.device = device
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self._image_features: Optional[torch.Tensor] = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "CLIP is required for semantic evidence. Install transformers "
                "(`pip install 'transformers>=4.40'`)."
            ) from exc
        dev = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(self.config.clip_model_name).to(dev).eval()
        self.processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)
        self.device = dev

    def encode_image(self, image: Any) -> torch.Tensor:
        """Encode and cache the image-side CLIP features (L2-normalised)."""
        self.load()
        from PIL import Image

        if self._image_features is not None:
            return self._image_features
        if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError(f"CLIP expects a PIL image or path, got {type(image)!r}")
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = _as_tensor(self.model.get_image_features(**inputs))
        self._image_features = F.normalize(feats.detach().float(), dim=-1)
        return self._image_features

    def raw_similarities(self, image: Any, texts: Sequence[str]) -> List[float]:
        """Raw CLIP cosine similarities between ``image`` and each text."""
        self.load()
        if not texts:
            return []
        img = self.encode_image(image)
        inputs = self.processor(
            text=[_clip_prompt(t) for t in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k in {"input_ids", "attention_mask"}}
        with torch.no_grad():
            txt = _as_tensor(self.model.get_text_features(**inputs))
        txt = F.normalize(txt.detach().float(), dim=-1)
        sims = (txt @ img.T).squeeze(-1)
        return [float(s) for s in sims]

    def score(self, image: Any, texts: Sequence[str]) -> List[float]:
        """Normalised semantic evidence in ``[0, 1]`` for each text."""
        if not texts:
            return []
        raw = self.raw_similarities(image, texts)
        return normalise_values(raw, self.config.normalize)

    def reset_image(self) -> None:
        """Drop the cached image features (call when the image changes)."""
        self._image_features = None


class RegionEvidence:
    """Region/object grounding channel."""

    def __init__(self, config: EvidenceConfig, backend: Optional[GroundingBackend] = None) -> None:
        self.config = config
        self.backend: GroundingBackend = backend or NullGroundingBackend()
        self._detections: Optional[List[Any]] = None

    @property
    def available(self) -> bool:
        return self.backend.available

    def prepare(self, image: Any, phrases: Sequence[str]) -> None:
        """Run the detector once for a batch of phrases on one image."""
        if not self.backend.available:
            self._detections = None
            return
        self._detections = self.backend.detect(image, list(phrases))

    def score(self, phrase: str) -> float:
        """Best detector confidence supporting ``phrase`` (already in [0, 1])."""
        if self._detections is None:
            return 0.0
        score, _ = match_detection(self._detections, phrase)
        return float(min(max(score, 0.0), 1.0))

    @property
    def reason(self) -> Optional[str]:
        return getattr(self.backend, "reason", None)


# ---------------------------------------------------------------------------
# CLIP prompt helper
# ---------------------------------------------------------------------------


def _clip_prompt(text: str) -> str:
    """Wrap a bare phrase in the CLIP text prompt template.

    CLIP was trained on captions, so a bare noun such as ``"dog"`` scores
    poorly and inconsistently across candidates. A minimal prompt template
    keeps the comparison fair between candidates.
    """
    cleaned = text.strip()
    if not cleaned:
        return "a photo"
    return f"a photo of {cleaned}"


# ---------------------------------------------------------------------------
# top-level scorer
# ---------------------------------------------------------------------------


class VisualEvidenceScorer:
    """Combines the three evidence channels into ``VES(t)`` per candidate.

    Example
    -------
    >>> scorer = VisualEvidenceScorer(config)
    >>> bundle = scorer.score_candidates(image, candidates, step)
    >>> bundle.candidates[0].ves > bundle.candidates[1].ves
    True
    """

    def __init__(
        self,
        config: EvidenceConfig,
        backend: Optional[LVLMBackendProtocol] = None,  # noqa: F821 - forward ref
        grounding: Optional[GroundingBackend] = None,
        semantic: Optional[SemanticEvidence] = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.attention = AttentionEvidence(config)
        self.semantic = semantic or SemanticEvidence(config)
        self.region = RegionEvidence(config, grounding)
        self._embeddings_cache: Optional[torch.Tensor] = None
        self._input_embeddings: Optional[torch.Tensor] = None

    # -- embedding access --------------------------------------------

    def bind_embeddings(
        self,
        input_embeddings: Optional[torch.Tensor],
        image_token_embeddings: Optional[torch.Tensor],
    ) -> None:
        """Provide LM-space embeddings for candidate-level attention evidence."""
        self._input_embeddings = input_embeddings
        self._embeddings_cache = image_token_embeddings

    # -- main entry point --------------------------------------------

    def score_candidates(
        self,
        image: Any,
        candidates: Sequence[Candidate],
        state_image_attention: Optional[float] = None,
    ) -> EvidenceBundle:
        """Score a candidate set.

        Args:
            image: PIL image or path (used by the semantic/region channels).
            candidates: Candidate continuations at this step.
            state_image_attention: Pre-computed attention value for the current
                decoding state. If omitted and a backend is bound, it is
                derived from the last model step when available.
        """
        notes: List[str] = []
        n = len(candidates)
        bundle = EvidenceBundle(candidates=list(candidates))
        if n == 0:
            return bundle

        # 1) Attention evidence ------------------------------------------------
        w = self.config.attention_candidate_mix
        state_attn = (
            state_image_attention
            if state_image_attention is not None
            else getattr(self.backend, "last_image_attention", 0.0)
            if self.backend is not None
            else 0.0
        )
        state_attn = float(min(max(state_attn, 0.0), 1.0))
        bundle.state_image_attention = state_attn

        cand_attn = self.attention.candidate_embed_cos(
            self._embeddings_cache,
            self._input_embeddings,
            [c.token_id for c in candidates],
        )
        attn_raw = [(1.0 - w) * state_attn + w * a for a in cand_attn]
        attention_ev = normalise_values(attn_raw, self.config.normalize)
        bundle.attention_source = "image_attention+embed_cos" if w > 0 else "image_attention"

        # 2) Semantic evidence -------------------------------------------------
        if self.config.beta > 0:
            try:
                semantic_ev = self.semantic.score(image, [c.text for c in candidates])
            except Exception as exc:  # pragma: no cover - defensive
                notes.append(f"semantic_evidence_unavailable: {exc}")
                semantic_ev = [0.5] * n
        else:
            semantic_ev = [0.5] * n

        # 3) Region evidence ---------------------------------------------------
        if self.config.gamma > 0 and self.region.available:
            self.region.prepare(image, [c.text for c in candidates])
            region_ev = [self.region.score(c.text) for c in candidates]
        else:
            if self.config.gamma > 0:
                notes.append(
                    f"region_evidence_disabled: {self.region.reason or 'backend unavailable'}"
                )
            region_ev = [0.0] * n
        bundle.grounding_reason = self.region.reason

        # 4) Combine -----------------------------------------------------------
        ves = combine_evidence(attention_ev, semantic_ev, region_ev, self.config)

        for cand, a, s, r, v in zip(candidates, attention_ev, semantic_ev, region_ev, ves):
            cand.attention_evidence = a
            cand.semantic_evidence = s
            cand.region_evidence = r
            cand.ves = v
            # Only content-bearing tokens are penalised; otherwise we would
            # rewrite function words and destroy fluency.
            cand.penalty = (
                hallucination_penalty(v, self.config)
                if self.config.penalise_function_words or cand.is_content
                else 0.0
            )

        bundle.notes = notes
        return bundle


# Kept as a loose alias so the module does not import the backend module at
# runtime (avoids a circular import when only evidence code is needed).
LVLMBackendProtocol = Any
