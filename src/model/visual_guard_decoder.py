# -*- coding: utf-8 -*-
"""
VisualGuard decoding: a decoding-time logit intervention.

The intervention happens *inside* the autoregressive loop, before a token is
selected::

    Image
      -> pretrained LVLM
      -> candidate token set (top-k by logit)
      -> visual evidence VES(t) per candidate
      -> hallucination penalty  logit_t -= lambda * deficit(VES(t))
      -> argmax (or sample) over adjusted logits
      -> repeat

Baselines for comparison share the same underlying LVLM and the same prompt,
so any difference is attributable to the decoding rule alone:

==============  ==========================================================
method          decoding rule
==============  ==========================================================
``greedy``      unmodified greedy
``sampling``    unmodified sampling at temperature T
``beam``        unmodified beam search (``num_beams`` configurable)
``attention``   VisualGuard with ``beta = gamma = 0``
``semantic``    VisualGuard with ``alpha = gamma = 0``
``visualguard`` all channels enabled
==============  ==========================================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .grounding import GroundingConfig, GroundingBackend, build_grounding_backend
from .llava_backend import LVLMBackend, LVLMConfig, build_backend
from .visual_evidence import (
    Candidate,
    EvidenceConfig,
    SemanticEvidence,
    VisualEvidenceScorer,
    is_content_bearing,
)

logger = logging.getLogger(__name__)

#: Methods that do not apply any VisualGuard logit modification.
BASELINE_METHODS = {"greedy", "sampling", "beam", "baseline"}

#: Every method the runner understands.
ALL_METHODS = BASELINE_METHODS | {"attention", "semantic", "region", "visualguard", "unidirectional"}


@dataclass
class DecodingConfig:
    """Decoding-level hyper-parameters.

    Attributes
    ----------
    max_new_tokens:
        Generation budget.
    top_k:
        Number of candidate tokens considered per step for evidence scoring.
        Candidates outside the top-k are never penalised — this is a compute
        bound, not a heuristic.
    num_beams:
        Beam width for the ``beam`` baseline.
    temperature / top_p:
        Sampling parameters for the ``sampling`` baseline.
    """

    max_new_tokens: int = 64
    top_k: int = 50
    num_beams: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    early_stop_on_newline: bool = False


@dataclass
class GenerationResult:
    """A generated answer plus timing and intervention diagnostics."""

    text: str
    token_ids: List[int] = field(default_factory=list)
    num_prompt_tokens: int = 0
    num_generated_tokens: int = 0
    method: str = ""
    latency_s: float = 0.0
    per_token_latency_s: float = 0.0
    interventions: int = 0
    interventions_detail: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "num_generated_tokens": self.num_generated_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "method": self.method,
            "latency_s": self.latency_s,
            "per_token_latency_s": self.per_token_latency_s,
            "interventions": self.interventions,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------


def trailing_word_fragment(text: str) -> str:
    """Return the incomplete trailing word of ``text``.

    CLIP should score the *word being formed*, not the whole sentence, so that
    ``"dog"`` competes against ``"cat"`` rather than against a partially
    written clause. Returns ``""`` when ``text`` ends at a word boundary.
    """
    if not text:
        return ""
    if text[-1].isspace():
        return ""
    stripped = text.rstrip()
    head, _, _ = stripped.rpartition(" ")
    return stripped[len(head) + 1 :] if head else stripped


def candidate_fragments(
    tokenizer: Any, generated_ids: Sequence[int], candidate_id: int
) -> str:
    """Trailing word fragment for ``generated_ids + [candidate_id]``."""
    try:
        text = tokenizer.decode(list(generated_ids) + [candidate_id], skip_special_tokens=True)
    except Exception:  # pragma: no cover - tokenizer specific
        return ""
    return trailing_word_fragment(text)


def make_candidates(
    logits: torch.Tensor,
    tokenizer: Any,
    generated_ids: Sequence[int],
    top_k: int,
    score: bool = True,
) -> List[Candidate]:
    """Build a candidate set from a logit row.

    Only content-bearing fragments are scored; the rest keep zero penalty so
    function words are emitted untouched.
    """
    logits = logits[0] if logits.dim() > 1 else logits
    k = min(int(top_k), logits.numel())
    top_logits, top_ids = torch.topk(logits.float(), k)
    candidates: List[Candidate] = []
    for logit, token_id in zip(top_logits.tolist(), top_ids.tolist()):
        fragment = candidate_fragments(tokenizer, generated_ids, token_id)
        candidates.append(
            Candidate(
                token_id=int(token_id),
                text=fragment,
                logit=float(logit),
                is_content=is_content_bearing(fragment),
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# method -> evidence configuration
# ---------------------------------------------------------------------------


def evidence_config_for_method(method: str, base: EvidenceConfig) -> EvidenceConfig:
    """Zero out the channels a method is not supposed to use.

    This guarantees the ablations differ *only* in which evidence channels are
    active, while sharing weights, thresholds and the same LVLM.
    """
    if method in BASELINE_METHODS:
        return EvidenceConfig(
            alpha=0.0, beta=0.0, gamma=0.0, lam=0.0,
            layer_fraction=base.layer_fraction,
            head_aggregation=base.head_aggregation,
            layer_aggregation=base.layer_aggregation,
            attention_candidate_mix=base.attention_candidate_mix,
            normalize=base.normalize,
            clip_model_name=base.clip_model_name,
            clip_device=base.clip_device,
            penalise_function_words=base.penalise_function_words,
            threshold=1.0,
        )
    if method == "attention":
        return _replace(base, alpha=1.0, beta=0.0, gamma=0.0)
    if method == "semantic":
        return _replace(base, alpha=0.0, beta=1.0, gamma=0.0)
    if method == "region":
        return _replace(base, alpha=0.0, beta=0.0, gamma=1.0)
    if method == "unidirectional":  # attention + semantic
        return _replace(base, alpha=1.0, beta=1.0, gamma=0.0)
    if method == "visualguard":
        return base
    raise ValueError(f"Unknown method {method!r}. Valid: {sorted(ALL_METHODS)}")


def _replace(cfg: EvidenceConfig, **kwargs: Any) -> EvidenceConfig:
    from dataclasses import replace

    return replace(cfg, **kwargs)


# ---------------------------------------------------------------------------
# decoder
# ---------------------------------------------------------------------------


class VisualGuardDecoder:
    """Runs generation, optionally with VisualGuard logit modification.

    Example
    -------
    >>> dec = VisualGuardDecoder(LVLMConfig(), DecodingConfig(), EvidenceConfig())
    >>> dec.load()
    >>> out = dec.generate(image, "Is there a dog?", method="visualguard")
    >>> out.interventions >= 0
    True
    """

    def __init__(
        self,
        lvlm_config: Optional[LVLMConfig] = None,
        decoding_config: Optional[DecodingConfig] = None,
        evidence_config: Optional[EvidenceConfig] = None,
        grounding_config: Optional[GroundingConfig] = None,
        method: str = "visualguard",
    ) -> None:
        self.lvlm_config = lvlm_config
        self.decoding_config = decoding_config or DecodingConfig()
        self.evidence_config = evidence_config or EvidenceConfig()
        self.grounding_config = grounding_config or GroundingConfig()
        self.method = method

        self.backend: Optional[LVLMBackend] = None
        self.scorer: Optional[VisualEvidenceScorer] = None
        self._clip_cache: Dict[str, float] = {}
        self._grounding_backend: Optional[GroundingBackend] = None

    # -- lifecycle ----------------------------------------------------

    def load(self) -> None:
        """Load the LVLM and (lazily) the optional evidence backends."""
        self.backend = build_backend(self.lvlm_config)
        self.backend.load()

        if self.method in BASELINE_METHODS:
            # Baselines must not pay for CLIP or a detector.
            self.scorer = None
            return

        grounding_cfg = self.grounding_config
        if self.evidence_config.gamma > 0 and grounding_cfg.backend == "none":
            # gamma requested but no backend configured -> make the intent explicit
            grounding_cfg = GroundingConfig(
                backend="hf_grounding_dino",
                model_id=grounding_cfg.model_id,
                box_threshold=grounding_cfg.box_threshold,
                text_threshold=grounding_cfg.text_threshold,
                device=grounding_cfg.device,
                cache=grounding_cfg.cache,
            )
        self._grounding_backend = build_grounding_backend(grounding_cfg)

        device = self.backend.device
        semantic = SemanticEvidence(self.evidence_config, device=device)
        self.scorer = VisualEvidenceScorer(
            config=self.evidence_config,
            backend=self.backend,
            grounding=self._grounding_backend,
            semantic=semantic,
        )
        self._bind_embeddings()

    def _bind_embeddings(self) -> None:
        """Expose LM-space embeddings for candidate-level attention evidence."""
        if self.scorer is None or self.backend is None:
            return
        model = getattr(self.backend, "model", None)
        if model is None:
            return
        try:
            base = getattr(model, "model", model)
            embed = getattr(getattr(base, "language_model", base), "embed_tokens", None)
            if embed is not None:
                self.scorer.bind_embeddings(
                    input_embeddings=embed.weight.data, image_token_embeddings=None
                )
        except Exception as exc:  # pragma: no cover
            logger.debug("Could not bind token embeddings: %s", exc)

    def set_method(self, method: str) -> None:
        """Switch method. Backend is reused; evidence channels are re-derived."""
        if method not in ALL_METHODS:
            raise ValueError(f"Unknown method {method!r}. Valid: {sorted(ALL_METHODS)}")
        if method in BASELINE_METHODS and self.method not in BASELINE_METHODS:
            self.scorer = None
        elif method not in BASELINE_METHODS:
            self.method = method
            self.evidence_config = evidence_config_for_method(method, self.evidence_config)
            self.load_evidence()
        self.method = method

    def load_evidence(self) -> None:
        """(Re)initialise evidence backends for the current method."""
        if self.backend is None:
            raise RuntimeError("Call load() before load_evidence()")
        grounding_cfg = self.grounding_config
        if self.evidence_config.gamma > 0 and grounding_cfg.backend == "none":
            grounding_cfg = GroundingConfig(
                backend="hf_grounding_dino",
                model_id=grounding_cfg.model_id,
                device=grounding_cfg.device,
                cache=grounding_cfg.cache,
            )
        self._grounding_backend = build_grounding_backend(grounding_cfg)
        semantic = SemanticEvidence(self.evidence_config, device=self.backend.device)
        self.scorer = VisualEvidenceScorer(
            config=self.evidence_config,
            backend=self.backend,
            grounding=self._grounding_backend,
            semantic=semantic,
        )
        self._bind_embeddings()
        self._clip_cache.clear()

    # -- generation ---------------------------------------------------

    def generate(self, image: Any, question: str, method: Optional[str] = None) -> GenerationResult:
        """Generate an answer to ``question`` about ``image`` using ``method``."""
        if self.backend is None:
            raise RuntimeError("Decoder not loaded; call load() first.")
        method = method or self.method
        if method in BASELINE_METHODS:
            return self._generate_baseline(image, question, method)
        return self._generate_visualguard(image, question, method)

    # -- baselines ----------------------------------------------------

    def _generate_baseline(self, image: Any, question: str, method: str) -> GenerationResult:
        """Unmodified decoding via ``model.generate``.

        Deliberately uses the HF generate path so the baselines are the
        standard, well-understood reference points rather than a re-implementation.
        """
        assert self.backend is not None
        model = getattr(self.backend, "model", None)
        if model is None:
            raise RuntimeError("Backend has no HF model; cannot run baseline decode")

        enc = self.backend.encode_prompt(question)
        input_ids = enc["input_ids"]
        pixel_values = self.backend.preprocess_image(image)
        attention_mask = enc.get("attention_mask")

        kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.decoding_config.max_new_tokens,
            pixel_values=pixel_values,
            repetition_penalty=self.decoding_config.repetition_penalty,
            pad_token_id=self.backend.tokenizer.pad_token_id,
        )
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if method in {"greedy", "baseline"}:
            kwargs.update(do_sample=False, num_beams=1)
        elif method == "sampling":
            kwargs.update(
                do_sample=True,
                temperature=self.decoding_config.temperature,
                top_p=self.decoding_config.top_p,
            )
        elif method == "beam":
            kwargs.update(
                do_sample=False,
                num_beams=self.decoding_config.num_beams,
                early_stopping=True,
            )

        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(input_ids=input_ids, **kwargs)
        latency = time.perf_counter() - start

        new_ids = output_ids[0, input_ids.shape[1]:]
        text = self.backend.tokenizer.decode(new_ids, skip_special_tokens=True)
        n_new = int(new_ids.shape[0])
        return GenerationResult(
            text=text,
            token_ids=[int(t) for t in new_ids.tolist()],
            num_prompt_tokens=int(input_ids.shape[1]),
            num_generated_tokens=n_new,
            method=method,
            latency_s=latency,
            per_token_latency_s=latency / max(n_new, 1),
            notes=["baseline: model.generate without VisualGuard logit modification"],
        )

    # -- visualguard --------------------------------------------------

    def _generate_visualguard(self, image: Any, question: str, method: str) -> GenerationResult:
        """Manual autoregressive loop with logit modification."""
        assert self.backend is not None and self.scorer is not None
        tokenizer = self.backend.tokenizer

        enc = self.backend.encode_prompt(question)
        input_ids = enc["input_ids"]
        pixel_values = self.backend.preprocess_image(image)

        self.scorer.semantic.reset_image()
        self._clip_cache.clear()

        eos_ids = self._eos_token_ids()
        generated: List[int] = []
        interventions = 0
        details: List[Dict[str, Any]] = []
        notes: List[str] = []

        start = time.perf_counter()
        step = self.backend.initial_step(input_ids, pixel_values, output_attentions=True)
        state_attn = self.scorer.attention.state_image_attention(
            step.attentions, step.image_token_span
        )
        if step.attentions is None:
            notes.append(
                "attention unavailable from this model; AttentionEvidence fell back "
                "to image_attention=0 and candidate embedding similarity only"
            )
        past = step.past_key_values

        for _ in range(self.decoding_config.max_new_tokens):
            candidates = make_candidates(
                step.logits, tokenizer, generated, self.decoding_config.top_k
            )
            bundle = self.scorer.score_candidates(
                image, candidates, state_image_attention=state_attn
            )
            notes.extend(bundle.notes)

            chosen, did_intervene = self._select(candidates)
            if did_intervene:
                interventions += 1
                if len(details) < 32:  # keep the artefact small
                    details.append(
                        {
                            "step": len(generated),
                            "chosen": chosen.text,
                            "chosen_ves": round(chosen.ves, 4),
                            "runner_up": _runner_up(candidates, chosen),
                            "penalty": round(chosen.penalty, 4),
                        }
                    )

            generated.append(chosen.token_id)
            if chosen.token_id in eos_ids:
                break

            step = self.backend.next_step(
                torch.tensor([[chosen.token_id]], device=self.backend.device),
                past,
                output_attentions=True,
            )
            past = step.past_key_values
            state_attn = self.scorer.attention.state_image_attention(
                step.attentions, None
            ) or state_attn

        latency = time.perf_counter() - start
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            token_ids=list(generated),
            num_prompt_tokens=int(input_ids.shape[1]),
            num_generated_tokens=len(generated),
            method=method,
            latency_s=latency,
            per_token_latency_s=latency / max(len(generated), 1),
            interventions=interventions,
            interventions_detail=details,
            notes=sorted(set(notes)),
        )

    def _select(self, candidates: Sequence[Candidate]) -> Tuple[Candidate, bool]:
        """Pick the token with the highest *adjusted* logit.

        Returns the chosen candidate and whether the evidence penalty actually
        changed the outcome relative to unpenalised greedy decoding. The second
        element is what makes the intervention auditable.
        """
        if not candidates:
            raise ValueError("empty candidate set")
        chosen = max(candidates, key=lambda c: c.adjusted_logit)
        unpenalised = max(candidates, key=lambda c: c.logit)
        return chosen, chosen.token_id != unpenalised.token_id

    def _eos_token_ids(self) -> set:
        tok = self.backend.tokenizer
        ids = set()
        for attr in ("eos_token_id",):
            value = getattr(tok, attr, None)
            if isinstance(value, int):
                ids.add(value)
            elif isinstance(value, (list, tuple)):
                ids.update(int(v) for v in value)
        extra = getattr(tok, "additional_special_tokens_ids", None)
        return ids or {tok.eos_token_id}


def _runner_up(candidates: Sequence[Candidate], chosen: Candidate) -> Dict[str, Any]:
    """Best alternative candidate, for inspecting interventions."""
    others = [c for c in candidates if c.token_id != chosen.token_id]
    if not others:
        return {}
    best = max(others, key=lambda c: c.logit)
    return {
        "text": best.text,
        "ves": round(best.ves, 4),
        "logit": round(best.logit, 4),
        "adjusted_logit": round(best.adjusted_logit, 4),
    }
