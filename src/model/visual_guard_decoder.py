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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..utils.config import accepts_keyword
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
    length_penalty: Optional[float] = None


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
        self._base_evidence_config = evidence_config or EvidenceConfig()
        #: Current (possibly ablation-adjusted) weights. Derived from the base on
        #: every :meth:`set_method` so switching methods is idempotent.
        self.evidence_config = self._base_evidence_config
        self.grounding_config = grounding_config or GroundingConfig()
        self.method = method

        self.backend: Optional[LVLMBackend] = None
        self.scorer: Optional[VisualEvidenceScorer] = None
        self._grounding_backend: Optional[GroundingBackend] = None
        #: Cheap fingerprint of the image the bound features were computed from.
        #: Compared by value, not by ``is``: ``prepare_inputs`` allocates a fresh
        #: tensor per sample, so an identity check never hits and re-ran the
        #: vision tower for every image.
        self._bound_image_key: Optional[Any] = None

    # -- lifecycle ----------------------------------------------------

    def load(self) -> None:
        """Load the LVLM and (lazily) the optional evidence backends."""
        self.backend = build_backend(self.lvlm_config)
        self.backend.load()

        if self.method in BASELINE_METHODS:
            # Baselines must not pay for CLIP or a detector.
            self.scorer = None
            return

        self.load_evidence()

    def _bind_embeddings(self, pixel_values: Optional[torch.Tensor] = None) -> None:
        """Expose LM-space embeddings for candidate-level attention evidence.

        Both halves are required for the channel to rank candidates:

        * ``input_embeddings``      -- the LM's token-embedding matrix, so a
          candidate token id can be turned into a vector.
        * ``image_token_embeddings`` -- the projected visual features the model
          actually sees, obtained from ``get_image_features``.

        Binding only the first leaves ``candidate_embed_cos`` returning ``None``
        (channel unavailable), which silently reduces the whole attention channel
        to a no-op. ``pixel_values`` may be supplied here, or later via
        :meth:`ensure_image_bindings` once the image for the current sample is
        known.
        """
        if self.scorer is None or self.backend is None:
            return
        # Both accessors are optional: ``LVLMBackend`` is an interface other
        # backends may implement partially, and a backend without them simply
        # cannot supply candidate-level attention evidence.
        token_embeddings = _optional_call(self.backend, "token_embedding_matrix")
        image_embeddings = (
            _optional_call(self.backend, "image_token_embeddings", pixel_values)
            if pixel_values is not None
            else None
        )
        if token_embeddings is None:
            logger.debug("Backend exposes no token embedding matrix; "
                         "candidate-level attention evidence unavailable.")
        self.scorer.bind_embeddings(
            input_embeddings=token_embeddings,
            image_token_embeddings=image_embeddings,
        )

    def ensure_image_bindings(
        self, pixel_values: Optional[torch.Tensor], image_key: Optional[Any] = None
    ) -> None:
        """Bind image features for the current sample, once per distinct image.

        Projected visual features depend only on the image, so this is cached and
        recomputed only when the image actually changes.

        ``image_key`` should identify the source image (its path, say). It is
        compared by value because the pixel tensor itself is freshly allocated
        per sample and so cannot be compared by identity; without it we fall back
        to a cheap content fingerprint of the tensor.
        """
        if self.scorer is None:
            return
        key = image_key if image_key is not None else _tensor_fingerprint(pixel_values)
        if pixel_values is None:
            # Clear the *bindings*, not just the cache key. Returning early with
            # the key cleared left the previous sample's projected features in
            # the scorer, so a sample without an image was scored against the
            # prior image.
            if self._bound_image_key is not None:
                self._bound_image_key = None
                self._bind_embeddings(None)
            return
        if key is not None and key == self._bound_image_key:
            return
        self._bound_image_key = key
        self._bind_embeddings(pixel_values)

    def set_method(self, method: str) -> None:
        """Switch method. Backend is reused; evidence channels are re-derived."""
        if method not in ALL_METHODS:
            raise ValueError(f"Unknown method {method!r}. Valid: {sorted(ALL_METHODS)}")
        if method in BASELINE_METHODS:
            if self.scorer is not None:
                self._close_grounding()
            self.scorer = None
        else:
            self.method = method
            # Derive from the *pristine* config, not the current one. Assigning
            # the ablated config back to ``self.evidence_config`` made this
            # non-idempotent: set_method("attention") followed by
            # set_method("visualguard") returned the attention-only weights,
            # permanently losing beta and gamma.
            self.evidence_config = evidence_config_for_method(
                method, self._base_evidence_config
            )
            self.load_evidence()
        self.method = method

    def _close_grounding(self) -> None:
        """Release the grounding backend, if one is loaded."""
        if self._grounding_backend is not None:
            try:
                self._grounding_backend.close()
            except Exception as exc:  # pragma: no cover - backend specific
                logger.debug("grounding backend close failed: %s", exc)
        self._grounding_backend = None

    def load_evidence(self) -> None:
        """(Re)initialise evidence backends for the current method.

        The grounding backend is taken exactly as configured. In particular
        ``backend="none"`` means "region evidence is off", even when
        ``gamma > 0``: the channel then contributes the neutral value and the
        scorer records ``region_evidence_disabled``, so the run cannot be
        mistaken for a region-grounded one. Silently force-enabling a detector the
        user disabled would instead make the run fail on machines without it.

        A region-only method with no backend is rejected outright -- see
        :meth:`_validate_evidence_reachable`.
        """
        if self.backend is None:
            raise RuntimeError("Call load() before load_evidence()")
        self._validate_evidence_reachable()
        # Drop the previous backend before building a new one; overwriting the
        # field leaked a loaded detector (and its pinned cache) on every
        # set_method.
        self._close_grounding()
        self._grounding_backend = build_grounding_backend(self.grounding_config)
        semantic = SemanticEvidence(self.evidence_config, device=self.backend.device)
        self.scorer = VisualEvidenceScorer(
            config=self.evidence_config,
            backend=self.backend,
            grounding=self._grounding_backend,
            semantic=semantic,
        )
        self._bound_image_key = None
        self._bind_embeddings()

    def _validate_evidence_reachable(self) -> None:
        """Reject configurations whose active channels cannot rank candidates.

        VisualGuard works by re-ordering candidates within a step. A channel that
        assigns every candidate the same value therefore cannot affect the
        outcome, whatever its weight. Two such configurations are refused here
        rather than silently reported as an ablation that measured nothing:

        * ``gamma > 0`` with no grounding backend -- region evidence is constant.
        * ``alpha > 0`` with ``attention_candidate_mix == 0`` -- image attention
          is a property of the decoding *state*, so it is identical for every
          candidate at a step. With no candidate-level term the attention channel
          is a constant by construction.
        """
        cfg = self.evidence_config
        if cfg.alpha > 0 and cfg.attention_candidate_mix <= 0.0:
            raise ValueError(
                f"method {self.method!r} has alpha={cfg.alpha} but "
                "attention_candidate_mix=0, which leaves AttentionEvidence a "
                "per-step constant: image attention describes the decoding "
                "state, not the candidate, so every candidate would score "
                "identically and the channel could not re-order anything. Set "
                "attention_candidate_mix > 0 (0.5 is the default) for the "
                "candidate-level term, or --alpha 0 for a run that genuinely "
                "does not use attention evidence."
            )
        if self.grounding_config.backend == "none":
            active = [n for n, w in (("alpha", cfg.alpha), ("beta", cfg.beta),
                                     ("gamma", cfg.gamma)) if w > 0]
            unreachable = [n for n in active if n == "gamma"]
            if unreachable:
                raise ValueError(
                    f"method {self.method!r} has gamma={cfg.gamma} but "
                    "--grounding-backend is 'none', so region evidence cannot be "
                    "measured and the run would be identical to greedy decoding. "
                    "Pass --grounding-backend hf_grounding_dino, or set --gamma 0 "
                    "for a run that genuinely does not use region evidence."
                )

    # -- generation ---------------------------------------------------

    def generate(
        self,
        image: Any,
        question: str,
        method: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """Generate an answer to ``question`` about ``image`` using ``method``.

        ``max_new_tokens`` overrides :class:`DecodingConfig` for this call. The
        evaluators rely on it: POPE answers are a single word, so the benchmark's
        budget (16) must actually reach the decoding loop rather than being
        stored on the evaluator and ignored.
        """
        if self.backend is None:
            raise RuntimeError("Decoder not loaded; call load() first.")
        method = method or self.method
        if method not in ALL_METHODS:
            raise ValueError(f"Unknown method {method!r}. Valid: {sorted(ALL_METHODS)}")

        # Applying the requested method's ablation here, not only in
        # set_method()/run.py. Without it, generate(..., method="attention") on a
        # visualguard decoder ran all three channels while reporting the result
        # as "attention" -- a mislabelled row in the ablation table.
        previous_weights = self.evidence_config
        previous_scorer = self.scorer
        if method not in BASELINE_METHODS:
            requested = evidence_config_for_method(method, self._base_evidence_config)
            if requested != previous_weights or self.scorer is None:
                self.evidence_config = requested
                try:
                    self.load_evidence()
                except Exception:
                    self.evidence_config = previous_weights
                    raise
        elif self.scorer is not None:
            self._close_grounding()
            self.scorer = None

        # The token budget is passed down explicitly instead of being written
        # into the shared DecodingConfig. Mutating it for the duration of the
        # call was not reentrant: two overlapping calls each captured the
        # other's value and the last restore won, permanently corrupting the
        # config for every later sample.
        budget = (
            int(max_new_tokens)
            if max_new_tokens is not None
            else self.decoding_config.max_new_tokens
        )
        try:
            if method in BASELINE_METHODS:
                return self._generate_baseline(image, question, method, budget)
            return self._generate_visualguard(
                image, question, method, budget, self.evidence_config
            )
        finally:
            if previous_scorer is not self.scorer and method not in BASELINE_METHODS:
                self.evidence_config = previous_weights
                self.scorer = previous_scorer
                self._bound_image_key = None
                self._bind_embeddings()
            elif method in BASELINE_METHODS and self.scorer is None:
                self.scorer = previous_scorer
                self.evidence_config = previous_weights

    # -- baselines ----------------------------------------------------

    def _generate_baseline(
        self,
        image: Any,
        question: str,
        method: str,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """Unmodified decoding via ``model.generate``.

        Deliberately uses the HF generate path so the baselines are the
        standard, well-understood reference points rather than a re-implementation.
        """
        assert self.backend is not None
        model = getattr(self.backend, "model", None)
        if model is None:
            raise RuntimeError("Backend has no HF model; cannot run baseline decode")

        inputs = self.backend.prepare_inputs(question, image)
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")

        kwargs: Dict[str, Any] = dict(
            max_new_tokens=(
                self.decoding_config.max_new_tokens
                if max_new_tokens is None
                else max_new_tokens
            ),
            pixel_values=pixel_values,
            repetition_penalty=self.decoding_config.repetition_penalty,
            length_penalty=self.decoding_config.length_penalty,
            pad_token_id=self.backend.tokenizer.pad_token_id,
        )
        if "attention_mask" in inputs:
            kwargs["attention_mask"] = inputs["attention_mask"]
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

    def _generate_visualguard(
        self,
        image: Any,
        question: str,
        method: str,
        max_new_tokens: Optional[int] = None,
        evidence_config: Optional[EvidenceConfig] = None,
    ) -> GenerationResult:
        """Manual autoregressive loop with logit modification."""
        assert self.backend is not None and self.scorer is not None
        tokenizer = self.backend.tokenizer
        cfg = evidence_config or self.evidence_config

        enc = self.backend.prepare_inputs(question, image)
        input_ids = enc["input_ids"]
        pixel_values = enc.get("pixel_values")
        attention_mask = enc.get("attention_mask")

        self.scorer.semantic.reset_image()
        # Drop the previous image's detections and detector cache. Without this,
        # any path that skipped re-preparation scored candidates against the
        # *previous* sample's boxes -- and since RegionEvidence.score() takes no
        # image argument, nothing downstream could notice.
        self.scorer.region.reset()
        reset_grounding_cache(self._grounding_backend)

        notes: List[str] = []

        # Projected visual features for this image. Without them the
        # candidate-level half of the attention channel cannot rank candidates.
        self.ensure_image_bindings(pixel_values, image_key=_image_identity(image))
        if cfg.alpha > 0 and not self.scorer.candidate_embeddings_available:
            notes.append(
                "image_token_embeddings unavailable: AttentionEvidence is "
                "state-level only and cannot rank candidates"
            )

        # Region evidence is prepared once per image, from the question's content
        # words, so the detector is not re-run on every decoding step.
        if cfg.gamma > 0 and self.scorer.region.available:
            self.scorer.region.prepare_for_image(image, question)

        eos_ids = self._eos_token_ids()
        generated: List[int] = []
        interventions = 0
        details: List[Dict[str, Any]] = []

        start = time.perf_counter()
        step = _call_step(
            self.backend.initial_step,
            input_ids,
            pixel_values,
            output_attentions=True,
            attention_mask=attention_mask,
        )
        # The image-token span is fixed by the prompt and stays valid for the
        # whole generation: cached steps only append to the key/value axis.
        image_span = step.image_token_span
        state_attn = self.scorer.attention.state_image_attention(
            step.attentions, image_span
        )
        if step.attentions is None:
            notes.append(
                "attention unavailable from this model; AttentionEvidence fell back "
                "to image_attention=0 and candidate embedding similarity only"
            )
        past = step.past_key_values

        budget = (
            self.decoding_config.max_new_tokens
            if max_new_tokens is None
            else max_new_tokens
        )
        for _ in range(budget):
            # Apply length penalty (same formula as HF's LengthPenalty):
            #   adjusted_logit = logit / (len(generated) + 1) ** length_penalty
            generated_len = len(generated) + 1  # +1 because we count the token about to be generated
            length_penalty = self.decoding_config.length_penalty
            if length_penalty is not None and length_penalty != 1.0:
                # HF applies: logit / (len + 1) ** penalty
                # We keep the original logits and apply penalty in candidate creation
                pass  # applied below per-candidate
            candidates = make_candidates(
                step.logits, tokenizer, generated, self.decoding_config.top_k
            )
            # Apply length penalty to candidate logits
            if length_penalty is not None and length_penalty != 1.0:
                lp = length_penalty
                for c in candidates:
                    # HF length penalty: divide logit by (gen_len + 1) ** penalty
                    # but we've already added 1 to generated_len, so use generated_len
                    c.logit = c.logit / (generated_len ** lp)
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

            step_attention_mask = _extend_attention_mask(attention_mask, past, self.backend)
            step = _call_step(
                self.backend.next_step,
                torch.tensor([[chosen.token_id]], device=self.backend.device),
                past,
                output_attentions=True,
                attention_mask=step_attention_mask,
            )
            past = step.past_key_values
            # Re-read attention against the *same* image-token span. Passing
            # ``None`` here would silently return 0.0 and freeze the state-level
            # signal at its prefill value, which is what turns the feedback loop
            # into a constant.
            #
            step_attn = self.scorer.attention.state_image_attention(
                step.attentions, image_span
            )
            # Adopt any value the call produced, including a genuine 0.0.
            # Gating the adoption on ``step_attn > 0.0`` (to tell "no data" from
            # "no mass") meant a later step could never *reduce* the signal, only
            # replace it with another positive one -- reintroducing a weaker
            # version of the very freeze this re-read exists to prevent. "No
            # data" is ``attentions is None``, which returns 0.0 above and is
            # reported as a note rather than silently treated as a measurement.
            state_attn = step_attn
            if step.attentions is None:
                notes.append(
                    "attention unavailable on a cached decode step; "
                    "image_attention for that step is 0.0 (not measured)"
                )

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


def _call_step(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a backend step, dropping optional keywords it does not accept.

    ``attention_mask`` is an optional capability: a backend or a test double that
    predates it must keep working. Decided by signature, never by catching
    ``TypeError`` -- see :func:`src.utils.config.accepts_keyword`.
    """
    if "attention_mask" in kwargs and not accepts_keyword(fn, "attention_mask"):
        kwargs = {k: v for k, v in kwargs.items() if k != "attention_mask"}
    return fn(*args, **kwargs)


def _extend_attention_mask(
    attention_mask: Optional[torch.Tensor],
    past: Any,
    backend: LVLMBackend,
) -> Optional[torch.Tensor]:
    """Grow a prefill ``attention_mask`` by one column per decoded token.

    A cached forward pass needs a mask covering the whole sequence, cache
    included. Only relevant once prompts are padded; with batch size 1 and no
    padding the mask is all ones and this is a no-op passthrough.
    """
    if attention_mask is None:
        return None
    try:
        cached = past.get_seq_length() if hasattr(past, "get_seq_length") else None
        if not cached:
            return attention_mask
        ones = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        return torch.cat([attention_mask, ones], dim=-1)
    except Exception as exc:  # pragma: no cover - cache backend specific
        logger.debug("could not extend attention_mask: %s", exc)
        return attention_mask


def _image_identity(image: Any) -> Optional[Any]:
    """Cheap comparable identity for the *source* image.

    Path-like inputs identify themselves exactly, which both avoids re-running
    the vision tower for a repeated image and keeps the binding correct when two
    samples share a path. For in-memory images there is no stable cheap key, so
    ``None`` is returned and the caller falls back to a tensor fingerprint.
    """
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        return ("path", str(image))
    return None


def reset_grounding_cache(backend: Optional[GroundingBackend]) -> None:
    """Clear a grounding backend's per-image cache, if it has one.

    The detector is now run once per image with a question-derived vocabulary, so
    its cache almost never hits -- but it retains the images it anchored and
    would serve detections for an identical (image, phrase) pair across samples.
    Resetting at the sample boundary keeps the cache's scope explicit.
    """
    reset = getattr(backend, "reset", None)
    if callable(reset):
        reset()


def _tensor_fingerprint(tensor: Optional[torch.Tensor]) -> Optional[Any]:
    """Cheap value-based identity for a pixel tensor, for cache comparison.

    ``prepare_inputs`` allocates a fresh tensor per sample, so comparing
    ``pixel_values is self._bound_pixel_values`` never hit and the vision tower
    was re-run for every image. Hashing the raw bytes would be exact but costs a
    full pass over the data, so this uses shape, dtype and a small deterministic
    sample of values -- enough to distinguish different images, cheap enough to
    do every step.
    """
    if tensor is None:
        return None
    with torch.no_grad():
        flat = tensor.detach().reshape(-1)
        n = flat.numel()
        if n == 0:
            return (tuple(tensor.shape), str(tensor.dtype))
        # Sample at most 64 evenly spaced positions.
        idx = torch.linspace(0, n - 1, steps=min(64, n), dtype=torch.long)
        sample = flat[idx].to(torch.float32).tolist()
        return (tuple(tensor.shape), str(tensor.dtype), tuple(sample))


def _optional_call(obj: Any, name: str, *args: Any) -> Any:
    """Call ``obj.name(*args)`` if it exists, else return ``None``.

    Keeps optional backend capabilities optional instead of turning a partially
    implemented backend into an ``AttributeError``.
    """
    fn = getattr(obj, name, None)
    if not callable(fn):
        return None
    try:
        return fn(*args)
    except Exception as exc:  # pragma: no cover - backend specific
        logger.debug("backend.%s failed: %s", name, exc)
        return None


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
