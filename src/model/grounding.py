# -*- coding: utf-8 -*-
"""
Optional object/region grounding backends for region-level visual evidence.

The region evidence signal answers: "is there a detected region in the image
whose label supports the candidate phrase?"

Grounding DINO is the preferred backend because it performs open-vocabulary
detection from free-form text. It is, however, an *optional* dependency: the
default backend is :class:`NullGroundingBackend`, which returns neutral
evidence and a clear reason string. The whole pipeline runs with grounding
disabled.

Implementations must be real detectors. There is no synthetic-detection
fallback, because fabricated region evidence would silently corrupt the
evidence combination.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detected region."""

    label: str
    score: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel coords)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class GroundingConfig:
    """Configuration for the grounding backend.

    Attributes
    ----------
    backend:
        ``"none"``, ``"grounding_dino"``, or ``"hf_grounding_dino"``.
    model_id:
        Detector checkpoint.
    box_threshold:
        Minimum detection confidence.
    text_threshold:
        Minimum confidence for the matched phrase within a caption.
    cache:
        Cache detections per image so repeated candidate phrases do not
        re-run the detector. This matters: the detector is the most expensive
        component in the pipeline.
    """

    backend: str = "none"
    model_id: str = "IDEA-Research/grounding-dino-base"
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    device: str = "auto"
    dtype: str = "float32"
    #: Explicit aliases for the *detector's* dtype/device in the flat config
    #: namespace. An unqualified ``dtype:`` / ``device:`` in a YAML is routed to
    #: the LVLM (which is what a user means), so the detector needs a distinct
    #: name to be configured from the same file.
    grounding_dtype: Optional[str] = None
    grounding_device_alias: Optional[str] = None
    cache: bool = True
    max_detections: int = 64
    #: Max cached (image, phrase-set) detection results. Bounded because the
    #: detector is re-run for every distinct phrase set the decoder produces.
    cache_size: int = 8

    def __post_init__(self) -> None:
        if self.grounding_dtype is not None:
            self.dtype = self.grounding_dtype
        if self.grounding_device_alias is not None:
            self.device = self.grounding_device_alias


class GroundingUnavailable(RuntimeError):
    """Raised when a requested grounding backend cannot be initialised."""


class GroundingBackend(ABC):
    """Interface for open-vocabulary region detectors."""

    @abstractmethod
    def detect(self, image: Any, phrases: List[str]) -> List[Detection]:
        """Return detections relevant to ``phrases`` for ``image``."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether this backend can produce real detections."""

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


class NullGroundingBackend(GroundingBackend):
    """No-op grounding backend used when grounding is disabled.

    Every phrase receives a neutral region evidence of 0.0 and the pipeline
    records ``"grounding_disabled"`` as the reason. This keeps the ablation
    grid meaningful (region evidence contributes nothing) without pretending
    that regions were detected.
    """

    reason = "grounding_disabled"

    def detect(self, image: Any, phrases: List[str]) -> List[Detection]:
        return []

    @property
    def available(self) -> bool:
        return False


class GroundingDINOBackend(GroundingBackend):
    """Grounding DINO open-vocabulary detector (transformers implementation).

    Two loading paths are supported:

    ``hf_grounding_dino``
        Uses ``transformers.AutoProcessor`` + ``GroundingDinoForObjectDetection``
        (transformers >= 4.40). Preferred: no extra dependency.
    ``grounding_dino``
        Uses the original ``groundingdino`` package. Provide this path only if
        you prefer the upstream implementation.

    Note on caching
    ---------------
    ``detect`` accepts a list of phrases for one image. The decoder should
    batch candidate phrases into a single call: running the detector once per
    candidate token would dominate runtime.
    """

    def __init__(self, config: GroundingConfig) -> None:
        self.config = config
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.device: torch.device = self._resolve_device()
        self._cache: "OrderedDict[Any, List[Detection]]" = OrderedDict()
        #: Strong references backing identity-based cache keys. Per instance and
        #: pruned with the cache, so a long run cannot accumulate every image it
        #: ever grounded. See :meth:`GroundingDINOBackend._image_key`.
        self._anchors: Dict[int, Any] = {}
        self._load()

    # -- loading ------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        if self.config.device != "auto":
            dev = torch.device(self.config.device)
            if dev.type == "cuda" and not torch.cuda.is_available():
                raise GroundingUnavailable(
                    "Grounding DINO requested on CUDA but CUDA is unavailable. "
                    "Use --grounding-backend none to disable region evidence, or "
                    "run grounding on CPU with --grounding-device cpu."
                )
            return dev
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self) -> None:
        if self.config.backend == "grounding_dino":
            self._load_upstream()
        else:
            self._load_hf()

    def _load_hf(self) -> None:
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError as exc:
            raise GroundingUnavailable(
                "Grounding DINO requires transformers>=4.40. "
                "Install with `pip install 'transformers>=4.40'`, or disable "
                "region evidence with --grounding-backend none."
            ) from exc
        try:
            self.processor = AutoProcessor.from_pretrained(self.config.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.config.model_id
            ).to(self.device)
            self.model.eval()
        except Exception as exc:
            raise GroundingUnavailable(
                f"Failed to load Grounding DINO checkpoint {self.config.model_id!r}: {exc}. "
                "Disable region evidence with --grounding-backend none to run the "
                "attention/semantic-only variants."
            ) from exc

    def _load_upstream(self) -> None:
        try:
            from groundingdino.util.inference import load_model  # type: ignore
        except ImportError as exc:
            raise GroundingUnavailable(
                "The `groundingdino` package is not installed. Either "
                "`pip install groundingdino` or switch to "
                "--grounding-backend hf_grounding_dino, or disable region "
                "evidence with --grounding-backend none."
            ) from exc
        self.model = load_model(self.config.model_id, device=str(self.device))

    # -- detection ----------------------------------------------------

    def detect(self, image: Any, phrases: List[str]) -> List[Detection]:
        """Detect regions for ``phrases`` in ``image``, memoised per phrase set.

        The cache key must include the phrases. Keying on the image alone means
        a later request for *different* phrases silently returns the earlier
        detections, which makes every step after the first score against stale
        vocabulary.
        """
        phrases = [p for p in phrases if p and p.strip()]
        if not phrases:
            return []
        key = (self._image_key(image), tuple(sorted({normalise_phrase(p) for p in phrases})))
        if self.config.cache:
            cached = self._cache.get(key)
            if cached is not None:
                # Refresh recency. Without this, insertion-order eviction dropped
                # hot entries while keeping cold ones, so it was FIFO by name
                # only.
                self._cache.move_to_end(key)
                return list(cached)
        dets = self._detect_uncached(image, phrases)
        if self.config.cache:
            self._cache[key] = dets
            self._evict()
        # A copy per caller: the cached list is shared, so a consumer that
        # mutated it would poison the cache for every later sample.
        return list(dets)

    def _image_key(self, image: Any) -> Any:
        """Stable cache key for an image.

        ``id(image)`` is unsafe on its own: CPython reuses addresses after
        garbage collection, so a freed image can alias a different one and
        return the wrong detections. Strings are hashed by value; for anything
        else the key is the object's id *plus* a strong reference held in this
        instance's ``_anchors`` map, so the address cannot be recycled while the
        entry is live.

        The anchor is per-instance, not module-global, and is pruned together
        with the entry it anchors (see :meth:`_evict`). A global map that was
        never pruned retained every image the process ever grounded for the
        lifetime of the run -- gigabytes over a full benchmark.
        """
        if isinstance(image, (str, bytes, Path)):
            return ("path", str(image))
        key = id(image)
        self._anchors[key] = image
        return ("id", key)

    def _evict(self) -> None:
        """Keep the memo bounded, dropping each evicted entry's anchor with it."""
        limit = max(1, int(self.config.cache_size))
        while len(self._cache) > limit:
            old_key, _ = self._cache.popitem(last=False)
            self._drop_anchor(old_key)

    def _drop_anchor(self, cache_key: Any) -> None:
        """Release the strong reference backing an identity cache key.

        A cache key is ``(image_key, phrases)`` where ``image_key`` is itself
        ``("path", str)`` or ``("id", int)``, so the marker is at ``[0][0]``.
        """
        if not isinstance(cache_key, tuple) or not cache_key:
            return
        image_key = cache_key[0]
        if (
            isinstance(image_key, tuple)
            and len(image_key) == 2
            and image_key[0] == "id"
        ):
            self._anchors.pop(image_key[1], None)

    def reset(self) -> None:
        """Drop all cached detections and their anchors (call between samples)."""
        for cache_key in list(self._cache):
            self._drop_anchor(cache_key)
        self._cache.clear()
        self._anchors.clear()

    def _detect_uncached(self, image: Any, phrases: List[str]) -> List[Detection]:
        if self.config.backend == "grounding_dino":
            return self._detect_upstream(image, phrases)
        return self._detect_hf(image, phrases)

    def _detect_hf(self, image: Any, phrases: List[str]) -> List[Detection]:
        from PIL import Image

        if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type for grounding: {type(image)!r}")

        # Grounding DINO expects a period-separated lowercase caption.
        caption = ". ".join(p.strip().lower().rstrip(".") for p in phrases if p.strip())
        if not caption:
            return []
        caption += "."

        inputs = self.processor(images=pil_image, text=caption, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        detections: List[Detection] = []
        for score, label, box in zip(
            results["scores"].tolist(),
            results.get("text_labels", results.get("labels")).tolist(),
            results["boxes"].tolist(),
        ):
            detections.append(
                Detection(label=str(label), score=float(score), box=tuple(box))
            )
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[: self.config.max_detections]

    def _detect_upstream(self, image: Any, phrases: List[str]) -> List[Detection]:
        from groundingdino.util.inference import predict  # type: ignore

        caption = ". ".join(p.strip().lower().rstrip(".") for p in phrases if p.strip())
        detections, _ = predict(
            model=self.model,
            image=image,
            caption=caption + ".",
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
        )

        # Each box must be labelled with the phrase it actually matched.
        #
        # The upstream API returns ``detections.phrases``: the list of phrases
        # whose tokens overlap that box. Using the whole caption as the label
        # instead -- which is what this did -- made every box claim to support
        # every phrase, so all of a question's content words received the *same*
        # top box confidence. Region evidence became a constant that could not
        # rank candidates and actively rewarded uttering any word from the
        # question, whether or not the object was present.
        per_box_phrases = getattr(detections, "phrases", None)

        out: List[Detection] = []
        for idx, boxes in enumerate(detections.logits):
            if per_box_phrases is not None and idx < len(per_box_phrases):
                matched = [str(p).strip() for p in per_box_phrases[idx] if str(p).strip()]
            else:
                # No per-box phrase attribution available. Emitting the caption
                # would reintroduce the false-support bug, so decline instead:
                # an unlabelled box matches nothing and scores 0.0, which is
                # the honest result given the backend cannot say what it saw.
                matched = []
                logger.warning(
                    "grounding_dino backend returned no per-box phrase "
                    "attribution; boxes will be unlabelled and region evidence "
                    "will score 0.0 rather than claim support for every phrase. "
                    "Use --grounding-backend hf_grounding_dino for labelled boxes."
                )
            if not matched:
                continue
            out.append(
                Detection(
                    label=" ".join(matched),
                    score=float(boxes[idx].max().item()),
                    box=tuple(float(v) for v in boxes[idx].tolist()),
                )
            )
        out.sort(key=lambda d: d.score, reverse=True)
        return out[: self.config.max_detections]

    @property
    def available(self) -> bool:
        return self.model is not None

    def close(self) -> None:
        """Release cached detections and their anchors."""
        self.reset()
        self.model = None
        self.processor = None


def build_grounding_backend(config: GroundingConfig) -> GroundingBackend:
    """Construct the configured grounding backend.

    ``backend="none"`` always succeeds and returns the null backend. Any other
    value raises :class:`GroundingUnavailable` with installation guidance if
    the dependency or checkpoint is missing — we never silently degrade a
    requested backend into a fake one.
    """
    if config.backend in {"none", "", None}:
        return NullGroundingBackend()
    if config.backend not in {"grounding_dino", "hf_grounding_dino"}:
        raise GroundingUnavailable(
            f"Unknown grounding backend {config.backend!r}. "
            "Valid choices: none, hf_grounding_dino, grounding_dino."
        )
    return GroundingDINOBackend(config)


# ---------------------------------------------------------------------------
# phrase / label matching
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "of", "and", "or",
    "to", "there", "this", "that", "these", "those", "it", "its", "with", "for",
}


def normalise_phrase(phrase: str) -> str:
    """Lowercase, strip articles/punctuation, collapse whitespace."""
    text = phrase.strip().lower()
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)
    tokens = [t for t in text.split() if t and t not in _STOPWORDS]
    return " ".join(tokens)


def is_token_subsequence(short: Sequence[str], long: Sequence[str]) -> bool:
    """Whether ``short`` appears in ``long`` as a run of whole tokens.

    Matching on tokens rather than raw substrings is what stops ``"cat"``
    matching ``"cattle"`` and ``"bus"`` matching ``"business"``, either of which
    would hand a hallucination a confident region-evidence score.
    """
    if not short or not long or len(short) > len(long):
        return False
    n, m = len(short), len(long)
    return any(list(long[i : i + n]) == list(short) for i in range(m - n + 1))


def _token_matches(phrase_tokens: List[str], label_tokens: List[str]) -> bool:
    """Whether a candidate phrase is supported by a detector label.

    Matching is on *whole tokens*, not raw substrings: substring matching makes
    ``"cat"`` match ``"cattle"`` and ``"bus"`` match ``"business"``, which would
    hand a hallucination a confident region-evidence score. Containment is only
    accepted when the shorter phrase covers a contiguous run of the longer
    one's tokens (articles are already stripped by :func:`normalise_phrase`),
    which is the common case for caption-fragment labels.
    """
    if not phrase_tokens or not label_tokens:
        return False
    if len(phrase_tokens) <= len(label_tokens):
        return is_token_subsequence(phrase_tokens, label_tokens)
    return is_token_subsequence(label_tokens, phrase_tokens)


def match_detection(detections: List[Detection], phrase: str) -> Tuple[float, Optional[Detection]]:
    """Best (score, detection) supporting ``phrase``.

    A detection supports a phrase when the normalised phrase equals the
    normalised label, or when one is a contiguous whole-token subsequence of the
    other. Containment is the common case for open-vocabulary detectors, whose
    labels are caption fragments (``"dog"`` vs ``"brown dog on grass"``).
    """
    target = normalise_phrase(phrase)
    if not target or not detections:
        return 0.0, None
    target_tokens = target.split()
    best_score, best_det = 0.0, None
    for det in detections:
        label = normalise_phrase(det.label)
        if not label:
            continue
        if _token_matches(target_tokens, label.split()) and det.score > best_score:
            best_score, best_det = det.score, det
    return (best_score, best_det)
