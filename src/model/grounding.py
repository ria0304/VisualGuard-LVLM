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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    cache: bool = True
    max_detections: int = 64


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
        self._cache: Dict[int, List[Detection]] = {}
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
        if not phrases:
            return []
        if self.config.cache:
            key = id(image)
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        dets = self._detect_uncached(image, phrases)
        if self.config.cache:
            self._cache[id(image)] = dets
        return dets

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
        out = [
            Detection(
                label=caption,
                score=float(boxes[idx].max().item()),
                box=tuple(float(v) for v in boxes[idx].tolist()),
            )
            for idx, boxes in enumerate(detections.logits)
        ]
        out.sort(key=lambda d: d.score, reverse=True)
        return out[: self.config.max_detections]

    @property
    def available(self) -> bool:
        return self.model is not None


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


def match_detection(detections: List[Detection], phrase: str) -> Tuple[float, Optional[Detection]]:
    """Best (score, detection) supporting ``phrase``.

    A detection supports a phrase when the normalised phrase equals, is
    contained in, or contains the normalised detection label. Containing a
    phrase (``"dog"`` vs ``"brown dog on grass"``) is the common case for
    open-vocabulary detectors, whose labels are caption fragments.
    """
    target = normalise_phrase(phrase)
    if not target or not detections:
        return 0.0, None
    best_score, best_det = 0.0, None
    for det in detections:
        label = normalise_phrase(det.label)
        if not label:
            continue
        if target == label or target in label or label in target:
            if det.score > best_score:
                best_score, best_det = det.score, det
    return (best_score, best_det)
