# -*- coding: utf-8 -*-
"""POPE loader (Li et al., 2023) with strict validation.

Expected layout::

    <pope_root>/coco_pope_random.jsonl
    <pope_root>/coco_pope_popular.jsonl
    <pope_root>/coco_pope_adversarial.jsonl

Each row: ``question_id`` (optional), ``image``, ``text``, ``label`` (yes/no).
Images are resolved against ``image_root`` first, then ``<pope_root>/images``.
Missing images raise :class:`POPEDataError`; nothing is ever fabricated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

POPE_SETTINGS = ("random", "popular", "adversarial")

_YES = {"yes", "true", "1"}
_NO = {"no", "false", "0"}


class POPEDataError(RuntimeError):
    """Raised when POPE annotations or images are missing or malformed."""


def _label_to_bool(label) -> bool:
    key = str(label).strip().lower()
    if key in _YES:
        return True
    if key in _NO:
        return False
    raise POPEDataError(f"Unrecognised POPE ground-truth label: {label!r} (expected yes/no)")


@dataclass
class POEPSample:
    """One POPE question. (Spelling kept: it is the name the codebase imports.)"""

    question_id: str
    setting: str
    image_id: str
    question: str
    label: str
    image_path: Path

    def __post_init__(self) -> None:
        self.image_path = Path(self.image_path)

    @property
    def ground_truth(self) -> bool:
        return _label_to_bool(self.label)


def locate_annotation_file(pope_root: Path, setting: str) -> Path:
    """Return ``<pope_root>/coco_pope_<setting>.jsonl`` or raise with guidance."""
    if setting not in POPE_SETTINGS:
        raise POPEDataError(
            f"Unknown POPE setting {setting!r}. Choose one of {list(POPE_SETTINGS)}."
        )
    path = Path(pope_root) / f"coco_pope_{setting}.jsonl"
    if not path.is_file():
        raise POPEDataError(
            f"POPE annotation file not found: {path}\n"
            f"Expected coco_pope_{setting}.jsonl inside {pope_root} "
            "(official POPE release)."
        )
    return path


def _probe_paths(
    image_root: Optional[Path], image_name: str, fallback_root: Path
) -> List[Path]:
    name = Path(image_name)
    bases: List[Path] = []
    if image_root is not None:
        bases.append(Path(image_root))
    bases.append(Path(fallback_root) / "images")
    bases.append(Path(fallback_root))
    probes: List[Path] = []
    for base in bases:
        for rel in (name, Path(name.name)):
            cand = base / rel
            if cand not in probes:
                probes.append(cand)
    return probes


def resolve_image_path(
    image_root: Optional[Path], image_name: str, fallback_root: Path
) -> Path:
    """Find a real image file, preferring ``image_root``; raise listing probes."""
    probes = _probe_paths(image_root, image_name, fallback_root)
    for cand in probes:
        if cand.is_file():
            return cand
    raise POPEDataError(
        f"Image {image_name!r} not found. Searched: " + ", ".join(str(p) for p in probes)
    )


def load_pope(
    pope_root: Path,
    setting: str,
    image_root: Optional[Path] = None,
    max_samples: Optional[int] = None,
    require_images: bool = True,
) -> List[POEPSample]:
    """Load one POPE split, validating every row eagerly."""
    pope_root = Path(pope_root)
    ann = locate_annotation_file(pope_root, setting)

    samples: List[POEPSample] = []
    missing: List[str] = []
    with ann.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            if max_samples is not None and len(samples) >= max_samples:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise POPEDataError(f"{ann}:{lineno}: invalid JSON ({exc})") from exc
            for field_name in ("image", "text", "label"):
                if field_name not in row:
                    raise POPEDataError(
                        f"{ann}:{lineno}: row is missing required field {field_name!r}"
                    )
            _label_to_bool(row["label"])  # eager validation
            image_name = str(row["image"])
            try:
                image_path = resolve_image_path(image_root, image_name, pope_root)
            except POPEDataError:
                if require_images:
                    missing.append(image_name)
                    continue
                base = Path(image_root) if image_root else pope_root / "images"
                image_path = base / image_name
            samples.append(
                POEPSample(
                    question_id=str(row.get("question_id", lineno - 1)),
                    setting=setting,
                    image_id=Path(image_name).stem,
                    question=str(row["text"]),
                    label=str(row["label"]),
                    image_path=image_path,
                )
            )

    if missing:
        probes = _probe_paths(image_root, missing[0], pope_root)
        raise POPEDataError(
            f"{len(missing)} POPE image(s) could not be resolved (first: {missing[0]!r}). "
            "Pass --image-root pointing at the COCO images. Searched e.g.: "
            + ", ".join(str(p) for p in probes)
        )
    if not samples:
        raise POPEDataError(f"No POPE samples read from {ann}")
    return samples
