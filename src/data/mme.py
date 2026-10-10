# -*- coding: utf-8 -*-
"""MME loader with strict validation.

Expected layout::

    <mme_root>/<subtask>/<subtask>.jsonl
    <mme_root>/<subtask>/images/<file>      (or next to the jsonl)

Each row: ``question_id``, ``image``, ``text``, ``answer`` (Yes/No).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

PERCEPTION_SUBTASKS = (
    "existence", "count", "position", "color", "posters", "celebrity",
    "scene", "landmark", "artwork", "OCR",
)
COGNITION_SUBTASKS = (
    "commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning",
)
MME_SUBTASKS = PERCEPTION_SUBTASKS + COGNITION_SUBTASKS

_YES = {"yes", "true", "1"}
_NO = {"no", "false", "0"}


class MMEDataError(RuntimeError):
    """Raised when MME annotations or images are missing or malformed."""


def _answer_to_bool(answer) -> bool:
    key = str(answer).strip().lower()
    if key in _YES:
        return True
    if key in _NO:
        return False
    raise MMEDataError(f"Unrecognised MME answer: {answer!r} (expected Yes/No)")


def _category_of(subtask: str) -> str:
    if subtask in PERCEPTION_SUBTASKS:
        return "perception"
    if subtask in COGNITION_SUBTASKS:
        return "cognition"
    return ""


@dataclass
class MMESample:
    question_id: str
    subtask: str
    image_path: Path
    question: str
    answer: str
    category: str = ""

    def __post_init__(self) -> None:
        self.image_path = Path(self.image_path)

    @property
    def ground_truth(self) -> bool:
        return _answer_to_bool(self.answer)


def _find_jsonl(directory: Path, subtask: str) -> Optional[Path]:
    named = directory / f"{subtask}.jsonl"
    if named.is_file():
        return named
    others = sorted(directory.glob("*.jsonl"))
    return others[0] if others else None


def discover_subtasks(mme_root: Path) -> List[str]:
    """Subtask directories under ``mme_root`` that hold an annotation file."""
    root = Path(mme_root)
    if not root.is_dir():
        return []
    found = [
        p.name for p in root.iterdir()
        if p.is_dir() and _find_jsonl(p, p.name) is not None
    ]
    order = {name: i for i, name in enumerate(MME_SUBTASKS)}
    return sorted(found, key=lambda n: (order.get(n, len(order)), n))


def _resolve_image(directory: Path, image: str) -> Optional[Path]:
    name = Path(image)
    for cand in (
        directory / name,
        directory / "images" / name,
        directory / name.name,
        directory / "images" / name.name,
    ):
        if cand.is_file():
            return cand
    return None


def load_mme(
    mme_root: Path,
    subtasks: Optional[Sequence[str]] = None,
    max_samples_per_subtask: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> List[MMESample]:
    """Load MME samples; ``max_samples`` caps the total across subtasks."""
    root = Path(mme_root)
    layout = (
        "Expected layout: <mme_root>/<subtask>/<subtask>.jsonl with images under "
        "<mme_root>/<subtask>/images/"
    )
    if not root.is_dir():
        raise MMEDataError(f"MME root {root} does not exist. {layout}")

    available = discover_subtasks(root)
    chosen = list(subtasks) if subtasks else available
    if not chosen:
        raise MMEDataError(f"No MME subtasks found under {root}. {layout}")

    samples: List[MMESample] = []
    for subtask in chosen:
        if max_samples is not None and len(samples) >= max_samples:
            break
        directory = root / subtask
        ann = _find_jsonl(directory, subtask) if directory.is_dir() else None
        if ann is None:
            raise MMEDataError(f"MME subtask {subtask!r} has no annotation file in {directory}. {layout}")

        taken = 0
        with ann.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                if max_samples_per_subtask is not None and taken >= max_samples_per_subtask:
                    break
                if max_samples is not None and len(samples) >= max_samples:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MMEDataError(f"{ann}:{lineno}: invalid JSON ({exc})") from exc
                for field_name in ("image", "text", "answer"):
                    if field_name not in row:
                        raise MMEDataError(
                            f"{ann}:{lineno}: row is missing required field {field_name!r}"
                        )
                _answer_to_bool(row["answer"])  # eager validation
                image_path = _resolve_image(directory, str(row["image"]))
                if image_path is None:
                    raise MMEDataError(
                        f"MME image {row['image']!r} not found for subtask {subtask!r}; "
                        f"looked under {directory} and {directory / 'images'}"
                    )
                samples.append(
                    MMESample(
                        question_id=str(row.get("question_id", lineno - 1)),
                        subtask=subtask,
                        image_path=image_path,
                        question=str(row["text"]),
                        answer=str(row["answer"]),
                        category=_category_of(subtask),
                    )
                )
                taken += 1

    if not samples:
        raise MMEDataError(f"No MME samples loaded from {root}. {layout}")
    return samples
