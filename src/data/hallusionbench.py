# -*- coding: utf-8 -*-
"""HallusionBench loader.

Reads ``<root>/HallusionBench.json`` (official release): a list of objects with
``question``, ``gt_answer`` ("1"/"0"), ``filename`` and ``category``/``subcategory``.
Questions without an image (``filename`` null) are skipped, since this evaluator
is visual. A ``.jsonl`` file with the same fields is also accepted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_YES = {"yes", "true", "1"}
_NO = {"no", "false", "0"}


class HallusionBenchDataError(RuntimeError):
    """Raised when HallusionBench annotations or images are missing or malformed."""


def _to_bool(value: Any) -> bool:
    key = str(value).strip().lower()
    if key in _YES:
        return True
    if key in _NO:
        return False
    raise HallusionBenchDataError(f"Unrecognised HallusionBench answer: {value!r}")


@dataclass
class HallusionBenchSample:
    question_id: str
    image_path: Path
    question: str
    answer: str
    hallucination_type: str = ""

    def __post_init__(self) -> None:
        self.image_path = Path(self.image_path)

    @property
    def ground_truth(self) -> bool:
        return _to_bool(self.answer)


def _read_rows(root: Path) -> List[Dict[str, Any]]:
    js = root / "HallusionBench.json"
    jl = root / "HallusionBench.jsonl"
    if js.is_file():
        data = json.loads(js.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise HallusionBenchDataError(f"{js} must contain a JSON list")
        return data
    if jl.is_file():
        return [json.loads(l) for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]
    raise HallusionBenchDataError(
        f"HallusionBench annotations not found in {root}. "
        "Expected HallusionBench.json (official release)."
    )


def _resolve(root: Path, filename: str) -> Optional[Path]:
    rel = Path(filename.lstrip("./"))
    for cand in (root / rel, root / rel.name, root / "images" / rel.name):
        if cand.is_file():
            return cand
    return None


def load_hallusionbench(
    hallusionben_root: Path,
    max_samples: Optional[int] = None,
    require_images: bool = True,
) -> List[HallusionBenchSample]:
    root = Path(hallusionben_root)
    if not root.is_dir():
        raise HallusionBenchDataError(f"HallusionBench root {root} does not exist")

    samples: List[HallusionBenchSample] = []
    for idx, row in enumerate(_read_rows(root)):
        if max_samples is not None and len(samples) >= max_samples:
            break
        for field_name in ("question", "gt_answer"):
            if field_name not in row:
                raise HallusionBenchDataError(
                    f"row {idx} is missing required field {field_name!r}"
                )
        filename = row.get("filename")
        if not filename:
            continue  # no image: not a visual sample
        _to_bool(row["gt_answer"])
        path = _resolve(root, str(filename))
        if path is None:
            if require_images:
                raise HallusionBenchDataError(
                    f"HallusionBench image {filename!r} could not be resolved under {root}"
                )
            path = root / str(filename).lstrip("./")
        qid = "{}_{}_{}_{}".format(
            row.get("category", ""), row.get("subcategory", ""),
            row.get("set_id", ""), row.get("question_id", idx),
        )
        samples.append(
            HallusionBenchSample(
                question_id=qid,
                image_path=path,
                question=str(row["question"]),
                answer=str(row["gt_answer"]),
                hallucination_type=str(row.get("category", "")),
            )
        )
    if not samples:
        raise HallusionBenchDataError(f"No HallusionBench samples loaded from {root}")
    return samples
