# -*- coding: utf-8 -*-
"""
Real MME evaluation.

Implements the official MME scoring:

* 14 yes/no subtasks, grouped into Perception (10) and Cognition (4).
* Each image is asked exactly two questions.
* ``accuracy``    — fraction of questions answered correctly.
* ``accuracy_plus`` — fraction of *images* where both questions are correct.
* ``score = 100 * (accuracy + accuracy_plus) / 2`` per subtask, capped at 200.
* Category totals are the sum of their subtask scores; the overall MME score is
  the sum over the categories present.

This is the standard MME protocol and is reproduced here in full — no external
evaluator service is required for the yes/no benchmark, and no score is
simulated. See ``docs`` note in the README about MME variants that use a
GPT-based judge; those are not part of the classic 14-subtask benchmark and are
not claimed to be reproduced here.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..data.mme import MMEDataError, MMESample, load_mme
from .metrics import (
    accepts_keyword,
    extract_verdict,
    mme_category_totals,
    mme_subtask_scores,
)

logger = logging.getLogger(__name__)

MME_PROMPT_TEMPLATE = "{question}\nAnswer the question using a single word or phrase."


@dataclass
class MMERecord:
    """One evaluated MME sample."""

    question_id: str
    subtask: str
    image: str
    question: str
    ground_truth: bool
    prediction: Optional[bool]
    raw_output: str
    correct: bool
    parseable: bool
    category: str = ""
    latency_s: float = 0.0
    generated_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "subtask": self.subtask,
            "category": self.category,
            "image": self.image,
            "question": self.question,
            "ground_truth": "yes" if self.ground_truth else "no",
            "prediction": (
                None if self.prediction is None
                else ("yes" if self.prediction else "no")
            ),
            "raw_output": self.raw_output,
            "correct": self.correct,
            "parseable": self.parseable,
            "latency_s": self.latency_s,
            "generated_tokens": self.generated_tokens,
        }


@dataclass
class MMEResult:
    """Aggregated MME metrics plus run metadata."""

    method: str
    metrics: Dict[str, float]
    category_totals: Dict[str, float]
    n_samples: int
    latency_s: float
    per_token_latency_s: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": "mme",
            "method": self.method,
            "metrics": self.metrics,
            "category_totals": self.category_totals,
            "n_samples": self.n_samples,
            "latency_s": self.latency_s,
            "per_token_latency_s": self.per_token_latency_s,
            "notes": self.notes,
        }


class MMEEvaluator:
    """Runs a decoder over MME and scores it with the official formula.

    Example
    -------
    >>> evaluator = MMEEvaluator(decoder.generate, method="visualguard")
    >>> result = evaluator.evaluate(mme_root=Path("data/MME"))
    >>> result.category_totals["total_score"] >= 0
    True
    """

    def __init__(
        self,
        generate_fn: Callable[[Any, str], Any],
        method: str = "greedy",
        prompt_template: str = MME_PROMPT_TEMPLATE,
        max_new_tokens: int = 32,
    ) -> None:
        self.generate_fn = generate_fn
        self.method = method
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        # Signature-inspected once; see :func:`accepts_keyword`.
        self._generate_takes_budget = accepts_keyword(generate_fn, "max_new_tokens")

    def _predict(self, sample: MMESample) -> MMERecord:
        question = self.prompt_template.format(question=sample.question)
        started = time.perf_counter()
        if self._generate_takes_budget:
            out = self.generate_fn(
                sample.image_path, question, max_new_tokens=self.max_new_tokens
            )
        else:
            out = self.generate_fn(sample.image_path, question)
        latency = time.perf_counter() - started

        raw = getattr(out, "text", str(out))
        verdict = extract_verdict(raw)
        return MMERecord(
            question_id=sample.question_id,
            subtask=sample.subtask,
            image=str(sample.image_path),
            question=sample.question,
            ground_truth=sample.ground_truth,
            prediction=verdict,
            raw_output=raw,
            correct=(verdict is not None and verdict == sample.ground_truth),
            parseable=verdict is not None,
            category=sample.category,
            latency_s=latency,
            generated_tokens=int(getattr(out, "num_generated_tokens", 0)),
        )

    def evaluate(
        self,
        mme_root: Path,
        subtasks: Optional[Sequence[str]] = None,
        max_samples_per_subtask: Optional[int] = None,
        max_samples: Optional[int] = None,
        output_dir: Optional[Path] = None,
        progress_every: int = 100,
    ) -> MMEResult:
        """Evaluate MME.

        Raises:
            MMEDataError: if data/images are missing (never substituted).
            ValueError: if no samples were evaluated.
        """
        samples = load_mme(
            mme_root=Path(mme_root),
            subtasks=subtasks,
            max_samples_per_subtask=max_samples_per_subtask,
            max_samples=max_samples,
        )
        logger.info("MME %s: %d samples", self.method, len(samples))

        records: List[MMERecord] = []
        run_start = time.perf_counter()
        total_tokens = 0
        for idx, sample in enumerate(samples, start=1):
            record = self._predict(sample)
            records.append(record)
            total_tokens += record.generated_tokens
            if progress_every and idx % progress_every == 0:
                logger.info("[MME %s] %d/%d", self.method, idx, len(samples))
        total_latency = time.perf_counter() - run_start

        score_records = [
            {
                "image_id": _image_key(r.image, r.subtask, r.question_id),
                "correct": r.correct,
                "subtask": r.subtask,
            }
            for r in records
        ]
        metrics = mme_subtask_scores(score_records)
        totals = mme_category_totals(metrics)

        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self.save_predictions(records, out_dir / f"mme_{self.method}_predictions.jsonl")

        notes: List[str] = []
        unparseable = sum(1 for r in records if not r.parseable)
        if unparseable:
            notes.append(
                f"{unparseable}/{len(records)} answers were not parseable as yes/no "
                "and were scored as incorrect"
            )
        return MMEResult(
            method=self.method,
            metrics=metrics,
            category_totals=totals,
            n_samples=len(records),
            latency_s=total_latency,
            per_token_latency_s=total_latency / max(total_tokens, 1),
            notes=notes,
        )

    @staticmethod
    def save_predictions(records: Sequence[MMERecord], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Wrote %d predictions to %s", len(records), path)


def _image_key(image: str, subtask: str, question_id: str) -> str:
    """Group questions that belong to the same image.

    MME asks two questions per image, and ``accuracy_plus`` is defined over those
    pairs, so the grouping key has to identify the *file*, not just its basename.

    The basename alone is not enough: two genuinely different images that share a
    filename (across subtask directories, or nested layouts) collapsed into one
    group, so a 2-question image was scored as a 4-question group and
    ``accuracy_plus`` was computed over the wrong denominator. The full relative
    path is used, and the ``question_id`` fallback keeps the grouping total when
    no usable path is available.
    """
    text = str(image or "").strip()
    if not text:
        return f"{subtask}::qid:{question_id}"
    # Normalise separators and strip a leading "./" so the same file referenced
    # two ways yields one group.
    normalised = Path(text).as_posix()
    while normalised.startswith("./"):
        normalised = normalised[2:]
    if not normalised or normalised == ".":
        return f"{subtask}::qid:{question_id}"
    return f"{subtask}::{normalised}"
