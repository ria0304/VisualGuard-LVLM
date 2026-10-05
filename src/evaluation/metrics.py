# -*- coding: utf-8 -*-
"""
Metric computation for POPE and MME.

Everything here is a pure function over ``(prediction, ground_truth)`` pairs, so
metrics can be unit-tested without any model or dataset. No function in this
module may fabricate a value: if there are no samples, they raise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# answer normalisation
# ---------------------------------------------------------------------------

_AFFIRMATIVE = {"yes", "yeah", "yep", "yup", "true", "correct", "right", "ok", "okay"}
_NEGATIVE = {"no", "nope", "not", "false", "incorrect", "wrong", "absent"}

_LEADING = re.compile(
    r"^\s*(?:answer\s*[:\-]\s*|the\s+answer\s+is\s*|a\s*:\s*|assistant\s*[:\-]\s*)",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^a-z0-9\s]+")


def normalise_answer(raw: str) -> str:
    """Lowercase, strip common boilerplate and punctuation."""
    if raw is None:
        return ""
    text = str(raw).strip().lower()
    text = _LEADING.sub("", text)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def extract_verdict(raw: str) -> Optional[bool]:
    """Map a free-form answer to ``True`` (yes) / ``False`` (no) / ``None``.

    Returns ``None`` when the answer contains neither an affirmative nor a
    negative token, or when it contains both. Such answers are *unparseable*:
    they are reported separately and scored as errors rather than being
    silently coerced into one class.

    Rules
    -----
    * No decisive token -> ``None``.
    * Decisive tokens all agreeing -> that value. So "No, there is no dog"
      yields ``False`` even though "no" appears twice.
    * Decisive tokens disagreeing -> ``None``. "yes and no" is genuinely
      contradictory and guessing one side would bias the metrics.
    """
    text = normalise_answer(raw)
    if not text:
        return None

    verdicts = [
        token in _AFFIRMATIVE for token in text.split()
        if token in _AFFIRMATIVE or token in _NEGATIVE
    ]
    if not verdicts:
        return None
    if all(verdicts):
        return True
    if not any(verdicts):
        return False
    return None


# ---------------------------------------------------------------------------
# binary classification metrics
# ---------------------------------------------------------------------------


@dataclass
class BinaryMetrics:
    """Counts and derived metrics for a binary yes/no task.

    Positive class = "yes" (an object was asserted to be present).
    """

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    unparseable: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        denom = self.total
        return (self.tp + self.tn) / denom if denom else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom else 0.0

    @property
    def fpr(self) -> float:
        """False-positive rate = rate of hallucinating a nonexistent object."""
        return 1.0 - self.specificity

    @property
    def yes_ratio(self) -> float:
        """Share of all answers that answered "yes"."""
        denom = self.total
        return (self.tp + self.fp) / denom if denom else 0.0

    @property
    def no_ratio(self) -> float:
        return 1.0 - self.yes_ratio if denom_safe(self.total) else 0.0

    @property
    def hallucination_rate(self) -> float:
        """POPE-style hallucination rate: FP / (all "no" ground-truth items).

        This isolates the failure mode the benchmark targets: asserting the
        presence of an object that is absent.
        """
        denom = self.tn + self.fp
        return self.fp / denom if denom else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "n": self.total,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
            "fpr": self.fpr,
            "yes_ratio": self.yes_ratio,
            "no_ratio": self.no_ratio,
            "hallucination_rate": self.hallucination_rate,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "unparseable": self.unparseable,
        }


def denom_safe(n: int) -> bool:
    return n > 0


def confusion_counts(
    predictions: Sequence[Optional[bool]],
    ground_truths: Sequence[bool],
) -> BinaryMetrics:
    """Accumulate a confusion matrix, counting unparseable answers as ``fn``/``fp``.

    An unparseable answer to a "yes" item is a miss; to a "no" item it is a
    hallucination-adjacent error. Either way it is counted against the model and
    surfaced separately via ``unparseable``.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have equal length")
    m = BinaryMetrics()
    for pred, truth in zip(predictions, ground_truths):
        if pred is None:
            m.unparseable += 1
            if truth:
                m.fn += 1
            else:
                m.fp += 1
            continue
        if truth and pred:
            m.tp += 1
        elif truth and not pred:
            m.fn += 1
        elif not truth and pred:
            m.fp += 1
        else:
            m.tn += 1
    return m


def pope_metrics(
    predictions: Sequence[Optional[bool]],
    ground_truths: Sequence[bool],
) -> Dict[str, float]:
    """Full POPE metric bundle. Raises on an empty input rather than faking 0."""
    if not predictions:
        raise ValueError(
            "pope_metrics called with no samples; refusing to report fabricated scores"
        )
    return confusion_counts(predictions, ground_truths).to_dict()


# ---------------------------------------------------------------------------
# MME metrics
# ---------------------------------------------------------------------------


@dataclass
class MMEImageScore:
    """Per-image bookkeeping needed for MME's ``accuracy_plus``.

    MME scores an image correct only if *both* questions for that image are
    answered correctly, so results must be grouped by image before aggregating.
    """

    image_id: str
    correct: List[bool] = field(default_factory=list)

    def add(self, ok: bool) -> None:
        self.correct.append(ok)

    @property
    def accuracy(self) -> float:
        return sum(self.correct) / len(self.correct) if self.correct else 0.0

    @property
    def accuracy_plus(self) -> int:
        """1 if every question for this image is correct, else 0."""
        return 1 if self.correct and all(self.correct) else 0


def mme_subtask_scores(records: Sequence[Dict[str, object]]) -> Dict[str, float]:
    """MME per-subtask score.

    Implements the officially specified MME formula::

        score = 100 * (accuracy + accuracy_plus) / 2

    where ``accuracy`` is over all questions in the subtask and
    ``accuracy_plus`` is the fraction of images for which *both* questions are
    correct. The final MME total is the sum of per-subtask scores.

    Args:
        records: Each dict must contain ``image_id``, ``correct`` (bool) and
            ``subtask`` (str).

    Raises:
        ValueError: if ``records`` is empty, or a record is missing a field.
    """
    if not records:
        raise ValueError(
            "mme_subtask_scores called with no records; refusing to fabricate scores"
        )
    for i, rec in enumerate(records):
        for key in ("image_id", "correct", "subtask"):
            if key not in rec:
                raise ValueError(f"record {i} missing required key {key!r}")

    by_subtask: Dict[str, List[Dict[str, object]]] = {}
    for rec in records:
        by_subtask.setdefault(str(rec["subtask"]), []).append(rec)

    out: Dict[str, float] = {}
    for subtask, recs in sorted(by_subtask.items()):
        groups: Dict[str, MMEImageScore] = {}
        for rec in recs:
            key = str(rec["image_id"])
            groups.setdefault(key, MMEImageScore(image_id=key)).add(bool(rec["correct"]))
        accuracy = sum(g.accuracy for g in groups.values()) / len(groups)
        accuracy_plus = sum(g.accuracy_plus for g in groups.values()) / len(groups)
        out[f"{subtask}/accuracy"] = 100.0 * accuracy
        out[f"{subtask}/accuracy_plus"] = 100.0 * accuracy_plus
        out[f"{subtask}/score"] = 100.0 * (accuracy + accuracy_plus) / 2.0
        out[f"{subtask}/n_images"] = len(groups)
    return out


def mme_category_totals(subtask_scores: Dict[str, float]) -> Dict[str, float]:
    """Sum subtask scores into MME's two published category totals.

    Perception and Cognition are the official groupings; each subtask score is
    capped at its maximum (200 = accuracy and accuracy_plus both 100) before
    summing, which is how MME is defined.
    """
    perception = [
        "existence", "count", "position", "color", "posters", "celebrity",
        "scene", "landmark", "artwork", "OCR",
    ]
    cognition = [
        "commonsense_reasoning", "numerical_calculation",
        "text_translation", "code_reasoning",
    ]
    totals: Dict[str, float] = {}
    for name, members in (("perception", perception), ("cognition", cognition)):
        total = 0.0
        for subtask in members:
            score = subtask_scores.get(f"{subtask}/score")
            if score is None:
                continue
            total += min(score, 200.0)
        totals[f"{name}_score"] = total
    present = [
        v for k, v in totals.items() if v > 0
    ]
    totals["total_score"] = sum(present) if present else 0.0
    return totals
