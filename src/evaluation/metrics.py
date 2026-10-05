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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Re-exported so callers that already import from this module can reach the
# shared capability probe without a second import path.
from ..utils.config import accepts_keyword

__all__ = [
    "BinaryMetrics",
    "MMEImageScore",
    "accepts_keyword",
    "confusion_counts",
    "extract_verdict",
    "mme_category_totals",
    "mme_subtask_scores",
    "normalise_answer",
    "pope_metrics",
]

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
        """Every scored item, including answers that could not be parsed.

        Unparseable answers count here so they still count *against* accuracy,
        but they are kept out of ``fp``/``fn``: an unparseable answer is not an
        assertion that the object is present, so folding it into ``fp`` turned
        a model emitting garbage into a reported 100% hallucination rate.
        """
        return self.tp + self.fp + self.tn + self.fn + self.unparseable

    @property
    def parseable(self) -> int:
        """Items whose answer resolved to a yes/no verdict."""
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
    def has_negatives(self) -> bool:
        """Whether any ground-truth "no" item was scored."""
        return (self.tn + self.fp) > 0

    @property
    def has_positives(self) -> bool:
        """Whether any ground-truth "yes" item was scored."""
        return (self.tp + self.fn) > 0

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom else 0.0

    @property
    def fpr(self) -> float:
        """False-positive rate: share of "no" items the model failed to reject.

        Computed directly rather than as ``1 - specificity``. Composing the two
        made ``specificity``'s "no negatives" sentinel of ``0.0`` invert into an
        ``fpr`` of ``1.0``, i.e. a run with nothing but "yes" items was reported
        as hallucinating on 100% of them.
        """
        denom = self.tn + self.fp
        return self.fp / denom if denom else 0.0

    @property
    def yes_ratio(self) -> float:
        """Share of *parseable* answers that answered "yes".

        The denominator is the parseable count, not ``total``. Including
        unparseable answers made this a measure of parse failures rather than
        of yes-tilting, which in turn made the ``yes_ratio > 0.95`` degeneracy
        check fire on a model that had simply produced no valid verdicts.
        """
        denom = self.parseable
        return (self.tp + self.fp) / denom if denom else 0.0

    @property
    def no_ratio(self) -> float:
        denom = self.parseable
        return (self.tn + self.fn) / denom if denom else 0.0

    @property
    def hallucination_rate(self) -> float:
        """POPE-style hallucination rate: FP / (all "no" ground-truth items).

        This isolates the failure mode the benchmark targets: asserting the
        presence of an object that is absent. Unparseable answers are excluded
        from the numerator, so a parse failure cannot be reported as a
        hallucination.
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
            "parseable": self.parseable,
        }


def confusion_counts(
    predictions: Sequence[Optional[bool]],
    ground_truths: Sequence[bool],
) -> BinaryMetrics:
    """Accumulate a confusion matrix, tracking unparseable answers separately.

    An unparseable answer is counted in ``total`` (so it lowers accuracy, which
    is right) and reported via ``unparseable``, but it is deliberately *not*
    folded into ``fp``/``fn``. It contains no yes/no assertion, so it is not
    evidence that the model asserted a present or an absent object. Putting it
    in ``fp`` made a model that emitted pure garbage score a hallucination rate
    of 1.0 and a yes-ratio of 1.0 simultaneously — two contradictory
    diagnoses from the same output.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have equal length")
    m = BinaryMetrics()
    for pred, truth in zip(predictions, ground_truths):
        if pred is None:
            m.unparseable += 1
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

        accuracy      = correct questions / all questions
        accuracy_plus = images with BOTH questions correct / all images
        score         = 100 * (accuracy + accuracy_plus)     # [0, 200]

    The final MME total is the sum of per-subtask scores, so a subtask can
    contribute at most 200.

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
        # ``accuracy`` is over *questions*, as MME specifies. Averaging the
        # per-image accuracies instead gave every image equal weight regardless
        # of how many questions it contributed, so a truncated run -- one where
        # the final image has a single question -- reported a different number
        # from the official protocol while looking entirely normal.
        n_questions = sum(len(g.correct) for g in groups.values())
        accuracy = (
            sum(sum(1 for c in g.correct if c) for g in groups.values()) / n_questions
            if n_questions
            else 0.0
        )
        accuracy_plus = sum(g.accuracy_plus for g in groups.values()) / len(groups)
        out[f"{subtask}/accuracy"] = 100.0 * accuracy
        out[f"{subtask}/accuracy_plus"] = 100.0 * accuracy_plus
        # ``accuracy`` and ``accuracy_plus`` are fractions in [0, 1], so
        # ``100 * (accuracy + accuracy_plus)`` ranges over [0, 200] -- matching the
        # documented 200 maximum for a subtask. Dividing by 2 as well capped the
        # result at 100, which made every reported MME number exactly half the
        # official value and left the documented 200 cap unreachable.
        out[f"{subtask}/score"] = min(100.0 * (accuracy + accuracy_plus), 200.0)
        out[f"{subtask}/n_images"] = len(groups)
        out[f"{subtask}/n_questions"] = n_questions
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
        totals[f"{name}_subtasks_measured"] = sum(
            1 for subtask in members
            if subtask_scores.get(f"{subtask}/score") is not None
        )
    # Sum the named categories rather than filtering ``v > 0``. The filter made
    # a category that genuinely scored 0.0 indistinguishable from one that was
    # never measured at all, so a run with no subtasks and a run where the model
    # failed every question reported the same total. The per-category measured
    # counts above make the two cases separable.
    totals["total_score"] = sum(
        totals.get(f"{name}_score", 0.0) for name, _ in (("perception", 0), ("cognition", 0))
    )
    return totals
