# -*- coding: utf-8 -*-
"""
Unit tests for POPE / MME metric computation.

The functions under test are pure, so these tests need neither a model nor a
dataset. They double as executable documentation of the metric definitions.
"""

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    BinaryMetrics,
    confusion_counts,
    extract_verdict,
    mme_category_totals,
    mme_subtask_scores,
    normalise_answer,
    pope_metrics,
)


# ---------------------------------------------------------------------------
# answer normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Yes", "yes"),
        ("  YES  ", "yes"),
        ("Answer: no", "no"),
        ("The answer is yes.", "yes"),
        ("no.", "no"),
        ("yes, definitely", "yes definitely"),
        ("", ""),
    ],
)
def test_normalise_answer(raw, expected):
    assert normalise_answer(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Yes", True),
        ("yes", True),
        ("Yes, there is a dog.", True),
        ("No", False),
        ("no, there is no dog", False),
        ("Answer: no", False),
        ("The answer is yes", True),
    ],
)
def test_extract_verdict_decisive(raw, expected):
    assert extract_verdict(raw) is expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "maybe",
        "the image is unclear",
        "42",
    ],
)
def test_extract_verdict_returns_none_for_unparseable(raw):
    """Unparseable answers must be surfaced, not guessed."""
    assert extract_verdict(raw) is None


def test_first_decisive_token_wins():
    # "No, there is no dog" contains two decisive tokens, both "no", so the
    # repeated negation must not flip the answer.
    assert extract_verdict("No, there is no dog") is False


def test_repeated_affirmation_is_still_yes():
    assert extract_verdict("Yes, yes there is a dog") is True


def test_conflicting_verdicts_are_unparseable():
    assert extract_verdict("yes and no") is None


# ---------------------------------------------------------------------------
# confusion matrix
# ---------------------------------------------------------------------------


def test_perfect_predictions():
    m = confusion_counts([True, False], [True, False])
    assert (m.tp, m.fp, m.tn, m.fn) == (1, 0, 1, 0)
    assert m.accuracy == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)
    assert m.hallucination_rate == pytest.approx(0.0)


def test_counts_are_correct():
    # truth:  T T F F ; pred:  T F T F
    m = confusion_counts([True, False, True, False], [True, True, False, False])
    assert (m.tp, m.fn, m.fp, m.tn) == (1, 1, 1, 1)


def test_metric_values():
    m = BinaryMetrics(tp=2, fp=1, tn=3, fn=2)  # n = 8
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))
    assert m.specificity == pytest.approx(0.75)
    assert m.fpr == pytest.approx(0.25)
    assert m.accuracy == pytest.approx(5 / 8)
    assert m.yes_ratio == pytest.approx(3 / 8)
    assert m.no_ratio == pytest.approx(5 / 8)


def test_hallucination_rate_is_fp_over_negatives():
    """POPE targets false "yes" answers, so the denominator is negatives."""
    m = BinaryMetrics(tp=5, fp=2, tn=6, fn=5)
    assert m.hallucination_rate == pytest.approx(2 / 8)


def test_all_yes_is_detectable_via_yes_ratio():
    m = confusion_counts([True] * 10, [True] * 5 + [False] * 5)
    assert m.yes_ratio == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.precision == pytest.approx(0.5)


def test_unparseable_counted_as_error_and_surfaced():
    m = confusion_counts([None, False], [True, False])
    assert m.unparseable == 1
    assert m.fn == 1  # missed a "yes"
    assert m.tn == 1
    assert m.total == 2


def test_unparseable_negative_counted_as_false_positive():
    m = confusion_counts([None], [False])
    assert m.fp == 1
    assert m.unparseable == 1


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        confusion_counts([True], [True, False])


def test_pope_metrics_returns_expected_keys():
    out = pope_metrics([True, False], [True, False])
    for key in (
        "accuracy", "precision", "recall", "f1", "yes_ratio",
        "no_ratio", "hallucination_rate", "n",
    ):
        assert key in out


def test_pope_metrics_refuses_empty_input():
    """Reporting 0.0 for zero samples would be indistinguishable from a real score."""
    with pytest.raises(ValueError, match="refusing to report fabricated scores"):
        pope_metrics([], [])


def test_all_negative_truth_has_zero_precision_without_dividing_by_zero():
    m = confusion_counts([False, False, False], [False, False, False])
    assert m.precision == 0.0
    assert m.f1 == 0.0
    assert m.hallucination_rate == pytest.approx(0.0)


def test_all_positive_truth_has_zero_recall_without_dividing_by_zero():
    m = confusion_counts([False, False], [True, True])
    assert m.recall == 0.0
    assert m.specificity == 0.0


def test_empty_metrics_are_zero_not_nan():
    m = BinaryMetrics()
    for value in (m.accuracy, m.precision, m.recall, m.f1, m.yes_ratio):
        assert value == 0.0


# ---------------------------------------------------------------------------
# MME
# ---------------------------------------------------------------------------


def _records():
    # Two images, two questions each: image A fully correct, image B fully
    # wrong. accuracy = 2/4 = 0.5; accuracy_plus = 1 of 2 images = 0.5;
    # score = 100 * (0.5 + 0.5) / 2 = 50.
    return [
        {"image_id": "sub::a.jpg", "correct": True, "subtask": "existence"},
        {"image_id": "sub::a.jpg", "correct": True, "subtask": "existence"},
        {"image_id": "sub::b.jpg", "correct": False, "subtask": "existence"},
        {"image_id": "sub::b.jpg", "correct": False, "subtask": "existence"},
    ]


def test_mme_subtask_score_formula():
    out = mme_subtask_scores(_records())
    assert out["existence/accuracy"] == pytest.approx(50.0)
    assert out["existence/accuracy_plus"] == pytest.approx(50.0)
    assert out["existence/score"] == pytest.approx(50.0)
    assert out["existence/n_images"] == 2


def test_mme_all_images_wrong_scores_zero():
    records = [r for r in _records() if not r["correct"]]
    out = mme_subtask_scores(records)
    assert out["existence/score"] == pytest.approx(0.0)


def test_accuracy_plus_requires_both_questions_correct():
    # One image with one wrong answer must not earn accuracy_plus.
    records = [
        {"image_id": "sub::a.jpg", "correct": True, "subtask": "existence"},
        {"image_id": "sub::a.jpg", "correct": False, "subtask": "existence"},
    ]
    out = mme_subtask_scores(records)
    assert out["existence/accuracy"] == pytest.approx(50.0)
    assert out["existence/accuracy_plus"] == pytest.approx(0.0)


def test_accuracy_plus_all_correct():
    records = [
        {"image_id": "sub::a.jpg", "correct": True, "subtask": "existence"},
        {"image_id": "sub::a.jpg", "correct": True, "subtask": "existence"},
    ]
    out = mme_subtask_scores(records)
    assert out["existence/accuracy"] == pytest.approx(100.0)
    assert out["existence/accuracy_plus"] == pytest.approx(100.0)
    assert out["existence/score"] == pytest.approx(100.0)


def test_mme_perfect_subtask_caps_at_200():
    records = [
        {"image_id": f"sub::{i}.jpg", "correct": True, "subtask": "count"}
        for i in range(3)
        for _ in range(2)
    ]
    out = mme_subtask_scores(records)
    assert out["count/score"] == pytest.approx(100.0)
    totals = mme_category_totals(out)
    assert totals["perception_score"] == pytest.approx(100.0)


def test_mme_category_totals_sum_present_subtasks():
    scores = {
        "existence/score": 100.0,
        "count/score": 90.0,
        "commonsense_reasoning/score": 80.0,
    }
    totals = mme_category_totals(scores)
    assert totals["perception_score"] == pytest.approx(190.0)
    assert totals["cognition_score"] == pytest.approx(80.0)
    assert totals["total_score"] == pytest.approx(270.0)


def test_mme_category_totals_cap_each_subtask_at_200():
    scores = {"existence/score": 250.0}
    assert mme_category_totals(scores)["perception_score"] == pytest.approx(200.0)


def test_mme_refuses_empty_records():
    with pytest.raises(ValueError, match="refusing to fabricate scores"):
        mme_subtask_scores([])


def test_mme_rejects_incomplete_records():
    with pytest.raises(ValueError, match="missing required key"):
        mme_subtask_scores([{"image_id": "a", "correct": True}])
