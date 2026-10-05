# -*- coding: utf-8 -*-
"""
End-to-end evaluator tests.

These exercise the *real* loaders and the *real* metric computation over
fixture datasets written to ``tmp_path``. The only stub is the generator
function, which stands in for the LVLM so the tests stay offline and fast.

The images written here are 2x2 PNGs used purely as file fixtures so the
loaders' path resolution is genuinely tested; they are never used to produce a
benchmark claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.data.pope import POPEDataError, load_pope
from src.evaluation.mme_eval import MMEEvaluator, _image_key
from src.evaluation.pope_eval import POPEEvaluator
from src.model.visual_guard_decoder import GenerationResult


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _write_image(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), (90, 120, 150)).save(path)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _pope_fixture(root: Path, n_yes: int = 3, n_no: int = 3) -> None:
    rows = []
    for i in range(n_yes):
        rows.append({"question_id": f"y{i}", "image": f"img_y{i}.jpg",
                     "text": "Is there an object present?", "label": "yes"})
    for i in range(n_no):
        rows.append({"question_id": f"n{i}", "image": f"img_n{i}.jpg",
                     "text": "Is there an absent thing?", "label": "no"})
    _write_jsonl(root / "coco_pope_random.jsonl", rows)
    for i in range(n_yes):
        _write_image(root / "images" / f"img_y{i}.jpg")
    for i in range(n_no):
        _write_image(root / "images" / f"img_n{i}.jpg")


class _ScriptedGenerator:
    """Returns a fixed script of answers, one per call."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, image, question):
        self.calls.append((str(image), question))
        text = self.answers.pop(0) if self.answers else "yes"
        return GenerationResult(
            text=text,
            token_ids=[1],
            num_generated_tokens=1,
            method="stub",
            latency_s=0.001,
            interventions=0,
        )


# ---------------------------------------------------------------------------
# POPE end to end
# ---------------------------------------------------------------------------


def test_pope_perfect_predictions(tmp_path):
    _pope_fixture(tmp_path)
    gen = _ScriptedGenerator(["Yes"] * 3 + ["No"] * 3)
    ev = POPEEvaluator(gen, method="stub")
    result = ev.evaluate(pope_root=tmp_path, setting="random", progress_every=0)

    assert result.n_samples == 6
    assert result.metrics["accuracy"] == pytest.approx(1.0)
    assert result.metrics["f1"] == pytest.approx(1.0)
    assert result.metrics["hallucination_rate"] == pytest.approx(0.0)
    assert result.metrics["yes_ratio"] == pytest.approx(0.5)


def test_pope_all_yes_is_flagged_as_degenerate(tmp_path):
    """An always-yes model must be reported as such, not celebrated."""
    _pope_fixture(tmp_path)
    gen = _ScriptedGenerator(["yes"] * 6)
    ev = POPEEvaluator(gen, method="stub")
    result = ev.evaluate(pope_root=tmp_path, setting="random", progress_every=0)

    assert result.metrics["recall"] == pytest.approx(1.0)
    assert result.metrics["precision"] == pytest.approx(0.5)
    assert any("yes_ratio" in n for n in result.notes)


def test_pope_hallucination_rate_reflects_false_yes(tmp_path):
    _pope_fixture(tmp_path)
    # Every answer "yes": all 3 negatives become hallucinations.
    gen = _ScriptedGenerator(["yes"] * 6)
    ev = POPEEvaluator(gen, method="stub")
    result = ev.evaluate(pope_root=tmp_path, setting="random", progress_every=0)
    assert result.metrics["hallucination_rate"] == pytest.approx(1.0)


def test_pope_unparseable_answers_counted_as_errors(tmp_path):
    _pope_fixture(tmp_path, n_yes=1, n_no=1)
    gen = _ScriptedGenerator(["maybe", "no"])
    ev = POPEEvaluator(gen, method="stub")
    result = ev.evaluate(pope_root=tmp_path, setting="random", progress_every=0)
    assert result.metrics["unparseable"] == 1
    assert result.metrics["accuracy"] == pytest.approx(0.5)
    assert any("could not be parsed" in n for n in result.notes)


def test_pope_prompt_template_is_applied_to_every_question(tmp_path):
    _pope_fixture(tmp_path, n_yes=1, n_no=1)
    gen = _ScriptedGenerator(["yes", "no"])
    ev = POPEEvaluator(gen, method="stub")
    ev.evaluate(pope_root=tmp_path, setting="random", progress_every=0)
    for _, question in gen.calls:
        assert "one word" in question


def test_pope_writes_predictions_for_audit(tmp_path):
    _pope_fixture(tmp_path, n_yes=1, n_no=1)
    out_dir = tmp_path / "out"
    gen = _ScriptedGenerator(["yes", "no"])
    ev = POPEEvaluator(gen, method="stub")
    ev.evaluate(pope_root=tmp_path, setting="random", output_dir=out_dir,
                progress_every=0)
    pred = out_dir / "pope_random_stub_predictions.jsonl"
    assert pred.is_file()
    lines = [json.loads(x) for x in pred.read_text().splitlines()]
    assert len(lines) == 2
    assert {"question_id", "ground_truth", "prediction", "correct"} <= set(lines[0])


def test_pope_evaluator_uses_the_real_image_paths(tmp_path):
    """The generator must receive the resolved on-disk image path."""
    _pope_fixture(tmp_path, n_yes=1, n_no=1)
    gen = _ScriptedGenerator(["yes", "no"])
    ev = POPEEvaluator(gen, method="stub")
    ev.evaluate(pope_root=tmp_path, setting="random", progress_every=0)
    for image, _ in gen.calls:
        assert Path(image).is_file(), f"evaluator passed a non-existent image: {image}"


def test_pope_evaluation_aborts_if_an_image_is_missing(tmp_path):
    _pope_fixture(tmp_path, n_yes=1, n_no=1)
    (tmp_path / "images" / "img_y0.jpg").unlink()
    ev = POPEEvaluator(_ScriptedGenerator(["yes"] * 2), method="stub")
    with pytest.raises(POPEDataError, match="could not be resolved"):
        ev.evaluate(pope_root=tmp_path, setting="random", progress_every=0)


def test_pope_max_samples_limits_evaluation(tmp_path):
    _pope_fixture(tmp_path)
    gen = _ScriptedGenerator(["yes"] * 10)
    ev = POPEEvaluator(gen, method="stub")
    result = ev.evaluate(pope_root=tmp_path, setting="random", max_samples=2,
                         progress_every=0)
    assert result.n_samples == 2
    assert len(gen.calls) == 2


# ---------------------------------------------------------------------------
# MME end to end
# ---------------------------------------------------------------------------


def _mme_fixture(root: Path) -> None:
    rows = [
        # image A: both correct
        {"question_id": 0, "image": "images/a.jpg", "text": "Is there a car?", "answer": "Yes"},
        {"question_id": 1, "image": "images/a.jpg", "text": "Is it moving?", "answer": "No"},
        # image B: one wrong -> no accuracy_plus
        {"question_id": 2, "image": "images/b.jpg", "text": "Is there a dog?", "answer": "Yes"},
        {"question_id": 3, "image": "images/b.jpg", "text": "Is it a cat?", "answer": "No"},
    ]
    _write_jsonl(root / "existence" / "existence.jsonl", rows)
    _write_image(root / "existence" / "images" / "a.jpg")
    _write_image(root / "existence" / "images" / "b.jpg")


def test_mme_accuracy_plus_requires_both_questions(tmp_path):
    _mme_fixture(tmp_path)
    # Ground truth is [Yes, No, Yes, No]. Image A (q0,q1) both correct;
    # image B (q2,q3) has q3 wrong, so image B earns no accuracy_plus.
    gen = _ScriptedGenerator(["yes", "no", "yes", "yes"])
    ev = MMEEvaluator(gen, method="stub")
    result = ev.evaluate(mme_root=tmp_path, progress_every=0)

    assert result.metrics["existence/accuracy"] == pytest.approx(75.0)      # 3 of 4
    assert result.metrics["existence/accuracy_plus"] == pytest.approx(50.0)  # 1 of 2 images
    assert result.metrics["existence/score"] == pytest.approx(62.5)
    assert result.metrics["existence/n_images"] == 2


def test_mme_perfect_subtask(tmp_path):
    _mme_fixture(tmp_path)
    gen = _ScriptedGenerator(["yes", "no", "yes", "no"])  # all four correct
    ev = MMEEvaluator(gen, method="stub")
    result = ev.evaluate(mme_root=tmp_path, progress_every=0)
    assert result.metrics["existence/accuracy"] == pytest.approx(100.0)
    assert result.metrics["existence/accuracy_plus"] == pytest.approx(100.0)
    assert result.metrics["existence/score"] == pytest.approx(100.0)
    assert result.category_totals["perception_score"] == pytest.approx(100.0)
    assert result.category_totals["total_score"] == pytest.approx(100.0)


def test_mme_groups_by_image_not_by_question(tmp_path):
    """If questions were grouped by question_id, accuracy_plus would be 100%."""
    _mme_fixture(tmp_path)
    gen = _ScriptedGenerator(["yes", "no", "yes", "yes"])  # q3 wrong
    ev = MMEEvaluator(gen, method="stub")
    result = ev.evaluate(mme_root=tmp_path, progress_every=0)
    assert result.metrics["existence/accuracy_plus"] < 100.0


def test_mme_image_key_includes_subtask(tmp_path):
    """Identical filenames in different subtasks must not be merged."""
    assert _image_key("images/a.jpg", "existence", "0") != _image_key("images/a.jpg", "count", "0")


def test_mme_uses_the_real_image_paths(tmp_path):
    _mme_fixture(tmp_path)
    gen = _ScriptedGenerator(["yes", "no", "yes", "yes"])
    ev = MMEEvaluator(gen, method="stub")
    ev.evaluate(mme_root=tmp_path, progress_every=0)
    for image, _ in gen.calls:
        assert Path(image).is_file()


def test_mme_result_serialises(tmp_path):
    _mme_fixture(tmp_path)
    gen = _ScriptedGenerator(["yes", "no", "yes", "yes"])
    ev = MMEEvaluator(gen, method="stub")
    payload = ev.evaluate(mme_root=tmp_path, progress_every=0).to_dict()
    assert payload["benchmark"] == "mme"
    assert "category_totals" in payload
    assert payload["n_samples"] == 4


def test_mme_writes_predictions(tmp_path):
    _mme_fixture(tmp_path)
    out_dir = tmp_path / "out"
    gen = _ScriptedGenerator(["yes", "no", "yes", "yes"])
    ev = MMEEvaluator(gen, method="stub")
    ev.evaluate(mme_root=tmp_path, output_dir=out_dir, progress_every=0)
    assert (out_dir / "mme_stub_predictions.jsonl").is_file()
