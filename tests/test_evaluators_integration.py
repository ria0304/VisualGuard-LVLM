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
    # 100 * (0.75 + 0.50) = 125.0
    assert result.metrics["existence/score"] == pytest.approx(125.0)
    assert result.metrics["existence/n_images"] == 2


def test_mme_perfect_subtask(tmp_path):
    _mme_fixture(tmp_path)
    gen = _ScriptedGenerator(["yes", "no", "yes", "no"])  # all four correct
    ev = MMEEvaluator(gen, method="stub")
    result = ev.evaluate(mme_root=tmp_path, progress_every=0)
    assert result.metrics["existence/accuracy"] == pytest.approx(100.0)
    assert result.metrics["existence/accuracy_plus"] == pytest.approx(100.0)
    # A flawless subtask reaches MME's documented 200 maximum.
    assert result.metrics["existence/score"] == pytest.approx(200.0)
    assert result.category_totals["perception_score"] == pytest.approx(200.0)
    assert result.category_totals["total_score"] == pytest.approx(200.0)


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


# ---------------------------------------------------------------------------
# the benchmark token budget must reach the decoding loop
# ---------------------------------------------------------------------------
#
# POPEEvaluator/MMEEvaluator stored ``max_new_tokens`` and never used it, so the
# decoder fell back to DecodingConfig's 64-token default. POPE answers are a
# single word, so every sample generated 4x the intended budget.


class _BudgetTokenizer:
    eos_token_id = 999
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return "yes"


class _BudgetResult:
    def __init__(self, n=1):
        self.text = "yes"
        self.num_generated_tokens = n
        self.interventions = 0
        self.interventions_detail = []


def test_pope_evaluator_passes_its_token_budget_to_the_decoder(tmp_path):
    """The evaluator's max_new_tokens must not be silently ignored."""
    seen = {}

    def generate_fn(image, question, max_new_tokens=None):
        seen["max_new_tokens"] = max_new_tokens
        return _BudgetResult()

    from src.evaluation.pope_eval import POPEEvaluator

    evaluator = POPEEvaluator(generate_fn=generate_fn, method="greedy",
                              max_new_tokens=16)
    assert evaluator.max_new_tokens == 16

    from src.data.pope import POEPSample

    sample = POEPSample(
        question_id="q1", setting="random", image_id="1",
        question="Is there a dog?", label="yes",
        image_path=tmp_path / "000000000001.jpg",
    )
    evaluator._predict(sample)
    assert seen["max_new_tokens"] == 16, (
        "POPE's 16-token budget never reached generate_fn; the decoder would "
        "have used its 64-token default"
    )


def test_mme_evaluator_passes_its_token_budget_to_the_decoder():
    seen = {}

    def generate_fn(image, question, max_new_tokens=None):
        seen["max_new_tokens"] = max_new_tokens
        return _BudgetResult()

    from src.data.mme import MMESample
    from src.evaluation.mme_eval import MMEEvaluator

    evaluator = MMEEvaluator(generate_fn=generate_fn, method="greedy",
                             max_new_tokens=32)
    sample = MMESample(
        question_id="q1", subtask="existence", image_path="a.jpg",
        question="Is there a dog?", answer="yes", category="perception",
    )
    evaluator._predict(sample)
    assert seen["max_new_tokens"] == 32


def test_decoder_generate_honours_per_call_token_override():
    import torch

    from src.model.visual_evidence import EvidenceConfig
    from src.model.visual_guard_decoder import DecodingConfig, VisualGuardDecoder

    class _StubStep:
        def __init__(self, logits):
            self.logits = logits
            self.attentions = None
            self.past_key_values = None
            self.image_token_span = None

    class _Loop:
        """Never emits EOS, so generation only stops at the token budget."""

        def __init__(self):
            self.device = torch.device("cpu")
            self.model = None
            self.calls = 0

        def prepare_inputs(self, question, image=None):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def initial_step(self, input_ids, pixel_values=None, output_attentions=False):
            return _StubStep(torch.tensor([[5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

        def next_step(self, input_ids, past_key_values=None, output_attentions=False):
            self.calls += 1
            return _StubStep(torch.tensor([[5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

    class _Tok(_BudgetTokenizer):
        eos_token_id = -1  # unreachable, so only max_new_tokens stops the loop

        def decode(self, ids, skip_special_tokens=True):
            return "yes"

    dec = VisualGuardDecoder(
        lvlm_config=None,
        decoding_config=DecodingConfig(max_new_tokens=64),
        evidence_config=EvidenceConfig(alpha=1.0, beta=0.0, gamma=0.0, lam=0.0,
                                       threshold=0.35,
                                       attention_candidate_mix=1.0),
        method="attention",
    )
    backend = _Loop()
    backend.tokenizer = _Tok()
    dec.backend = backend
    from src.model.visual_evidence import VisualEvidenceScorer

    dec.scorer = VisualEvidenceScorer(config=dec.evidence_config)

    out = dec.generate("img.jpg", "q", method="attention", max_new_tokens=3)
    assert out.num_generated_tokens == 3, out.num_generated_tokens

    # the override must not leak into later calls
    out2 = dec.generate("img.jpg", "q", method="attention")
    assert out2.num_generated_tokens == 64, out2.num_generated_tokens
    assert dec.decoding_config.max_new_tokens == 64


# ---------------------------------------------------------------------------
# regression: the token-budget probe must not swallow runtime errors
# ---------------------------------------------------------------------------


def test_budget_probe_does_not_swallow_runtime_type_errors():
    """A TypeError from inside the decoder must propagate, not be retried.

    The evaluator used to call ``generate_fn(..., max_new_tokens=...)`` inside a
    ``try/except TypeError`` to detect an unsupported signature. ``TypeError`` is
    also what the decoder raises for ordinary faults (a bad image type, a CLIP
    dtype mismatch), so those were silently swallowed and the sample was decoded a
    second time at the decoder's own default budget -- 64 tokens where POPE
    specifies 16 -- with the original error thrown away.
    """
    from src.data.pope import POEPSample

    attempts = []

    def gen(image, question, max_new_tokens=None):
        attempts.append(max_new_tokens)
        raise TypeError("Unsupported image input type: <class 'Tensor'>")

    ev = POPEEvaluator(generate_fn=gen, method="visualguard", max_new_tokens=16)
    sample = POEPSample(
        question_id="1", image_id="1", image_path=Path("x.jpg"),
        question="Is there a dog?", label="yes", setting="random",
    )
    with pytest.raises(TypeError, match="Unsupported image input"):
        ev._predict(sample)
    assert attempts == [16], f"the sample was decoded {len(attempts)} times"


def test_two_argument_generate_fn_still_works_and_is_reported():
    """A decoder that cannot take the override is used as-is, and says so."""
    from src.data.pope import POEPSample

    def gen(image, question):
        return GenerationResult(text="Yes", num_generated_tokens=4)

    ev = POPEEvaluator(generate_fn=gen, method="visualguard", max_new_tokens=16)
    sample = POEPSample(
        question_id="1", image_id="1", image_path=Path("x.jpg"),
        question="Is there a dog?", label="yes", setting="random",
    )
    record = ev._predict(sample)
    assert record.prediction is True
    notes = ev._diagnostics([record])
    assert any("max_new_tokens" in n for n in notes), (
        f"the degraded token budget was not reported: {notes}"
    )


def test_unparseable_only_run_is_flagged_not_silently_zeroed():
    from src.data.pope import POEPSample

    ev = POPEEvaluator(
        generate_fn=lambda i, q: GenerationResult(text="???"), method="visualguard"
    )
    samples = [
        POEPSample(
            question_id=str(n), image_id=str(n), image_path=Path("x.jpg"),
            question="Is there a dog?", label="no", setting="random",
        )
        for n in range(4)
    ]
    records = [ev._predict(s) for s in samples]
    notes = ev._diagnostics(records)
    assert any("could not be parsed" in n for n in notes)
    assert any("nothing was parseable" in n for n in notes), (
        f"an all-unparseable run was not called out: {notes}"
    )


def test_mme_images_are_grouped_by_path_not_basename():
    """Same-named images in different directories must not collapse."""
    from src.evaluation.mme_eval import _image_key

    a = _image_key("images/001.jpg", "existence", "q0")
    b = _image_key("other/001.jpg", "existence", "q1")
    assert a != b, "distinct images sharing a filename were grouped together"

    # The same file referenced two ways is still one group.
    assert _image_key("./images/001.jpg", "existence", "q0") == a
    assert _image_key("images/001.jpg", "existence", "q1") == a

    # Different subtasks never collide.
    assert _image_key("images/001.jpg", "count", "q0") != a

    # No usable path falls back to the question id rather than colliding.
    assert _image_key("", "existence", "q9") != _image_key("", "existence", "q8")
