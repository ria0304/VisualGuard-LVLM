# -*- coding: utf-8 -*-
"""
HallusionBench evaluation.

HallusionBench is a hallucination benchmark for LVLMs. This module provides
a evaluator compatible with the VisualGuard framework, following the same
pattern as :mod:`src.evaluation.pope_eval`.

The HallusionBench dataset consists of images with questions and answers
that test different types of hallucination. See
<https://github.com/MMMU-Bench/HallusionBench> for the original benchmark.

Example
-------
>>> evaluator = HallusionBenchEvaluator(decoder.generate, method="visualguard")
>>> result = evaluator.evaluate(
...     hallusionben_root=Path("data/hallusionbench"),
...     max_samples=100,
... )
>>> result.metrics["f1"] >= 0
True
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..data.hallusionbench import HallusionBenchSample, load_hallusionbench
from .metrics import (
    BinaryMetrics,
    extract_verdict,
    hallusionbench_metrics,
)
from .pope_eval import POPE_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

#: Prompt used for every HallusionBench question.
HB_PROMPT_TEMPLATE = "{question} ASSISTANT:"


@dataclass
class HallusionBenchRecord:
    """One evaluated HallusionBench sample."""

    question_id: str
    image: str
    question: str
    ground_truth: bool
    prediction: Optional[bool]
    raw_output: str
    correct: bool
    parseable: bool
    generated_tokens: int = 0
    latency_s: float = 0.0
    interventions: int = 0
    ves: Optional[float] = None
    hallucination_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
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
            "generated_tokens": self.generated_tokens,
            "latency_s": self.latency_s,
            "interventions": self.interventions,
            "ves": self.ves,
            "hallucination_type": self.hallucination_type,
        }


@dataclass
class HallusionBenchResult:
    """Aggregated HallusionBench metrics plus run metadata."""

    method: str
    metrics: Dict[str, float]
    n_samples: int
    latency_s: float
    per_token_latency_s: float
    peak_memory_mb: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": "hallusionbench",
            "method": self.method,
            "metrics": self.metrics,
            "n_samples": self.n_samples,
            "latency_s": self.latency_s,
            "per_token_latency_s": self.per_token_latency_s,
            "peak_memory_mb": self.peak_memory_mb,
            "notes": self.notes,
        }


class HallusionBenchEvaluator:
    """Runs a decoder over a HallusionBench split and scores it.

    Example
    -------
    >>> evaluator = HallusionBenchEvaluator(decoder.generate, method="visualguard")
    >>> result = evaluator.evaluate(
    ...     hallusionben_root=Path("data/hallusionbench"), max_samples=100,
    ... )
    >>> result.metrics["f1"] >= 0
    True
    """

    def __init__(
        self,
        generate_fn: Callable[[Any, str], Any],
        method: str = "greedy",
        prompt_template: str = HB_PROMPT_TEMPLATE,
        max_new_tokens: int = 32,
    ) -> None:
        """
        Args:
            generate_fn: ``(image, question) -> GenerationResult``. Injected so
                the evaluator is decoupled from the decoder implementation and
                is trivially testable with a stub.
            method: Label recorded in results; must match what the decoder does.
            prompt_template: Applied to every question, identically for all
                methods, so comparisons are not confounded by prompt changes.
            max_new_tokens: HallusionBench generation budget.
        """
        self.generate_fn = generate_fn
        self.method = method
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        # Decided once, from the signature. Probing this by calling the
        # function and catching TypeError also catches genuine runtime errors
        # raised inside the decoder, and silently re-runs the sample at the
        # wrong token budget.
        self._generate_takes_budget = accepts_keyword(generate_fn, "max_new_tokens")

    # -- single sample -------------------------------------------------

    def _predict(self, sample: HallusionBenchSample) -> HallusionBenchRecord:
        question = self.prompt_template.format(question=sample.question)
        started = time.perf_counter()
        # The benchmark's token budget is passed through to the decoder; without this
        # the decoder falls back to its own (much larger) default. Whether the
        # override is supported is known from the signature, not from catching
        # TypeError around the call.
        if self._generate_takes_budget:
            out = self.generate_fn(
                sample.image_path, question, max_new_tokens=self.max_new_tokens
            )
        else:
            out = self.generate_fn(sample.image_path, question)
        latency = time.perf_counter() - started

        raw = getattr(out, "text", str(out))
        verdict = extract_verdict(raw)
        details = getattr(out, "interventions_detail", None) or []
        ves = details[0].get("chosen_ves") if details else None

        return HallusionBenchRecord(
            question_id=sample.question_id,
            image=str(sample.image_path),
            question=sample.question,
            ground_truth=sample.ground_truth,
            prediction=verdict,
            raw_output=raw,
            correct=(verdict is not None and verdict == sample.ground_truth),
            parseable=verdict is not None,
            generated_tokens=int(getattr(out, "num_generated_tokens", 0)),
            latency_s=latency,
            interventions=int(getattr(out, "interventions", 0)),
            ves=ves,
            hallucination_type=getattr(sample, "hallucination_type", ""),
        )

    # -- full split ----------------------------------------------------

    def evaluate(
        self,
        hallusionben_root: Path,
        max_samples: Optional[int] = None,
        output_dir: Optional[Path] = None,
        progress_every: int = 50,
    ) -> HallusionBenchResult:
        """Evaluate one HallusionBench split end to end.

        Raises:
            HallusionBenchDataError: if the data or images are missing.
            ValueError: if zero samples were evaluated.
        """
        samples = load_hallusionbench(
            hallusionben_root=Path(hallusionben_root),
            max_samples=max_samples,
            require_images=True,
        )
        logger.info(
            "HallusionBench: %d samples",
            len(samples),
        )

        records: List[HallusionBenchRecord] = []
        run_start = time.perf_counter()
        total_tokens = 0
        for idx, sample in enumerate(samples, start=1):
            record = self._predict(sample)
            records.append(record)
            total_tokens += record.generated_tokens
            if progress_every and idx % progress_every == 0:
                running = confusion_counts(
                    [r.prediction for r in records],
                    [r.ground_truth for r in records],
                )
                logger.info(
                    "[%d/%d acc=%.3f" % (idx, len(samples), running.accuracy),
                )
        total_latency = time.perf_counter() - run_start

        metrics = hallusionbench_metrics(
            [r.prediction for r in records],
            [r.ground_truth for r in records],
        )

        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / f"hallusionbench_{self.method}_predictions.jsonl"
            self.save_predictions(records, pred_path)

        return HallusionBenchResult(
            method=self.method,
            metrics=metrics,
            n_samples=len(records),
            latency_s=total_latency,
            per_token_latency_s=total_latency / max(total_tokens, 1),
            notes=self._diagnostics(records),
        )

    def _diagnostics(self, records: Sequence[HallusionBenchRecord]) -> List[str]:
        """Surface signs of degenerate behaviour."""
        notes: List[str] = []
        m = confusion_counts([r.prediction for r in records], [r.ground_truth for r in records])
        if m.unparseable:
            notes.append(
                f"{m.unparseable} answers could not be parsed as yes/no "
                "and were scored as errors"
            )
        if m.yes_ratio > 0.95:
            notes.append(
                f"yes_ratio={m.yes_ratio:.3f}: the model answers 'yes' almost "
                "always, which inflates recall and deflates precision"
            )
        elif m.yes_ratio < 0.05:
            notes.append(
                f"yes_ratio={m.yes_ratio:.3f}: the model answers 'no' almost "
                "always; a strongly conservative decoding rule can cause this"
            )
        return notes

    @staticmethod
    def save_predictions(records: Sequence[HallusionBenchRecord], path: Path) -> None:
        """Write per-sample predictions for auditing / re-scoring."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Wrote %d predictions to %s", len(records), path)


def evaluate_hallusionbench(
    evaluator: HallusionBenchEvaluator,
    hallusionben_root: Path,
    max_samples: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> HallusionBenchResult:
    """Run HallusionBench evaluation, skipping (and reporting) failures."""
    try:
        return evaluator.evaluate(
            hallusionben_root=hallusionben_root,
            max_samples=max_samples,
            output_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001 - broad, data-availability error
        logger.error("HallusionBench evaluation failed: %s", exc)
        raise