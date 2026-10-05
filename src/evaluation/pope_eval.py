# -*- coding: utf-8 -*-
"""
Real POPE evaluation.

For every POPE question this module:

1. loads the *real* referenced image from disk,
2. builds the prompt with the same template used by every other method,
3. generates an answer with the configured decoder (baseline or VisualGuard),
4. normalises the answer to a yes/no verdict,
5. compares it against the ground-truth label,
6. aggregates Accuracy / Precision / Recall / F1 / yes-ratio / no-ratio /
   hallucination rate.

Every sample is written to a JSONL prediction file so results can be audited
and re-scored without re-running the model. Nothing here fabricates a score:
if no samples were evaluated the evaluator raises.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..data.pope import POPEDataError, POEPSample, load_pope
from .metrics import BinaryMetrics, confusion_counts, extract_verdict, pope_metrics

logger = logging.getLogger(__name__)

#: Prompt used for every POPE question, identical across all methods.
POPE_PROMPT_TEMPLATE = '{question}\nPlease answer this question with one word.'


@dataclass
class POPERecord:
    """One evaluated POPE sample."""

    question_id: str
    setting: str
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "setting": self.setting,
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
        }


@dataclass
class POPEResult:
    """Aggregated POPE metrics plus run metadata."""

    setting: str
    method: str
    metrics: Dict[str, float]
    n_samples: int
    latency_s: float
    per_token_latency_s: float
    peak_memory_mb: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": "pope",
            "setting": self.setting,
            "method": self.method,
            "metrics": self.metrics,
            "n_samples": self.n_samples,
            "latency_s": self.latency_s,
            "per_token_latency_s": self.per_token_latency_s,
            "peak_memory_mb": self.peak_memory_mb,
            "notes": self.notes,
        }


class POPEEvaluator:
    """Runs a decoder over a POPE split and scores it.

    Example
    -------
    >>> evaluator = POPEEvaluator(decoder.generate, method="visualguard")
    >>> result = evaluator.evaluate(
    ...     pope_root=Path("data/POPE"), setting="random",
    ...     image_root=Path("data/coco/val2017"), max_samples=100,
    ... )
    >>> result.metrics["f1"] >= 0
    True
    """

    def __init__(
        self,
        generate_fn: Callable[[Any, str], Any],
        method: str = "greedy",
        prompt_template: str = POPE_PROMPT_TEMPLATE,
        max_new_tokens: int = 16,
    ) -> None:
        """
        Args:
            generate_fn: ``(image, question) -> GenerationResult``. Injected so
                the evaluator is decoupled from the decoder implementation and
                is trivially testable with a stub.
            method: Label recorded in results; must match what the decoder does.
            prompt_template: Applied to every question, identically for all
                methods, so comparisons are not confounded by prompt changes.
            max_new_tokens: POPE answers are single words.
        """
        self.generate_fn = generate_fn
        self.method = method
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens

    # -- single sample -------------------------------------------------

    def _predict(self, sample: POEPSample) -> POPERecord:
        question = self.prompt_template.format(question=sample.question)
        started = time.perf_counter()
        out = self.generate_fn(sample.image_path, question)
        latency = time.perf_counter() - started

        raw = getattr(out, "text", str(out))
        verdict = extract_verdict(raw)
        details = getattr(out, "interventions_detail", None) or []
        ves = details[0].get("chosen_ves") if details else None

        return POPERecord(
            question_id=sample.question_id,
            setting=sample.setting,
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
        )

    # -- full split ----------------------------------------------------

    def evaluate(
        self,
        pope_root: Path,
        setting: str,
        image_root: Optional[Path] = None,
        max_samples: Optional[int] = None,
        output_dir: Optional[Path] = None,
        progress_every: int = 50,
    ) -> POPEResult:
        """Evaluate one POPE setting end to end.

        Raises:
            POPEDataError: if the data or images are missing (never substituted).
            ValueError: if zero samples were evaluated.
        """
        samples = load_pope(
            pope_root=Path(pope_root),
            setting=setting,
            image_root=Path(image_root) if image_root else None,
            max_samples=max_samples,
            require_images=True,
        )
        logger.info(
            "POPE %s/%s: %d samples (images verified present)",
            self.method, setting, len(samples),
        )

        records: List[POPERecord] = []
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
                    "[%s/%s] %d/%d acc=%.3f yes_ratio=%.3f",
                    self.method, setting, idx, len(samples),
                    running.accuracy, running.yes_ratio,
                )
        total_latency = time.perf_counter() - run_start

        metrics = pope_metrics(
            [r.prediction for r in records],
            [r.ground_truth for r in records],
        )

        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / f"pope_{setting}_{self.method}_predictions.jsonl"
            self.save_predictions(records, pred_path)

        return POPEResult(
            setting=setting,
            method=self.method,
            metrics=metrics,
            n_samples=len(records),
            latency_s=total_latency,
            per_token_latency_s=total_latency / max(total_tokens, 1),
            notes=self._diagnostics(records),
        )

    def _diagnostics(self, records: Sequence[POPERecord]) -> List[str]:
        """Surface signs of degenerate behaviour (e.g. all-yes answering)."""
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
        if m.yes_ratio < 0.05:
            notes.append(
                f"yes_ratio={m.yes_ratio:.3f}: the model answers 'no' almost "
                "always; a strongly conservative decoding rule can cause this"
            )
        return notes

    @staticmethod
    def save_predictions(records: Sequence[POPERecord], path: Path) -> None:
        """Write per-sample predictions for auditing / re-scoring."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Wrote %d predictions to %s", len(records), path)


def evaluate_pope_settings(
    evaluator: POPEEvaluator,
    pope_root: Path,
    settings: Sequence[str],
    image_root: Optional[Path] = None,
    max_samples: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, POPEResult]:
    """Run several POPE settings, skipping (and reporting) failures."""
    results: Dict[str, POPEResult] = {}
    for setting in settings:
        try:
            results[setting] = evaluator.evaluate(
                pope_root=pope_root,
                setting=setting,
                image_root=image_root,
                max_samples=max_samples,
                output_dir=output_dir,
            )
        except POPEDataError as exc:
            logger.error("POPE setting %s unavailable: %s", setting, exc)
            raise
    return results
