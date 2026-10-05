#!/usr/bin/env python
"""
Build the ablation comparison table from result JSONs.

Reads every ``*.json`` under ``--results-dir`` that ``src/run.py`` produced and
emits a markdown table of the form::

    | Method | POPE Random F1 | POPE Popular F1 | POPE Adv F1 | Hallucination | MME |

Missing runs are rendered as ``not run`` — never as a zero, a dash that could be
mistaken for a measured value, or an estimate. If no results exist, the script
says so instead of printing an empty table.

Usage::

    python scripts/make_results_table.py --results-dir results
    python scripts/make_results_table.py --results-dir results --format csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NOT_RUN = "not run"
POPE_SETTINGS = ("random", "popular", "adversarial")

#: Order the ablation rows appear in the table.
METHOD_ORDER = [
    "baseline", "greedy", "sampling", "beam",
    "attention", "semantic", "region", "unidirectional", "visualguard",
]
METHOD_LABELS = {
    "baseline": "A. baseline (greedy)",
    "greedy": "greedy",
    "sampling": "sampling",
    "beam": "beam search",
    "attention": "B. attention only",
    "semantic": "C. semantic only",
    "region": "D. region only",
    "unidirectional": "E. attention + semantic",
    "visualguard": "G. full VisualGuard",
}

METRIC_KEYS = ["f1", "precision", "recall", "accuracy", "hallucination_rate"]


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load every result JSON in ``results_dir``, skipping malformed files."""
    payloads = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            print(f"warning: skipping {path.name}: not a JSON object", file=sys.stderr)
            continue
        data["_path"] = str(path)
        payloads.append(data)
    return payloads


def pope_cell(payload: Dict[str, Any], setting: str, key: str) -> str:
    """One POPE cell, or ``not run``."""
    pope = payload.get("pope")
    if not isinstance(pope, dict):
        return NOT_RUN
    block = pope.get(setting)
    if not isinstance(block, dict):
        return NOT_RUN
    metrics = block.get("metrics")
    if not isinstance(metrics, dict) or key not in metrics:
        return NOT_RUN
    value = metrics[key]
    if not isinstance(value, (int, float)):
        return NOT_RUN
    return f"{value:.4f}"


def mme_total(payload: Dict[str, Any]) -> str:
    mme = payload.get("mme")
    if not isinstance(mme, dict):
        return NOT_RUN
    totals = mme.get("category_totals")
    if not isinstance(totals, dict) or "total_score" not in totals:
        return NOT_RUN
    value = totals["total_score"]
    return f"{value:.2f}" if isinstance(value, (int, float)) else NOT_RUN


def hallucination_summary(payload: Dict[str, Any]) -> str:
    """Mean hallucination rate across whichever POPE settings were run."""
    pope = payload.get("pope")
    if not isinstance(pope, dict):
        return NOT_RUN
    values = []
    for setting in POPE_SETTINGS:
        block = pope.get(setting)
        if isinstance(block, dict):
            metrics = block.get("metrics") or {}
            value = metrics.get("hallucination_rate")
            if isinstance(value, (int, float)):
                values.append(value)
    if not values:
        return NOT_RUN
    return f"{sum(values) / len(values):.4f}"


def worst_pope_f1(payload: Dict[str, Any]) -> str:
    """Lowest F1 across the settings that ran (guards against cherry-picking)."""
    pope = payload.get("pope")
    if not isinstance(pope, dict):
        return NOT_RUN
    values = []
    for block in pope.values():
        if isinstance(block, dict):
            metrics = block.get("metrics") or {}
            if isinstance(metrics.get("f1"), (int, float)):
                values.append(metrics["f1"])
    return f"{min(values):.4f}" if values else NOT_RUN


def provenance_note(payloads: List[Dict[str, Any]]) -> List[str]:
    """Warnings about comparability that a reader must see."""
    notes: List[str] = []
    models = {p.get("provenance", {}).get("model") for p in payloads if p.get("provenance")}
    models.discard(None)
    if len(models) > 1:
        notes.append(
            "WARNING: rows use different models "
            f"({sorted(models)}); they are NOT comparable."
        )
    capped = [p for p in payloads if p.get("provenance", {}).get("max_samples")]
    if capped:
        notes.append(
            f"WARNING: {len(capped)} run(s) used --max-samples; those rows are "
            "not comparable to a full-dataset run."
        )
    seeds = {p.get("provenance", {}).get("seed") for p in payloads if p.get("provenance")}
    if len(seeds) > 1:
        notes.append(f"WARNING: rows use different seeds {sorted(seeds)}.")
    return notes


def build_rows(payloads: List[Dict[str, Any]]) -> List[Tuple[str, ...]]:
    by_method: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        method = payload.get("provenance", {}).get("method") or payload.get("method")
        if method is None:
            continue
        # Prefer a payload with actual benchmark data.
        if method not in by_method or (
            "pope" not in by_method[method] and "pope" in payload
        ):
            by_method[method] = payload

    ordered = [m for m in METHOD_ORDER if m in by_method]
    ordered += sorted(m for m in by_method if m not in METHOD_ORDER)

    rows: List[Tuple[str, ...]] = []
    for method in ordered:
        payload = by_method[method]
        rows.append(
            (
                METHOD_LABELS.get(method, method),
                pope_cell(payload, "random", "f1"),
                pope_cell(payload, "popular", "f1"),
                pope_cell(payload, "adversarial", "f1"),
                hallucination_summary(payload),
                worst_pope_f1(payload),
                mme_total(payload),
            )
        )
    return rows


HEADER = (
    "Method",
    "POPE Random F1",
    "POPE Popular F1",
    "POPE Adv F1",
    "Hallucination (mean, lower better)",
    "POPE worst-setting F1",
    "MME total",
)


def render_markdown(rows: List[Tuple[str, ...]]) -> str:
    """Render rows as a markdown table.

    Values never contain commas (they are fixed-precision floats or labels), so
    joining on ``|`` is safe and no CSV escaping is needed.
    """
    lines = ["| " + " | ".join(HEADER) + " |"]
    lines.append("| " + " | ".join(["---"] * len(HEADER)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--format", default="markdown", choices=["markdown", "csv"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    payloads = load_results(results_dir)
    if not payloads:
        print(
            f"No result JSON files in {results_dir}.\n"
            "No results are available yet. Run an experiment first, e.g.\n"
            "  python -m src.run --benchmark pope --method baseline \\\n"
            "      --data-root /path/to/POPE --image-root /path/to/coco/val2017",
            file=sys.stderr,
        )
        return 1

    rows = build_rows(payloads)
    if not rows:
        print("Result files contained no recognisable benchmark rows.", file=sys.stderr)
        return 1

    if args.format == "csv":
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
    else:
        print(render_markdown(rows))
        print()

    notes = provenance_note(payloads)
    if notes:
        print("Notes:")
        for note in notes:
            print(f"- {note}")
    print()
    print("Cells marked 'not run' were not measured. They are not zeros.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
