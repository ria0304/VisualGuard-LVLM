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

#: Order the standard ablation rows appear in the table.
METHOD_ORDER = [
    "baseline", "greedy", "sampling", "beam",
    "attention", "semantic", "region", "unidirectional", "visualguard",
]
#: The shipped full-method weights (``configs/visualguard.yaml``).
FULL_WEIGHTS = (1.0, 1.0, 0.0)

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

#: Canonical evidence weights per single-channel method, used only to decide
#: whether a row needs a descriptive suffix. See :func:`row_label`.
CANONICAL_WEIGHTS = {
    "baseline": (0.0, 0.0, 0.0),
    "greedy": (0.0, 0.0, 0.0),
    "sampling": (0.0, 0.0, 0.0),
    "beam": (0.0, 0.0, 0.0),
    "attention": (1.0, 0.0, 0.0),
    "semantic": (0.0, 1.0, 0.0),
    "region": (0.0, 0.0, 1.0),
    "unidirectional": (1.0, 1.0, 0.0),
}

METRIC_KEYS = ["f1", "precision", "recall", "accuracy", "hallucination_rate"]


def provenance_of(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The payload's provenance block, or an empty dict.

    ``payload.get("provenance", {})`` only defaults when the key is *absent*; a
    JSON ``"provenance": null`` returns ``None`` and every ``.get(...)`` on it
    then raises. Normalising here keeps the comparability checks total.
    """
    block = payload.get("provenance")
    return block if isinstance(block, dict) else {}


def _num(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def describe_evidence(payload: Dict[str, Any]) -> Optional[str]:
    """Compact, comparable description of a run's evidence configuration.

    Two runs can share a ``method`` and still differ: ablation row F is
    ``visualguard`` with ``gamma=1``, row G is ``visualguard`` with the config
    defaults, and every lambda-sweep row is ``visualguard`` too. Keying the table
    on ``method`` alone therefore collapsed all of them into one row. Including
    the weights in the row label makes every run distinguishable.
    """
    evidence = _evidence_block(payload)
    alpha, beta, gamma = (_num(evidence.get(k)) for k in ("alpha", "beta", "gamma"))
    if alpha is None and beta is None and gamma is None:
        return None
    lam = _num(evidence.get("lam"))
    text = "a{:.3g} b{:.3g} g{:.3g}".format(alpha or 0.0, beta or 0.0, gamma or 0.0)
    if lam is not None:
        text += f" lam{lam:.3g}"
    return text


def row_label(payload: Dict[str, Any]) -> str:
    """Human label for one run, disambiguated by its evidence weights.

    Rows whose weights are the canonical ones for their method keep the plain
    "B. attention only" label. Anything else gets its weights appended, so a
    ``visualguard`` row configured as attention+region reads as distinct from the
    full method rather than silently replacing it.
    """
    prov = provenance_of(payload)
    method = prov.get("method") or payload.get("method")
    base = METHOD_LABELS.get(str(method), str(method) if method else "unknown run")
    description = describe_evidence(payload)
    if description is None:
        return base
    evidence = _evidence_block(payload)
    weights = (_num(evidence.get("alpha")), _num(evidence.get("beta")),
               _num(evidence.get("gamma")))
    is_full_method = str(method) == "visualguard" and weights == FULL_WEIGHTS
    if CANONICAL_WEIGHTS.get(str(method)) == weights or is_full_method:
        return base
    if str(method) == "visualguard":
        # Labeling a non-full visualguard configuration "G. full VisualGuard"
        # would be self-contradictory, and rows F and the lambda sweep both land
        # here. They are variants, so say that.
        return f"visualguard variant [{description}]"
    return f"{base} [{description}]"


def _evidence_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    resolved = payload.get("resolved_config")
    evidence = resolved.get("evidence") if isinstance(resolved, dict) else None
    return evidence if isinstance(evidence, dict) else {}


def run_identity(payload: Dict[str, Any]) -> str:
    """A key that is unique per *run*, not per method.

    ``method`` is not unique: rows F, G and the whole lambda sweep all record
    ``"visualguard"``. Keying on the run name keeps them as separate rows.
    """
    prov = provenance_of(payload)
    for candidate in (prov.get("run_name"), payload.get("_path")):
        if candidate:
            return str(Path(str(candidate)).stem)
    return str(prov.get("method") or payload.get("method") or id(payload))


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
    provs = [provenance_of(p) for p in payloads]
    present = [p for p in provs if p]

    # Every set below discards ``None`` before sorting. Sorting a mixed set of
    # int and None raises TypeError, which is how a payload missing its seed
    # broke the very check meant to flag incomparable rows.
    models = {p.get("model") for p in present}
    missing_models = None in models
    models.discard(None)
    if len(models) > 1:
        notes.append(
            "WARNING: rows use different models "
            f"({sorted(models)}); they are NOT comparable."
        )
    if missing_models:
        notes.append(
            "WARNING: some result files have no provenance.model; which model "
            "they used could not be verified."
        )

    capped = [p for p in present if p.get("max_samples")]
    if capped:
        notes.append(
            f"WARNING: {len(capped)} run(s) used --max-samples; those rows are "
            "not comparable to a full-dataset run."
        )

    seeds = {p.get("seed") for p in present if p.get("seed") is not None}
    missing_seeds = any(p.get("seed") is None for p in present)
    if len(seeds) > 1:
        notes.append(
            f"WARNING: rows use different seeds ({sorted(seeds)}); they are NOT "
            "strictly comparable."
        )
    if missing_seeds:
        notes.append(
            "WARNING: some result files have no provenance.seed; their seeding "
            "could not be verified."
        )

    # Deduplicate identical labels, which can now legitimately repeat across runs.
    return list(dict.fromkeys(notes))


def build_rows(payloads: List[Dict[str, Any]]) -> List[Tuple[str, ...]]:
    """One table row per run.

    Keyed on the run identity rather than the method: rows F, G and every
    lambda-sweep row all record ``method="visualguard"``, so keying on the
    method silently dropped all but one of them from the table.
    """
    by_run: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        prov = provenance_of(payload)
        method = prov.get("method") or payload.get("method")
        if method is None:
            continue
        key = run_identity(payload)
        existing = by_run.get(key)
        # Prefer a payload with actual benchmark data if two files share a stem.
        if existing is None or (
            "pope" not in existing and "pope" in payload
        ):
            by_run[key] = payload

    def sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, int, str]:
        key, payload = item
        prov = provenance_of(payload)
        method = str(prov.get("method") or payload.get("method"))
        order = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
        canonical = method in CANONICAL_WEIGHTS or method == "visualguard"
        return (order, 0 if canonical else 1, key)

    rows: List[Tuple[str, ...]] = []
    ordered_runs = sorted(by_run.items(), key=sort_key)

    # Two runs can share a label even with different keys -- e.g. the full method
    # and a lambda-sweep row that happens to sweep to the default lambda. Append
    # the run identity to every colliding label so no two rows look identical.
    label_counts: Dict[str, int] = {}
    for _key, payload in ordered_runs:
        label_counts[row_label(payload)] = label_counts.get(row_label(payload), 0) + 1

    for key, payload in ordered_runs:
        label = row_label(payload)
        if label_counts[label] > 1:
            label = f"{label} ({key})"
        rows.append(
            (
                label,
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
