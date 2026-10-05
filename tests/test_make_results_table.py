# -*- coding: utf-8 -*-
"""
Tests for ``scripts/make_results_table.py``.

The table is the deliverable of an ablation study, so a row that silently
disappears is worse than a wrong number: the reader cannot tell it is missing.
These tests pin the two failure modes that were live -- rows collapsing by method
name, and the comparability checks crashing -- plus the guarantee that
unmeasured cells never render as a number.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_results_table.py"


def _load():
    spec = importlib.util.spec_from_file_location("make_results_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mrt = _load()


def _run(
    name: str,
    method: str,
    weights=(1.0, 1.0, 0.0),
    lam: float = 0.5,
    f1: float = 0.5,
    seed=42,
    model="m",
    provenance=True,
) -> dict:
    payload = {
        "pope": {"random": {"metrics": {"f1": f1, "hallucination_rate": 0.1}}},
        "resolved_config": {
            "evidence": {
                "alpha": weights[0], "beta": weights[1], "gamma": weights[2],
                "lam": lam,
            }
        },
    }
    if provenance:
        payload["provenance"] = {
            "method": method, "model": model, "seed": seed, "run_name": name,
        }
    return payload


def _write(directory: Path, *payloads) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index, payload in enumerate(payloads):
        path = directory / f"run_{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
    return directory


# ---------------------------------------------------------------------------
# every run gets its own row
# ---------------------------------------------------------------------------


def test_runs_sharing_a_method_name_each_get_a_row(tmp_path):
    """Rows F, G and the lambda sweep all record ``method="visualguard"``.

    Keying the table on the method collapsed them into a single row, so the
    attention+region ablation and every lambda point vanished from the report
    without a ``not run`` marker or a warning.
    """
    directory = _write(
        tmp_path / "r",
        _run("ablation_A", "baseline", weights=(0.0, 0.0, 0.0)),
        _run("ablation_B", "attention", weights=(1.0, 0.0, 0.0)),
        _run("ablation_F", "visualguard", weights=(1.0, 0.0, 1.0)),
        _run("ablation_G", "visualguard", weights=(1.0, 1.0, 0.0)),
        _run("ablation_L_025", "visualguard", weights=(1.0, 1.0, 0.0), lam=0.25),
        _run("ablation_L_1", "visualguard", weights=(1.0, 1.0, 0.0), lam=1.0),
    )
    rows = mrt.build_rows(mrt.load_results(directory))
    assert len(rows) == 6, f"only {len(rows)} of 6 runs appeared in the table"


def test_distinct_labels_for_distinct_visualguard_configurations(tmp_path):
    directory = _write(
        tmp_path / "r",
        _run("ablation_F", "visualguard", weights=(1.0, 0.0, 1.0)),
        _run("ablation_G", "visualguard", weights=(1.0, 1.0, 0.0)),
    )
    labels = [row[0] for row in mrt.build_rows(mrt.load_results(directory))]
    assert len(set(labels)) == 2, f"two configurations shared one label: {labels}"
    # The shipped full configuration keeps its canonical label...
    assert any(label == "G. full VisualGuard" for label in labels)
    # ...while the attention+region variant must not be called "full".
    variant = next(label for label in labels if label != "G. full VisualGuard")
    assert "full VisualGuard" not in variant
    assert "a1 b0 g1" in variant, f"the variant is not self-describing: {variant!r}"


def test_canonical_rows_keep_their_plain_labels(tmp_path):
    directory = _write(
        tmp_path / "r",
        _run("ablation_A", "baseline", weights=(0.0, 0.0, 0.0)),
        _run("ablation_B", "attention", weights=(1.0, 0.0, 0.0)),
        _run("ablation_E", "unidirectional", weights=(1.0, 1.0, 0.0)),
    )
    labels = [row[0] for row in mrt.build_rows(mrt.load_results(directory))]
    assert labels == [
        "A. baseline (greedy)", "B. attention only", "E. attention + semantic",
    ]


def test_identical_configurations_stay_distinguishable(tmp_path):
    """Two runs with the same weights must not render as identical rows."""
    directory = _write(
        tmp_path / "r",
        _run("ablation_G", "visualguard", weights=(1.0, 1.0, 0.0)),
        _run("ablation_L_05", "visualguard", weights=(1.0, 1.0, 0.0)),
    )
    labels = [row[0] for row in mrt.build_rows(mrt.load_results(directory))]
    assert len(set(labels)) == 2, f"colliding labels were not disambiguated: {labels}"


# ---------------------------------------------------------------------------
# comparability checks must not crash
# ---------------------------------------------------------------------------


def test_missing_seed_warns_instead_of_crashing(tmp_path):
    """A payload without a seed is exactly the case the check exists for."""
    directory = _write(
        tmp_path / "r",
        _run("a", "baseline", weights=(0.0, 0.0, 0.0), seed=42),
        _run("b", "attention", weights=(1.0, 0.0, 0.0), seed=None),
    )
    notes = mrt.provenance_note(mrt.load_results(directory))
    assert any("seed" in n for n in notes), f"no seed warning: {notes}"


def test_null_provenance_is_tolerated(tmp_path):
    """``"provenance": null`` must not raise where a missing key would not."""
    directory = _write(
        tmp_path / "r",
        {"provenance": None, "pope": {"random": {"metrics": {"f1": 0.5}}}},
    )
    payloads = mrt.load_results(directory)
    assert mrt.provenance_note(payloads) == []


def test_missing_provenance_is_tolerated(tmp_path):
    directory = _write(
        tmp_path / "r",
        {"pope": {"random": {"metrics": {"f1": 0.5}}}},
    )
    payloads = mrt.load_results(directory)
    assert mrt.provenance_note(payloads) == []


def test_different_seeds_are_reported_and_sortable(tmp_path):
    directory = _write(
        tmp_path / "r",
        _run("a", "baseline", weights=(0.0, 0.0, 0.0), seed=1),
        _run("b", "attention", weights=(1.0, 0.0, 0.0), seed=2),
    )
    notes = mrt.provenance_note(mrt.load_results(directory))
    assert any("different seeds" in n for n in notes)


def test_different_models_are_reported(tmp_path):
    directory = _write(
        tmp_path / "r",
        _run("a", "baseline", weights=(0.0, 0.0, 0.0), model="llava-7b"),
        _run("b", "attention", weights=(1.0, 0.0, 0.0), model="llava-13b"),
    )
    notes = mrt.provenance_note(mrt.load_results(directory))
    assert any("different models" in n for n in notes)


def test_max_samples_is_reported(tmp_path):
    directory = _write(tmp_path / "r", _run("a", "baseline", weights=(0.0, 0.0, 0.0)))
    payloads = mrt.load_results(directory)
    payloads[0]["provenance"]["max_samples"] = 100
    assert any("max-samples" in n for n in mrt.provenance_note(payloads))


# ---------------------------------------------------------------------------
# unmeasured cells
# ---------------------------------------------------------------------------


def test_missing_benchmarks_render_as_not_run():
    payload = _run("a", "baseline", weights=(0.0, 0.0, 0.0))
    payload.pop("pope")
    assert mrt.pope_cell(payload, "random", "f1") == mrt.NOT_RUN
    assert mrt.mme_total(payload) == mrt.NOT_RUN
    assert mrt.hallucination_summary(payload) == mrt.NOT_RUN


def test_absent_setting_renders_as_not_run():
    payload = _run("a", "baseline", weights=(0.0, 0.0, 0.0))
    assert mrt.pope_cell(payload, "adversarial", "f1") == mrt.NOT_RUN


def test_non_numeric_metric_renders_as_not_run():
    payload = _run("a", "baseline", weights=(0.0, 0.0, 0.0))
    payload["pope"]["random"]["metrics"]["f1"] = "n/a"
    assert mrt.pope_cell(payload, "random", "f1") == mrt.NOT_RUN


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def test_cli_renders_every_run_and_exits_zero(tmp_path):
    directory = _write(
        tmp_path / "r",
        _run("ablation_A", "baseline", weights=(0.0, 0.0, 0.0)),
        _run("ablation_F", "visualguard", weights=(1.0, 0.0, 1.0)),
        _run("ablation_G", "visualguard", weights=(1.0, 1.0, 0.0)),
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--results-dir", str(directory)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    body = [line for line in result.stdout.splitlines() if line.startswith("| ")]
    data_rows = body[2:]
    assert len(data_rows) == 3, f"table rendered {len(data_rows)} rows:\n{result.stdout}"
    assert "not run" in result.stdout


def test_cli_reports_an_empty_results_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--results-dir", str(empty)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "No result JSON files" in result.stderr