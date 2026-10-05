#!/usr/bin/env python
"""
Run the VisualGuard ablation grid.

Every configuration uses the *same* LVLM, the same prompts and the same seed, so
the only difference between rows is which evidence channels are active and the
penalty strength. That is what makes the comparison meaningful.

Grid
----
A. baseline          unmodified greedy decoding
B. attention         alpha only
C. semantic          beta only
D. region            gamma only (requires a grounding backend)
E. attention+semantic
F. attention+region  (requires a grounding backend)
G. visualguard       all channels

plus a lambda sweep on the full method.

Usage
-----
::

    python scripts/run_ablations.py \\
        --benchmark pope \\
        --data-root /path/to/POPE \\
        --image-root /path/to/coco/val2017 \\
        --model llava-hf/llava-1.5-7b-hf \\
        --max-samples 500

    python scripts/run_ablations.py --smoke ...    # tiny pipeline check

Every row is run against the shipped YAML for that ablation (``--config
<name>``) rather than the bare dataclass defaults, so the row that is measured is
the row that is documented.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

#: (label, method, extra CLI args, YAML config name). The config name is passed
#: as ``--config`` so each row runs its documented configuration. Without it the
#: rows silently used the ``EvidenceConfig`` *dataclass* defaults instead -- which
#: differ from the shipped YAML, e.g. ``visualguard.yaml`` sets ``gamma: 0.0``
#: while the dataclass defaults to ``0.5``, so row G ran a different weighting
#: than the one documented and measured.
ABLATION_ROWS = [
    ("A. baseline", "baseline", [], "baseline"),
    ("B. attention only", "attention", [], "attention"),
    ("C. semantic only", "semantic", [], "semantic"),
    ("D. region only", "region", [], "region"),
    ("E. attention + semantic", "unidirectional", [], "unidirectional"),
    (
        "F. attention + region", "visualguard",
        ["--alpha", "1.0", "--beta", "0.0", "--gamma", "1.0"], "visualguard",
    ),
    ("G. full visualguard", "visualguard", [], "visualguard"),
]

#: Extra samples for ``--smoke``: enough to exercise every code path.
SMOKE_MAX_SAMPLES = 2


def build_command(
    args: argparse.Namespace,
    label: str,
    method: str,
    extra: Sequence[str],
    config: Optional[str] = None,
) -> List[str]:
    slug = label.split(". ")[-1].replace(" ", "_").replace("+", "and")
    # Keep run names filename-safe: a lambda row's label contains "=" and ".",
    # which would otherwise end up in the result filename.
    slug = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in slug)
    cmd = [
        sys.executable, "-m", "src.run",
        "--benchmark", args.benchmark,
        "--method", method,
        "--model", args.model,
        "--data-root", str(args.data_root),
        "--output-dir", str(args.output_dir),
        "--seed", str(args.seed),
        "--run-name", f"ablation_{slug}",
        "--log-level", args.log_level,
    ]
    # Only pass --device/--dtype when the user asked for them, so the runner's
    # own "auto" defaults still apply. Forcing --device cuda made the sweep
    # unusable on a CPU-only host.
    if args.device:
        cmd += ["--device", args.device]
    if args.dtype:
        cmd += ["--dtype", args.dtype]
    if config:
        cmd += ["--config", config]
    if args.image_root:
        cmd += ["--image-root", str(args.image_root)]
    if args.setting and args.benchmark == "pope":
        # --setting is POPE-only; run_mme never reads it, so passing it for MME
        # was a silently ignored flag.
        cmd += ["--setting", args.setting]
    if args.max_samples:
        cmd += ["--max-samples", str(args.max_samples)]
    if args.grounding_backend:
        cmd += ["--grounding-backend", args.grounding_backend]
    cmd += list(extra)
    cmd += list(args.passthrough)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default="pope", choices=["pope", "mme"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--setting", default=None)
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default=None,
                        help="Default: the runner's auto detection.")
    parser.add_argument("--dtype", default=None,
                        help="Default: the runner's auto detection.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run each row on a couple of samples to check the "
                             "pipeline. Results are NOT comparable to a real run.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grounding-backend", default="none",
                        choices=["none", "hf_grounding_dino", "grounding_dino"],
                        help="Required for rows D and F.")
    parser.add_argument("--lambda-sweep", default="",
                        help="Comma-separated lambda values, e.g. 0.25,0.5,1.0")
    parser.add_argument("--skip-rows", default="",
                        help="Comma-separated row labels to skip, e.g. 'D,F'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without running them.")
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    if args.passthrough and args.passthrough[0] == "--":
        args.passthrough = args.passthrough[1:]

    skip = {s.strip().upper() for s in args.skip_rows.split(",") if s.strip()}
    if args.smoke and not args.max_samples:
        args.max_samples = SMOKE_MAX_SAMPLES

    plans: List[tuple] = [
        (label, method, extra, config)
        for label, method, extra, config in ABLATION_ROWS
        if label.split(".")[0].upper() not in skip
    ]

    for lam in [x for x in args.lambda_sweep.split(",") if x.strip()]:
        plans.append(
            (f"L. visualguard lambda={lam.strip()}", "visualguard",
             ["--lambda", lam.strip()], "visualguard")
        )

    # Skip the detector-dependent rows, but only those the user did NOT already
    # ask to skip. The old filter removed D *and* F whenever either was absent
    # from --skip-rows, so "--skip-rows D" silently also dropped F.
    needs_grounding = {"D", "F"}
    blocked = needs_grounding - skip
    if blocked and args.grounding_backend == "none":
        names = ", ".join(sorted(blocked))
        noun = "row requires" if len(blocked) == 1 else "rows require"
        print(
            f"WARNING: {names} {noun} a grounding backend, but "
            "--grounding-backend none was given.\n"
            "         Without a detector those rows would measure no regions at "
            "all. Either\n"
            "         enable a backend, or accept that they are skipped (the "
            "runner also refuses\n"
            "         a gamma>0 run without a backend, rather than reporting it "
            "as a region ablation).\n",
            file=sys.stderr,
        )
        plans = [
            p for p in plans
            if p[0].split(".")[0].upper() not in blocked
        ]

    print(f"Planned runs: {len(plans)}")
    failures = []
    if args.smoke:
        print(
            f"SMOKE MODE: {args.max_samples} samples per split. These numbers "
            "are NOT comparable\nto a full run and must not be reported as "
            "results.",
            file=sys.stderr,
        )
    for label, method, extra, config in plans:
        cmd = build_command(args, label, method, extra, config)
        print("\n" + "=" * 72)
        print(f"RUNNING {label}  (method={method} config={config})")
        print("  " + " ".join(cmd))
        print("=" * 72, flush=True)
        if args.dry_run:
            continue
        started = time.perf_counter()
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            failures.append((label, result.returncode))
            print(f"FAILED {label} (exit {result.returncode}) after {elapsed:.0f}s")
        else:
            print(f"DONE {label} in {elapsed:.0f}s")

    print("\n" + "=" * 72)
    print("Ablation sweep finished.")
    print(f"Results in {args.output_dir}")
    print("Generate the comparison table with:")
    print("  python scripts/make_results_table.py --results-dir results")
    if failures:
        print(f"\n{len(failures)} run(s) FAILED: {failures}")
        print("Do not report a table with missing rows as if they had run.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
