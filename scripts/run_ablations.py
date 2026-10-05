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

Add ``--smoke`` for a tiny run suitable for checking the pipeline (results are
NOT comparable to a real run and are labelled as such).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

#: (label, method, extra CLI args). ``None`` means the row needs a detector.
ABLATION_ROWS = [
    ("A. baseline", "baseline", []),
    ("B. attention only", "attention", []),
    ("C. semantic only", "semantic", []),
    ("D. region only", "region", []),
    ("E. attention + semantic", "unidirectional", []),
    ("F. attention + region", "visualguard", ["--alpha", "1.0", "--beta", "0.0", "--gamma", "1.0"]),
    ("G. full visualguard", "visualguard", []),
]


def build_command(
    args: argparse.Namespace, label: str, method: str, extra: Sequence[str]
) -> List[str]:
    slug = label.split(". ")[-1].replace(" ", "_").replace("+", "and")
    cmd = [
        sys.executable, "-m", "src.run",
        "--benchmark", args.benchmark,
        "--method", method,
        "--model", args.model,
        "--device", args.device,
        "--dtype", args.dtype,
        "--data-root", str(args.data_root),
        "--output-dir", str(args.output_dir),
        "--seed", str(args.seed),
        "--run-name", f"ablation_{slug}",
        "--log-level", args.log_level,
    ]
    if args.image_root:
        cmd += ["--image-root", str(args.image_root)]
    if args.setting:
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
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
    plans: List[tuple] = [
        (label, method, extra) for label, method, extra in ABLATION_ROWS
        if label.split(".")[0].upper() not in skip
    ]

    for lam in [x for x in args.lambda_sweep.split(",") if x.strip()]:
        slug = f"lambda_{lam.strip()}"
        plans.append(
            (f"L. visualguard lambda={lam.strip()}", "visualguard",
             ["--lambda", lam.strip()])
        )

    needs_grounding = {"D", "F"}
    if needs_grounding - skip and args.grounding_backend == "none":
        print(
            "WARNING: rows D (region only) and F (attention + region) require a\n"
            "         grounding backend, but --grounding-backend none was given.\n"
            "         Those rows would measure nothing. Either enable a backend or\n"
            "         pass --skip-rows D,F. They are being skipped.\n",
            file=sys.stderr,
        )
        plans = [p for p in plans if p[0].split(".")[0].upper() not in needs_grounding]

    print(f"Planned runs: {len(plans)}")
    failures = []
    for label, method, extra in plans:
        cmd = build_command(args, label, method, extra)
        print("\n" + "=" * 72)
        print(f"RUNNING {label}  (method={method})")
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
