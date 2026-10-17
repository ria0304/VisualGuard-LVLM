#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end experiment workflow for the VisualGuard-LVLM conference paper.

One command reproduces every number that goes into the paper. Each stage is
resumable (a run whose result JSON already exists is skipped unless --force),
every run is logged, and a final gate refuses to call the results "paper-ready"
if anything was capped, missing, or run on a dirty git tree.

Stages (run in this order)
--------------------------
 1. preflight   compileall + pytest + input-path checks + environment snapshot
 2. smoke       scripts/smoke_test.py on a tiny LVLM (plumbing only, no numbers)
 3. pope        POPE random/popular/adversarial: baseline, sampling, beam,
                visualguard   (full splits)
 4. ablations   rows A-G from scripts/run_ablations.py on POPE adversarial
                (the hardest split)
 5. sweeps      lambda and threshold sweeps of full VisualGuard on adversarial
 6. mme         MME: baseline, visualguard
 7. chair       CHAIR: baseline, VCD, visualguard (same image subset per seed)
 8. report      results_table.md/csv, mean +/- std across seeds, paper gate

Typical use
-----------
Rehearsal on a tiny sample (output goes to a separate dir, flagged NOT reportable)::

    python scripts/run_paper_pipeline.py --quick \\
        --pope-root /data/POPE --image-root /data/coco/val2017 \\
        --mme-root /data/MME \\
        --chair-ann-dir /data/coco/annotations --chair-images /data/coco/val2014 \\
        --chair-synonyms /data/chair/synonyms.txt

Real run (LLaVA-1.5-7B, 3 seeds, GPU)::

    python scripts/run_paper_pipeline.py \\
        --model llava-hf/llava-1.5-7b-hf --device cuda --dtype float16 \\
        --seeds 42,43,44 --grounding-backend hf_grounding_dino \\
        --pope-root ... --image-root ... --mme-root ... \\
        --chair-ann-dir ... --chair-images ... --chair-synonyms ...

Run only some stages::

    python scripts/run_paper_pipeline.py --stages pope,report ...
    python scripts/run_paper_pipeline.py --skip-stages smoke,mme ...

See the commands without running anything::

    python scripts/run_paper_pipeline.py --dry-run ...

Design rules (what keeps the paper defensible)
----------------------------------------------
* Every method loads the SAME checkpoint, uses the same prompt template and the
  same seed list, so differences come from the decoding rule alone.
* Deterministic methods (greedy, beam, visualguard) are run on the first seed
  only; stochastic ones (sampling, VCD noise, CHAIR image subset) run on every
  seed so the paper can report mean +/- std.
* --quick writes to <output-dir>_quick and the report marks it NOT reportable.
* Missing cells are reported as "not run", never as zero.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

STAGES = ["preflight", "smoke", "pope", "ablations", "sweeps", "mme", "chair", "report"]
POPE_SETTINGS = ["random", "popular", "adversarial"]
SWEEP_SETTING = "adversarial"
QUICK_SAMPLES = 20
QUICK_CHAIR_IMAGES = 10

#: method tag -> (src.run --method, --config, extra CLI args, stochastic?)
POPE_METHODS: Dict[str, Tuple[str, Optional[str], List[str], bool]] = {
    "baseline": ("baseline", "baseline", [], False),
    "sampling": ("sampling", "baseline", [], True),
    "beam": ("beam", "baseline", [], False),
    "visualguard": ("visualguard", "visualguard", [], False),
}

MME_METHODS = ["baseline", "visualguard"]
CHAIR_METHODS: Dict[str, Tuple[str, Optional[str]]] = {
    "baseline": ("baseline", "baseline"),
    "vcd": ("vcd", None),
    "visualguard": ("visualguard", "visualguard"),
}


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------


@dataclass
class Job:
    stage: str
    name: str                # result file stem
    cmd: List[str]
    result_path: Path        # JSON that marks the job as finished
    log_path: Path


@dataclass
class Ctx:
    args: argparse.Namespace
    out: Path
    logs: Path
    seeds: List[int]
    max_samples: Optional[int]
    chair_images: int
    jobs: List[Job] = field(default_factory=list)
    failures: List[Tuple[str, int]] = field(default_factory=list)
    skipped_for_missing_backend: List[str] = field(default_factory=list)


def banner(msg: str) -> None:
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78, flush=True)


def py() -> str:
    return sys.executable


def common_model_args(a: argparse.Namespace) -> List[str]:
    cmd = ["--model", a.model]
    if a.device:
        cmd += ["--device", a.device]
    if a.dtype:
        cmd += ["--dtype", a.dtype]
    if a.quantization:
        cmd += ["--quantization", a.quantization]
    if a.cache_dir:
        cmd += ["--cache-dir", a.cache_dir]
    return cmd


def run_job(ctx: Ctx, job: Job) -> bool:
    a = ctx.args
    if job.result_path.exists() and not a.force:
        print(f"[skip] {job.name} (result exists)")
        return True
    print(f"[run ] {job.name}\n       {' '.join(job.cmd)}", flush=True)
    if a.dry_run:
        return True
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with open(job.log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            job.cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True
        )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        ctx.failures.append((job.name, proc.returncode))
        print(f"[FAIL] {job.name} exit={proc.returncode} after {elapsed:.0f}s -> {job.log_path}")
        return False
    if not job.result_path.exists():
        ctx.failures.append((job.name, -1))
        print(f"[FAIL] {job.name}: exited 0 but wrote no {job.result_path.name}")
        return False
    print(f"[done] {job.name} in {elapsed:.0f}s")
    return True


def run_cmd(ctx: Ctx, name: str, cmd: Sequence[str], log_name: str) -> bool:
    print(f"[run ] {name}\n       {' '.join(cmd)}", flush=True)
    if ctx.args.dry_run:
        return True
    log_path = ctx.logs / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(list(cmd), cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        ctx.failures.append((name, proc.returncode))
        print(f"[FAIL] {name} exit={proc.returncode} -> {log_path}")
        return False
    print(f"[done] {name}")
    return True


def pope_cmd(ctx: Ctx, tag: str, method: str, config: Optional[str], setting: str,
             seed: int, extra: Sequence[str] = ()) -> Job:
    a = ctx.args
    name = f"pope_{tag}_{setting}_s{seed}"
    cmd = [
        py(), "-m", "src.run",
        "--benchmark", "pope", "--method", method,
        "--setting", setting,
        "--data-root", str(a.pope_root), "--image-root", str(a.image_root),
        "--output-dir", str(ctx.out), "--run-name", name,
        "--seed", str(seed), "--log-level", "INFO",
        *common_model_args(a),
    ]
    if config:
        cmd += ["--config", config]
    if ctx.max_samples:
        cmd += ["--max-samples", str(ctx.max_samples)]
    if a.grounding_backend != "none":
        cmd += ["--grounding-backend", a.grounding_backend]
    cmd += list(extra)
    return Job("pope", name, cmd, ctx.out / f"{name}.json", ctx.logs / f"{name}.log")


def load_ablation_rows() -> List[Tuple[str, str, List[str], Optional[str]]]:
    """Reuse the repo's own ablation grid so the paper never drifts from it."""
    path = REPO_ROOT / "scripts" / "run_ablations.py"
    spec = importlib.util.spec_from_file_location("run_ablations", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return list(mod.ABLATION_ROWS)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_preflight(ctx: Ctx) -> bool:
    a = ctx.args
    banner("STAGE 1/8  preflight")
    ok = True

    needed = {
        "--pope-root": a.pope_root, "--image-root": a.image_root,
    }
    if "mme" in ctx.args.active:
        needed["--mme-root"] = a.mme_root
    if "chair" in ctx.args.active:
        needed.update({
            "--chair-ann-dir": a.chair_ann_dir,
            "--chair-images": a.chair_images,
            "--chair-synonyms": a.chair_synonyms,
        })
    for flag, value in needed.items():
        if not value:
            print(f"[FAIL] {flag} is required for the selected stages")
            ok = False
        elif not Path(value).exists():
            print(f"[FAIL] {flag} path does not exist: {value}")
            ok = False
    for s in POPE_SETTINGS:
        f = Path(a.pope_root or ".") / f"coco_pope_{s}.jsonl"
        if a.pope_root and not f.exists():
            print(f"[FAIL] missing POPE file {f}")
            ok = False

    try:
        import torch  # noqa: WPS433
        cuda = torch.cuda.is_available()
        print(f"[info] torch {torch.__version__} cuda={cuda}")
        if (a.device or "").startswith("cuda") and not cuda:
            print("[FAIL] --device cuda requested but CUDA is not available")
            ok = False
    except Exception as exc:  # pragma: no cover
        if a.dry_run:
            print(f"[warn] torch not importable here ({exc}); fine for --dry-run")
        else:
            print(f"[FAIL] cannot import torch: {exc}")
            ok = False

    if a.grounding_backend == "none":
        print("[warn] --grounding-backend none: ablation rows D and F (region) "
              "will be skipped. A paper that claims region evidence needs them.")

    if not a.skip_tests:
        ok &= run_cmd(ctx, "compileall", [py(), "-m", "compileall", "-q", "src", "tests", "scripts"],
                      "00_compileall.log")
        ok &= run_cmd(ctx, "pytest", [py(), "-m", "pytest", "tests", "-q"], "01_pytest.log")

    if not a.dry_run:
        ctx.out.mkdir(parents=True, exist_ok=True)
        snap = {
            "python": sys.version, "platform": platform.platform(),
            "argv": sys.argv, "model": a.model, "seeds": ctx.seeds,
            "quick": a.quick, "git_commit": _git(["rev-parse", "HEAD"]),
            "git_dirty": bool(_git(["status", "--porcelain"])),
        }
        (ctx.out / "environment.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")
        _git_freeze(ctx)
    return ok


def _git(args: Sequence[str]) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    except Exception:
        return ""


def _git_freeze(ctx: Ctx) -> None:
    try:
        out = subprocess.run([py(), "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        (ctx.out / "pip_freeze.txt").write_text(out, encoding="utf-8")
    except Exception:
        pass


def stage_smoke(ctx: Ctx) -> bool:
    banner("STAGE 2/8  smoke (plumbing only, numbers are meaningless)")
    cmd = [py(), "scripts/smoke_test.py", "--device", ctx.args.smoke_device]
    return run_cmd(ctx, "smoke_test", cmd, "02_smoke.log")


def stage_pope(ctx: Ctx) -> bool:
    banner("STAGE 3/8  POPE main comparison")
    ok = True
    for tag, (method, config, extra, stochastic) in POPE_METHODS.items():
        seeds = ctx.seeds if stochastic else ctx.seeds[:1]
        for setting in POPE_SETTINGS:
            for seed in seeds:
                ok &= run_job(ctx, pope_cmd(ctx, tag, method, config, setting, seed, extra))
    return ok


def _ablation_tag(label: str) -> str:
    slug = label.split(". ")[-1].replace(" ", "_").replace("+", "and")
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in slug)


def stage_ablations(ctx: Ctx) -> bool:
    banner(f"STAGE 4/8  ablations on POPE {SWEEP_SETTING}")
    ok = True
    seed = ctx.seeds[0]
    for label, method, extra, config in load_ablation_rows():
        letter = label.split(".")[0].upper()
        if letter in ("D", "F") and ctx.args.grounding_backend == "none":
            ctx.skipped_for_missing_backend.append(label)
            print(f"[skip] {label}: needs --grounding-backend")
            continue
        if letter == "A":
            continue  # identical to the stage-3 baseline run
        tag = f"abl_{letter}_{_ablation_tag(label)}"
        ok &= run_job(ctx, pope_cmd(ctx, tag, method, config, SWEEP_SETTING, seed, extra))
    return ok


def stage_sweeps(ctx: Ctx) -> bool:
    banner(f"STAGE 5/8  hyper-parameter sweeps on POPE {SWEEP_SETTING}")
    ok = True
    seed = ctx.seeds[0]
    for lam in _floats(ctx.args.lambda_sweep):
        ok &= run_job(ctx, pope_cmd(ctx, f"sweep_lambda{lam:g}", "visualguard", "visualguard",
                                    SWEEP_SETTING, seed, ["--lambda", str(lam)]))
    for thr in _floats(ctx.args.threshold_sweep):
        ok &= run_job(ctx, pope_cmd(ctx, f"sweep_thr{thr:g}", "visualguard", "visualguard",
                                    SWEEP_SETTING, seed, ["--threshold", str(thr)]))
    return ok


def stage_mme(ctx: Ctx) -> bool:
    banner("STAGE 6/8  MME")
    a = ctx.args
    ok = True
    seed = ctx.seeds[0]
    for method in MME_METHODS:
        name = f"mme_{method}_s{seed}"
        cmd = [
            py(), "-m", "src.run", "--benchmark", "mme", "--method", method,
            "--data-root", str(a.mme_root), "--output-dir", str(ctx.out),
            "--run-name", name, "--seed", str(seed), "--log-level", "INFO",
            "--config", "visualguard" if method == "visualguard" else "baseline",
            *common_model_args(a),
        ]
        if ctx.max_samples:
            cmd += ["--max-samples", str(ctx.max_samples)]
        if a.grounding_backend != "none":
            cmd += ["--grounding-backend", a.grounding_backend]
        ok &= run_job(ctx, Job("mme", name, cmd, ctx.out / f"{name}.json", ctx.logs / f"{name}.log"))
    return ok


def stage_chair(ctx: Ctx) -> bool:
    banner("STAGE 7/8  CHAIR (open-ended captioning)")
    a = ctx.args
    ok = True
    for tag, (method, config) in CHAIR_METHODS.items():
        for seed in ctx.seeds:
            name = f"chair_{tag}_s{seed}"
            run_dir = ctx.out / "chair" / f"{tag}_s{seed}"   # keeps predictions.jsonl per run
            cmd = [
                py(), "scripts/run_chair.py", "--method", method,
                "--ann-dir", str(a.chair_ann_dir), "--image-root", str(a.chair_images),
                "--synonyms", str(a.chair_synonyms), "--coco-split", a.chair_split,
                "--num-images", str(ctx.chair_images),
                "--output-dir", str(run_dir), "--run-name", name,
                "--seed", str(seed), "--log-level", "INFO",
                *common_model_args(a),
            ]
            if config:
                cmd += ["--config", config]
            if method == "visualguard" and a.grounding_backend != "none":
                cmd += ["--grounding-backend", a.grounding_backend]
            if not a.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
            ok &= run_job(ctx, Job("chair", name, cmd, run_dir / f"{name}.json", ctx.logs / f"{name}.log"))
    return ok


# ---------------------------------------------------------------------------
# report + paper gate
# ---------------------------------------------------------------------------


def _floats(csv: str) -> List[float]:
    return [float(x) for x in csv.split(",") if x.strip()]


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return math.nan, math.nan
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def _fmt(xs: List[float], pct: bool = False) -> str:
    if not xs:
        return "not run"
    m, s = _mean_std(xs)
    k = 100.0 if pct else 1.0
    return f"{m * k:.2f}" if len(xs) == 1 else f"{m * k:.2f} ± {s * k:.2f} (n={len(xs)})"


def _load_all(out: Path) -> List[Dict[str, Any]]:
    payloads = []
    for p in sorted(out.rglob("*.json")):
        if p.name in ("environment.json", "manifest.json"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and "provenance" in d:
            d["_file"] = str(p.relative_to(out))
            payloads.append(d)
    return payloads


def _run_name(d: Dict[str, Any]) -> str:
    return str(d["provenance"].get("run_name") or Path(d["_file"]).stem)


def _split_name(name: str) -> Tuple[str, str, str]:
    """pope_<tag>_<setting>_s<seed> -> (tag, setting, seed)."""
    head, _, seed = name.rpartition("_s")
    parts = head.split("_")
    if parts[0] == "pope" and parts[-1] in POPE_SETTINGS:
        return "_".join(parts[1:-1]), parts[-1], seed
    return head, "", seed


def stage_report(ctx: Ctx) -> bool:
    banner("STAGE 8/8  report + paper gate")
    out = ctx.out
    if ctx.args.dry_run:
        print("[dry-run] would write results_table.md/.csv, summary.md, manifest.json")
        return True
    payloads = _load_all(out)
    if not payloads:
        print("[FAIL] no result files found")
        return False

    # 1) the repo's own table (kept as-is for continuity)
    table_script = REPO_ROOT / "scripts" / "make_results_table.py"
    for fmt, ext in (("markdown", "md"), ("csv", "csv")):
        res = subprocess.run([py(), str(table_script), "--results-dir", str(out), "--format", fmt],
                             cwd=REPO_ROOT, capture_output=True, text=True)
        (out / f"results_table.{ext}").write_text(res.stdout, encoding="utf-8")

    # 2) mean +/- std across seeds
    pope: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    mme: Dict[str, List[float]] = {}
    chair: Dict[str, Dict[str, List[float]]] = {}
    effic: Dict[str, List[float]] = {}
    capped, dirty = [], []
    for d in payloads:
        prov = d["provenance"]
        name = _run_name(d)
        if prov.get("max_samples"):
            capped.append(name)
        if prov.get("git_dirty"):
            dirty.append(name)
        if "pope" in d:
            tag, _, _ = _split_name(name)
            for setting, block in d["pope"].items():
                m = block["metrics"]
                slot = pope.setdefault((tag, setting), {"f1": [], "acc": [], "hall": []})
                slot["f1"].append(m["f1"]); slot["acc"].append(m["accuracy"])
                slot["hall"].append(m["hallucination_rate"])
        if "mme" in d:
            tag = name.replace("mme_", "").rsplit("_s", 1)[0]
            mme.setdefault(tag, []).append(d["mme"]["category_totals"].get("total_score", math.nan))
        if "chair" in d:
            tag = name.replace("chair_", "").rsplit("_s", 1)[0]
            slot = chair.setdefault(tag, {"chair_s": [], "chair_i": [], "recall": [], "len": []})
            m = d["chair"]["metrics"]
            slot["chair_s"].append(m["chair_s"]); slot["chair_i"].append(m["chair_i"])
            slot["recall"].append(m["recall"]); slot["len"].append(m.get("avg_length_words", math.nan))
        eff = d.get("efficiency", {})
        if eff.get("total_runtime_s") and "pope" in d:
            tag, _, _ = _split_name(name)
            effic.setdefault(tag, []).append(eff["total_runtime_s"])

    lines: List[str] = ["# Paper summary (auto-generated)", ""]
    if ctx.args.quick:
        lines += ["> **QUICK REHEARSAL: capped samples. NOT reportable.**", ""]

    lines += ["## POPE (F1 %, mean ± std over seeds)", "",
              "| Method | Random | Popular | Adversarial | Worst | Halluc. rate (adv) |",
              "| --- | --- | --- | --- | --- | --- |"]
    tags = sorted({t for t, _ in pope}, key=lambda t: (t.startswith(("abl_", "sweep_")), t))
    for t in tags:
        cells = [_fmt(pope.get((t, s), {}).get("f1", []), pct=True) for s in POPE_SETTINGS]
        means = [statistics.fmean(pope[(t, s)]["f1"]) for s in POPE_SETTINGS if (t, s) in pope]
        worst = f"{min(means) * 100:.2f}" if means else "not run"
        hall = _fmt(pope.get((t, "adversarial"), {}).get("hall", []), pct=True)
        lines.append(f"| {t} | " + " | ".join(cells) + f" | {worst} | {hall} |")

    if mme:
        lines += ["", "## MME (total score)", "", "| Method | Total |", "| --- | --- |"]
        lines += [f"| {t} | {_fmt(v)} |" for t, v in sorted(mme.items())]
    if chair:
        lines += ["", "## CHAIR (lower is better, recall higher is better)", "",
                  "| Method | CHAIRs | CHAIRi | Recall | Avg words |", "| --- | --- | --- | --- | --- |"]
        for t, v in sorted(chair.items()):
            lines.append(f"| {t} | {_fmt(v['chair_s'])} | {_fmt(v['chair_i'])} | "
                         f"{_fmt(v['recall'])} | {_fmt(v['len'])} |")
    if effic:
        base = statistics.fmean(effic.get("baseline", [math.nan]))
        lines += ["", "## Cost (wall-clock per POPE run, seconds)", "",
                  "| Method | Seconds | x baseline |", "| --- | --- | --- |"]
        for t, v in sorted(effic.items()):
            m = statistics.fmean(v)
            lines.append(f"| {t} | {m:.0f} | {m / base:.2f}x |" if base == base else f"| {t} | {m:.0f} | n/a |")
    if ctx.skipped_for_missing_backend:
        lines += ["", "Skipped (no grounding backend): " + ", ".join(ctx.skipped_for_missing_backend)]

    # 3) paper gate
    problems = _gate(ctx, pope, mme, chair, capped, dirty)
    lines += ["", "## Paper gate", ""]
    lines += [f"- FAIL: {p}" for p in problems] or ["- PASS: all required rows present, full splits, clean tree"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {"files": [d["_file"] for d in payloads], "seeds": ctx.seeds,
                "model": ctx.args.model, "problems": problems, "quick": ctx.args.quick}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {out/'summary.md'}, {out/'results_table.md'}, {out/'results_table.csv'}")
    return not problems or ctx.args.quick


def _gate(ctx: Ctx, pope, mme, chair, capped, dirty) -> List[str]:
    problems: List[str] = []
    if ctx.args.quick:
        problems.append("--quick run: results are for rehearsal only")
    if capped and not ctx.args.quick:
        problems.append(f"{len(capped)} run(s) used --max-samples (e.g. {capped[0]})")
    if dirty:
        problems.append(f"{len(dirty)} run(s) were produced on a dirty git tree (commit first)")
    for tag in POPE_METHODS:
        for s in POPE_SETTINGS:
            if (tag, s) not in pope:
                problems.append(f"missing POPE row: {tag}/{s}")
    if "ablations" in ctx.args.active and not any(t.startswith("abl_") for t, _ in pope):
        problems.append("ablation rows missing")
    if "mme" in ctx.args.active:
        for m in MME_METHODS:
            if m not in mme:
                problems.append(f"missing MME row: {m}")
    if "chair" in ctx.args.active:
        for m in CHAIR_METHODS:
            if m not in chair:
                problems.append(f"missing CHAIR row: {m}")
            elif len(chair[m]["chair_s"]) < len(ctx.seeds):
                problems.append(f"CHAIR {m}: {len(chair[m]['chair_s'])}/{len(ctx.seeds)} seeds")
    if len(ctx.seeds) < 3 and not ctx.args.quick:
        problems.append("fewer than 3 seeds: std is not meaningful for stochastic rows")
    if ctx.skipped_for_missing_backend:
        problems.append("region ablations skipped (no grounding backend)")
    if ctx.failures:
        problems.append(f"{len(ctx.failures)} job(s) failed: {[n for n, _ in ctx.failures]}")
    return problems


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

STAGE_FUNCS = {
    "preflight": stage_preflight, "smoke": stage_smoke, "pope": stage_pope,
    "ablations": stage_ablations, "sweeps": stage_sweeps, "mme": stage_mme,
    "chair": stage_chair, "report": stage_report,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("model")
    g.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    g.add_argument("--device", default=None, help="cuda | cpu (default: runner auto)")
    g.add_argument("--dtype", default=None, help="float16 | bfloat16 | float32")
    g.add_argument("--quantization", default=None, choices=[None, "none", "4bit", "8bit"])
    g.add_argument("--cache-dir", default=None)
    g.add_argument("--grounding-backend", default="none",
                   choices=["none", "hf_grounding_dino", "grounding_dino"])
    g.add_argument("--smoke-device", default="cpu")

    g = p.add_argument_group("data")
    g.add_argument("--pope-root", help="dir with coco_pope_<setting>.jsonl")
    g.add_argument("--image-root", help="COCO val2017 dir (POPE images)")
    g.add_argument("--mme-root", help="MME root (one subdir per subtask)")
    g.add_argument("--chair-ann-dir", help="dir with instances_/captions_<split>.json")
    g.add_argument("--chair-images", help="COCO image dir for CHAIR")
    g.add_argument("--chair-synonyms", help="official CHAIR synonyms.txt")
    g.add_argument("--chair-split", default="val2014")
    g.add_argument("--chair-num-images", type=int, default=500)

    g = p.add_argument_group("experiment")
    g.add_argument("--seeds", default="42,43,44")
    g.add_argument("--lambda-sweep", default="0.25,0.5,1.0,2.0")
    g.add_argument("--threshold-sweep", default="0.2,0.35,0.5")
    g.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "paper"))
    g.add_argument("--stages", default="", help=f"comma list of {STAGES} (default: all)")
    g.add_argument("--skip-stages", default="")
    g.add_argument("--skip-tests", action="store_true")
    g.add_argument("--quick", action="store_true",
                   help=f"{QUICK_SAMPLES} samples / {QUICK_CHAIR_IMAGES} CHAIR images, 1 seed, "
                        "writes to <output-dir>_quick. NOT reportable.")
    g.add_argument("--force", action="store_true", help="re-run jobs that already have results")
    g.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    chosen = [s.strip() for s in args.stages.split(",") if s.strip()] or list(STAGES)
    skip = {s.strip() for s in args.skip_stages.split(",") if s.strip()}
    bad = [s for s in chosen + list(skip) if s not in STAGES]
    if bad:
        print(f"Unknown stage(s): {bad}. Valid: {STAGES}", file=sys.stderr)
        return 2
    args.active = [s for s in STAGES if s in chosen and s not in skip]

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out = Path(args.output_dir)
    max_samples, chair_images = None, args.chair_num_images
    if args.quick:
        seeds = seeds[:1]
        max_samples, chair_images = QUICK_SAMPLES, QUICK_CHAIR_IMAGES
        out = out.parent / (out.name + "_quick")
        banner("QUICK MODE: capped samples. Numbers are NOT reportable.")
    if not seeds:
        print("--seeds is empty", file=sys.stderr)
        return 2

    ctx = Ctx(args=args, out=out, logs=out / "logs", seeds=seeds,
              max_samples=max_samples, chair_images=chair_images)

    t0 = time.perf_counter()
    for stage in args.active:
        if not STAGE_FUNCS[stage](ctx) and stage == "preflight":
            print("\nPreflight failed: fix the above before spending GPU hours.")
            return 1
    banner(f"Pipeline finished in {(time.perf_counter() - t0) / 60:.1f} min")
    if ctx.failures:
        print(f"{len(ctx.failures)} job(s) failed: {ctx.failures}")
        print("Re-run the same command: finished jobs are skipped, failed ones retry.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
