# -*- coding: utf-8 -*-
"""
VisualGuard experiment runner.

Usage
-----
Baseline (unmodified greedy decoding of the same LVLM)::

    python -m src.run --benchmark pope --method baseline

Ablations on the same LVLM::

    python -m src.run --benchmark pope --method attention
    python -m src.run --benchmark pope --method semantic
    python -m src.run --benchmark pope --method visualguard

Every method loads the *same* checkpoint and uses the same prompt template, so
any difference is attributable to the decoding rule alone.

Results are written to ``--output-dir`` as JSON with full provenance (seed,
model id, git commit, device, library versions, resolved config), so a number
can be traced back to the exact command that produced it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.pope import POPE_SETTINGS, POPEDataError  # noqa: E402
from src.data.mme import MMEDataError, MME_SUBTASKS  # noqa: E402
from src.evaluation.mme_eval import MMEEvaluator  # noqa: E402
from src.evaluation.pope_eval import POPEEvaluator, evaluate_pope_settings  # noqa: E402
from src.model.grounding import GroundingConfig, GroundingUnavailable  # noqa: E402
from src.model.llava_backend import LVLMConfig  # noqa: E402
from src.model.visual_evidence import EvidenceConfig  # noqa: E402
from src.model.visual_guard_decoder import (  # noqa: E402
    ALL_METHODS,
    BASELINE_METHODS,
    DecodingConfig,
    VisualGuardDecoder,
    evidence_config_for_method,
)
from src.utils.config import (  # noqa: E402
    ConfigError,
    build_config,
    build_configs,
    deep_update,
    resolve_configs,
    save_json,
    to_dict,
)
from src.utils.reproducibility import (  # noqa: E402
    ReproducibilityConfig,
    apply_reproducibility,
    collect_metadata,
    peak_memory_mb,
    reset_peak_memory,
)

logger = logging.getLogger("visualguard")

CONFIG_DIR = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.run",
        description="VisualGuard: dynamic visual evidence-guided decoding "
        "for hallucination reduction in LVLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_argument_group("model")
    group.add_argument(
        "--model",
        default="llava-hf/llava-1.5-7b-hf",
        help="Hugging Face id of the LVLM. Must be the SAME across methods "
        "you intend to compare.",
    )
    group.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    group.add_argument("--dtype", default="auto", help="auto | float32 | float16 | bfloat16")
    group.add_argument(
        "--quantization", default=None, choices=["none", "4bit", "8bit"],
        help="bitsandbytes quantization (requires a CUDA device). Default: none.",
    )
    group.add_argument(
        "--attn-implementation", default="eager",
        help="Attention kernel. 'eager' is required for attention evidence, "
        "since fused kernels do not return attention matrices.",
    )
    group.add_argument("--revision", default="main", help="HF revision / branch.")
    group.add_argument("--cache-dir", default=None, help="HF cache directory.")
    group.add_argument(
        "--trust-remote-code", action="store_true",
        help="Allow custom modelling code from the Hub (audit the repo first).",
    )

    group = parser.add_argument_group("data")
    group.add_argument(
        "--benchmark", default="pope", choices=["pope", "mme"],
        help="Which benchmark to run.",
    )
    group.add_argument(
        "--data-root", default=None,
        help="Benchmark root. For POPE: the directory holding "
        "coco_pope_<setting>.jsonl. For MME: the directory holding one "
        "subdirectory per subtask.",
    )
    group.add_argument(
        "--image-root", default=None,
        help="Directory containing the benchmark images. For POPE this is "
        "normally the COCO val2017 directory.",
    )
    group.add_argument("--setting", default=None, choices=list(POPE_SETTINGS),
                       help="Single POPE setting. Omit to run all three.")
    group.add_argument("--mme-subtask", default=None, action="append",
                       help="Restrict MME to these subtasks (repeatable).")
    group.add_argument("--max-samples", type=int, default=None,
                       help="Cap samples per split. Use for smoke tests; "
                       "results are NOT comparable to a full run.")

    group = parser.add_argument_group("method")
    group.add_argument("--method", default="baseline", choices=sorted(ALL_METHODS),
                       help="Decoding method. 'baseline' = unmodified greedy.")
    group.add_argument("--config", default=None,
                       help=f"YAML config under {CONFIG_DIR.name}/ (e.g. visualguard).")
    group.add_argument("--alpha", type=float, default=None, help="Attention evidence weight.")
    group.add_argument("--beta", type=float, default=None, help="Semantic evidence weight.")
    group.add_argument("--gamma", type=float, default=None, help="Region evidence weight.")
    group.add_argument("--lambda", dest="lam", type=float, default=None,
                       help="Hallucination penalty strength.")
    group.add_argument("--threshold", type=float, default=None,
                       help="VES below this counts as insufficient support.")
    group.add_argument("--top-k", dest="top_k", type=int, default=None,
                       help="Candidate tokens scored per step.")
    group.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None)
    group.add_argument("--num-beams", type=int, default=None, help="Beam width for --method beam.")
    group.add_argument("--temperature", type=float, default=None)
    group.add_argument("--top-p", dest="top_p", type=float, default=None)

    group = parser.add_argument_group("evidence backends")
    group.add_argument(
        "--clip-model", dest="clip_model_name", default=None,
        help="CLIP checkpoint for semantic evidence. Default: the value in the "
             "YAML config, else openai/clip-vit-base-patch32.",
    )
    group.add_argument(
        "--grounding-backend", dest="grounding_backend", default=None,
        choices=["none", "hf_grounding_dino", "grounding_dino"],
        help="Region-evidence backend. 'none' disables region evidence. "
             "Default: the YAML config value, else none.",
    )
    group.add_argument(
        "--grounding-model", dest="grounding_model_id", default=None,
        help="Grounding detector checkpoint. Default: "
             "IDEA-Research/grounding-dino-base.",
    )
    group.add_argument(
        "--grounding-device", dest="grounding_device", default=None,
        help="Device for the detector. Default: auto.",
    )
    group.add_argument(
        "--grounding-box-threshold", dest="grounding_box_threshold",
        type=float, default=None,
        help="Minimum detection confidence. Default: 0.3.",
    )

    group = parser.add_argument_group("output / reproducibility")
    group.add_argument("--output-dir", default=str(REPO_ROOT / "results"))
    group.add_argument("--run-name", default=None,
                       help="Result file stem. Defaults to <benchmark>_<method>_<timestamp>.")
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--no-deterministic", action="store_true",
                       help="Allow non-deterministic kernels (faster, less reproducible).")
    group.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


# ---------------------------------------------------------------------------
# config assembly
# ---------------------------------------------------------------------------


def evidence_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect explicitly-provided CLI evidence flags (None = don't override)."""
    return {
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "lam": args.lam,
        "threshold": args.threshold,
        "clip_model_name": args.clip_model_name,
    }


def decoding_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "num_beams": args.num_beams,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }


def resolve_quantization(args: argparse.Namespace) -> Dict[str, Any]:
    if args.quantization == "4bit":
        return {"load_in_4bit": True}
    if args.quantization == "8bit":
        return {"load_in_8bit": True}
    return {}


def build_experiment_configs(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Resolve YAML config + CLI overrides into typed config objects.

    Method semantics
    ----------------
    For single-channel ablation methods (``attention`` / ``semantic`` /
    ``region``) the active channel is forced to weight 1.0 and the others to
    0.0, so the ablation isolates exactly one channel regardless of what the
    YAML file specified. For ``visualguard`` the full YAML/CLI weights apply.
    """
    yaml_path: Optional[Path] = None
    if args.config:
        candidate = Path(args.config)
        yaml_path = candidate if candidate.is_file() else CONFIG_DIR / f"{args.config}.yaml"

    merged = resolve_configs(yaml_path, evidence_overrides(args))

    # One flat namespace: CLI flags override YAML, and every key must be
    # claimed by one of the config dataclasses (typos raise).
    # ``deep_update`` (not a dict merge) is required here because unset CLI flags
    # are None, and a plain ``{**merged, "backend": None}`` would write the None
    # over the value the YAML just supplied, silently discarding the config file.
    flat = deep_update(
        merged,
        {
            **decoding_overrides(args),
            "backend": args.grounding_backend,
            "model_id": args.grounding_model_id,
            "box_threshold": args.grounding_box_threshold,
            "device": args.grounding_device,
        },
    )

    # ``dtype`` and ``device`` exist on both GroundingConfig and LVLMConfig, but
    # only the grounding ones are in the specs below -- so an unqualified
    # ``dtype: float16`` in a YAML silently configured the *detector* while the
    # LVLM kept its default. That is the reading nobody intends. Route the
    # unqualified keys to the model, and give the detector explicitly prefixed
    # aliases.
    model_dtype = (
        args.dtype if args.dtype != "auto" else flat.pop("dtype", None)
    )
    model_device = flat.pop("device", None)
    flat["dtype"] = flat.pop("grounding_dtype", None)
    if args.device != "auto":
        model_device = args.device
    flat["device"] = args.grounding_device or flat.get("device") or model_device

    built = build_configs(
        {
            "evidence": EvidenceConfig,
            "decoding": DecodingConfig,
            "grounding": GroundingConfig,
        },
        flat,
    )
    evidence = built["evidence"]
    decoding = built["decoding"]
    grounding = built["grounding"]

    if args.method in BASELINE_METHODS:
        # Baselines must resolve to an all-zero evidence config, otherwise the
        # logged weights would misrepresent what the run actually did.
        evidence = evidence_config_for_method(args.method, evidence)
    elif args.method in {"attention", "semantic", "region", "unidirectional"}:
        evidence = evidence_config_for_method(args.method, evidence)

    lvlm = build_config(
        LVLMConfig,
        {
            "model_name": args.model,
            "device": model_device or "auto",
            "dtype": model_dtype or "auto",
            "attn_implementation": args.attn_implementation,
            "revision": args.revision,
            "cache_dir": args.cache_dir,
            "trust_remote_code": args.trust_remote_code,
            **resolve_quantization(args),
        },
    )
    return {
        "lvlm": lvlm,
        "decoding": decoding,
        "evidence": evidence,
        "grounding": grounding,
        "yaml_path": yaml_path,
        "merged": merged,
    }


# ---------------------------------------------------------------------------
# experiment execution
# ---------------------------------------------------------------------------


def precheck_dataset(args: argparse.Namespace) -> None:
    """Validate dataset paths *before* loading the model.

    Loading a multi-billion-parameter checkpoint takes minutes and gigabytes, so
    discovering a missing annotation file afterwards is expensive and confusing.
    This raises :class:`SystemExit` with the loaders' guidance.
    """
    if not args.data_root:
        if args.benchmark == "pope":
            hint = (
                "--data-root is required for --benchmark pope.\n"
                "Expected a directory containing coco_pope_random.jsonl, "
                "coco_pope_popular.jsonl and coco_pope_adversarial.jsonl "
                "(official POPE release)."
            )
        else:
            hint = (
                "--data-root is required for --benchmark mme.\n"
                "Expected the official MME layout: "
                "<data-root>/<subtask>/<subtask>.jsonl with images under "
                "<data-root>/<subtask>/images/."
            )
        raise SystemExit(hint)

    root = Path(args.data_root)
    if args.benchmark == "pope":
        from src.data.pope import locate_annotation_file

        settings = [args.setting] if args.setting else list(POPE_SETTINGS)
        for setting in settings:
            locate_annotation_file(root, setting)
    else:
        from src.data.mme import discover_subtasks

        if not root.is_dir():
            raise SystemExit(
                f"MME root {root} does not exist.\n"
                "Expected <data-root>/<subtask>/<subtask>.jsonl per subtask."
            )
        found = discover_subtasks(root)
        if not found:
            raise SystemExit(
                f"No MME subtasks found under {root}. Expected subdirectories "
                f"{list(MME_SUBTASKS)} each containing <subtask>.jsonl.\n"
                f"Directory listing: {sorted(p.name for p in root.iterdir())[:40]}"
            )
        logger.info("MME subtasks discovered: %s", found)


#: Benchmark token budgets. POPE answers are one word; MME allows a phrase.
POPE_MAX_NEW_TOKENS = 16
MME_MAX_NEW_TOKENS = 32


def benchmark_budget(args: argparse.Namespace, benchmark: str) -> int:
    """The token budget this benchmark imposes, CLI flag or benchmark default."""
    if args.max_new_tokens:
        return int(args.max_new_tokens)
    return MME_MAX_NEW_TOKENS if benchmark == "mme" else POPE_MAX_NEW_TOKENS


def run_pope(
    args: argparse.Namespace,
    decoder: VisualGuardDecoder,
    configs: Dict[str, Any],
) -> Dict[str, Any]:
    settings = [args.setting] if args.setting else list(POPE_SETTINGS)
    # POPE answers are a single word, so the benchmark budget wins over whatever
    # the YAML asked for. The resolved decoding config is updated to match, so
    # provenance records the budget that actually ran rather than the one that
    # was configured and then silently overridden by the evaluator.
    budget = benchmark_budget(args, "pope")
    configs["decoding"].max_new_tokens = budget
    evaluator = POPEEvaluator(
        generate_fn=decoder.generate,
        method=args.method,
        max_new_tokens=budget,
    )
    output_dir = Path(args.output_dir)
    results = evaluate_pope_settings(
        evaluator=evaluator,
        pope_root=Path(args.data_root),
        settings=settings,
        image_root=Path(args.image_root) if args.image_root else None,
        max_samples=args.max_samples,
        output_dir=output_dir,
    )
    return {
        "pope": {k: v.to_dict() for k, v in results.items()},
    }


def run_mme(
    args: argparse.Namespace,
    decoder: VisualGuardDecoder,
    configs: Dict[str, Any],
) -> Dict[str, Any]:
    budget = benchmark_budget(args, "mme")
    configs["decoding"].max_new_tokens = budget
    evaluator = MMEEvaluator(
        generate_fn=decoder.generate,
        method=args.method,
        max_new_tokens=budget,
    )
    result = evaluator.evaluate(
        mme_root=Path(args.data_root),
        subtasks=args.mme_subtask,
        max_samples_per_subtask=args.max_samples,
        output_dir=Path(args.output_dir),
    )
    return {"mme": result.to_dict()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.method in BASELINE_METHODS and (args.alpha or args.beta or args.gamma or args.lam):
        logger.warning(
            "--method %s ignores alpha/beta/gamma/lambda; those flags only "
            "affect the VisualGuard variants.", args.method,
        )

    repro = ReproducibilityConfig(seed=args.seed, deterministic=not args.no_deterministic)
    apply_reproducibility(repro)

    try:
        configs = build_experiment_configs(args)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Method: %s", args.method)
    logger.info("LVLM: %s", configs["lvlm"].model_name)
    logger.info(
        "Evidence weights: alpha=%s beta=%s gamma=%s lambda=%s threshold=%s",
        configs["evidence"].alpha, configs["evidence"].beta,
        configs["evidence"].gamma, configs["evidence"].lam,
        configs["evidence"].threshold,
    )

    decoder = VisualGuardDecoder(
        lvlm_config=configs["lvlm"],
        decoding_config=configs["decoding"],
        evidence_config=configs["evidence"],
        grounding_config=configs["grounding"],
        method=args.method,
    )

    # Validate the dataset before spending minutes loading the checkpoint.
    try:
        precheck_dataset(args)
    except (POPEDataError, MMEDataError, SystemExit) as exc:
        if isinstance(exc, SystemExit):
            logger.error("%s", exc)
        else:
            logger.error("Dataset problem: %s", exc)
        return 5

    run_start = time.perf_counter()
    reset_peak_memory()
    try:
        decoder.load()
    except GroundingUnavailable as exc:
        logger.error(
            "Grounding backend unavailable: %s\n"
            "Fix by installing the backend, or rerun with "
            "--grounding-backend none (and --gamma 0) to disable region evidence.",
            exc,
        )
        return 3
    except Exception as exc:
        logger.error("Failed to load model/evidence backends: %s", exc)
        return 4

    try:
        if args.benchmark == "pope":
            payload = run_pope(args, decoder, configs)
        else:
            payload = run_mme(args, decoder, configs)
    except (POPEDataError, MMEDataError) as exc:
        logger.error("Dataset problem: %s", exc)
        return 5
    except KeyboardInterrupt:  # pragma: no cover
        logger.warning("Interrupted.")
        return 130
    finally:
        if decoder._grounding_backend is not None:
            decoder._grounding_backend.close()

    total_runtime = time.perf_counter() - run_start

    run_name = args.run_name or (
        f"{args.benchmark}_{args.method}_{int(time.time())}"
    )
    result_path = output_dir / f"{run_name}.json"

    results: Dict[str, Any] = {
        **payload,
        "efficiency": {
            "total_runtime_s": total_runtime,
            "peak_gpu_memory_mb": peak_memory_mb(),
        },
        "provenance": collect_metadata(
            seed=args.seed,
            extra={
                "command": " ".join([Path(sys.argv[0]).name] + list(argv or sys.argv[1:])),
                "method": args.method,
                # Recorded so the results table can key rows on the *run*, not on
                # the method. Several distinct ablations share a method name
                # (attention+region and the lambda sweep are all "visualguard").
                "run_name": run_name,
                "model": configs["lvlm"].model_name,
                "config_file": str(configs["yaml_path"]) if configs["yaml_path"] else None,
                "max_samples": args.max_samples,
                "note": (
                    "max_samples was set: these numbers are NOT comparable to a "
                    "full-dataset run"
                ) if args.max_samples else None,
            },
        ),
        "resolved_config": {
            "lvlm": to_dict(configs["lvlm"]),
            "decoding": to_dict(configs["decoding"]),
            "evidence": to_dict(configs["evidence"]),
            "grounding": to_dict(configs["grounding"]),
        },
    }
    save_json(results, result_path)

    logger.info("Wrote results to %s", result_path)
    _print_summary(args, results)
    return 0


def _print_summary(args: argparse.Namespace, results: Dict[str, Any]) -> None:
    """Print the numbers that actually ran. No placeholders."""
    print("\n" + "=" * 68)
    print(f"VisualGuard | benchmark={args.benchmark} method={args.method} "
          f"model={args.model}")
    print("=" * 68)
    if "pope" in results:
        header = f"{'setting':<14}{'n':>6}{'acc':>9}{'F1':>9}{'prec':>9}{'rec':>9}{'halluc':>9}"
        print(header)
        print("-" * len(header))
        for setting, block in sorted(results["pope"].items()):
            m = block["metrics"]
            print(
                f"{setting:<14}{m['n']:>6}{m['accuracy']:>9.4f}{m['f1']:>9.4f}"
                f"{m['precision']:>9.4f}{m['recall']:>9.4f}{m['hallucination_rate']:>9.4f}"
            )
            for note in block.get("notes", []):
                print(f"  note: {note}")
    if "mme" in results:
        totals = results["mme"]["category_totals"]
        print(f"MME total: {totals.get('total_score', 0.0):.2f} "
              f"(perception {totals.get('perception_score', 0.0):.2f}, "
              f"cognition {totals.get('cognition_score', 0.0):.2f})")
    eff = results["efficiency"]
    print(f"runtime {eff['total_runtime_s']:.1f}s | "
          f"peak GPU mem {eff['peak_gpu_memory_mb']}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
