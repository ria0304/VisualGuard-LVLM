#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run CHAIR (open-ended captioning hallucination) for any method.

Reuses the main runner's CLI and config resolution, so every flag from
``python -m src.run`` works here (``--method``, ``--alpha``, ``--lambda``,
``--config``, ``--model``, ``--dtype`` ...). Adds ``--method vcd`` and the CHAIR
data flags.

Examples
--------
Greedy baseline, 500 images::

    python scripts/run_chair.py --method baseline \\
        --ann-dir /data/coco/annotations --image-root /data/coco/val2014 \\
        --synonyms /data/chair/synonyms.txt --num-images 500

VisualGuard (same images, same seed)::

    python scripts/run_chair.py --method visualguard --config visualguard \\
        --ann-dir ... --image-root ... --synonyms ...

VCD baseline::

    python scripts/run_chair.py --method vcd --ann-dir ... --image-root ... --synonyms ...

All methods see the same sampled image subset (``--seed``), the same prompt and
the same checkpoint, so rows are directly comparable.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.chair_eval import CHAIR_PROMPT, CHAIRDataError, CHAIREvaluator  # noqa: E402
from src.model.grounding import GroundingUnavailable  # noqa: E402
from src.model.vcd import VCDConfig, VCDDecoder  # noqa: E402
from src.model.visual_guard_decoder import BASELINE_METHODS, VisualGuardDecoder  # noqa: E402
from src.run import build_experiment_configs, build_parser  # noqa: E402
from src.utils.config import ConfigError, save_json, to_dict  # noqa: E402
from src.utils.reproducibility import (  # noqa: E402
    ReproducibilityConfig,
    apply_reproducibility,
    collect_metadata,
    peak_memory_mb,
    reset_peak_memory,
)

logger = logging.getLogger("visualguard.chair")

DEFAULT_CHAIR_TOKENS = 512


def build_chair_parser():
    parser = build_parser()
    parser.prog = "python scripts/run_chair.py"

    # Allow --method vcd on top of the main runner's choices.
    for action in parser._actions:
        if action.dest == "method":
            action.choices = sorted(set(action.choices) | {"vcd"})

    g = parser.add_argument_group("chair")
    g.add_argument("--ann-dir", required=True,
                   help="Directory with instances_<split>.json and captions_<split>.json")
    g.add_argument("--synonyms", required=True,
                   help="Path to the official CHAIR synonyms.txt")
    g.add_argument("--coco-split", default="val2014", help="val2014 | val2017")
    g.add_argument("--num-images", type=int, default=500)
    g.add_argument("--prompt", default=CHAIR_PROMPT)

    g = parser.add_argument_group("vcd")
    g.add_argument("--vcd-alpha", type=float, default=1.0)
    g.add_argument("--vcd-beta", type=float, default=0.1)
    g.add_argument("--vcd-noise-step", type=int, default=500)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_chair_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if not args.image_root:
        logger.error("--image-root is required for CHAIR")
        return 2

    apply_reproducibility(
        ReproducibilityConfig(seed=args.seed, deterministic=not args.no_deterministic)
    )

    requested_method = args.method
    is_vcd = requested_method == "vcd"
    if is_vcd:
        # VCD needs only the LVLM: resolve configs as an unmodified baseline so
        # no evidence scorer or detector is built.
        args.method = "baseline"

    try:
        configs = build_experiment_configs(args)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 2

    budget = args.max_new_tokens or DEFAULT_CHAIR_TOKENS
    configs["decoding"].max_new_tokens = budget

    decoder = VisualGuardDecoder(
        lvlm_config=configs["lvlm"],
        decoding_config=configs["decoding"],
        evidence_config=configs["evidence"],
        grounding_config=configs["grounding"],
        method=args.method,
    )

    run_start = time.perf_counter()
    reset_peak_memory()
    try:
        decoder.load()
    except GroundingUnavailable as exc:
        logger.error("Grounding backend unavailable: %s", exc)
        return 3
    except Exception as exc:
        logger.error("Failed to load model/evidence backends: %s", exc)
        return 4

    if is_vcd:
        vcd_cfg = VCDConfig(
            alpha=args.vcd_alpha, beta=args.vcd_beta,
            noise_step=args.vcd_noise_step, seed=args.seed,
        )
        generate_fn = VCDDecoder(decoder.backend, configs["decoding"], vcd_cfg).generate
    else:
        generate_fn = decoder.generate

    evaluator = CHAIREvaluator(
        generate_fn, method=requested_method, prompt=args.prompt, max_new_tokens=budget
    )
    try:
        result = evaluator.evaluate(
            ann_dir=Path(args.ann_dir),
            image_root=Path(args.image_root),
            synonyms_path=Path(args.synonyms),
            split=args.coco_split,
            num_images=args.num_images,
            seed=args.seed,
            output_dir=Path(args.output_dir),
        )
    except CHAIRDataError as exc:
        logger.error("Dataset problem: %s", exc)
        return 5
    finally:
        if decoder._grounding_backend is not None:
            decoder._grounding_backend.close()

    run_name = args.run_name or f"chair_{requested_method}_{int(time.time())}"
    out_path = Path(args.output_dir) / f"{run_name}.json"
    payload: Dict[str, Any] = {
        "chair": result.to_dict(),
        "efficiency": {
            "total_runtime_s": time.perf_counter() - run_start,
            "peak_gpu_memory_mb": peak_memory_mb(),
        },
        "provenance": collect_metadata(
            seed=args.seed,
            extra={
                "command": " ".join([Path(sys.argv[0]).name] + list(argv or sys.argv[1:])),
                "method": requested_method,
                "run_name": run_name,
                "model": configs["lvlm"].model_name,
                "num_images": args.num_images,
                "coco_split": args.coco_split,
                "prompt": args.prompt,
                "max_new_tokens": budget,
            },
        ),
        "resolved_config": {
            "lvlm": to_dict(configs["lvlm"]),
            "decoding": to_dict(configs["decoding"]),
            "evidence": to_dict(configs["evidence"]),
            "grounding": to_dict(configs["grounding"]),
            **({"vcd": to_dict(vcd_cfg)} if is_vcd else {}),
        },
    }
    save_json(payload, out_path)
    logger.info("Wrote %s", out_path)
    print(json.dumps(result.metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
