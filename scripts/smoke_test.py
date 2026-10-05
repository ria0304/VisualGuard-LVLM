#!/usr/bin/env python
"""
Plumbing smoke test for the VisualGuard pipeline.

What this verifies
------------------
* a *real* Hugging Face LLaVA checkpoint loads through ``LLaVAHFBackend``
  (processor, tokenizer, weights, device placement),
* the image processor turns a real image file into ``pixel_values``,
* ``initial_step`` / ``next_step`` return logits, attentions and a KV cache,
* attention evidence extracts a bounded, non-degenerate value from real
  attention tensors,
* the VisualGuard loop runs autoregressively and its ``_select`` reports
  whether the evidence penalty changed the chosen token,
* the baseline path calls ``model.generate`` and returns real decoded text.

What this does NOT verify
-------------------------
Anything about hallucination reduction. The default checkpoint is a small but
*real, coherently configured* LLaVA-family LVLM (0.5B), used to keep the
download to ~1.7 GB so the pipeline can be exercised on CPU.
**Numbers from this script are meaningless and must never be reported**, and
small models hallucinate far more than the 7B model used for experiments. For
real experiments use the intended checkpoint, e.g.::

    python -m src.run --benchmark pope --method visualguard \\
        --model llava-hf/llava-1.5-7b-hf

Usage::

    python scripts/smoke_test.py                       # small real LVLM
    python scripts/smoke_test.py --model <hf-id>       # any other checkpoint
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.model.llava_backend import LLaVAHFBackend, LVLMConfig  # noqa: E402
from src.model.visual_evidence import EvidenceConfig, VisualEvidenceScorer  # noqa: E402
from src.model.visual_guard_decoder import (  # noqa: E402
    DecodingConfig,
    VisualGuardDecoder,
    make_candidates,
)

#: Small, real, coherently-configured LLaVA-family LVLM (~1.7GB). Chosen so the
#: pipeline can be exercised on CPU. NOT the model used for experiments.
SMOKE_TEST_MODEL = "llava-hf/llava-interleave-qwen-0.5b-hf"

PASS, FAIL = "PASS", "FAIL"
_results: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    _results.append((PASS if condition else FAIL, name, detail))
    marker = "  ok " if condition else " FAIL"
    print(f"[{marker}] {name}" + (f"  ({detail})" if detail else ""))


def make_test_image(path: Path) -> Image.Image:
    """Create a small real image file on disk (a photo-like gradient + shapes)."""
    import numpy as np

    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(array, mode="RGB")
    img.save(path)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=SMOKE_TEST_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--skip-clip", action="store_true",
                        help="Skip the CLIP semantic channel (avoids a ~600MB download).")
    args = parser.parse_args()

    print("=" * 70)
    print("VisualGuard plumbing smoke test")
    print(f"model: {args.model}")
    print(f"NOTE: the default checkpoint is a small real LVLM ({args.model}).")
    print("      This validates plumbing only. It says NOTHING about accuracy,")
    print("      and small models hallucinate more than the 7B experiment model.")
    print("=" * 70)

    tmpdir = Path(tempfile.mkdtemp(prefix="visualguard_smoke_"))
    image_path = tmpdir / "image.jpg"
    make_test_image(image_path)
    check("test image written to disk", image_path.is_file(), str(image_path))

    # ---- 1. load a real HF checkpoint -------------------------------
    backend = LLaVAHFBackend(
        LVLMConfig(model_name=args.model, device=args.device, dtype="float32")
    )
    try:
        backend.load()
    except Exception as exc:  # pragma: no cover
        print(f"[ FAIL] backend.load() raised: {exc}")
        return 1
    check("real HF checkpoint loaded", backend.model is not None)
    check("tokenizer available", backend.tokenizer is not None)
    check("image processor available", backend.image_processor is not None)
    check("model on expected device", backend.device.type == torch.device(args.device).type,
          str(backend.device))

    # ---- 2. encoding ------------------------------------------------
    inputs = backend.prepare_inputs("Is there a dog in the image?", image_path)
    check("prompt + image encoded together", "input_ids" in inputs)
    n_tokens = inputs["input_ids"].shape[-1]
    check("prompt encoded", n_tokens > 0, f"tokens={n_tokens}")

    pixel_values = inputs.get("pixel_values")
    check("pixel_values present", pixel_values is not None,
          f"shape={None if pixel_values is None else tuple(pixel_values.shape)}")
    # The processor must expand <image> into one placeholder per vision patch,
    # otherwise the model rejects the call (token/feature mismatch).
    span = backend.image_token_span(inputs["input_ids"], pixel_values)
    check("image token span resolved", span is not None and span[1] > span[0], f"span={span}")
    check("image span fits the sequence", span is None or span[1] <= n_tokens,
          f"span={span} n_tokens={n_tokens}")

    # ---- 3. forward steps -------------------------------------------
    step = backend.initial_step(inputs["input_ids"], pixel_values, output_attentions=True)
    check("initial_step returned logits", step.logits.shape[0] == 1,
          f"vocab={step.logits.shape[-1]}")
    check("initial_step returned a KV cache", step.past_key_values is not None)
    check("attentions returned", step.attentions is not None,
          f"layers={0 if step.attentions is None else len(step.attentions)}")

    if step.attentions is not None and step.past_key_values is not None:
        ev_cfg = EvidenceConfig(alpha=1.0, beta=0.0, gamma=0.0)
        scorer = VisualEvidenceScorer(config=ev_cfg, backend=backend)
        span = step.image_token_span
        attn_value = scorer.attention.state_image_attention(step.attentions, span)
        check("attention evidence is bounded", 0.0 <= attn_value <= 1.0,
              f"value={attn_value:.4f} span={span}")
        check("attention evidence is non-degenerate", attn_value > 0.0,
              f"value={attn_value:.4f}; 0.0 means the image-token span was wrong")

        # cached step
        top = torch.argmax(step.logits, dim=-1, keepdim=True)
        step2 = backend.next_step(top, step.past_key_values, output_attentions=True)
        check("cached next_step returned logits", step2.logits.shape[0] == 1)

        # candidates + intervention audit
        cands = make_candidates(step.logits, backend.tokenizer, [], top_k=5)
        check("candidate set built", len(cands) == 5)
        check("candidates are content-flagged",
              any(c.is_content for c in cands))
        bundle = scorer.score_candidates(image_path, cands, state_image_attention=attn_value)
        check("VES values bounded", all(0.0 <= c.ves <= 1.0 for c in bundle.candidates))
        check("penalties non-negative", all(c.penalty >= 0.0 for c in bundle.candidates))
        check("scorer reported whether image features were bound",
              isinstance(scorer.candidate_embeddings_available, bool),
              f"available={scorer.candidate_embeddings_available}")

    # ---- 4. baseline path uses model.generate ----------------------
    base = VisualGuardDecoder(
        lvlm_config=LVLMConfig(model_name=args.model, device=args.device, dtype="float32"),
        decoding_config=DecodingConfig(max_new_tokens=args.max_new_tokens),
        method="greedy",
    )
    base.backend = backend
    out = base.generate(image_path, "Describe the image.")
    check("baseline generate() returned text", isinstance(out.text, str) and len(out.text) > 0,
          f"text={out.text!r}")
    check("baseline produced no interventions", out.interventions == 0)
    check("baseline latency recorded", out.latency_s > 0)

    # ---- 5. full VisualGuard decoding loop (attention channel only) ----
    # This is the central claim: the intervention runs inside the real
    # autoregressive loop and can change which token is selected.
    #
    # ``lam`` is deliberately huge and ``threshold`` 1.0 so any evidence deficit
    # produces a penalty. A soft configuration would let the loop pass with zero
    # interventions and prove nothing, which is exactly how the attention
    # channel's no-op bug went unnoticed.
    try:
        vg = VisualGuardDecoder(
            lvlm_config=LVLMConfig(
                model_name=args.model, device=args.device, dtype="float32"
            ),
            decoding_config=DecodingConfig(max_new_tokens=args.max_new_tokens, top_k=5),
            evidence_config=EvidenceConfig(
                alpha=1.0, beta=0.0, gamma=0.0, lam=50.0, threshold=1.0,
                attention_candidate_mix=1.0,
                penalise_function_words=False,
            ),
            method="visualguard",
        )
        vg.backend = backend
        vg.evidence_config = EvidenceConfig(
            alpha=1.0, beta=0.0, gamma=0.0, lam=50.0, threshold=1.0,
            attention_candidate_mix=1.0, penalise_function_words=False,
        )
        vg.scorer = VisualEvidenceScorer(config=vg.evidence_config, backend=backend)
        # Bind embeddings exactly as load() does. Skipping this is what let the
        # channel silently degenerate into a constant.
        vg._bound_pixel_values = None
        vg.ensure_image_bindings(pixel_values)

        check("image features bound for candidate-level attention",
              vg.scorer.candidate_embeddings_available,
              "without these the attention channel cannot rank candidates")

        out = vg.generate(image_path, "Describe the image.")
        check("VisualGuard loop generated text", isinstance(out.text, str),
              f"text={out.text!r}")
        check("VisualGuard produced per-step candidates", out.num_generated_tokens > 0,
              f"tokens={out.num_generated_tokens}")
        check("VisualGuard recorded a latency", out.per_token_latency_s > 0,
              f"{out.per_token_latency_s*1000:.1f} ms/token")

        # --- non-vacuous assertions on the intervention itself ---
        check("VisualGuard actually intervened", out.interventions > 0,
              f"interventions={out.interventions}; 0 means the evidence "
              "penalty never changed the selected token")
        check("intervention detail is auditable", len(out.interventions_detail) > 0,
              f"recorded={len(out.interventions_detail)}")
        if out.interventions_detail:
            d = out.interventions_detail[0]
            check("intervention recorded a chosen token with VES",
                  "chosen_ves" in d and 0.0 <= d["chosen_ves"] <= 1.0, str(d))

        # The whole point: VisualGuard must not reproduce the baseline verbatim.
        # Same loop, same channel, but lam=0 -> no penalty at all, so this
        # decodes greedily while `vg` decodes with evidence applied.
        vg_loose = VisualGuardDecoder(
            lvlm_config=LVLMConfig(
                model_name=args.model, device=args.device, dtype="float32"
            ),
            decoding_config=DecodingConfig(max_new_tokens=args.max_new_tokens, top_k=5),
            evidence_config=EvidenceConfig(
                alpha=1.0, beta=0.0, gamma=0.0, lam=0.0, threshold=1.0,
                attention_candidate_mix=1.0, penalise_function_words=False,
            ),
            method="visualguard",
        )
        vg_loose.backend = backend
        vg_loose.scorer = VisualEvidenceScorer(
            config=vg_loose.evidence_config, backend=backend)
        vg_loose._bound_pixel_values = None
        vg_loose.ensure_image_bindings(pixel_values)
        out_loose = vg_loose.generate(image_path, "Describe the image.")
        check("unpenalised control applies no penalty", out_loose.interventions == 0,
              f"interventions={out_loose.interventions}")
        check("penalised decoding differs from unpenalised decoding",
              out_loose.token_ids != out.token_ids,
              f"penalised={out.token_ids[:8]} unpenalised={out_loose.token_ids[:8]}")
    except Exception as exc:  # pragma: no cover
        check("VisualGuard decoding loop", False, f"{type(exc).__name__}: {exc}")

    # ---- 6. optional CLIP channel ----------------------------------
    if not args.skip_clip:
        try:
            from src.model.visual_evidence import SemanticEvidence

            sem = SemanticEvidence(EvidenceConfig(clip_model_name="openai/clip-vit-base-patch32"))
            sem.encode_image(image_path)
            sims = sem.raw_similarities(image_path, ["dog", "cat"])
            check("CLIP produced similarities", len(sims) == 2 and all(
                isinstance(s, float) for s in sims), f"sims={[round(s, 3) for s in sims]}")
        except Exception as exc:  # pragma: no cover
            check("CLIP semantic channel", False, f"{type(exc).__name__}: {exc}")

    # ---- summary ----------------------------------------------------
    failures = [r for r in _results if r[0] == FAIL]
    print("=" * 70)
    print(f"{len(_results) - len(failures)}/{len(_results)} checks passed")
    if failures:
        print("FAILED checks:")
        for _, name, detail in failures:
            print(f"  - {name} {detail}")
    print("Reminder: this smoke test validates plumbing, not method quality.")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
