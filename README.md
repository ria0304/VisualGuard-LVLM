# VisualGuard: Dynamic Visual Evidence-Guided Decoding for Hallucination Reduction in LVLMs

> **Status: implementation complete, results not yet available.**
> This repository implements the proposed method and a reproducible evaluation
> framework. **No experimental results are reported here, because no full
> benchmark run has been performed.** Every results table is generated from
> actual result JSONs on disk; unmeasured cells are rendered as `not run` rather
> than as zeros or estimates.

---

## 1. Overview

Large Vision-Language Models hallucinate: they describe objects that are absent
from the image. The dominant failure is *confident* hallucination — the model
assigns high probability to a token that the image does not support, and greedy
decoding commits to it.

VisualGuard is a **decoding-time intervention**. It leaves the model weights
untouched and instead modifies the logit of each candidate token at every
autoregressive step, based on how well the image supports that candidate:

```
Image
  -> pretrained LVLM (unchanged weights)
  -> candidate token set (top-k by logit)
  -> estimate visual evidence VES(t) per candidate
  -> penalty(t) = lambda * deficit(VES(t))
  -> logit_t -= penalty(t)
  -> select next token
  -> repeat autoregressively
```

Because the intervention is training-free and confined to decoding, every
baseline and every variant runs against **the same checkpoint with the same
prompts**, so measured differences are attributable to the decoding rule alone.

## 2. Motivation

- Post-hoc editing of a completed answer ("generate, then check, then rewrite")
  is expensive, can change the answer's meaning, and cannot prevent a
  hallucinated token from entering the context.
- Fine-tuning-based mitigations change the model, which makes comparison against
  the unmodified model unfair and costs training compute.
- Existing attention-based analyses conflate *where the model looks* with *what
  the model is justified in saying*. Attention is a weak, contested explanation
  signal, so VisualGuard does not rely on it alone.

## 3. Research hypothesis

**Hypothesis.** For a candidate continuation `t`, the degree to which the image
corroborates `t` is measurable from complementary, partially independent signals
(LVLM attention over image tokens, image-text semantic similarity, and
open-vocabulary region detections). Penalising candidates whose support falls
below a threshold should reduce object hallucination, and the *cost* in general
capability is governed by how narrowly the penalty is targeted.

This is a hypothesis, not a finding. It has not been tested on any benchmark in
this repository. The hypothesis being falsifiable matters more than it sounding
plausible: the ablation grid in §12 is designed so that it can fail.

## 4. Method

### 4.1 Visual evidence score

All channels are mapped into `[0, 1]` and combined as a weighted mean:

```
VES(t) = alpha * AttentionEvidence(t)
       + beta  * SemanticEvidence(t)
       + gamma * RegionEvidence(t)

         ------------------------------------------------------------
         alpha + beta + gamma
```

Dividing by the sum of *active* weights keeps `VES` in `[0, 1]` however many
channels are enabled, so an ablation with one channel is directly comparable to
the full method.

**AttentionEvidence.** Self-attention rows belong to sequence *positions*, not
to candidate tokens, so attention alone cannot rank candidates within a step.
This is stated plainly rather than papered over. The channel therefore has two
components, mixed by `attention_candidate_mix` (`w`):

```
AttentionEvidence(t) = (1-w) * image_attention  +  w  * embed_cos(t)

  image_attention : attention mass the current query position places on the
                    image-token span, aggregated over a configurable fraction of
                    layers and heads (mean / max / median), normalised by total
                    attention mass
  embed_cos(t)    : cosine similarity between the candidate token's embedding
                    and the mean image-token embedding in LM hidden space
```

`w = 0` gives a purely state-level signal; `w = 1` gives a purely
candidate-level one. The default `w = 0.5`.

**SemanticEvidence.** CLIP image-text similarity between the image and the word
fragment being formed (`image ↔ "dog"` vs `image ↔ "cat"`), normalised *within
the candidate set* (min-max by default) so the value is scale-free. Candidates
are wrapped in a minimal caption template, because CLIP was trained on captions
and scores bare nouns inconsistently.

**RegionEvidence.** Best detector confidence for an open-vocabulary detection
whose label supports the candidate phrase. Requires a grounding backend; when
disabled it contributes exactly `0.0` and the run records
`region_evidence_disabled` as the reason. There is no synthetic-detection
fallback.

### 4.2 Hallucination penalty

```
penalty(t) = lambda * max(0, threshold - VES(t)) / threshold
logit'_t   = logit_t - penalty(t)
```

Two deliberate design choices:

- **The penalty is thresholded, not uniform.** `penalty = 0` whenever
  `VES(t) >= threshold`. Penalising every token equally would push the whole
  distribution and damage fluency for no hallucination benefit.
- **Only content-bearing tokens are penalised.** Articles, auxiliaries,
  prepositions and punctuation are flagged by a conservative lexical filter and
  exempt by default (`penalise_function_words: false`). Rewriting function words
  is the fastest way to break generation quality.

### 4.3 Baselines

All share one LVLM, one prompt template, one seed:

| method | decoding rule |
| --- | --- |
| `greedy` / `baseline` | unmodified greedy |
| `sampling` | unmodified sampling at temperature `T` |
| `beam` | unmodified beam search |
| `attention` | VisualGuard, `beta = gamma = 0` |
| `semantic` | VisualGuard, `alpha = gamma = 0` |
| `region` | VisualGuard, `alpha = beta = 0` (needs a grounding backend) |
| `unidirectional` | attention + semantic |
| `visualguard` | all channels |

Baselines run through `model.generate`, i.e. the standard well-understood
reference implementation, and never build an evidence scorer — they do not pay
the evidence cost and cannot be modified.

## 5. Architecture

```
configs/*.yaml ──┐
CLI flags ───────┤
                 v
           src/run.py  (experiment runner, provenance capture)
                 |
      +----------+-----------+
      |                      |
      v                      v
LLaVAHFBackend        VisualEvidenceScorer
  (src/model/          (src/model/visual_evidence.py)
   llava_backend.py)     |- AttentionEvidence   (LM attention + embeddings)
      |                  |- SemanticEvidence    (CLIP)
      |                  `- RegionEvidence      (Grounding DINO, optional)
      |                             |
      +------------+----------------+
                   v
        VisualGuardDecoder  (src/model/visual_guard_decoder.py)
          manual autoregressive loop, KV-cached,
          evidence scored per step, logits modified before selection
                   |
      +------------+------------+
      v                         v
POPEEvaluator              MMEEvaluator
(src/evaluation/)          (src/evaluation/)
      |                         |
      `--> src/evaluation/metrics.py  (pure, unit-tested)
                   |
                   v
        results/<run-name>.json  + per-sample predictions .jsonl
```

```
src/
├── run.py                     experiment runner / CLI
├── model/
│   ├── llava_backend.py       real pretrained LVLM loading + cached decode steps
│   ├── visual_evidence.py     VES channels, combination, penalty
│   ├── visual_guard_decoder.py decoding-time intervention + baselines
│   └── grounding.py           optional Grounding DINO backend
├── data/
│   ├── pope.py                POPE loading, strict label + image validation
│   ├── mme.py                 MME loading
│   └── coco.py                MS COCO 2017 helpers
├── evaluation/
│   ├── pope_eval.py           real POPE evaluation
│   ├── mme_eval.py            real MME evaluation
│   └── metrics.py             accuracy / P / R / F1 / hallucination rate / MME
└── utils/
    ├── config.py              YAML + CLI config resolution
    └── reproducibility.py     seeding, versions, device, git commit
```

## 6. Installation

```bash
git clone <this-repo>
cd VisualGuard-LVLM

python -m venv .venv && source .venv/bin/activate   # recommended
pip install --upgrade pip

pip install -r requirements.txt
```

Optional extras:

```bash
pip install bitsandbytes                 # for --quantization 4bit / 8bit (CUDA only)
pip install 'transformers>=4.40'        # already required; Grounding DINO uses it
# upstream detector, only if you prefer it over the transformers backend:
pip install groundingdino
```

Verify the install (no download required):

```bash
python -m pytest tests -q
```

## 7. Dataset preparation

**No datasets or checkpoints are committed to this repository.** Obtain each
source from its official release.

### POPE (primary hallucination benchmark)

From the official POPE release (Li et al., 2023). Expected layout:

```
<data-root>/
    coco_pope_random.jsonl
    coco_pope_popular.jsonl
    coco_pope_adversarial.jsonl
```

Each JSONL row has `question_id`, `image`, `text`, `label` (`yes`/`no`).
`--image-root` must point at the COCO images the `image` field references
(typically `val2017`).

The three settings sample the absent object differently — uniformly at random
(`random`), biased toward frequently co-occurring objects (`popular`), and by
ground-truth co-occurrence maximum (`adversarial`). All three are required for
a complete table.

### MS COCO 2017 (development / image source)

COCO is the image source for POPE and an optional set for qualitative checks.
It is **not** a hallucination benchmark, and nothing here trains on it.

```bash
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip && unzip annotations_trainval2017.zip
```

### MME (capability benchmark)

From the official MME release (Yin et al., 2023). Expected layout:

```
<data-root>/<subtask>/<subtask>.jsonl
<data-root>/<subtask>/images/*.jpg
```

with the 10 Perception and 4 Cognition subtasks.

### Optional: HallusionBench

Not yet wired into the runner. No loader or evaluator is provided, and none is
claimed.

## 8. Running baseline

```bash
python -m src.run \
    --benchmark pope \
    --method baseline \
    --model llava-hf/llava-1.5-7b-hf \
    --device cuda \
    --dtype float16 \
    --data-root /path/to/POPE \
    --image-root /path/to/coco/val2017 \
    --output-dir results
```

Other baselines:

```bash
python -m src.run --benchmark pope --method sampling --temperature 1.0 ...
python -m src.run --benchmark pope --method beam --num-beams 4 ...
```

## 9. Running VisualGuard

```bash
python -m src.run \
    --benchmark pope \
    --method visualguard \
    --config visualguard \
    --alpha 1.0 --beta 1.0 --gamma 0.0 \
    --lambda 0.5 --threshold 0.35 --top-k 50 \
    --model llava-hf/llava-1.5-7b-hf \
    --data-root /path/to/POPE \
    --image-root /path/to/coco/val2017
```

With region evidence (requires a grounding backend):

```bash
python -m src.run --benchmark pope --method visualguard \
    --gamma 0.5 --grounding-backend hf_grounding_dino ...
```

If the backend is unavailable the run **fails with install guidance** rather
than silently degrading. `--grounding-backend none` disables the channel
explicitly, and the run records that fact.

## 10. POPE evaluation

Computed by `src/evaluation/metrics.py`:

- **Accuracy**, **Precision**, **Recall**, **F1** (positive class = "yes")
- **Yes ratio** / **No ratio** — exposes degenerate all-yes answering
- **Hallucination rate** — `FP / (all "no" items)`, isolating the targeted
  failure mode of asserting a nonexistent object

Per-sample predictions are written to
`results/pope_<setting>_<method>_predictions.jsonl` so any run can be re-scored
or audited without re-running the model.

Two honesty guards: an answer that cannot be parsed as yes/no is counted as an
error and counted separately as `unparseable` (never coerced to a class), and
runs whose yes-ratio exceeds 0.95 are flagged, because such a model can look
good on recall while being useless.

## 11. MME evaluation

Implements the official MME protocol in full — no external evaluator service is
required and no score is simulated:

```
accuracy      = correct questions / all questions
accuracy_plus = images where BOTH questions are correct / all images
score         = 100 * (accuracy + accuracy_plus) / 2     # per subtask, capped at 200
category      = sum of subtask scores                   # perception / cognition
total         = sum of present categories
```

14 subtasks: Perception (existence, count, position, color, posters, celebrity,
scene, landmark, artwork, OCR) and Cognition (commonsense_reasoning,
numerical_calculation, text_translation, code_reasoning).

**Scope note.** This reproduces the classic 14-subtask yes/no MME benchmark.
MME variants that use a GPT-based judge are a different protocol and are *not*
reproduced or approximated here.

## 12. Ablation studies

```bash
python scripts/run_ablations.py \
    --benchmark pope \
    --data-root /path/to/POPE \
    --image-root /path/to/coco/val2017 \
    --model llava-hf/llava-1.5-7b-hf \
    --device cuda \
    --lambda-sweep 0.25,0.5,1.0 \
    --grounding-backend hf_grounding_dino
```

Rows: A baseline, B attention only, C semantic only, D region only,
E attention+semantic, F attention+region, G full, plus the lambda sweep.
Rows D and F are skipped with a warning if no grounding backend is enabled,
because without a detector they would measure nothing.

Build the table:

```bash
python scripts/make_results_table.py --results-dir results
```

producing

```
| Method | POPE Random F1 | POPE Popular F1 | POPE Adv F1 | Hallucination | POPE worst-setting F1 | MME total |
```

`POPE worst-setting F1` is included on purpose: reporting only the best setting
would hide the method's weak cases. Missing runs render as `not run`, and the
script warns when rows used different models, seeds, or `--max-samples`, since
such rows are not comparable.

## 13. Reproducibility

Every result JSON embeds a `provenance` block: UTC timestamp, seed, model id,
exact command, resolved config, platform, CPU count, GPU name and memory,
library versions (Python, torch, transformers, accelerate, bitsandbytes,
numpy, PIL), git commit and dirty-tree flag, plus `total_runtime_s` and peak GPU
memory.

```bash
python -m src.run ... --seed 42
```

`--no-deterministic` relaxes determinism for speed. Note the honest limit:
PyTorch does not guarantee bit-identical results across different GPUs, CUDA
versions, or kernels, so seeding reduces variance rather than eliminating it.

## 14. Limitations

**Method-level**

- Attention is not a faithful explanation of model behaviour, and the
  implementation does not pretend otherwise. `image_attention` is a
  *state-level* signal; candidate ranking genuinely comes from the embedding
  and semantic channels. The split is exposed as
  `attention_candidate_mix` rather than hidden.
- Semantic evidence is applied to the *word fragment being formed*, so evidence
  for a multi-token word is only available mid-word. This is why the CLIP call
  dominates the per-step cost.
- Grounding-DINO detections are open-vocabulary and error-prone; a missed
  detection is indistinguishable from an absent object, which pushes the
  penalty the wrong way.
- POPE answers are a single token, so attention evidence for POPE rests almost
  entirely on the prefill row. Conclusions about POPE do not automatically
  transfer to long-form captioning.
- CLIP is itself trained on web image-text data and carries its own biases; it
  is a noisy judge, not ground truth.
- Region evidence is off by default because the detector is the dominant cost
  and the most fragile dependency.

**Engineering**

- The LVLM backend is LLaVA-family only. `LVLMBackend` defines the extension
  point, but no second LVLM is implemented, so the pluggability claim is
  structural rather than demonstrated.
- VisualGuard is far slower than plain greedy decoding: one forward pass per
  step plus a CLIP forward per step. See `efficiency` in any result JSON.
- MME capability cost is measured but no trade-off has been quantified yet,
  because no experiment has been run.

## 15. Citation

The method in this repository is not yet published, so there is no citation to
give. Please cite the benchmarks and prior work it builds on:

```bibtex
% POPE: Polling-based Object Hallucination Evaluation
@inproceedings{li2023evaluating,
  title     = {Evaluating Object Hallucination in Large Vision-Language Models},
  author    = {Li, Yuhang and others},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  year      = {2023},
  note      = {Introduces POPE: random / popular / adversarial settings}
}

% MME: the 14-subtask multimodal evaluation benchmark
@inproceedings{yin2023halting,
  title     = {Halting, Not Deciding: Vision-Language Models Towards
               Determining and Evaluating Hallucinations},
  author    = {Yin, Shukang and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  note      = {Introduces MME: accuracy, accuracy+ and the perception/cognition split}
}

% LLaVA, the default backbone family
@article{liu2023visual,
  title   = {Visual Instruction Tuning},
  author  = {Liu, Haotian and others},
  journal = {Advances in Neural Information Processing Systems (NeurIPS)},
  year    = {2023}
}

% CLIP, used for the semantic evidence channel
@article{radford2021learning,
  title   = {Learning Transferable Visual Models From Natural Language
             Supervision},
  author  = {Radford, Alec and others},
  journal = {Proceedings of the 40th International Conference on Machine
             Learning (ICML)},
  year    = {2021}
}
```

Author lists are abbreviated to "and others" deliberately: consult the original
papers before copying these entries into a manuscript.

Once experiments exist, the honest claims for this repository will be exactly
what those experiments show — no more. In particular, **no claim of state of the
art, priority, superiority, or statistical significance is made anywhere in this
repository**, because none has been demonstrated.

---

## Appendix: repository guarantees

These are enforced in code, not merely intended:

| guarantee | enforcement |
| --- | --- |
| no random/fake images | `src/data/*.py` resolve real paths and raise with the probed locations; `require_images=True` by default |
| no fabricated metrics | `pope_metrics` / `mme_subtask_scores` raise on empty input rather than returning 0.0 |
| no hidden synthetic evidence | a requested grounding backend raises `GroundingUnavailable`; it never falls back to the null backend |
| unparseable answers surfaced | counted as errors and reported as `unparseable`, never coerced |
| unknown config keys rejected | `build_configs` raises on any key no dataclass claims |
| interventions auditable | `GenerationResult.interventions` and `interventions_detail` record whether evidence changed the chosen token |
| datasets/checkpoints uncommitted | `.gitignore` excludes `data/`, `*.jsonl`, `*.safetensors`, `results/*` |
| mocks confined to tests | stubs and synthetic tensors appear only under `tests/` and `scripts/smoke_test.py`; benchmarks use real data exclusively |

### Smoke test

```bash
python scripts/smoke_test.py            # ~1.7GB download, CPU-friendly
python scripts/smoke_test.py --skip-clip
```

Validates the real plumbing end to end against a small **real** LLaVA-family
checkpoint: loading, the processor, image-token span detection, attention
extraction, the KV-cached decoding loop, baseline `generate`, and the CLIP
channel. It reports no accuracy figures, because the 0.5B checkpoint it uses for
speed is not the experiment model.
