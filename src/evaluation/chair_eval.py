# -*- coding: utf-8 -*-
"""
CHAIR evaluation for open-ended image captioning.

Reference: Rohrbach et al., "Object Hallucination in Image Captioning",
EMNLP 2018.

For every sampled COCO image this module

1. asks the configured decoder for a long-form caption (same prompt for every
   method),
2. extracts the COCO objects mentioned in the caption using the official CHAIR
   synonym list,
3. compares them against the objects actually present (COCO instance
   annotations *and* the reference captions, as in the official script),
4. aggregates::

       CHAIR_s = #captions with >= 1 hallucinated object / #captions
       CHAIR_i = #hallucinated object mentions / #object mentions
       Recall  = #ground-truth objects mentioned / #ground-truth objects
       Length  = mean caption length in words

Every caption is written to JSONL so results can be re-scored without
re-running the model. Nothing here fabricates a number: with zero samples, or
with a missing annotation / synonym file, the evaluator raises.

Data needed (official downloads, nothing is bundled):

* ``instances_<split>.json`` and ``captions_<split>.json`` from the COCO
  annotations archive (``<split>`` = ``val2014`` or ``val2017``),
* ``synonyms.txt`` from the CHAIR release
  (https://github.com/LisaAnne/Hallucination, ``data/synonyms.txt``): one object
  per line, comma-separated synonyms, the first being the COCO category name.

Simplifications versus the official script, stated rather than hidden:

* Singularisation is rule based (no ``pattern`` dependency), so a few irregular
  plurals can differ. Absolute numbers may therefore differ slightly from the
  official script; comparisons between methods inside this repo are unaffected
  because every method is scored by the same code.
* The official script's coarse-word handling is not reproduced; only synonym
  matches (single words and multi-word phrases up to three words) count.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .metrics import accepts_keyword

logger = logging.getLogger(__name__)

#: Long-form prompt, identical across all methods (convention of VCD / OPERA).
CHAIR_PROMPT = "Please describe this image in detail."

_WORD = re.compile(r"[a-z]+")
_IRREGULAR = {
    "people": "person", "men": "man", "women": "woman", "children": "child",
    "mice": "mouse", "geese": "goose", "feet": "foot", "teeth": "tooth",
    "skis": "ski", "knives": "knife", "leaves": "leaf", "shelves": "shelf",
    "wolves": "wolf", "loaves": "loaf", "sheep": "sheep", "scissors": "scissors",
}


class CHAIRDataError(RuntimeError):
    """Raised when annotations, synonyms or images are missing or malformed."""


# ---------------------------------------------------------------------------
# text helpers (pure, unit-testable)
# ---------------------------------------------------------------------------


def singularize(word: str) -> str:
    """Conservative rule-based singulariser for COCO object nouns."""
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "zes", "ches", "shes", "oes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> List[str]:
    return _WORD.findall(str(text).lower())


def load_synonyms(path: Path) -> Dict[str, str]:
    """Map every synonym phrase to its canonical COCO object name.

    File format: one object per line, comma-separated, canonical name first.
    """
    path = Path(path)
    if not path.is_file():
        raise CHAIRDataError(
            f"CHAIR synonyms file not found: {path}\n"
            "Download data/synonyms.txt from https://github.com/LisaAnne/Hallucination"
        )
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = [p.strip().lower() for p in line.strip().split(",") if p.strip()]
            if not parts:
                continue
            canonical = parts[0]
            for phrase in parts:
                mapping[phrase] = canonical
    if not mapping:
        raise CHAIRDataError(f"No synonyms parsed from {path}")
    return mapping


def extract_objects(
    caption: str, synonyms: Dict[str, str], max_ngram: int = 3
) -> List[str]:
    """COCO objects mentioned in ``caption``, one entry per mention.

    Greedy longest-phrase-first matching, so "teddy bear" is one mention of
    ``teddy bear`` and not a mention of ``bear``. The last word of a phrase is
    tried both as written and singularised.
    """
    words = tokenize(caption)
    mentions: List[str] = []
    i = 0
    while i < len(words):
        matched = False
        for n in range(min(max_ngram, len(words) - i), 0, -1):
            head = words[i:i + n - 1]
            last = words[i + n - 1]
            for form in (last, singularize(last)):
                phrase = " ".join(head + [form])
                if phrase in synonyms:
                    mentions.append(synonyms[phrase])
                    i += n
                    matched = True
                    break
            if matched:
                break
        if not matched:
            i += 1
    return mentions


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------


def load_ground_truth(
    ann_dir: Path, split: str, synonyms: Dict[str, str]
) -> Tuple[Dict[int, Set[str]], Dict[int, str]]:
    """Per-image GT object sets and image file names.

    GT = COCO instance categories  U  objects mentioned in the reference
    captions (both are used by the official script).
    """
    ann_dir = Path(ann_dir)
    inst_path = ann_dir / f"instances_{split}.json"
    cap_path = ann_dir / f"captions_{split}.json"
    for p in (inst_path, cap_path):
        if not p.is_file():
            raise CHAIRDataError(f"COCO annotation file not found: {p}")

    with inst_path.open("r", encoding="utf-8") as fh:
        inst = json.load(fh)
    with cap_path.open("r", encoding="utf-8") as fh:
        caps = json.load(fh)

    cat_name = {c["id"]: c["name"].lower() for c in inst["categories"]}
    files = {im["id"]: im["file_name"] for im in inst["images"]}
    gt: Dict[int, Set[str]] = {im_id: set() for im_id in files}

    for ann in inst["annotations"]:
        name = cat_name.get(ann["category_id"])
        if name is None or ann["image_id"] not in gt:
            continue
        gt[ann["image_id"]].add(synonyms.get(name, name))
    for ann in caps["annotations"]:
        if ann["image_id"] in gt:
            gt[ann["image_id"]].update(extract_objects(ann["caption"], synonyms))
    return gt, files


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@dataclass
class CHAIRRecord:
    image_id: int
    image: str
    caption: str
    mentioned: List[str]
    hallucinated: List[str]
    covered: List[str]
    gt_objects: List[str]
    length_words: int
    generated_tokens: int = 0
    latency_s: float = 0.0
    interventions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def chair_metrics(records: Sequence[CHAIRRecord]) -> Dict[str, float]:
    """Aggregate CHAIR metrics. Raises on an empty record set."""
    if not records:
        raise ValueError("chair_metrics: no records")
    n = len(records)
    mentions = sum(len(r.mentioned) for r in records)
    halluc_mentions = sum(len(r.hallucinated) for r in records)
    halluc_caps = sum(1 for r in records if r.hallucinated)
    gt_total = sum(len(r.gt_objects) for r in records)
    covered = sum(len(r.covered) for r in records)
    return {
        "chair_s": 100.0 * halluc_caps / n,
        "chair_i": 100.0 * halluc_mentions / mentions if mentions else 0.0,
        "recall": 100.0 * covered / gt_total if gt_total else 0.0,
        "avg_length_words": sum(r.length_words for r in records) / n,
        "avg_mentions": mentions / n,
    }


def score_caption(
    caption: str, gt: Set[str], synonyms: Dict[str, str]
) -> Tuple[List[str], List[str], List[str]]:
    """Return ``(mentioned, hallucinated, covered)`` for one caption.

    ``hallucinated`` keeps one entry per hallucinated mention (CHAIR_i counts
    mentions); ``covered`` is the set of GT objects the caption mentions.
    """
    mentioned = extract_objects(caption, synonyms)
    hallucinated = [m for m in mentioned if m not in gt]
    covered = sorted(gt.intersection(mentioned))
    return mentioned, hallucinated, covered


@dataclass
class CHAIRResult:
    method: str
    metrics: Dict[str, float]
    n_samples: int
    latency_s: float
    per_token_latency_s: float
    seed: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": "chair",
            "method": self.method,
            "metrics": self.metrics,
            "n_samples": self.n_samples,
            "latency_s": self.latency_s,
            "per_token_latency_s": self.per_token_latency_s,
            "seed": self.seed,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------


class CHAIREvaluator:
    """Runs a decoder over sampled COCO images and scores object hallucination.

    Example
    -------
    >>> ev = CHAIREvaluator(decoder.generate, method="visualguard",
    ...                     max_new_tokens=512)
    >>> res = ev.evaluate(ann_dir="coco/annotations", image_root="coco/val2014",
    ...                   synonyms_path="synonyms.txt", num_images=500)
    """

    def __init__(
        self,
        generate_fn: Callable[..., Any],
        method: str = "greedy",
        prompt: str = CHAIR_PROMPT,
        max_new_tokens: int = 512,
    ) -> None:
        self.generate_fn = generate_fn
        self.method = method
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self._takes_budget = accepts_keyword(generate_fn, "max_new_tokens")

    def _generate(self, image_path: Path) -> Any:
        if self._takes_budget:
            return self.generate_fn(
                str(image_path), self.prompt, max_new_tokens=self.max_new_tokens
            )
        return self.generate_fn(str(image_path), self.prompt)

    def evaluate(
        self,
        ann_dir: Path,
        image_root: Path,
        synonyms_path: Path,
        split: str = "val2014",
        num_images: int = 500,
        seed: int = 42,
        output_dir: Optional[Path] = None,
        progress_every: int = 25,
    ) -> CHAIRResult:
        synonyms = load_synonyms(synonyms_path)
        gt, files = load_ground_truth(ann_dir, split, synonyms)

        ids = sorted(files)
        if num_images > len(ids):
            raise CHAIRDataError(
                f"Requested {num_images} images but split has only {len(ids)}"
            )
        # Same seed -> same image subset for every method, so rows are comparable.
        chosen = random.Random(seed).sample(ids, num_images)

        image_root = Path(image_root)
        missing = [files[i] for i in chosen if not (image_root / files[i]).is_file()]
        if missing:
            raise CHAIRDataError(
                f"{len(missing)} sampled images missing under {image_root}, "
                f"e.g. {missing[:3]}"
            )

        records: List[CHAIRRecord] = []
        total_tokens = 0
        run_start = time.perf_counter()
        for idx, image_id in enumerate(chosen, start=1):
            path = image_root / files[image_id]
            started = time.perf_counter()
            out = self._generate(path)
            latency = time.perf_counter() - started
            caption = getattr(out, "text", str(out)).strip()

            mentioned, halluc, covered = score_caption(caption, gt[image_id], synonyms)
            n_tok = int(getattr(out, "num_generated_tokens", 0))
            total_tokens += n_tok
            records.append(
                CHAIRRecord(
                    image_id=image_id,
                    image=str(path),
                    caption=caption,
                    mentioned=mentioned,
                    hallucinated=halluc,
                    covered=covered,
                    gt_objects=sorted(gt[image_id]),
                    length_words=len(tokenize(caption)),
                    generated_tokens=n_tok,
                    latency_s=latency,
                    interventions=int(getattr(out, "interventions", 0)),
                )
            )
            if progress_every and idx % progress_every == 0:
                m = chair_metrics(records)
                logger.info(
                    "[%s/chair] %d/%d CHAIRs=%.1f CHAIRi=%.1f recall=%.1f len=%.1f",
                    self.method, idx, len(chosen),
                    m["chair_s"], m["chair_i"], m["recall"], m["avg_length_words"],
                )
        total_latency = time.perf_counter() - run_start

        if output_dir is not None:
            self.save_predictions(
                records, Path(output_dir) / f"chair_{self.method}_predictions.jsonl"
            )

        return CHAIRResult(
            method=self.method,
            metrics=chair_metrics(records),
            n_samples=len(records),
            latency_s=total_latency,
            per_token_latency_s=total_latency / max(total_tokens, 1),
            seed=seed,
            notes=self._diagnostics(records),
        )

    def _diagnostics(self, records: Sequence[CHAIRRecord]) -> List[str]:
        notes: List[str] = []
        if not self._takes_budget:
            notes.append(
                "generate_fn does not accept max_new_tokens; the decoder's own "
                "default budget was used instead of the benchmark's"
            )
        m = chair_metrics(records)
        if m["avg_length_words"] < 20:
            notes.append(
                f"mean caption length is {m['avg_length_words']:.1f} words: very short "
                "captions lower CHAIR trivially, so read CHAIR together with "
                "length and recall"
            )
        capped = sum(1 for r in records if r.generated_tokens >= self.max_new_tokens)
        if capped:
            notes.append(
                f"{capped} captions hit the {self.max_new_tokens}-token budget "
                "and may be truncated mid-sentence"
            )
        return notes

    @staticmethod
    def save_predictions(records: Sequence[CHAIRRecord], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Wrote %d captions to %s", len(records), path)
