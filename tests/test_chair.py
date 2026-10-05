# -*- coding: utf-8 -*-
"""Tests for the CHAIR evaluator. No downloads required."""
import json
from types import SimpleNamespace

import pytest

from src.evaluation.chair_eval import (
    CHAIREvaluator,
    CHAIRRecord,
    chair_metrics,
    extract_objects,
    load_ground_truth,
    load_synonyms,
    score_caption,
    singularize,
)

SYN_LINES = [
    "person,man,woman,people,child,boy,girl",
    "dog,puppy",
    "cat,kitten",
    "teddy bear,teddy",
    "bear",
    "dining table,table",
    "cup,mug",
    "scissors",
]


@pytest.fixture()
def syn(tmp_path):
    p = tmp_path / "synonyms.txt"
    p.write_text("\n".join(SYN_LINES), encoding="utf-8")
    return load_synonyms(p)


def test_singularize():
    assert singularize("dogs") == "dog"
    assert singularize("people") == "person"
    assert singularize("sandwiches") == "sandwich"
    assert singularize("glass") == "glass"
    assert singularize("scissors") == "scissors"
    assert singularize("puppies") == "puppy"


def test_extract_prefers_longest_phrase(syn):
    assert extract_objects("A teddy bear on a table.", syn) == ["teddy bear", "dining table"]
    assert extract_objects("A bear in the woods", syn) == ["bear"]


def test_extract_plurals_and_synonyms(syn):
    assert extract_objects("Two puppies and a kitten", syn) == ["dog", "cat"]


def test_score_caption(syn):
    mentioned, halluc, covered = score_caption(
        "A man walks a dog next to a cat.", {"person", "dog"}, syn
    )
    assert mentioned == ["person", "dog", "cat"]
    assert halluc == ["cat"]
    assert covered == ["dog", "person"]


def _rec(m, h, c, gt, n=10):
    return CHAIRRecord(1, "x", "", m, h, c, gt, n)


def test_chair_metrics():
    recs = [
        _rec(["dog", "cat"], ["cat"], ["dog"], ["dog", "person"], 10),
        _rec(["dog"], [], ["dog"], ["dog"], 20),
    ]
    m = chair_metrics(recs)
    assert m["chair_s"] == pytest.approx(50.0)
    assert m["chair_i"] == pytest.approx(100 * 1 / 3)
    assert m["recall"] == pytest.approx(100 * 2 / 3)
    assert m["avg_length_words"] == 15


def test_chair_metrics_empty_raises():
    with pytest.raises(ValueError):
        chair_metrics([])


def _write_coco(tmp_path):
    ann = tmp_path / "ann"
    ann.mkdir()
    img = tmp_path / "img"
    img.mkdir()
    ids = list(range(1, 6))
    inst = {
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
        "images": [{"id": i, "file_name": f"{i}.jpg"} for i in ids],
        "annotations": [{"image_id": i, "category_id": 2} for i in ids],
    }
    caps = {"annotations": [{"image_id": i, "caption": "A man with a dog."} for i in ids]}
    (ann / "instances_val2014.json").write_text(json.dumps(inst))
    (ann / "captions_val2014.json").write_text(json.dumps(caps))
    for i in ids:
        (img / f"{i}.jpg").write_bytes(b"x")
    return ann, img


def test_ground_truth_unions_instances_and_captions(tmp_path, syn):
    ann, _ = _write_coco(tmp_path)
    gt, files = load_ground_truth(ann, "val2014", syn)
    assert gt[1] == {"dog", "person"}
    assert files[1] == "1.jpg"


def test_evaluator_end_to_end_and_same_subset(tmp_path, syn):
    ann, img = _write_coco(tmp_path)
    syn_path = tmp_path / "synonyms.txt"
    syn_path.write_text("\n".join(SYN_LINES))
    seen = []

    def gen(image, prompt, max_new_tokens=None):
        seen.append(image)
        return SimpleNamespace(text="A man, a dog and a cat.", num_generated_tokens=8)

    kw = dict(ann_dir=ann, image_root=img, synonyms_path=syn_path, num_images=3, seed=7)
    r1 = CHAIREvaluator(gen, method="a").evaluate(**kw, output_dir=tmp_path / "out")
    first = list(seen)
    seen.clear()
    CHAIREvaluator(gen, method="b").evaluate(**kw)
    assert first == seen  # same seed -> identical image subset across methods
    assert r1.metrics["chair_s"] == 100.0  # 'cat' hallucinated in every caption
    assert (tmp_path / "out" / "chair_a_predictions.jsonl").is_file()


def test_evaluator_missing_images_raises(tmp_path, syn):
    from src.evaluation.chair_eval import CHAIRDataError

    ann, img = _write_coco(tmp_path)
    (img / "1.jpg").unlink()
    (tmp_path / "s.txt").write_text("\n".join(SYN_LINES))
    ev = CHAIREvaluator(lambda i, p: SimpleNamespace(text="x"), method="m")
    with pytest.raises(CHAIRDataError):
        ev.evaluate(ann, img, tmp_path / "s.txt", num_images=5)


