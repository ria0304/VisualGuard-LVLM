# -*- coding: utf-8 -*-
"""
Tests for config loading and data validation.

Data tests write tiny JSONL/image fixtures to ``tmp_path`` so the real loaders
can be exercised end to end without downloading COCO or MME. The images written
are 1x1 PNGs used purely as *file fixtures* to exercise path resolution — no
benchmark result is computed from them anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.mme import MMEDataError, load_mme
from src.data.pope import POPEDataError, load_pope, resolve_image_path
from src.model.grounding import (
    GroundingConfig,
    NullGroundingBackend,
    match_detection,
    normalise_phrase,
)
from src.model.grounding import Detection
from src.model.llava_backend import LVLMConfig
from src.model.visual_guard_decoder import (
    ALL_METHODS,
    DecodingConfig,
    evidence_config_for_method,
    make_candidates,
    trailing_word_fragment,
)
from src.model.visual_evidence import EvidenceConfig
from src.utils.config import (
    ConfigError,
    build_config,
    build_configs,
    deep_update,
    filter_known,
    load_yaml,
    resolve_configs,
)


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------


def test_yaml_config_loads():
    data = load_yaml(Path("configs/visualguard.yaml"))
    assert data["alpha"] == 1.0
    assert data["beta"] == 1.0
    assert "lam" in data


def test_shipped_configs_are_valid():
    """Every shipped config must instantiate cleanly across all config classes."""
    specs = {
        "evidence": EvidenceConfig,
        "decoding": DecodingConfig,
        "grounding": GroundingConfig,
    }
    for name in ("baseline", "attention", "semantic", "region", "unidirectional", "visualguard"):
        merged = resolve_configs(Path("configs") / f"{name}.yaml")
        built = build_configs(specs, merged)
        assert isinstance(built["evidence"], EvidenceConfig)
        assert isinstance(built["decoding"], DecodingConfig)


def test_baseline_config_has_all_channels_disabled():
    """A baseline config must not carry any active evidence weight."""
    merged = resolve_configs(Path("configs/baseline.yaml"))
    evidence = build_config(EvidenceConfig, merged)
    assert evidence.total_weight == 0.0
    assert evidence.lam == 0.0


def test_build_configs_rejects_unclaimed_keys():
    """Typos must fail even in the shared flat namespace."""
    with pytest.raises(ConfigError, match="Unknown config keys"):
        build_configs({"evidence": EvidenceConfig}, {"alfa": 1.0})


def test_build_configs_accepts_cross_class_keys():
    """One YAML namespace legitimately holds keys for several dataclasses."""
    out = build_configs(
        {"evidence": EvidenceConfig, "decoding": DecodingConfig},
        {"alpha": 0.5, "max_new_tokens": 8},
    )
    assert out["evidence"].alpha == 0.5
    assert out["decoding"].max_new_tokens == 8


def test_config_inheritance_merges_parent():
    merged = resolve_configs(Path("configs/semantic.yaml"))
    # semantic.yaml sets beta=1.0 explicitly and inherits lam from visualguard
    assert merged["beta"] == 1.0
    assert merged["lam"] == pytest.approx(0.5)
    assert merged["clip_model_name"] == "openai/clip-vit-base-patch32"


def test_cli_overrides_win_over_yaml():
    merged = resolve_configs(Path("configs/visualguard.yaml"), {"lam": 2.5})
    assert merged["lam"] == 2.5


def test_none_overrides_are_ignored():
    merged = resolve_configs(Path("configs/visualguard.yaml"), {"lam": None})
    assert merged["lam"] == pytest.approx(0.5)


def test_deep_update_is_recursive():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_update(base, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}, "d": 3}


def test_missing_config_lists_alternatives():
    with pytest.raises(ConfigError, match="Available configs"):
        load_yaml(Path("configs/does_not_exist.yaml"))


def test_unknown_keys_are_rejected():
    """Typos must fail loudly rather than being silently ignored."""
    with pytest.raises(ConfigError, match="Unknown config keys"):
        filter_known(EvidenceConfig, {"alpha": 1.0, "alfa": 1.0})


def test_build_config_validates_values():
    with pytest.raises(ConfigError):
        build_config(EvidenceConfig, {"normalize": "bogus"})


def test_lvlm_config_rejects_double_quantisation():
    with pytest.raises(ValueError, match="mutually exclusive"):
        LVLMConfig(load_in_4bit=True, load_in_8bit=True)


def test_lvlm_config_rejects_bad_dtype():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        LVLMConfig(dtype="int4")


# ---------------------------------------------------------------------------
# method -> channel mapping
# ---------------------------------------------------------------------------


def test_attention_method_disables_other_channels():
    cfg = evidence_config_for_method("attention", EvidenceConfig())
    assert (cfg.alpha, cfg.beta, cfg.gamma) == (1.0, 0.0, 0.0)


def test_semantic_method_disables_other_channels():
    cfg = evidence_config_for_method("semantic", EvidenceConfig())
    assert (cfg.alpha, cfg.beta, cfg.gamma) == (0.0, 1.0, 0.0)


def test_region_method_disables_other_channels():
    cfg = evidence_config_for_method("region", EvidenceConfig())
    assert (cfg.alpha, cfg.beta, cfg.gamma) == (0.0, 0.0, 1.0)


def test_baseline_method_zeroes_everything():
    cfg = evidence_config_for_method("baseline", EvidenceConfig())
    assert cfg.total_weight == 0.0
    assert cfg.lam == 0.0


def test_visualguard_method_keeps_weights():
    base = EvidenceConfig(alpha=1.5, beta=0.5, gamma=0.25, lam=0.7)
    cfg = evidence_config_for_method("visualguard", base)
    assert (cfg.alpha, cfg.beta, cfg.gamma, cfg.lam) == (1.5, 0.5, 0.25, 0.7)


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        evidence_config_for_method("nonexistent", EvidenceConfig())


def test_all_methods_cover_required_baselines():
    for method in ("baseline", "greedy", "sampling", "beam", "attention", "semantic", "visualguard"):
        assert method in ALL_METHODS


# ---------------------------------------------------------------------------
# decoding text helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", ""),
        ("dog", "dog"),
        ("a dog", "dog"),
        ("a dog and", "and"),
        ("a dog and ", ""),
        ("brown dog", "dog"),
    ],
)
def test_trailing_word_fragment(text, expected):
    assert trailing_word_fragment(text) == expected


class _StubTokenizer:
    """Minimal tokenizer stand-in: token id N decodes to word 'wN'."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(f"w{int(i)}" for i in ids)


def test_make_candidates_returns_top_k():
    import torch

    logits = torch.tensor([[0.0, 5.0, 3.0, 1.0]])
    cands = make_candidates(logits, _StubTokenizer(), [], top_k=3)
    assert len(cands) == 3
    assert cands[0].logit == pytest.approx(5.0)
    # logits are sorted descending
    assert [c.logit for c in cands] == sorted([c.logit for c in cands], reverse=True)


def test_make_candidates_top_k_clamped_to_vocab():
    import torch

    logits = torch.tensor([[0.0, 1.0]])
    assert len(make_candidates(logits, _StubTokenizer(), [], top_k=999)) == 2


def test_make_candidates_flags_function_words():
    import torch

    class _FnTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return {0: "the", 1: "dog"}[int(ids[-1])]

    logits = torch.tensor([[0.0, 5.0]])
    cands = make_candidates(logits, _FnTokenizer(), [], top_k=2)
    by_id = {c.token_id: c for c in cands}
    assert by_id[0].is_content is False
    assert by_id[1].is_content is True


# ---------------------------------------------------------------------------
# grounding
# ---------------------------------------------------------------------------


def test_null_grounding_is_disabled_and_silent():
    backend = build_backend_for_test()
    assert isinstance(backend, NullGroundingBackend)
    assert backend.available is False
    assert backend.detect(None, ["dog"]) == []
    assert backend.reason == "grounding_disabled"


def build_backend_for_test():
    from src.model.grounding import build_grounding_backend

    return build_grounding_backend(GroundingConfig(backend="none"))


def test_grounding_none_is_always_available():
    """Disabling grounding must never fail."""
    from src.model.grounding import build_grounding_backend

    backend = build_grounding_backend(GroundingConfig(backend="none"))
    assert backend.available is False


def test_grounding_unknown_backend_raises():
    from src.model.grounding import GroundingUnavailable, build_grounding_backend

    with pytest.raises(GroundingUnavailable, match="Unknown grounding backend"):
        build_grounding_backend(GroundingConfig(backend="magic"))


def test_grounding_cuda_request_without_cuda_is_refused():
    """Asking for CUDA grounding on a CPU-only box must fail with guidance."""
    import torch
    from src.model.grounding import GroundingDINOBackend, GroundingUnavailable

    cfg = GroundingConfig(backend="hf_grounding_dino", device="cuda")
    if torch.cuda.is_available():
        pytest.skip("CUDA present; this guard only applies to CPU-only hosts")
    with pytest.raises(GroundingUnavailable, match="disable region evidence"):
        GroundingDINOBackend(cfg)


def test_requested_grounding_backend_is_never_silently_degraded():
    """A requested backend must raise rather than fall back to the null backend.

    Checked without touching the network: the guard lives in the factory, which
    rejects unknown names before any checkpoint is fetched.
    """
    from src.model.grounding import GroundingUnavailable, build_grounding_backend

    with pytest.raises(GroundingUnavailable):
        build_grounding_backend(GroundingConfig(backend="hf_grounding_dinO_typo"))

    # Sanity: the null backend is only produced for an explicit "none".
    assert isinstance(
        build_grounding_backend(GroundingConfig(backend="none")), NullGroundingBackend
    )


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("dog", "dog"),
        ("the dog", "dog"),
        ("A Dog!", "dog"),
        ("in the image", "image"),
        ("", ""),
    ],
)
def test_normalise_phrase(phrase, expected):
    assert normalise_phrase(phrase) == expected


def test_match_detection_finds_substring_label():
    dets = [Detection(label="brown dog on grass", score=0.9, box=(0, 0, 10, 10))]
    score, det = match_detection(dets, "dog")
    assert score == pytest.approx(0.9)
    assert det is not None


def test_match_detection_returns_zero_when_unsupported():
    dets = [Detection(label="cat", score=0.9, box=(0, 0, 10, 10))]
    assert match_detection(dets, "tractor")[0] == 0.0


def test_match_detection_keeps_best_score():
    dets = [
        Detection(label="dog", score=0.3, box=(0, 0, 1, 1)),
        Detection(label="dog", score=0.8, box=(0, 0, 1, 1)),
    ]
    assert match_detection(dets, "dog")[0] == pytest.approx(0.8)


def test_match_detection_empty_inputs():
    assert match_detection([], "dog")[0] == 0.0
    assert match_detection([Detection(label="dog", score=0.5, box=(0, 0, 1, 1))], "")[0] == 0.0


# ---------------------------------------------------------------------------
# data loaders: fixtures + failure modes
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _write_dummy_image(path: Path) -> None:
    """Create a real (tiny) image file so existence checks pass."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(128, 128, 128)).save(path)


def test_pope_missing_annotation_raises_with_guidance(tmp_path):
    with pytest.raises(POPEDataError, match="coco_pope_random"):
        load_pope(pope_root=tmp_path, setting="random")


def test_pope_unknown_setting_raises(tmp_path):
    with pytest.raises(POPEDataError, match="Unknown POPE setting"):
        load_pope(pope_root=tmp_path, setting="nonexistent")


def test_pope_missing_images_raise_rather_than_substitute(tmp_path):
    """The core honesty guarantee: no placeholder images, ever."""
    _write_jsonl(
        tmp_path / "coco_pope_random.jsonl",
        [{"question_id": 1, "image": "000001.jpg", "text": "Is there a dog?", "label": "yes"}],
    )
    with pytest.raises(POPEDataError, match="could not be resolved"):
        load_pope(pope_root=tmp_path, setting="random")


def test_pope_loads_real_rows(tmp_path):
    _write_jsonl(
        tmp_path / "coco_pope_random.jsonl",
        [
            {"question_id": 1, "image": "a.jpg", "text": "Is there a dog?", "label": "yes"},
            {"question_id": 2, "image": "b.jpg", "text": "Is there a cat?", "label": "no"},
        ],
    )
    images = tmp_path / "images"
    _write_dummy_image(images / "a.jpg")
    _write_dummy_image(images / "b.jpg")

    samples = load_pope(pope_root=tmp_path, setting="random")
    assert len(samples) == 2
    assert samples[0].ground_truth is True
    assert samples[1].ground_truth is False
    assert samples[0].image_path.is_file()


def test_pope_max_samples_truncates(tmp_path):
    rows = [
        {"question_id": i, "image": f"{i}.jpg", "text": "Is there a dog?", "label": "yes"}
        for i in range(5)
    ]
    _write_jsonl(tmp_path / "coco_pope_popular.jsonl", rows)
    images = tmp_path / "images"
    for i in range(5):
        _write_dummy_image(images / f"{i}.jpg")
    assert len(load_pope(pope_root=tmp_path, setting="popular", max_samples=2)) == 2


def test_pope_rejects_malformed_row(tmp_path):
    _write_jsonl(tmp_path / "coco_pope_random.jsonl", [{"image": "a.jpg"}])
    with pytest.raises(POPEDataError, match="missing required field"):
        load_pope(pope_root=tmp_path, setting="random", require_images=False)


def test_pope_rejects_unknown_label(tmp_path):
    _write_jsonl(
        tmp_path / "coco_pope_adversarial.jsonl",
        [{"image": "a.jpg", "text": "q", "label": "maybe"}],
    )
    _write_dummy_image(tmp_path / "images" / "a.jpg")
    with pytest.raises(POPEDataError, match="Unrecognised POPE ground-truth label"):
        load_pope(pope_root=tmp_path, setting="adversarial")


def test_resolve_image_path_prefers_image_root(tmp_path):
    _write_dummy_image(tmp_path / "a.jpg")
    resolved = resolve_image_path(tmp_path, "a.jpg", tmp_path / "unused")
    assert resolved == tmp_path / "a.jpg"


def test_resolve_image_path_reports_probes():
    with pytest.raises(POPEDataError, match="Searched"):
        resolve_image_path(None, "missing.jpg", Path("/nonexistent"))


def test_mme_missing_root_raises_with_layout(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(MMEDataError, match="Expected layout"):
        load_mme(mme_root=missing)


def test_mme_loads_subtasks_and_groups_by_image(tmp_path):
    _write_jsonl(
        tmp_path / "existence" / "existence.jsonl",
        [
            {"question_id": 0, "image": "images/a.jpg", "text": "q1?", "answer": "Yes"},
            {"question_id": 1, "image": "images/a.jpg", "text": "q2?", "answer": "No"},
            {"question_id": 2, "image": "images/b.jpg", "text": "q3?", "answer": "No"},
            {"question_id": 3, "image": "images/b.jpg", "text": "q4?", "answer": "Yes"},
        ],
    )
    imgs = tmp_path / "existence" / "images"
    _write_dummy_image(imgs / "a.jpg")
    _write_dummy_image(imgs / "b.jpg")

    samples = load_mme(mme_root=tmp_path)
    assert len(samples) == 4
    assert samples[0].ground_truth is True
    assert samples[1].ground_truth is False
    assert all(s.subtask == "existence" for s in samples)


def test_mme_missing_image_raises(tmp_path):
    _write_jsonl(
        tmp_path / "count" / "count.jsonl",
        [{"question_id": 0, "image": "images/gone.jpg", "text": "q?", "answer": "Yes"}],
    )
    with pytest.raises(MMEDataError, match="MME image"):
        load_mme(mme_root=tmp_path)


def test_mme_rejects_unknown_answer(tmp_path):
    _write_jsonl(
        tmp_path / "existence" / "existence.jsonl",
        [{"question_id": 0, "image": "a.jpg", "text": "q?", "answer": "perhaps"}],
    )
    _write_dummy_image(tmp_path / "existence" / "a.jpg")
    samples = load_mme(mme_root=tmp_path)
    with pytest.raises(MMEDataError, match="Unrecognised MME answer"):
        _ = samples[0].ground_truth
