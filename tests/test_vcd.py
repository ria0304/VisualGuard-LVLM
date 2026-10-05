# -*- coding: utf-8 -*-
"""Tests for the VCD baseline. No downloads required."""
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from src.model.vcd import (  # noqa: E402
    VCDConfig,
    VCDDecoder,
    add_diffusion_noise,
    contrastive_logits,
)


def test_noise_is_seeded_and_changes_image():
    x = torch.zeros(1, 3, 8, 8)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    a = add_diffusion_noise(x, 500, generator=g1)
    b = add_diffusion_noise(x, 500, generator=g2)
    assert torch.equal(a, b)
    assert a.abs().sum() > 0
    assert torch.equal(add_diffusion_noise(x, 0), x)


def test_contrastive_prefers_image_dependent_token():
    # token 0: model likes it with or without the image (language prior)
    # token 1: only likes it when the image is intact
    orig = torch.tensor([[5.0, 5.0, 0.0]])
    noisy = torch.tensor([[5.0, 1.0, 0.0]])
    cd = contrastive_logits(orig, noisy, alpha=1.0, beta=0.1)
    assert cd.argmax(-1).item() == 1


def test_plausibility_constraint_masks_unlikely_tokens():
    orig = torch.tensor([[10.0, 0.0]])
    noisy = torch.tensor([[0.0, 20.0]])  # contrast would pick token 1
    cd = contrastive_logits(orig, noisy, alpha=1.0, beta=0.1)
    assert cd[0, 1] == float("-inf")
    assert cd.argmax(-1).item() == 0


def test_vcd_config_validation():
    with pytest.raises(ValueError):
        VCDConfig(beta=0.0)
    with pytest.raises(ValueError):
        VCDConfig(noise_step=2000)


class _Tok:
    eos_token_id = 9

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


class _Backend:
    """Image-dependent toy LM: token 1 only with a clean (zero-mean) image."""
    tokenizer = _Tok()
    device = torch.device("cpu")

    def prepare_inputs(self, question, image=None):
        return {"input_ids": torch.tensor([[1, 2, 3]]), "pixel_values": torch.zeros(1, 3, 4, 4)}

    def _logits(self, pixel_values):
        noisy = pixel_values.abs().sum() > 0
        l = torch.zeros(1, 10)
        l[0, 0] = 5.0
        l[0, 1] = 1.0 if noisy else 5.0
        l[0, 9] = -10.0
        return l

    def initial_step(self, input_ids, pixel_values, output_attentions=False, attention_mask=None):
        self._last = pixel_values
        return SimpleNamespace(logits=self._logits(pixel_values), past_key_values=pixel_values)

    def next_step(self, input_ids, past_key_values, output_attentions=False, attention_mask=None):
        return SimpleNamespace(logits=self._logits(past_key_values), past_key_values=past_key_values)


def test_vcd_decoder_runs_and_reports_interventions():
    dec = VCDDecoder(_Backend(), vcd_config=VCDConfig(alpha=1.0, beta=0.1))
    out = dec.generate("img", "q", max_new_tokens=3)
    assert out.method == "vcd"
    assert out.token_ids == [1, 1, 1]       # contrast picks the image-dependent token
    assert out.interventions == 3            # greedy would have picked token 0 (tie -> first)
