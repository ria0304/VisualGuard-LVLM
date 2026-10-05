# -*- coding: utf-8 -*-
"""
Visual Contrastive Decoding (VCD) baseline.

Reference: Leng et al., "Mitigating Object Hallucinations in Large
Vision-Language Models through Visual Contrastive Decoding", CVPR 2024.

At every step the LVLM is run twice, once on the original image and once on a
diffusion-noised copy. Tokens that the model prefers *even without seeing the
image properly* (language priors) are down-weighted::

    cd_logits = (1 + alpha) * logits(original) - alpha * logits(noised)

An adaptive plausibility constraint restricts the choice to tokens whose
probability under the original image is at least ``beta`` times the maximum::

    keep(t)  iff  log p_orig(t) >= log(beta) + max_t' log p_orig(t')

This module uses the same :class:`LVLMBackend` step API as
:class:`VisualGuardDecoder`, the same prompt builder and the same
``GenerationResult``, so VCD plugs into every evaluator unchanged and is
compared under identical prompts, checkpoint and seed.

Differences from the authors' release, stated rather than hidden:

* Greedy selection only (the paper's headline setting).
* The noise is applied to the *processed* ``pixel_values`` tensor, as in the
  authors' code, using a DDPM linear beta schedule (1e-4 .. 0.02, 1000 steps).
* Two KV caches are kept, one per image, so each step costs two cached forward
  passes. That is the method's real cost and is reflected in ``latency_s``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from .llava_backend import LVLMBackend
from .visual_guard_decoder import (
    DecodingConfig,
    GenerationResult,
    _call_step,
)

logger = logging.getLogger(__name__)


@dataclass
class VCDConfig:
    """Hyper-parameters. Defaults follow the paper's LLaVA-1.5 setting."""

    alpha: float = 1.0          # contrast strength
    beta: float = 0.1           # plausibility cutoff (fraction of max prob)
    noise_step: int = 500       # diffusion step used to corrupt the image (of 1000)
    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    seed: int = 42

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must be in (0, 1]")
        if not 0 <= self.noise_step <= self.num_diffusion_steps:
            raise ValueError("noise_step must be within [0, num_diffusion_steps]")


def add_diffusion_noise(
    pixel_values: torch.Tensor,
    noise_step: int,
    num_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Forward-diffuse ``pixel_values`` to step ``noise_step``.

    ``x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps``. ``noise_step == 0``
    returns the input unchanged.
    """
    if noise_step <= 0:
        return pixel_values.clone()
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
    abar = torch.cumprod(1.0 - betas, dim=0)
    t = min(int(noise_step), num_steps) - 1
    a = abar[t].item()
    x = pixel_values.float()
    noise = torch.randn(
        x.shape, generator=generator, device=generator.device if generator else x.device,
        dtype=torch.float32,
    ).to(x.device)
    noisy = (a ** 0.5) * x + ((1.0 - a) ** 0.5) * noise
    return noisy.to(pixel_values.dtype)


def contrastive_logits(
    logits_orig: torch.Tensor,
    logits_noisy: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """VCD combination with the adaptive plausibility constraint.

    Both inputs are ``(batch, vocab)``. Returns logits with implausible tokens
    set to ``-inf``; ``argmax`` of the result is the VCD greedy token.
    """
    log_p = torch.log_softmax(logits_orig.float(), dim=-1)
    cutoff = torch.log(torch.tensor(beta, device=log_p.device)) + log_p.max(
        dim=-1, keepdim=True
    ).values
    cd = (1.0 + alpha) * logits_orig.float() - alpha * logits_noisy.float()
    return cd.masked_fill(log_p < cutoff, float("-inf"))


class VCDDecoder:
    """Greedy visual contrastive decoding over an :class:`LVLMBackend`.

    Example
    -------
    >>> vcd = VCDDecoder(decoder.backend, decoder.decoding_config, VCDConfig())
    >>> out = vcd.generate(image_path, "Describe this image.", max_new_tokens=64)
    """

    method = "vcd"

    def __init__(
        self,
        backend: LVLMBackend,
        decoding_config: Optional[DecodingConfig] = None,
        vcd_config: Optional[VCDConfig] = None,
    ) -> None:
        self.backend = backend
        self.decoding_config = decoding_config or DecodingConfig()
        self.cfg = vcd_config or VCDConfig()
        self._sample_counter = 0

    # -- helpers -------------------------------------------------------

    def _eos_ids(self) -> set:
        tok = self.backend.tokenizer
        value = getattr(tok, "eos_token_id", None)
        if isinstance(value, int):
            return {value}
        if isinstance(value, (list, tuple)):
            return {int(v) for v in value}
        return set()

    def _generator(self, device: torch.device) -> torch.Generator:
        # Seeded per sample (seed + running index) so the noise differs across
        # images but a whole run is exactly reproducible.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.cfg.seed + self._sample_counter)
        self._sample_counter += 1
        return gen

    # -- generation ----------------------------------------------------

    def generate(
        self,
        image: Any,
        question: str,
        method: Optional[str] = None,  # accepted for interface parity
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        budget = (
            int(max_new_tokens)
            if max_new_tokens is not None
            else self.decoding_config.max_new_tokens
        )
        enc = self.backend.prepare_inputs(question, image)
        input_ids = enc["input_ids"]
        pixel_values = enc.get("pixel_values")
        attention_mask = enc.get("attention_mask")
        if pixel_values is None:
            raise ValueError("VCD needs an image; got none.")

        noisy_pixels = add_diffusion_noise(
            pixel_values,
            noise_step=self.cfg.noise_step,
            num_steps=self.cfg.num_diffusion_steps,
            beta_start=self.cfg.beta_start,
            beta_end=self.cfg.beta_end,
            generator=self._generator(pixel_values.device),
        )

        eos_ids = self._eos_ids()
        generated: List[int] = []
        changed = 0  # steps where VCD picked a different token than greedy

        start = time.perf_counter()
        with torch.inference_mode():
            step_o = _call_step(
                self.backend.initial_step, input_ids, pixel_values,
                output_attentions=False, attention_mask=attention_mask,
            )
            step_n = _call_step(
                self.backend.initial_step, input_ids, noisy_pixels,
                output_attentions=False, attention_mask=attention_mask,
            )
            past_o, past_n = step_o.past_key_values, step_n.past_key_values

            for _ in range(budget):
                cd = contrastive_logits(
                    step_o.logits, step_n.logits, self.cfg.alpha, self.cfg.beta
                )
                token = int(cd.argmax(dim=-1).item())
                if token != int(step_o.logits.argmax(dim=-1).item()):
                    changed += 1
                generated.append(token)
                if token in eos_ids:
                    break

                tok_t = torch.tensor([[token]], device=self.backend.device)
                step_o = _call_step(
                    self.backend.next_step, tok_t, past_o,
                    output_attentions=False,
                    attention_mask=None,  # batch 1, no padding: HF assumes all ones
                )
                step_n = _call_step(
                    self.backend.next_step, tok_t, past_n,
                    output_attentions=False,
                    attention_mask=None,
                )
                past_o, past_n = step_o.past_key_values, step_n.past_key_values
        latency = time.perf_counter() - start

        text = self.backend.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            token_ids=list(generated),
            num_prompt_tokens=int(input_ids.shape[1]),
            num_generated_tokens=len(generated),
            method="vcd",
            latency_s=latency,
            per_token_latency_s=latency / max(len(generated), 1),
            interventions=changed,
            notes=[
                f"VCD alpha={self.cfg.alpha} beta={self.cfg.beta} "
                f"noise_step={self.cfg.noise_step}; two forward passes per step",
            ],
        )
