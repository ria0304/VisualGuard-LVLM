# -*- coding: utf-8 -*-
"""
LVLM backends for VisualGuard.

This module provides a thin, well-defined interface over *real pretrained*
LVLMs (LLaVA-compatible Hugging Face models) plus the generation plumbing
required by the decoding-time intervention in
:mod:`src.model.visual_guard_decoder`.

Design goals
------------
* Load a genuine pretrained LVLM (never a randomly initialised stand-in).
* Expose the pieces the evidence module needs:
    - tokenizer
    - image processor
    - logit distribution for the next token
    - self-attention over the *current* sequence (for attention evidence)
* Support a step-by-step decoding loop with KV caching so that logits can be
  modified *before* the token is selected (the core of the method).
* Be pluggable: a second LVLM only needs to implement :class:`LVLMBackend`.

Notes on attention
------------------
LLaVA-style models route visual tokens into the language model as ordinary
sequence positions, so "which visual patch is the model looking at" reduces to
"how much does the last query position attend to the image-token span". Getting
that requires eager attention, so we request ``attn_implementation="eager"``.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)

#: Legacy sentinel used by the original LLaVA repo. ``llava-hf`` checkpoints use
#: an in-vocabulary token id instead (32000 for LLaVA-1.5), which is read from
#: the model config at load time. This is only a last-resort fallback.
LEGACY_IMAGE_TOKEN_INDEX = -200


class LVLMBackendError(RuntimeError):
    """Raised for backend loading / inference failures."""


@dataclass
class LVLMConfig:
    """Configuration for loading a pretrained LVLM.

    Attributes
    ----------
    model_name:
        Hugging Face model id, e.g. ``"llava-hf/llava-1.5-7b-hf"``.
    device:
        ``"cuda"``, ``"cuda:0"``, ``"cpu"``, or ``"auto"``.
    dtype:
        One of ``{"float32", "float16", "bfloat16", "auto"}``.
    load_in_4bit / load_in_8bit:
        bitsandbytes quantisation. Mutually exclusive.
    attn_implementation:
        Forced to ``"eager"`` by default because attention evidence needs the
        attention matrices, which fused/SDPA kernels do not return.
    max_new_tokens:
        Default generation budget.
    """

    model_name: str = "llava-hf/llava-1.5-7b-hf"
    device: str = "auto"
    dtype: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    attn_implementation: str = "eager"
    max_new_tokens: int = 64
    revision: str = "main"
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are mutually exclusive.")
        if self.dtype not in {"float32", "float16", "bfloat16", "auto"}:
            raise ValueError(f"Unsupported dtype: {self.dtype!r}")


@dataclass
class StepOutput:
    """Result of one autoregressive step.

    Attributes
    ----------
    logits:
        ``(batch, vocab)`` float tensor of *unprocessed* next-token logits.
    attentions:
        Optional per-layer attention tensors. Each is
        ``(batch, heads, q_len, kv_len)``. ``q_len`` is 1 for cached steps.
    past_key_values:
        KV cache to feed into the next step.
    image_token_span:
        ``(start, end)`` half-open range of image-token positions in the
        *current* sequence, or ``None`` when the step carries no image.
    """

    logits: torch.Tensor
    attentions: Optional[Tuple[torch.Tensor, ...]]
    past_key_values: Any
    image_token_span: Optional[Tuple[int, int]]


class LVLMBackend(ABC):
    """Abstract interface over a pretrained LVLM."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights, tokenizer and image processor."""

    @abstractmethod
    def encode_prompt(self, question: str) -> Dict[str, torch.Tensor]:
        """Tokenise ``question`` into an LLaVA-style chat prompt."""

    @abstractmethod
    def preprocess_image(self, image: Any) -> torch.Tensor:
        """Convert a PIL image (or path) into a ``(1, 3, H, W)`` tensor."""

    @abstractmethod
    def initial_step(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        output_attentions: bool = False,
    ) -> StepOutput:
        """Run the prefill step for a full prompt."""

    @abstractmethod
    def next_step(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any,
        output_attentions: bool = False,
    ) -> StepOutput:
        """Run one cached step given the previously selected token."""

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def tokenizer(self) -> Any: ...

    def image_token_span(
        self, input_ids: torch.Tensor, pixel_values: Optional[torch.Tensor]
    ) -> Optional[Tuple[int, int]]:
        """Locate the image-token positions inside ``input_ids``.

        LLaVA expands a single ``<image>`` placeholder into one placeholder per
        vision patch, so the run is normally longer than one position; the whole
        contiguous run is returned.

        Backends whose models expose image positions differently should override
        this.
        """
        if pixel_values is None:
            return None
        ids = input_ids[0].tolist()
        sentinel = self._image_token_index
        if sentinel in ids:
            start = ids.index(sentinel)
            end = start
            while end < len(ids) and ids[end] == sentinel:
                end += 1
            return (start, end)
        # No placeholder found: the processor may splice image features in
        # directly. Fall back to the standard LLaVA convention of a leading
        # image block, using the configured image sequence length.
        if int(pixel_values.shape[0]) == 0:
            return None
        model_cfg = getattr(self.model, "config", None)
        n_image = getattr(model_cfg, "image_seq_length", None) or 1
        return (0, min(int(n_image), len(ids)))


class LLaVAHFBackend(LVLMBackend):
    """Backend for LLaVA-style models on Hugging Face.

    Example
    -------
    >>> backend = LLaVAHFBackend(LVLMConfig(model_name="llava-hf/llava-1.5-7b-hf"))
    >>> backend.load()
    >>> enc = backend.encode_prompt("Is there a dog in the image?")
    >>> pv = backend.preprocess_image("dog.jpg")
    >>> out = backend.initial_step(enc["input_ids"], pv)
    >>> out.logits.shape
    torch.Size([1, 32064])
    """

    #: Prompt template for LLaVA-1.5 style conversation formatting.
    PROMPT_TEMPLATE = (
        "USER: <image>\n{question} ASSISTANT:"
    )

    def __init__(self, config: LVLMConfig) -> None:
        self.config = config
        self.model: Optional[Any] = None
        # Private backing fields: `tokenizer` / `image_processor` are exposed as
        # properties on the backend interface.
        self._tokenizer: Optional[Any] = None
        self._processor: Optional[Any] = None
        #: Image placeholder token id, resolved from the model config in load().
        self._image_token_index: int = LEGACY_IMAGE_TOKEN_INDEX

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        if self.config.device != "auto":
            dev = torch.device(self.config.device)
            if dev.type == "cuda" and not torch.cuda.is_available():
                raise LVLMBackendError(
                    "device='cuda' requested but torch.cuda.is_available() is False. "
                    "Use --device cpu to run on CPU, or install a CUDA build of PyTorch."
                )
            return dev
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_dtype(self, device: torch.device) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        if device.type == "cuda":
            return torch.float16
        return torch.float32

    def load(self) -> None:
        try:
            from transformers import AutoProcessor, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise LVLMBackendError(
                "transformers is required. Install with `pip install -e .` or "
                "`pip install transformers>=4.40`."
            ) from exc

        device = self._resolve_device()
        dtype = self._resolve_dtype(device)

        quant_config = None
        if self.config.load_in_4bit or self.config.load_in_8bit:
            quant_config = self._build_quant_config()

        model_kwargs: Dict[str, Any] = dict(
            torch_dtype=dtype,
            revision=self.config.revision,
            attn_implementation=self.config.attn_implementation,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.config.cache_dir:
            model_kwargs["cache_dir"] = self.config.cache_dir
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            # bitsandbytes requires the model to be dispatched, not `.to(device)`.
            model_kwargs["device_map"] = (
                {"": 0} if device.type == "cuda" else "cpu"
            )

        logger.info(
            "Loading LVLM %s (dtype=%s, device=%s, quant=%s)",
            self.config.model_name,
            dtype,
            device,
            "4bit" if self.config.load_in_4bit
            else "8bit" if self.config.load_in_8bit else "none",
        )

        model_cls = self._resolve_model_class()
        try:
            self.model = model_cls.from_pretrained(self.config.model_name, **model_kwargs)
        except Exception as exc:
            raise LVLMBackendError(
                f"Failed to load LVLM {self.config.model_name!r}: {exc}\n"
                "Checklist:\n"
                "  * the model id is correct and publicly available\n"
                f"  * transformers version is new enough (found "
                f"{self._transformers_version()})\n"
                "  * you have accepted the model's licence on the Hub, if required\n"
                "  * sufficient disk space in the HF cache"
            ) from exc

        if quant_config is None:
            self.model.to(device)
        self.model.eval()

        try:
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                revision=self.config.revision,
                trust_remote_code=self.config.trust_remote_code,
                **({"cache_dir": self.config.cache_dir} if self.config.cache_dir else {}),
            )
        except Exception as exc:
            raise LVLMBackendError(
                f"Failed to load processor for {self.config.model_name!r}: {exc}"
            ) from exc

        self._tokenizer = getattr(self._processor, "tokenizer", None)
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, revision=self.config.revision
            )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._resolve_image_token_index()
        self._check_attention_support()

    def _resolve_image_token_index(self) -> None:
        """Determine which token id marks image positions in the prompt.

        ``llava-hf`` checkpoints use an in-vocabulary id (32000 for LLaVA-1.5),
        so it must be read from the config rather than assumed.
        """
        model_cfg = getattr(self.model, "config", None)
        for attr in ("image_token_index", "image_token_id", "image_token_id_"):
            value = getattr(model_cfg, attr, None)
            if isinstance(value, int) and value >= 0:
                self._image_token_index = value
                logger.debug("Resolved image token index %d", value)
                return
        if self._processor is not None:
            value = getattr(self._processor, "image_token", None)
            if isinstance(value, int) and value >= 0:
                self._image_token_index = value
                return
        warnings.warn(
            "Could not determine the image placeholder token id from the model "
            "config; falling back to the legacy LLaVA sentinel "
            f"{LEGACY_IMAGE_TOKEN_INDEX}. Attention evidence over image tokens "
            "may be unavailable for this checkpoint."
        )

    def _build_quant_config(self):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise LVLMBackendError(
                "Quantisation requested but bitsandbytes is not installed. "
                "Install it with `pip install bitsandbytes`, or drop "
                "--quantization and run in fp16/fp32."
            ) from exc
        if self.config.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    def _resolve_model_class(self):
        """Prefer the concrete LLaVA class; fall back to AutoModel."""
        try:
            from transformers import LlavaForConditionalGeneration

            return LlavaForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq

    def _check_attention_support(self) -> None:
        """Warn loudly if attention evidence will be unavailable.

        Attention evidence needs the raw attention matrices, which only the eager
        kernel returns. Note that ``config.output_attentions`` is simply its
        default (``False``) on almost every checkpoint and says nothing about
        availability, since we request ``output_attentions=True`` explicitly on
        each forward pass — so it is deliberately not checked here.
        """
        model_cfg = getattr(self.model, "config", None)
        impl = getattr(model_cfg, "_attn_implementation", None)
        if impl is not None and impl != "eager":
            warnings.warn(
                f"Model loaded with attn_implementation={impl!r}; attention "
                "matrices will not be returned, so AttentionEvidence falls back "
                "to image-attention=0 plus candidate embedding similarity. "
                "Re-run with --attn-implementation eager to enable true "
                "attention evidence."
            )

    @staticmethod
    def _transformers_version() -> str:
        try:
            import transformers

            return transformers.__version__
        except ImportError:  # pragma: no cover
            return "unknown"

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------

    def build_prompt(self, question: str) -> str:
        """Render the chat prompt for ``question``.

        Uses the checkpoint's own chat template when it provides one. This
        matters: LLaVA-1.5 expects ``USER: <image>\\n... ASSISTANT:``, whereas
        newer checkpoints built on Qwen/Llama-3 expect ChatML and will emit an
        immediate end-of-turn token if given the LLaVA-1.5 format. Hardcoding a
        single format silently produces empty answers on other models.
        """
        question = question.strip()
        if self._processor is not None:
            apply_template = getattr(self._processor, "apply_chat_template", None)
            if callable(apply_template):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
                try:
                    prompt = apply_template(messages, add_generation_prompt=True)
                    if isinstance(prompt, str) and prompt.strip():
                        return prompt
                except Exception as exc:
                    logger.debug(
                        "chat_template unavailable for %s (%s); using the "
                        "LLaVA-1.5 template", self.config.model_name, exc,
                    )
        return self.PROMPT_TEMPLATE.format(question=question)

    def prepare_inputs(
        self, question: str, image: Any = None
    ) -> Dict[str, torch.Tensor]:
        """Build model inputs for one (question, image) pair.

        The *processor* is used rather than the bare tokenizer on purpose: it
        expands the single ``<image>`` placeholder into one placeholder per
        vision patch. Tokenizing the prompt directly leaves a single placeholder,
        and the model then rejects the call because the number of image tokens
        does not match the number of image features.

        Returns a dict that may contain ``input_ids``, ``attention_mask`` and
        ``pixel_values``, already moved to the model device.
        """
        if self._processor is None:
            raise LVLMBackendError("Backend not loaded; call load() first.")
        prompt = self.build_prompt(question)

        processor = getattr(self._processor, "image_processor", self._processor)
        pil_image = self._to_pil(image) if image is not None else None

        inputs: Dict[str, torch.Tensor] = {}
        if pil_image is not None:
            try:
                raw = self._processor(
                    images=pil_image, text=prompt, return_tensors="pt"
                )
            except Exception:  # pragma: no cover - processor without text path
                raw = {}
            if "input_ids" in raw:
                inputs = dict(raw)
            else:
                # Fall back to manual assembly, expanding the image placeholder
                # to match the vision tower output when the config says so.
                inputs = self._manual_inputs(prompt, pil_image, processor)
        else:
            encoded = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v for k, v in encoded.items() if k in {"input_ids", "attention_mask"}}

        out: Dict[str, torch.Tensor] = {}
        dtype = self._model_dtype()
        for key, value in inputs.items():
            if key == "pixel_values":
                out[key] = value.to(self.device, dtype=dtype)
            elif key in {"input_ids", "attention_mask"}:
                out[key] = value.to(self.device)
        if "input_ids" not in out:
            raise LVLMBackendError(
                "Processor did not return input_ids; cannot build model inputs."
            )
        return out

    def _manual_inputs(
        self, prompt: str, pil_image: Image.Image, processor: Any
    ) -> Dict[str, torch.Tensor]:
        """Assemble inputs when the processor cannot handle text+image jointly.

        Expands ``<image>`` into ``config.image_seq_length`` placeholder tokens
        so the token count matches the vision tower output.
        """
        encoded = self._tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        model_cfg = getattr(self.model, "config", None)
        n_image = int(getattr(model_cfg, "image_seq_length", 0) or 0)
        sentinel = self._image_token_index
        if n_image > 0 and sentinel in input_ids[0].tolist():
            row = input_ids[0].tolist()
            pos = row.index(sentinel)
            row = row[:pos] + [sentinel] * n_image + row[pos + 1:]
            input_ids = torch.tensor([row], dtype=torch.long, device=input_ids.device)
        pixel_values = processor(images=pil_image, return_tensors="pt")["pixel_values"]
        result = {"input_ids": input_ids, "pixel_values": pixel_values}
        if "attention_mask" in encoded:
            mask = torch.ones_like(input_ids)
            result["attention_mask"] = mask
        return result

    @staticmethod
    def _to_pil(image: Any) -> "Image.Image":
        """Coerce a path / PIL image to RGB PIL."""
        if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            raise TypeError(
                "Expected a PIL image or a path, not a preprocessed tensor. "
                "Tensors must be produced by this backend's preprocess_image."
            )
        raise TypeError(f"Unsupported image input type: {type(image)!r}")

    def encode_prompt(self, question: str) -> Dict[str, torch.Tensor]:
        """Tokenise ``question`` into an LLaVA-style chat prompt.

        For text-only use. When an image is involved prefer
        :meth:`prepare_inputs`, which expands the image placeholder correctly.
        """
        if self._tokenizer is None:
            raise LVLMBackendError("Backend not loaded; call load() first.")
        prompt = self.build_prompt(question)
        encoded = self._tokenizer(prompt, return_tensors="pt")
        return {
            k: v.to(self.device)
            for k, v in encoded.items()
            if k in {"input_ids", "attention_mask"}
        }

    def preprocess_image(self, image: Any) -> torch.Tensor:
        """Return a pixel-values tensor on the model device.

        Prefer :meth:`prepare_inputs` for generation, since it guarantees the
        image placeholder count matches the vision tower output.
        """
        if self._processor is None:
            raise LVLMBackendError("Backend not loaded; call load() first.")
        pil_image = self._to_pil(image)
        processor = getattr(self._processor, "image_processor", self._processor)
        pixel_values = processor(images=pil_image, return_tensors="pt")["pixel_values"]
        return pixel_values.to(self.device, dtype=self._model_dtype())

    def _model_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except Exception:  # pragma: no cover
            return torch.float32

    @property
    def device(self) -> torch.device:
        """Device the model parameters live on."""
        if self.model is None:
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover - parameterless module
            return self._resolve_device()

    @property
    def tokenizer(self) -> Any:
        """Tokenizer used by this backend."""
        if self._tokenizer is None:
            raise LVLMBackendError(
                "Tokenizer unavailable; call load() first."
            )
        return self._tokenizer

    @property
    def image_processor(self) -> Any:
        """Image processor used by this backend."""
        if self._processor is None:
            raise LVLMBackendError(
                "Image processor unavailable; call load() first."
            )
        return getattr(self._processor, "image_processor", self._processor)

    # ------------------------------------------------------------------
    # decoding steps
    # ------------------------------------------------------------------

    def _forward(self, output_attentions: bool = False, **kwargs) -> StepOutput:
        out = self.model(
            output_attentions=output_attentions,
            use_cache=True,
            return_dict=True,
            **kwargs,
        )
        attentions = out.attentions if output_attentions else None
        return StepOutput(
            logits=out.logits[:, -1, :].float(),
            attentions=tuple(attentions) if attentions is not None else None,
            past_key_values=out.past_key_values,
            image_token_span=None,
        )

    def initial_step(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        output_attentions: bool = False,
    ) -> StepOutput:
        if self.model is None:
            raise LVLMBackendError("Backend not loaded; call load() first.")
        span = self.image_token_span(input_ids, pixel_values)
        out = self._forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            output_attentions=output_attentions,
        )
        out.image_token_span = span
        return out

    def next_step(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any,
        output_attentions: bool = False,
    ) -> StepOutput:
        if self.model is None:
            raise LVLMBackendError("Backend not loaded; call load() first.")
        return self._forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )


def build_backend(config: LVLMConfig) -> LVLMBackend:
    """Factory that returns a backend for ``config``.

    Extend this to register additional LVLM families; each must implement
    :class:`LVLMBackend`.
    """
    return LLaVAHFBackend(config)


def decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode a sequence of token ids to text, skipping special tokens."""
    return tokenizer.decode(list(token_ids), skip_special_tokens=True)
