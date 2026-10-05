# -*- coding: utf-8 -*-
"""
Reproducibility and environment capture.

Every result JSON produced by ``src.run`` carries the provenance recorded
here: seed, library versions, device, git commit. Without it a number cannot
be re-derived, so this module deliberately fails loudly rather than emitting
partial metadata that looks complete.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch


@dataclass
class ReproducibilityConfig:
    """Seed / determinism switches."""

    seed: int = 42
    deterministic: bool = True
    cudnn_deterministic: bool = True
    benchmark: bool = False


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that can influence a run.

    Also configures cuDNN determinism. Note the honest limitation: bit-exact
    reproducibility across different GPUs, CUDA versions, or kernels is not
    guaranteed by PyTorch, so ``deterministic`` reduces variance rather than
    eliminating it.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):  # pragma: no cover - old torch
            pass


def git_commit(repo_dir: Optional[str] = None) -> Optional[str]:
    """Return the current git commit, or ``None`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_dirty(repo_dir: Optional[str] = None) -> Optional[bool]:
    """Whether the working tree has uncommitted changes."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def device_info() -> Dict[str, Any]:
    """Describe the compute device(s) actually used."""
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_mb": props.total_memory / (1024 ** 2),
                "cuda_version": torch.version.cuda,
            }
        )
    return info


def peak_memory_mb() -> Optional[float]:
    """Peak CUDA memory for this process, if CUDA is in use."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def library_versions() -> Dict[str, str]:
    """Versions of the libraries that can change numerical results."""
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    for name in ("transformers", "accelerate", "bitsandbytes", "numpy", "PIL", "yaml"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover
        pass
    return versions


def collect_metadata(
    seed: int,
    extra: Optional[Dict[str, Any]] = None,
    repo_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the provenance block stored alongside every result."""
    metadata: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "device": device_info(),
        "library_versions": library_versions(),
        "git_commit": git_commit(repo_dir),
        "git_dirty": git_dirty(repo_dir),
    }
    if extra:
        metadata.update(extra)
    return metadata


def apply_reproducibility(config: ReproducibilityConfig) -> None:
    """Convenience wrapper: seed + determinism in one call."""
    set_seed(config.seed, deterministic=config.deterministic)
    if torch.backends.cudnn.is_available():  # pragma: no cover
        torch.backends.cudnn.deterministic = config.cudnn_deterministic
        torch.backends.cudnn.benchmark = config.benchmark
