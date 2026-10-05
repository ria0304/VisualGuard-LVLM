# -*- coding: utf-8 -*-
"""
Configuration loading.

Experimental parameters live in YAML files under ``configs/`` and are merged
with CLI overrides. Nothing scientific is hard-coded in the source tree: the
weights, thresholds and channel selections all come from here.

Merge precedence (lowest to highest)::

    dataclass defaults  <  YAML config  <  CLI flags
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Type, TypeVar

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised for malformed or contradictory configuration."""


def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore

        return yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigError(
            "PyYAML is required to read config files. "
            "Install with `pip install pyyaml`."
        ) from exc


def load_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML (or JSON) config file into a dict."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(
            f"Config file not found: {path}\n"
            f"Available configs in {path.parent}: "
            f"{sorted(p.name for p in path.parent.glob('*.yaml'))}"
        )
    yaml = _require_yaml()
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except Exception as exc:
            raise ConfigError(f"Failed to parse {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path} must contain a top-level mapping, got {type(data).__name__}"
        )
    return data


def deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def filter_known(cls: Type[T], values: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only keys that are actual fields of dataclass ``cls``.

    Unknown keys raise, so typos are never silently ignored.
    """
    if not is_dataclass(cls):
        raise ConfigError(f"{cls} is not a dataclass")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ConfigError(
            f"Unknown config keys for {cls.__name__}: {unknown}. "
            f"Valid keys: {sorted(known)}"
        )
    return {k: v for k, v in values.items() if k in known}


def build_config(cls: Type[T], values: Mapping[str, Any]) -> T:
    """Instantiate dataclass ``cls`` from a flat mapping.

    Keys belonging to other config classes are ignored, so one YAML file can
    describe the evidence, decoding and model settings together. Keys that no
    configured class recognises still raise (see :func:`build_configs`).
    """
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in values.items() if k in known and v is not None}
    try:
        return cls(**kwargs)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid values for {cls.__name__}: {exc}") from exc


def build_configs(
    specs: Mapping[str, Type[Any]], values: Mapping[str, Any]
) -> Dict[str, Any]:
    """Instantiate several config dataclasses from one flat mapping.

    Args:
        specs: ``{name: dataclass_type}``.
        values: The merged YAML + CLI mapping.

    Every key must be claimed by at least one dataclass in ``specs``; otherwise a
    ``ConfigError`` is raised listing the offending keys. This is what lets a
    single ``configs/*.yaml`` file cover evidence, decoding and grounding
    settings while still catching genuine typos.
    """
    claimed: set = set()
    for cls in specs.values():
        if not is_dataclass(cls):
            raise ConfigError(f"{cls} is not a dataclass")
        claimed.update(f.name for f in fields(cls))

    provided = {k for k, v in values.items() if v is not None}
    unknown = sorted(provided - claimed)
    if unknown:
        raise ConfigError(
            f"Unknown config keys: {unknown}. "
            f"Recognised keys across {sorted(specs)}: {sorted(claimed)}"
        )
    return {name: build_config(cls, values) for name, cls in specs.items()}


def resolve_configs(
    config_path: Optional[Path],
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a YAML config and apply CLI overrides.

    Supports a ``defaults: <name>`` key inside a YAML file so specialised
    configs can inherit from e.g. ``visualguard.yaml`` without duplicating
    every weight.
    """
    merged: Dict[str, Any] = {}
    if config_path is not None:
        merged = _load_with_inheritance(Path(config_path), seen=set())
    if overrides:
        merged = deep_update(merged, overrides)
    return merged


def _load_with_inheritance(path: Path, seen: set) -> Dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"Circular config inheritance detected at {path}")
    seen.add(path)

    data = load_yaml(path)
    parent_name = data.pop("defaults", None)
    if parent_name is None:
        return data

    parent_path = (path.parent / f"{parent_name}.yaml")
    if not parent_path.is_file():
        raise ConfigError(
            f"{path} declares defaults={parent_name!r} but "
            f"{parent_path} does not exist"
        )
    parent = _load_with_inheritance(parent_path, seen)
    return deep_update(parent, data)


def to_dict(config: Any) -> Dict[str, Any]:
    """Serialise dataclass configs (recursively) to plain dicts."""
    if is_dataclass(config):
        return {k: to_dict(v) for k, v in asdict(config).items()}
    if isinstance(config, dict):
        return {k: to_dict(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return [to_dict(v) for v in config]
    return config


def save_json(payload: Any, path: Path) -> None:
    """Write JSON with sorted keys for diff-friendly results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")


def load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
