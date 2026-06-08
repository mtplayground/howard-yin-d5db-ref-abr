"""Configuration loading, deterministic seeds, and split identity resolution."""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


CONFIG_RECORD_VERSION = 1
SEED_MIN = 0
SEED_MAX = 2**32 - 1
SPLIT_NAMES: tuple[str, ...] = ("train", "calibration", "final")


class ConfigError(ValueError):
    """Raised when a config file cannot be parsed or resolved."""


@dataclass(frozen=True)
class SeedSpec:
    """Resolved deterministic root seed."""

    value: int

    def as_payload(self) -> dict[str, int]:
        return {"value": self.value}


@dataclass(frozen=True)
class SplitIdentity:
    """Resolved identity and deterministic seed for a workload split."""

    name: str
    identity: str
    seed: int

    def as_payload(self) -> dict[str, str | int]:
        return {"name": self.name, "identity": self.identity, "seed": self.seed}


@dataclass(frozen=True)
class ResolvedConfig:
    """Fully resolved config record with a stable content-derived identifier."""

    config_id: str
    version: int
    seed: SeedSpec
    active_split: str
    splits: Mapping[str, SplitIdentity]
    values: Mapping[str, Any]
    source_path: Path | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "version": self.version,
            "seed": self.seed.as_payload(),
            "active_split": self.active_split,
            "splits": {name: self.splits[name].as_payload() for name in SPLIT_NAMES},
            "values": _canonicalize_json(self.values),
            "source_path": str(self.source_path) if self.source_path is not None else None,
        }

    def stable_payload(self) -> dict[str, Any]:
        payload = self.as_payload()
        payload.pop("config_id", None)
        payload.pop("source_path", None)
        return payload


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON, TOML, YAML, or YML config file into a mapping."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    if not config_path.is_file():
        raise ConfigError(f"Config path is not a file: {config_path}")

    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read config file {config_path}: {exc}") from exc

    suffix = config_path.suffix.lower()
    try:
        if suffix == ".json":
            loaded = json.loads(text)
        elif suffix == ".toml":
            loaded = tomllib.loads(text)
        elif suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(text)
        else:
            raise ConfigError(f"Unsupported config file extension '{suffix}' for {config_path}")
    except ConfigError:
        raise
    except (json.JSONDecodeError, tomllib.TOMLDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not parse config file {config_path}: {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config file {config_path} must contain a top-level mapping.")
    return _canonicalize_json(loaded)


def resolve_config(
    path: str | Path,
    overrides: Mapping[str, str] | None = None,
    split: str | None = None,
) -> ResolvedConfig:
    """Load and resolve a config file into a stable config record."""

    config_path = Path(path)
    values = load_config_file(config_path)
    if overrides:
        values = apply_overrides(values, overrides)

    seed = SeedSpec(_resolve_root_seed(values.get("seed", 0)))
    active_split = _resolve_active_split(values, split)
    splits = _resolve_splits(values.get("splits", {}), seed.value)
    resolved = ResolvedConfig(
        config_id="",
        version=CONFIG_RECORD_VERSION,
        seed=seed,
        active_split=active_split,
        splits=splits,
        values=values,
        source_path=config_path,
    )
    return ResolvedConfig(
        config_id=stable_config_id(resolved.stable_payload()),
        version=resolved.version,
        seed=resolved.seed,
        active_split=resolved.active_split,
        splits=resolved.splits,
        values=resolved.values,
        source_path=resolved.source_path,
    )


def apply_overrides(values: Mapping[str, Any], overrides: Mapping[str, str]) -> dict[str, Any]:
    """Apply dot-path KEY=VALUE overrides to a config mapping."""

    resolved = copy.deepcopy(dict(values))
    for dotted_key, raw_value in sorted(overrides.items()):
        path = dotted_key.split(".")
        if not dotted_key or any(segment == "" for segment in path):
            raise ConfigError(f"Override key '{dotted_key}' must be a non-empty dot path.")
        cursor: dict[str, Any] = resolved
        for segment in path[:-1]:
            existing = cursor.get(segment)
            if existing is None:
                existing = {}
                cursor[segment] = existing
            if not isinstance(existing, dict):
                raise ConfigError(f"Override key '{dotted_key}' conflicts with non-mapping value at '{segment}'.")
            cursor = existing
        cursor[path[-1]] = _parse_override_value(raw_value)
    return _canonicalize_json(resolved)


def stable_config_id(payload: Mapping[str, Any]) -> str:
    """Return a stable short ID for canonical JSON-compatible config content."""

    canonical = json.dumps(_canonicalize_json(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _resolve_root_seed(seed_value: Any) -> int:
    if isinstance(seed_value, dict):
        if "base" in seed_value:
            seed_value = seed_value["base"]
        elif "value" in seed_value:
            seed_value = seed_value["value"]
        else:
            raise ConfigError("Seed mapping must include 'base' or 'value'.")
    return _coerce_seed(seed_value, "seed")


def _resolve_active_split(values: Mapping[str, Any], split: str | None) -> str:
    raw_split = split or values.get("split") or values.get("active_split") or "train"
    if not isinstance(raw_split, str):
        raise ConfigError("Active split must be a string.")
    if raw_split not in SPLIT_NAMES:
        valid = ", ".join(SPLIT_NAMES)
        raise ConfigError(f"Unknown split '{raw_split}'. Expected one of: {valid}.")
    return raw_split


def _resolve_splits(raw_splits: Any, root_seed: int) -> dict[str, SplitIdentity]:
    if raw_splits is None:
        raw_splits = {}
    if not isinstance(raw_splits, dict):
        raise ConfigError("Config 'splits' must be a mapping.")

    unknown = sorted(str(name) for name in raw_splits if name not in SPLIT_NAMES)
    if unknown:
        valid = ", ".join(SPLIT_NAMES)
        raise ConfigError(f"Unknown split identity keys: {', '.join(unknown)}. Expected only: {valid}.")

    splits: dict[str, SplitIdentity] = {}
    for name in SPLIT_NAMES:
        node = raw_splits.get(name, name)
        if isinstance(node, str):
            identity = node
            seed = _derive_split_seed(root_seed, name, identity)
        elif isinstance(node, dict):
            identity_value = node.get("identity", name)
            if not isinstance(identity_value, str) or not identity_value:
                raise ConfigError(f"Split '{name}' identity must be a non-empty string.")
            identity = identity_value
            seed = (
                _coerce_seed(node["seed"], f"splits.{name}.seed")
                if "seed" in node
                else _derive_split_seed(root_seed, name, identity)
            )
        else:
            raise ConfigError(f"Split '{name}' must be a string or mapping.")
        splits[name] = SplitIdentity(name=name, identity=identity, seed=seed)
    return splits


def _coerce_seed(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be an integer seed, not a boolean.")
    if isinstance(value, int):
        seed = value
    elif isinstance(value, str) and value.isdecimal():
        seed = int(value)
    else:
        raise ConfigError(f"{field_name} must be an integer seed.")
    if not SEED_MIN <= seed <= SEED_MAX:
        raise ConfigError(f"{field_name} must be between {SEED_MIN} and {SEED_MAX}.")
    return seed


def _derive_split_seed(root_seed: int, name: str, identity: str) -> int:
    digest = hashlib.sha256(f"{root_seed}:{name}:{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (SEED_MAX + 1)


def _parse_override_value(value: str) -> Any:
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse override value '{value}': {exc}") from exc
    return value if parsed is None and value != "null" else parsed


def _canonicalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        canonical: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ConfigError("Config mappings must use string keys.")
            canonical[key] = _canonicalize_json(nested)
        return {key: canonical[key] for key in sorted(canonical)}
    if isinstance(value, list):
        return [_canonicalize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ConfigError(f"Config value of type {type(value).__name__} is not supported.")
