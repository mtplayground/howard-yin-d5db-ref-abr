"""Configuration loading, deterministic seeds, and split identity resolution."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


CONFIG_RECORD_VERSION = 1
SEED_MIN = 0
SEED_MAX = 2**32 - 1
SPLIT_NAMES: tuple[str, ...] = ("train", "calibration", "final")
ENV_DEFAULTS: dict[str, str] = {
    "REF_ABR_ARTIFACT_ROOT": "artifacts",
    "REF_ABR_DATASET_BASE_PATH": "data/datasets",
    "REF_ABR_TRACE_BASE_PATH": "data/traces",
    "REF_ABR_DEFAULT_RUN_NAME": "default",
    "REF_ABR_DEFAULT_SEED": "0",
    "REF_ABR_DEFAULT_SPLIT": "train",
    "REF_ABR_MAX_WORKERS": "1",
    "REF_ABR_OVERWRITE_OUTPUTS": "false",
    "REF_ABR_LOG_LEVEL": "INFO",
}
REQUIRED_ENV_KEYS: tuple[str, ...] = tuple(ENV_DEFAULTS)


class ConfigError(ValueError):
    """Raised when a config file cannot be parsed or resolved."""


@dataclass(frozen=True)
class EnvConfig:
    """Environment-derived paths and default run settings."""

    artifact_output_root: Path
    dataset_base_path: Path
    trace_base_path: Path
    default_run_name: str
    default_seed: int
    default_split: str
    max_workers: int
    overwrite_outputs: bool
    log_level: str

    def as_payload(self) -> dict[str, str | int | bool]:
        return {
            "artifact_output_root": str(self.artifact_output_root),
            "dataset_base_path": str(self.dataset_base_path),
            "trace_base_path": str(self.trace_base_path),
            "default_run_name": self.default_run_name,
            "default_seed": self.default_seed,
            "default_split": self.default_split,
            "max_workers": self.max_workers,
            "overwrite_outputs": self.overwrite_outputs,
            "log_level": self.log_level,
        }


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


def load_env_config(
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = None,
) -> EnvConfig:
    """Resolve environment configuration from defaults, an optional env file, and env vars."""

    values = dict(ENV_DEFAULTS)
    if env_file is not None:
        values.update(load_env_file(env_file))
    source_environ = os.environ if environ is None else environ
    values.update({key: value for key, value in source_environ.items() if key in REQUIRED_ENV_KEYS})

    return EnvConfig(
        artifact_output_root=_coerce_env_path(values["REF_ABR_ARTIFACT_ROOT"], "REF_ABR_ARTIFACT_ROOT"),
        dataset_base_path=_coerce_env_path(values["REF_ABR_DATASET_BASE_PATH"], "REF_ABR_DATASET_BASE_PATH"),
        trace_base_path=_coerce_env_path(values["REF_ABR_TRACE_BASE_PATH"], "REF_ABR_TRACE_BASE_PATH"),
        default_run_name=_coerce_env_string(values["REF_ABR_DEFAULT_RUN_NAME"], "REF_ABR_DEFAULT_RUN_NAME"),
        default_seed=_coerce_seed(values["REF_ABR_DEFAULT_SEED"], "REF_ABR_DEFAULT_SEED"),
        default_split=_coerce_env_split(values["REF_ABR_DEFAULT_SPLIT"], "REF_ABR_DEFAULT_SPLIT"),
        max_workers=_coerce_positive_int(values["REF_ABR_MAX_WORKERS"], "REF_ABR_MAX_WORKERS"),
        overwrite_outputs=_coerce_bool(values["REF_ABR_OVERWRITE_OUTPUTS"], "REF_ABR_OVERWRITE_OUTPUTS"),
        log_level=_coerce_log_level(values["REF_ABR_LOG_LEVEL"], "REF_ABR_LOG_LEVEL"),
    )


def load_env_file(path: str | Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file without mutating process environment."""

    env_path = Path(path)
    if not env_path.exists():
        raise ConfigError(f"Env file does not exist: {env_path}")
    if not env_path.is_file():
        raise ConfigError(f"Env path is not a file: {env_path}")
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"Could not read env file {env_path}: {exc}") from exc

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ConfigError(f"{env_path}:{line_number} must use KEY=VALUE format.")
        values[key] = _unquote_env_value(value.strip())
    return values


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


def _coerce_env_path(value: str, field_name: str) -> Path:
    cleaned = _coerce_env_string(value, field_name)
    return Path(cleaned).expanduser()


def _coerce_env_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _coerce_env_split(value: str, field_name: str) -> str:
    split = _coerce_env_string(value, field_name)
    if split not in SPLIT_NAMES:
        valid = ", ".join(SPLIT_NAMES)
        raise ConfigError(f"{field_name} must be one of: {valid}.")
    return split


def _coerce_positive_int(value: str, field_name: str) -> int:
    if not isinstance(value, str) or not value.isdecimal():
        raise ConfigError(f"{field_name} must be a positive integer.")
    parsed = int(value)
    if parsed < 1:
        raise ConfigError(f"{field_name} must be a positive integer.")
    return parsed


def _coerce_bool(value: str, field_name: str) -> bool:
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} must be a boolean string.")
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{field_name} must be one of: true, false, 1, 0, yes, no, on, off.")


def _coerce_log_level(value: str, field_name: str) -> str:
    normalized = _coerce_env_string(value, field_name).upper()
    if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError(f"{field_name} must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    return normalized


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


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
