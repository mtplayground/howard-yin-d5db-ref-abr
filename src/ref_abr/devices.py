"""Device profile loading and normalization."""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ref_abr.config import ConfigError, load_config_file, stable_config_id


class DeviceError(ValueError):
    """Raised when device profile metadata cannot be normalized."""


DEVICE_CLASSES: tuple[str, ...] = ("server", "edge", "desktop", "laptop", "mobile")


@dataclass(frozen=True)
class DeviceBudgets:
    """Timing, memory, and frame-rate budgets for a device profile."""

    generation_ms: float
    transfer_ms: float
    restoration_ms: float
    render_ms: float
    memory_mb: float
    fps: float

    def __post_init__(self) -> None:
        for field_name in ("generation_ms", "transfer_ms", "restoration_ms", "render_ms"):
            object.__setattr__(self, field_name, _non_negative_float(getattr(self, field_name), field_name))
        object.__setattr__(self, "memory_mb", _positive_float(self.memory_mb, "memory_mb"))
        object.__setattr__(self, "fps", _positive_float(self.fps, "fps"))

    def as_payload(self) -> dict[str, float]:
        return {
            "generation_ms": self.generation_ms,
            "transfer_ms": self.transfer_ms,
            "restoration_ms": self.restoration_ms,
            "render_ms": self.render_ms,
            "memory_mb": self.memory_mb,
            "fps": self.fps,
        }


@dataclass(frozen=True)
class DeviceProfile:
    """Normalized device profile used by schedulers and harnesses."""

    profile_id: str
    device_class: str
    budgets: DeviceBudgets
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise DeviceError("profile_id must be a non-empty string.")
        object.__setattr__(self, "profile_id", self.profile_id.strip())
        object.__setattr__(self, "device_class", _device_class(self.device_class, field_name="device_class"))
        if not isinstance(self.budgets, DeviceBudgets):
            raise DeviceError("budgets must be a DeviceBudgets record.")
        if self.description is not None:
            if not isinstance(self.description, str) or not self.description.strip():
                raise DeviceError("description must be a non-empty string when provided.")
            object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "device_class": self.device_class,
            "budgets": self.budgets.as_payload(),
            "description": self.description,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class DeviceProfileSet:
    """Collection of normalized device profiles keyed by profile_id."""

    profiles: tuple[DeviceProfile, ...]
    source_uri: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        profiles = tuple(self.profiles)
        if not profiles:
            raise DeviceError("profiles must contain at least one device profile.")
        profile_ids = [profile.profile_id for profile in profiles]
        duplicates = sorted({profile_id for profile_id in profile_ids if profile_ids.count(profile_id) > 1})
        if duplicates:
            raise DeviceError(f"profiles must not contain duplicate profile_id values: {', '.join(duplicates)}.")
        if self.source_uri is not None and not self.source_uri:
            raise DeviceError("source_uri must be non-empty when provided.")
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def by_id(self, profile_id: str) -> DeviceProfile:
        try:
            return {profile.profile_id: profile for profile in self.profiles}[profile_id]
        except KeyError as exc:
            raise DeviceError(f"Unknown device profile_id {profile_id!r}.") from exc

    def as_payload(self) -> dict[str, Any]:
        return {
            "profiles": [profile.as_payload() for profile in self.profiles],
            "source_uri": self.source_uri,
            "metadata": _to_payload(self.metadata),
        }


def load_device_profiles(path: str | Path) -> DeviceProfileSet:
    """Load server/edge/desktop/laptop/mobile device profiles from disk."""

    profile_path = Path(path)
    try:
        raw_profiles = load_config_file(profile_path)
    except ConfigError as exc:
        raise DeviceError(str(exc)) from exc
    return normalize_device_profiles(raw_profiles, source_uri=str(profile_path))


def normalize_device_profiles(
    raw_profiles: Mapping[str, Any],
    *,
    source_uri: str | None = None,
) -> DeviceProfileSet:
    """Normalize flexible device profile metadata into typed profile records."""

    root = _require_mapping(raw_profiles, "device_profiles")
    profile_entries = _profile_entries(root)
    profiles = tuple(
        _profile_from_raw(
            entry,
            index=index,
            profile_key=profile_key,
            source_uri=source_uri,
        )
        for index, (profile_key, entry) in enumerate(profile_entries)
    )
    metadata = {
        "provenance": {
            "source_uri": source_uri,
            "profile_count": len(profiles),
            "device_classes": sorted({profile.device_class for profile in profiles}),
        }
    }
    return DeviceProfileSet(profiles=profiles, source_uri=source_uri, metadata=metadata)


def _profile_from_raw(
    raw_profile: Mapping[str, Any],
    *,
    index: int,
    profile_key: str | None,
    source_uri: str | None,
) -> DeviceProfile:
    profile = _require_mapping(raw_profile, f"device_profiles[{index}]")
    raw_id = _string_or_none(_first_present(profile, ("profile_id", "device_id", "id", "name"))) or profile_key
    description = _string_or_none(_first_present(profile, ("description", "notes", "label")))
    device_class = _resolve_device_class(profile, raw_id, index)
    budgets = _budgets_from_profile(profile, index)
    profile_id = raw_id.strip() if raw_id else _generated_profile_id(device_class, budgets, description)
    metadata = _profile_metadata(profile, index=index, profile_key=profile_key, source_uri=source_uri)
    return DeviceProfile(
        profile_id=profile_id,
        device_class=device_class,
        budgets=budgets,
        description=description,
        metadata=metadata,
    )


def _budgets_from_profile(profile: Mapping[str, Any], index: int) -> DeviceBudgets:
    path = f"device_profiles[{index}]"
    generation_ms = _budget_ms(
        profile,
        ("generation_ms", "generation_budget_ms", "generate_ms", "compute_ms", "generation"),
        f"{path}.generation_ms",
    )
    transfer_ms = _budget_ms(
        profile,
        ("transfer_ms", "transfer_budget_ms", "network_ms", "network_transfer_ms", "transfer"),
        f"{path}.transfer_ms",
    )
    restoration_ms = _budget_ms(
        profile,
        ("restoration_ms", "restoration_budget_ms", "restore_ms", "decode_ms", "restoration", "restore"),
        f"{path}.restoration_ms",
    )
    render_ms = _budget_ms(
        profile,
        ("render_ms", "render_budget_ms", "rendering_ms", "frame_render_ms", "render"),
        f"{path}.render_ms",
    )
    memory_mb = _memory_mb(profile, path)
    fps = _positive_float(
        _required_value(profile, ("fps", "target_fps", "frame_rate", "frames_per_second"), f"{path}.fps"),
        f"{path}.fps",
    )
    return DeviceBudgets(
        generation_ms=generation_ms,
        transfer_ms=transfer_ms,
        restoration_ms=restoration_ms,
        render_ms=render_ms,
        memory_mb=memory_mb,
        fps=fps,
    )


def _profile_entries(root: Mapping[str, Any]) -> tuple[tuple[str | None, Mapping[str, Any]], ...]:
    container = _first_present(root, ("profiles", "devices", "device_profiles"))
    if container is None:
        return ((None, root),)
    if isinstance(container, MappingABC) and any(key in container for key in ("profiles", "devices", "device_profiles")):
        return _profile_entries(container)
    if isinstance(container, list):
        if not container:
            raise DeviceError("device_profiles.profiles must contain at least one profile.")
        return tuple(
            (None, _require_mapping(profile, f"device_profiles.profiles[{index}]"))
            for index, profile in enumerate(container)
        )
    if isinstance(container, MappingABC):
        if not container:
            raise DeviceError("device_profiles.profiles must contain at least one profile.")
        return tuple(
            (
                str(profile_key),
                _require_mapping(profile, f"device_profiles.profiles.{profile_key}"),
            )
            for profile_key, profile in sorted(container.items(), key=lambda item: str(item[0]))
        )
    raise DeviceError("device_profiles.profiles must be a list or mapping when present.")


def _resolve_device_class(profile: Mapping[str, Any], profile_id: str | None, index: int) -> str:
    raw_class = _first_present(profile, ("device_class", "class", "type", "kind", "category"))
    if raw_class is not None:
        return _device_class(str(raw_class), field_name=f"device_profiles[{index}].device_class")
    if profile_id:
        for token in _identifier_tokens(profile_id):
            try:
                return _device_class(token, field_name=f"device_profiles[{index}].device_class")
            except DeviceError:
                continue
    raise DeviceError(
        f"device_profiles[{index}].device_class is required when it cannot be inferred from profile_id."
    )


def _device_class(value: str, *, field_name: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "cloud": "server",
        "datacenter": "server",
        "gpu-server": "server",
        "server": "server",
        "edge": "edge",
        "edge-node": "edge",
        "desktop": "desktop",
        "workstation": "desktop",
        "pc": "desktop",
        "laptop": "laptop",
        "notebook": "laptop",
        "mobile": "mobile",
        "phone": "mobile",
        "smartphone": "mobile",
        "handset": "mobile",
        "tablet": "mobile",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in DEVICE_CLASSES:
        raise DeviceError(f"{field_name} must be one of: {', '.join(DEVICE_CLASSES)}.")
    return resolved


def _budget_ms(profile: Mapping[str, Any], keys: tuple[str, ...], field_name: str) -> float:
    return _non_negative_float(_required_value(profile, keys, field_name), field_name)


def _memory_mb(profile: Mapping[str, Any], path: str) -> float:
    value = _optional_value(profile, ("memory_mb", "memory_mib", "memory_budget_mb", "ram_mb", "vram_mb"))
    if value is not None:
        return _positive_float(value, f"{path}.memory_mb")
    value = _optional_value(profile, ("memory_gb", "memory_budget_gb", "ram_gb", "vram_gb"))
    if value is not None:
        return _positive_float(value, f"{path}.memory_gb") * 1024.0
    value = _optional_value(profile, ("memory_bytes", "memory_budget_bytes"))
    if value is not None:
        return _positive_float(value, f"{path}.memory_bytes") / 1_000_000.0
    raise DeviceError(f"{path}.memory_mb is required.")


def _required_value(profile: Mapping[str, Any], keys: tuple[str, ...], field_name: str) -> Any:
    value = _optional_value(profile, keys)
    if value is None:
        raise DeviceError(f"{field_name} is required.")
    return value


def _optional_value(profile: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    direct = _first_present(profile, keys)
    if direct is not None:
        return direct
    for section_name in ("budgets", "timing", "latency", "performance", "resources", "resource_budgets"):
        section = profile.get(section_name)
        if isinstance(section, MappingABC):
            nested = _first_present(section, keys)
            if nested is not None:
                return nested
    return None


def _profile_metadata(
    profile: Mapping[str, Any],
    *,
    index: int,
    profile_key: str | None,
    source_uri: str | None,
) -> dict[str, Any]:
    supplied = profile.get("metadata")
    metadata: dict[str, Any] = _plain_json_mapping(supplied, "metadata") if supplied is not None else {}
    metadata["provenance"] = {
        "source_uri": source_uri,
        "source_index": index,
        "profile_key": profile_key,
    }
    attributes = {
        str(key): _plain_json_value(value, f"attributes.{key}")
        for key, value in sorted(profile.items(), key=lambda item: str(item[0]))
        if key not in _KNOWN_PROFILE_KEYS
    }
    if attributes:
        metadata["attributes"] = attributes
    return metadata


def _generated_profile_id(device_class: str, budgets: DeviceBudgets, description: str | None) -> str:
    payload = {
        "device_class": device_class,
        "budgets": budgets.as_payload(),
        "description": description,
    }
    return f"{device_class}-{stable_config_id(payload)}"


def _identifier_tokens(value: str) -> tuple[str, ...]:
    normalized = value.strip().lower().replace("_", "-")
    return tuple(token for token in normalized.replace("/", "-").replace(".", "-").split("-") if token)


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DeviceError(f"{field_name} must be numeric.")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise DeviceError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise DeviceError(f"{field_name} must be finite.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0:
        raise DeviceError(f"{field_name} must be non-negative.")
    return parsed


def _positive_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed <= 0:
        raise DeviceError(f"{field_name} must be positive.")
    return parsed


def _plain_json_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, MappingABC):
        raise DeviceError(f"{field_name} must be a mapping.")
    return {
        str(key): _plain_json_value(nested, f"{field_name}.{key}")
        for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
    }


def _plain_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, MappingABC):
        return _plain_json_mapping(value, field_name)
    if isinstance(value, list | tuple):
        return [_plain_json_value(item, field_name) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise DeviceError(f"{field_name} must be finite.")
        return value
    raise DeviceError(f"{field_name} contains unsupported value type {type(value).__name__}.")


def _to_payload(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {key: _to_payload(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_to_payload(item) for item in value]
    if isinstance(value, list):
        return [_to_payload(item) for item in value]
    return value


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise DeviceError(f"{path} must be a mapping.")
    return value


_KNOWN_PROFILE_KEYS: frozenset[str] = frozenset(
    {
        "profile_id",
        "device_id",
        "id",
        "name",
        "description",
        "notes",
        "label",
        "device_class",
        "class",
        "type",
        "kind",
        "category",
        "metadata",
        "budgets",
        "timing",
        "latency",
        "performance",
        "resources",
        "resource_budgets",
        "generation_ms",
        "generation_budget_ms",
        "generate_ms",
        "compute_ms",
        "generation",
        "transfer_ms",
        "transfer_budget_ms",
        "network_ms",
        "network_transfer_ms",
        "transfer",
        "restoration_ms",
        "restoration_budget_ms",
        "restore_ms",
        "decode_ms",
        "restoration",
        "restore",
        "render_ms",
        "render_budget_ms",
        "rendering_ms",
        "frame_render_ms",
        "render",
        "memory_mb",
        "memory_mib",
        "memory_budget_mb",
        "ram_mb",
        "vram_mb",
        "memory_gb",
        "memory_budget_gb",
        "ram_gb",
        "vram_gb",
        "memory_bytes",
        "memory_budget_bytes",
        "fps",
        "target_fps",
        "frame_rate",
        "frames_per_second",
    }
)


__all__ = [
    "DEVICE_CLASSES",
    "DeviceBudgets",
    "DeviceError",
    "DeviceProfile",
    "DeviceProfileSet",
    "load_device_profiles",
    "normalize_device_profiles",
]
