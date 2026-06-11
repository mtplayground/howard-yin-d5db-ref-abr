"""Shared domain records used across workloads, methods, scheduling, and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


class DomainError(ValueError):
    """Raised when a domain record is constructed with invalid values."""


class MediaType(str, Enum):
    """Supported media object categories."""

    GAUSSIAN_SPLAT = "gaussian_splat"
    MESH = "mesh"
    TEXTURE = "texture"
    VIDEO_SEGMENT = "video_segment"
    METADATA = "metadata"


class LifecycleStatus(str, Enum):
    """Reference lifecycle statuses shared by schedulers and metrics."""

    CANDIDATE = "candidate"
    REQUESTED = "requested"
    IN_FLIGHT = "in_flight"
    AVAILABLE = "available"
    EXPIRED = "expired"
    DROPPED = "dropped"


@dataclass(frozen=True)
class MediaObject:
    """Addressable media or reference object that may be scheduled."""

    object_id: str
    uri: str
    media_type: MediaType | str
    size_bytes: int
    duration_ms: int | None = None
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.object_id, "object_id")
        _require_non_empty(self.uri, "uri")
        _require_non_negative_int(self.size_bytes, "size_bytes")
        if self.duration_ms is not None:
            _require_non_negative_int(self.duration_ms, "duration_ms")
        object.__setattr__(self, "media_type", _coerce_enum(MediaType, self.media_type, "media_type"))
        object.__setattr__(self, "dependencies", _freeze_string_tuple(self.dependencies, "dependencies"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "uri": self.uri,
            "media_type": self.media_type.value,
            "size_bytes": self.size_bytes,
            "duration_ms": self.duration_ms,
            "dependencies": list(self.dependencies),
            "metadata": _to_payload_value(self.metadata),
        }


@dataclass(frozen=True)
class WorkloadManifest:
    """Resolved workload manifest containing media objects for one split."""

    manifest_id: str
    config_id: str
    split: str
    seed: int
    media_objects: tuple[MediaObject, ...]
    source_uri: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.manifest_id, "manifest_id")
        _require_non_empty(self.config_id, "config_id")
        _require_non_empty(self.split, "split")
        _require_non_negative_int(self.seed, "seed")
        media_objects = tuple(self.media_objects)
        object_ids = [media_object.object_id for media_object in media_objects]
        if len(object_ids) != len(set(object_ids)):
            raise DomainError("media_objects must not contain duplicate object_id values.")
        if self.source_uri is not None:
            _require_non_empty(self.source_uri, "source_uri")
        object.__setattr__(self, "media_objects", media_objects)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "config_id": self.config_id,
            "split": self.split,
            "seed": self.seed,
            "media_objects": [media_object.as_payload() for media_object in self.media_objects],
            "source_uri": self.source_uri,
            "metadata": _to_payload_value(self.metadata),
        }


@dataclass(frozen=True)
class ReferenceLifecycleState:
    """Current lifecycle status for a reference object."""

    reference_id: str
    status: LifecycleStatus | str
    updated_at_ms: int
    deadline_ms: int | None = None
    attempts: int = 0
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.reference_id, "reference_id")
        _require_non_negative_int(self.updated_at_ms, "updated_at_ms")
        if self.deadline_ms is not None:
            _require_non_negative_int(self.deadline_ms, "deadline_ms")
        _require_non_negative_int(self.attempts, "attempts")
        object.__setattr__(self, "status", _coerce_enum(LifecycleStatus, self.status, "status"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "status": self.status.value,
            "updated_at_ms": self.updated_at_ms,
            "deadline_ms": self.deadline_ms,
            "attempts": self.attempts,
            "metadata": _to_payload_value(self.metadata),
        }


@dataclass(frozen=True)
class ControllerState:
    """Serializable controller state at a scheduling step."""

    controller_id: str
    method_name: str
    step_index: int
    active_split: str
    state: Mapping[str, JsonValue] = field(default_factory=dict)
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.controller_id, "controller_id")
        _require_non_empty(self.method_name, "method_name")
        _require_non_negative_int(self.step_index, "step_index")
        _require_non_empty(self.active_split, "active_split")
        object.__setattr__(self, "state", _freeze_mapping(self.state, "state"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "controller_id": self.controller_id,
            "method_name": self.method_name,
            "step_index": self.step_index,
            "active_split": self.active_split,
            "state": _to_payload_value(self.state),
            "metadata": _to_payload_value(self.metadata),
        }


@dataclass(frozen=True)
class ScheduleDecision:
    """Controller decision for objects to request or keep for a frame."""

    decision_id: str
    controller_id: str
    frame_id: str
    selected_object_ids: tuple[str, ...]
    decision_time_ms: int
    target_deadline_ms: int
    expected_utility: float | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.decision_id, "decision_id")
        _require_non_empty(self.controller_id, "controller_id")
        _require_non_empty(self.frame_id, "frame_id")
        _require_non_negative_int(self.decision_time_ms, "decision_time_ms")
        _require_non_negative_int(self.target_deadline_ms, "target_deadline_ms")
        if self.target_deadline_ms < self.decision_time_ms:
            raise DomainError("target_deadline_ms must be greater than or equal to decision_time_ms.")
        if self.expected_utility is not None:
            _require_finite_number(self.expected_utility, "expected_utility")
        object.__setattr__(
            self,
            "selected_object_ids",
            _freeze_string_tuple(self.selected_object_ids, "selected_object_ids"),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "controller_id": self.controller_id,
            "frame_id": self.frame_id,
            "selected_object_ids": list(self.selected_object_ids),
            "decision_time_ms": self.decision_time_ms,
            "target_deadline_ms": self.target_deadline_ms,
            "expected_utility": self.expected_utility,
            "metadata": _to_payload_value(self.metadata),
        }


@dataclass(frozen=True)
class FrameOutcome:
    """Observed result for a rendered or missed frame."""

    frame_id: str
    scheduled_time_ms: int
    rendered_time_ms: int | None
    deadline_ms: int
    delivered_object_ids: tuple[str, ...] = field(default_factory=tuple)
    missing_object_ids: tuple[str, ...] = field(default_factory=tuple)
    quality_score: float | None = None
    deadline_hit: bool | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.frame_id, "frame_id")
        _require_non_negative_int(self.scheduled_time_ms, "scheduled_time_ms")
        _require_non_negative_int(self.deadline_ms, "deadline_ms")
        if self.rendered_time_ms is not None:
            _require_non_negative_int(self.rendered_time_ms, "rendered_time_ms")
        if self.quality_score is not None:
            _require_unit_interval(self.quality_score, "quality_score")
        object.__setattr__(
            self,
            "delivered_object_ids",
            _freeze_string_tuple(self.delivered_object_ids, "delivered_object_ids"),
        )
        object.__setattr__(
            self,
            "missing_object_ids",
            _freeze_string_tuple(self.missing_object_ids, "missing_object_ids"),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        deadline_hit = self.deadline_hit
        if deadline_hit is None and self.rendered_time_ms is not None:
            deadline_hit = self.rendered_time_ms <= self.deadline_ms
        return {
            "frame_id": self.frame_id,
            "scheduled_time_ms": self.scheduled_time_ms,
            "rendered_time_ms": self.rendered_time_ms,
            "deadline_ms": self.deadline_ms,
            "delivered_object_ids": list(self.delivered_object_ids),
            "missing_object_ids": list(self.missing_object_ids),
            "quality_score": self.quality_score,
            "deadline_hit": deadline_hit,
            "metadata": _to_payload_value(self.metadata),
        }


@dataclass(frozen=True)
class MetricRecord:
    """Single metric value with deterministic tags."""

    metric_name: str
    value: int | float
    unit: str
    tags: Mapping[str, str] = field(default_factory=dict)
    frame_id: str | None = None
    split: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.metric_name, "metric_name")
        _require_non_empty(self.unit, "unit")
        if isinstance(self.value, bool):
            raise DomainError("value must be numeric, not boolean.")
        _require_finite_number(self.value, "value")
        if self.frame_id is not None:
            _require_non_empty(self.frame_id, "frame_id")
        if self.split is not None:
            _require_non_empty(self.split, "split")
        object.__setattr__(self, "tags", _freeze_string_mapping(self.tags, "tags"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "tags": dict(self.tags),
            "frame_id": self.frame_id,
            "split": self.split,
            "metadata": _to_payload_value(self.metadata),
        }


@dataclass(frozen=True)
class FrozenMethodManifest:
    """Manifest for a frozen method implementation and its artifacts."""

    method_id: str
    method_name: str
    version: str
    config_id: str
    artifact_uri: str
    entrypoint: str
    parameters: Mapping[str, JsonValue] = field(default_factory=dict)
    source_uri: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.method_name, "method_name")
        _require_non_empty(self.version, "version")
        _require_non_empty(self.config_id, "config_id")
        _require_non_empty(self.artifact_uri, "artifact_uri")
        _require_non_empty(self.entrypoint, "entrypoint")
        if self.source_uri is not None:
            _require_non_empty(self.source_uri, "source_uri")
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters, "parameters"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "method_name": self.method_name,
            "version": self.version,
            "config_id": self.config_id,
            "artifact_uri": self.artifact_uri,
            "entrypoint": self.entrypoint,
            "parameters": _to_payload_value(self.parameters),
            "source_uri": self.source_uri,
            "metadata": _to_payload_value(self.metadata),
        }


@dataclass(frozen=True)
class ExternalMeasurementRecord:
    """Validated timing, quality, and artifact measurements from an external backend.

    These records are the boundary between RefABR scheduling experiments and
    optional codec/render tooling.  They intentionally contain only normalized
    measurements and provenance; backend-specific algorithm state stays outside
    this repository.
    """

    generation_ms: int | float
    transfer_ms: int | float
    decode_ms: int | float
    restore_ms: int | float
    render_ms: int | float
    size_bytes: int
    visible_quality: int | float
    dropped_frame: bool
    deadline_hit: bool
    provenance: Mapping[str, JsonValue]
    record_id: str | None = None
    backend_id: str | None = None
    object_id: str | None = None
    frame_id: str | None = None
    candidate_kind: str | None = None
    artifact_uri: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("generation_ms", "transfer_ms", "decode_ms", "restore_ms", "render_ms"):
            _require_non_negative_number(getattr(self, field_name), field_name)
        _require_non_negative_int(self.size_bytes, "size_bytes")
        _require_unit_interval(self.visible_quality, "visible_quality")
        _require_bool(self.dropped_frame, "dropped_frame")
        _require_bool(self.deadline_hit, "deadline_hit")
        for field_name in ("record_id", "backend_id", "object_id", "frame_id", "candidate_kind", "artifact_uri"):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_empty(value, field_name)
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance, "provenance"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    @property
    def encoded_bytes(self) -> int:
        """Alias used by codec-oriented tools for the normalized size field."""

        return self.size_bytes

    def as_payload(self) -> dict[str, Any]:
        return {
            "generation_ms": self.generation_ms,
            "transfer_ms": self.transfer_ms,
            "decode_ms": self.decode_ms,
            "restore_ms": self.restore_ms,
            "render_ms": self.render_ms,
            "size_bytes": self.size_bytes,
            "visible_quality": self.visible_quality,
            "dropped_frame": self.dropped_frame,
            "deadline_hit": self.deadline_hit,
            "provenance": _to_payload_value(self.provenance),
            "record_id": self.record_id,
            "backend_id": self.backend_id,
            "object_id": self.object_id,
            "frame_id": self.frame_id,
            "candidate_kind": self.candidate_kind,
            "artifact_uri": self.artifact_uri,
            "metadata": _to_payload_value(self.metadata),
        }


def _coerce_enum(enum_type: type[Enum], value: Enum | str, field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            valid = ", ".join(member.value for member in enum_type)
            raise DomainError(f"{field_name} must be one of: {valid}.") from exc
    raise DomainError(f"{field_name} must be a string or {enum_type.__name__}.")


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise DomainError(f"{field_name} must be a non-empty string.")


def _require_non_negative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DomainError(f"{field_name} must be an integer.")
    if value < 0:
        raise DomainError(f"{field_name} must be non-negative.")


def _require_finite_number(value: int | float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainError(f"{field_name} must be numeric.")
    if value != value or value in {float("inf"), float("-inf")}:
        raise DomainError(f"{field_name} must be finite.")


def _require_non_negative_number(value: int | float, field_name: str) -> None:
    _require_finite_number(value, field_name)
    if value < 0:
        raise DomainError(f"{field_name} must be non-negative.")


def _require_unit_interval(value: float, field_name: str) -> None:
    _require_finite_number(value, field_name)
    if not 0.0 <= value <= 1.0:
        raise DomainError(f"{field_name} must be between 0 and 1.")


def _require_bool(value: bool, field_name: str) -> None:
    if not isinstance(value, bool):
        raise DomainError(f"{field_name} must be a boolean.")


def _freeze_string_tuple(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    frozen = tuple(values)
    for value in frozen:
        _require_non_empty(value, field_name)
    return frozen


def _freeze_string_mapping(values: Mapping[str, str], field_name: str) -> Mapping[str, str]:
    frozen: dict[str, str] = {}
    for key, value in values.items():
        _require_non_empty(key, f"{field_name} key")
        _require_non_empty(value, f"{field_name}.{key}")
        frozen[key] = value
    return MappingProxyType({key: frozen[key] for key in sorted(frozen)})


def _freeze_mapping(values: Mapping[str, JsonValue], field_name: str) -> Mapping[str, JsonValue]:
    frozen: dict[str, JsonValue] = {}
    for key, value in values.items():
        _require_non_empty(key, f"{field_name} key")
        frozen[key] = _freeze_json_value(value, f"{field_name}.{key}")
    return MappingProxyType({key: frozen[key] for key in sorted(frozen)})


def _freeze_json_value(value: JsonValue, field_name: str) -> JsonValue:
    if isinstance(value, dict):
        return _freeze_mapping(value, field_name)
    if isinstance(value, list):
        return tuple(_freeze_json_value(item, field_name) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json_value(item, field_name) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float):
            _require_finite_number(value, field_name)
        return value
    if isinstance(value, Path):
        return str(value)
    raise DomainError(f"{field_name} contains unsupported value type {type(value).__name__}.")


def _to_payload_value(value: JsonValue) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_payload_value(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_to_payload_value(item) for item in value]
    return value


__all__ = [
    "ControllerState",
    "DomainError",
    "ExternalMeasurementRecord",
    "FrameOutcome",
    "FrozenMethodManifest",
    "JsonScalar",
    "JsonValue",
    "LifecycleStatus",
    "MediaObject",
    "MediaType",
    "MetricRecord",
    "ReferenceLifecycleState",
    "ScheduleDecision",
    "WorkloadManifest",
]
