"""Schema version stamping and validation for shared domain records."""

from __future__ import annotations

import copy
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, Mapping

from ref_abr.domain import (
    ControllerState,
    DomainError,
    FrameOutcome,
    FrozenMethodManifest,
    MediaObject,
    MetricRecord,
    ReferenceLifecycleState,
    ScheduleDecision,
    WorkloadManifest,
)


DOMAIN_SCHEMA_VERSION = 1
SCHEMA_VERSION_FIELD = "schema_version"
RECORD_TYPE_FIELD = "record_type"


class SchemaError(ValueError):
    """Raised when a domain record payload violates the shared schema."""


@dataclass(frozen=True)
class SchemaSpec:
    """Required and optional payload fields for one record type."""

    record_type: str
    domain_type: type
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()

    @property
    def allowed_fields(self) -> frozenset[str]:
        return frozenset((*self.required_fields, *self.optional_fields))


SCHEMA_SPECS: Mapping[str, SchemaSpec] = {
    "media_object": SchemaSpec(
        record_type="media_object",
        domain_type=MediaObject,
        required_fields=("object_id", "uri", "media_type", "size_bytes"),
        optional_fields=("duration_ms", "dependencies", "metadata"),
    ),
    "workload_manifest": SchemaSpec(
        record_type="workload_manifest",
        domain_type=WorkloadManifest,
        required_fields=("manifest_id", "config_id", "split", "seed", "media_objects"),
        optional_fields=("source_uri", "metadata"),
    ),
    "reference_lifecycle_state": SchemaSpec(
        record_type="reference_lifecycle_state",
        domain_type=ReferenceLifecycleState,
        required_fields=("reference_id", "status", "updated_at_ms"),
        optional_fields=("deadline_ms", "attempts", "metadata"),
    ),
    "controller_state": SchemaSpec(
        record_type="controller_state",
        domain_type=ControllerState,
        required_fields=("controller_id", "method_name", "step_index", "active_split"),
        optional_fields=("state", "metadata"),
    ),
    "schedule_decision": SchemaSpec(
        record_type="schedule_decision",
        domain_type=ScheduleDecision,
        required_fields=(
            "decision_id",
            "controller_id",
            "frame_id",
            "selected_object_ids",
            "decision_time_ms",
            "target_deadline_ms",
        ),
        optional_fields=("expected_utility", "metadata"),
    ),
    "frame_outcome": SchemaSpec(
        record_type="frame_outcome",
        domain_type=FrameOutcome,
        required_fields=("frame_id", "scheduled_time_ms", "rendered_time_ms", "deadline_ms"),
        optional_fields=("delivered_object_ids", "missing_object_ids", "quality_score", "deadline_hit", "metadata"),
    ),
    "metric_record": SchemaSpec(
        record_type="metric_record",
        domain_type=MetricRecord,
        required_fields=("metric_name", "value", "unit"),
        optional_fields=("tags", "frame_id", "split", "metadata"),
    ),
    "frozen_method_manifest": SchemaSpec(
        record_type="frozen_method_manifest",
        domain_type=FrozenMethodManifest,
        required_fields=("method_id", "method_name", "version", "config_id", "artifact_uri", "entrypoint"),
        optional_fields=("parameters", "source_uri", "metadata"),
    ),
}


_DOMAIN_TYPE_TO_RECORD_TYPE: Mapping[type, str] = {
    spec.domain_type: record_type for record_type, spec in SCHEMA_SPECS.items()
}


def stamp_record(record: object, record_type: str | None = None) -> dict[str, Any]:
    """Return a schema-stamped payload for a domain record or raw payload."""

    resolved_record_type = record_type or infer_record_type(record)
    payload = _payload_from_record(record)
    validated = validate_record_payload(resolved_record_type, payload)
    stamped: dict[str, Any] = {
        SCHEMA_VERSION_FIELD: DOMAIN_SCHEMA_VERSION,
        RECORD_TYPE_FIELD: resolved_record_type,
    }
    stamped.update(validated)
    return stamped


def validate_stamped_record(
    stamped_payload: Mapping[str, Any],
    expected_record_type: str | None = None,
) -> dict[str, Any]:
    """Validate a schema-stamped record and return its canonical stamped payload."""

    payload = _require_mapping(stamped_payload, "record")
    version = payload.get(SCHEMA_VERSION_FIELD)
    if version != DOMAIN_SCHEMA_VERSION:
        raise SchemaError(
            f"record.{SCHEMA_VERSION_FIELD} must be {DOMAIN_SCHEMA_VERSION}; got {version!r}."
        )
    record_type = payload.get(RECORD_TYPE_FIELD)
    if not isinstance(record_type, str) or not record_type:
        raise SchemaError(f"record.{RECORD_TYPE_FIELD} must be a non-empty string.")
    if expected_record_type is not None and record_type != expected_record_type:
        raise SchemaError(
            f"record.{RECORD_TYPE_FIELD} must be {expected_record_type!r}; got {record_type!r}."
        )

    raw_record = {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key not in {SCHEMA_VERSION_FIELD, RECORD_TYPE_FIELD}
    }
    canonical = validate_record_payload(record_type, raw_record)
    return {
        SCHEMA_VERSION_FIELD: DOMAIN_SCHEMA_VERSION,
        RECORD_TYPE_FIELD: record_type,
        **canonical,
    }


def validate_record_payload(record_type: str, payload: Mapping[str, Any], path: str = "record") -> dict[str, Any]:
    """Validate an unstamped payload for a known record type."""

    spec = _schema_spec(record_type)
    raw_payload = _require_mapping(payload, path)
    _validate_field_set(spec, raw_payload, path)
    record = _build_domain_record(record_type, raw_payload, path)
    return record.as_payload()


def materialize_record(stamped_payload: Mapping[str, Any], expected_record_type: str | None = None) -> object:
    """Validate a stamped payload and return the corresponding domain record."""

    validated = validate_stamped_record(stamped_payload, expected_record_type=expected_record_type)
    record_type = validated[RECORD_TYPE_FIELD]
    raw_payload = {
        key: value
        for key, value in validated.items()
        if key not in {SCHEMA_VERSION_FIELD, RECORD_TYPE_FIELD}
    }
    return _build_domain_record(record_type, raw_payload, "record")


def infer_record_type(record: object) -> str:
    """Infer the schema record type for a domain object or stamped mapping."""

    if isinstance(record, MappingABC):
        record_type = record.get(RECORD_TYPE_FIELD)
        if isinstance(record_type, str) and record_type:
            _schema_spec(record_type)
            return record_type
    for domain_type, record_type in _DOMAIN_TYPE_TO_RECORD_TYPE.items():
        if isinstance(record, domain_type):
            return record_type
    raise SchemaError(f"Cannot infer schema record type for {type(record).__name__}.")


def _payload_from_record(record: object) -> Mapping[str, Any]:
    if isinstance(record, MappingABC):
        return {
            key: copy.deepcopy(value)
            for key, value in record.items()
            if key not in {SCHEMA_VERSION_FIELD, RECORD_TYPE_FIELD}
        }
    as_payload = getattr(record, "as_payload", None)
    if callable(as_payload):
        payload = as_payload()
        if not isinstance(payload, MappingABC):
            raise SchemaError(f"{type(record).__name__}.as_payload() must return a mapping.")
        return payload
    raise SchemaError(f"{type(record).__name__} is not a domain record payload.")


def _schema_spec(record_type: str) -> SchemaSpec:
    try:
        return SCHEMA_SPECS[record_type]
    except KeyError as exc:
        valid = ", ".join(sorted(SCHEMA_SPECS))
        raise SchemaError(f"Unknown record_type {record_type!r}. Expected one of: {valid}.") from exc


def _validate_field_set(spec: SchemaSpec, payload: Mapping[str, Any], path: str) -> None:
    missing = [field for field in spec.required_fields if field not in payload]
    if missing:
        raise SchemaError(f"{path} is missing required field(s): {', '.join(missing)}.")

    allowed = set(spec.allowed_fields)
    unknown = sorted(field for field in payload if field not in allowed)
    if unknown:
        raise SchemaError(f"{path} contains unknown field(s): {', '.join(unknown)}.")


def _build_domain_record(record_type: str, payload: Mapping[str, Any], path: str) -> object:
    try:
        if record_type == "media_object":
            return MediaObject(**payload)
        if record_type == "workload_manifest":
            media_objects = _materialize_media_objects(payload["media_objects"], f"{path}.media_objects")
            return WorkloadManifest(
                manifest_id=payload["manifest_id"],
                config_id=payload["config_id"],
                split=payload["split"],
                seed=payload["seed"],
                media_objects=media_objects,
                source_uri=payload.get("source_uri"),
                metadata=payload.get("metadata", {}),
            )
        if record_type == "reference_lifecycle_state":
            return ReferenceLifecycleState(**payload)
        if record_type == "controller_state":
            return ControllerState(**payload)
        if record_type == "schedule_decision":
            return ScheduleDecision(**payload)
        if record_type == "frame_outcome":
            return FrameOutcome(**payload)
        if record_type == "metric_record":
            return MetricRecord(**payload)
        if record_type == "frozen_method_manifest":
            return FrozenMethodManifest(**payload)
    except KeyError as exc:
        raise SchemaError(f"{path} is missing required field {exc.args[0]!r}.") from exc
    except TypeError as exc:
        raise SchemaError(f"{path} has malformed field types: {exc}") from exc
    except DomainError as exc:
        raise SchemaError(f"{path} is invalid: {exc}") from exc
    _schema_spec(record_type)
    raise SchemaError(f"Unsupported record_type {record_type!r}.")


def _materialize_media_objects(raw_media_objects: Any, path: str) -> tuple[MediaObject, ...]:
    if not isinstance(raw_media_objects, list):
        raise SchemaError(f"{path} must be a list of media_object payloads.")
    media_objects: list[MediaObject] = []
    for index, raw_media_object in enumerate(raw_media_objects):
        item_path = f"{path}[{index}]"
        validated = validate_record_payload("media_object", raw_media_object, path=item_path)
        media_objects.append(MediaObject(**validated))
    return tuple(media_objects)


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise SchemaError(f"{path} must be a mapping.")
    return value


__all__ = [
    "DOMAIN_SCHEMA_VERSION",
    "RECORD_TYPE_FIELD",
    "SCHEMA_SPECS",
    "SCHEMA_VERSION_FIELD",
    "SchemaError",
    "SchemaSpec",
    "infer_record_type",
    "materialize_record",
    "stamp_record",
    "validate_record_payload",
    "validate_stamped_record",
]
