"""External codec/render measurement record loading and validation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping as MappingABC, Sequence
from pathlib import Path
from typing import Any, Mapping

import yaml

from ref_abr.domain import ExternalMeasurementRecord
from ref_abr.schema import (
    RECORD_TYPE_FIELD,
    SCHEMA_VERSION_FIELD,
    SchemaError,
    materialize_record,
    stamp_record,
    validate_record_payload,
)


EXTERNAL_MEASUREMENT_RECORD_TYPE = "external_measurement"
ExternalMeasurementFormat = str


class ExternalMeasurementError(ValueError):
    """Raised when external measurement trace records cannot be loaded."""


def validate_external_measurement_payload(
    payload: Mapping[str, Any],
    *,
    path: str = "record",
) -> dict[str, Any]:
    """Validate and canonicalize one unstamped external-measurement payload."""

    try:
        return validate_record_payload(EXTERNAL_MEASUREMENT_RECORD_TYPE, payload, path=path)
    except SchemaError as exc:
        raise ExternalMeasurementError(str(exc)) from exc


def stamp_external_measurement_record(record: ExternalMeasurementRecord | Mapping[str, Any]) -> dict[str, Any]:
    """Return a schema-stamped external-measurement record."""

    try:
        return stamp_record(record, record_type=EXTERNAL_MEASUREMENT_RECORD_TYPE)
    except SchemaError as exc:
        raise ExternalMeasurementError(str(exc)) from exc


def materialize_external_measurement_record(
    payload: Mapping[str, Any],
    *,
    path: str = "record",
) -> ExternalMeasurementRecord:
    """Validate a stamped or unstamped payload and return its domain record."""

    try:
        if _is_stamped_record(payload):
            record = materialize_record(payload, expected_record_type=EXTERNAL_MEASUREMENT_RECORD_TYPE)
        else:
            canonical = validate_record_payload(EXTERNAL_MEASUREMENT_RECORD_TYPE, payload, path=path)
            record = ExternalMeasurementRecord(**canonical)
    except (SchemaError, TypeError) as exc:
        raise ExternalMeasurementError(str(exc)) from exc
    if not isinstance(record, ExternalMeasurementRecord):
        raise ExternalMeasurementError(f"{path} did not materialize to an external measurement record.")
    return record


def validate_external_measurement_records(
    records: Iterable[Mapping[str, Any] | ExternalMeasurementRecord],
) -> tuple[ExternalMeasurementRecord, ...]:
    """Validate a sequence of external-measurement records."""

    materialized: list[ExternalMeasurementRecord] = []
    for index, record in enumerate(records):
        if isinstance(record, ExternalMeasurementRecord):
            materialized.append(materialize_external_measurement_record(record.as_payload(), path=f"records[{index}]"))
        elif isinstance(record, MappingABC):
            materialized.append(materialize_external_measurement_record(record, path=f"records[{index}]"))
        else:
            raise ExternalMeasurementError(f"records[{index}] must be a mapping or ExternalMeasurementRecord.")
    return tuple(materialized)


def load_external_measurement_records(path: str | Path) -> tuple[ExternalMeasurementRecord, ...]:
    """Load versioned external measurements from JSON, JSONL, YAML, or YML."""

    trace_path = Path(path)
    if not trace_path.exists():
        raise ExternalMeasurementError(f"External measurement trace does not exist: {trace_path}")
    if not trace_path.is_file():
        raise ExternalMeasurementError(f"External measurement trace path is not a file: {trace_path}")
    try:
        text = trace_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExternalMeasurementError(f"Could not read external measurement trace {trace_path}: {exc}") from exc

    suffix = trace_path.suffix.lower()
    try:
        if suffix == ".jsonl":
            raw_records = _load_jsonl_records(text, trace_path)
        elif suffix == ".json":
            raw_records = _records_from_loaded(json.loads(text), trace_path)
        elif suffix in {".yaml", ".yml"}:
            raw_records = _records_from_loaded(yaml.safe_load(text), trace_path)
        else:
            raise ExternalMeasurementError(f"Unsupported external measurement trace extension '{suffix}' for {trace_path}")
    except ExternalMeasurementError:
        raise
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ExternalMeasurementError(f"Could not parse external measurement trace {trace_path}: {exc}") from exc

    return validate_external_measurement_records(raw_records)


def dump_external_measurement_records(
    records: Sequence[ExternalMeasurementRecord | Mapping[str, Any]],
    *,
    stamped: bool = True,
) -> list[dict[str, Any]]:
    """Return canonical payloads for deterministic test fixtures and exporters."""

    materialized = validate_external_measurement_records(records)
    if stamped:
        return [stamp_external_measurement_record(record) for record in materialized]
    return [record.as_payload() for record in materialized]


def _is_stamped_record(payload: Mapping[str, Any]) -> bool:
    return SCHEMA_VERSION_FIELD in payload or RECORD_TYPE_FIELD in payload


def _load_jsonl_records(text: str, trace_path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExternalMeasurementError(f"Could not parse {trace_path}:{line_number}: {exc}") from exc
        if not isinstance(loaded, MappingABC):
            raise ExternalMeasurementError(f"{trace_path}:{line_number} must contain a mapping record.")
        records.append(dict(loaded))
    return records


def _records_from_loaded(loaded: Any, trace_path: Path) -> list[Mapping[str, Any]]:
    if loaded is None:
        return []
    if isinstance(loaded, list):
        raw_records = loaded
    elif isinstance(loaded, MappingABC) and isinstance(loaded.get("records"), list):
        raw_records = loaded["records"]
    elif isinstance(loaded, MappingABC):
        raw_records = [loaded]
    else:
        raise ExternalMeasurementError(f"{trace_path} must contain a record mapping, a records list, or a list of records.")

    records: list[Mapping[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, MappingABC):
            raise ExternalMeasurementError(f"{trace_path} record {index} must be a mapping.")
        records.append(dict(raw_record))
    return records


__all__ = [
    "EXTERNAL_MEASUREMENT_RECORD_TYPE",
    "ExternalMeasurementError",
    "ExternalMeasurementFormat",
    "dump_external_measurement_records",
    "load_external_measurement_records",
    "materialize_external_measurement_record",
    "stamp_external_measurement_record",
    "validate_external_measurement_payload",
    "validate_external_measurement_records",
]
