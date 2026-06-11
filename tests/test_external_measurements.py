from __future__ import annotations

import json

import pytest

from ref_abr.domain import ExternalMeasurementRecord
from ref_abr.external_measurements import (
    ExternalMeasurementError,
    dump_external_measurement_records,
    load_external_measurement_records,
    materialize_external_measurement_record,
    stamp_external_measurement_record,
    validate_external_measurement_payload,
)
from ref_abr.schema import DOMAIN_SCHEMA_VERSION, SchemaError, materialize_record, stamp_record, validate_record_payload


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "generation_ms": 1.5,
        "transfer_ms": 2.0,
        "decode_ms": 3.0,
        "restore_ms": 4.0,
        "render_ms": 5.0,
        "encoded_bytes": 4096,
        "visible_quality": 0.92,
        "dropped_frame": False,
        "deadline_hit": True,
        "provenance": {
            "backend": "fixture",
            "source_uri": "file://trace/sample.mp4",
            "command": ["fixture", "--measure"],
        },
        "record_id": "measurement-a",
        "backend_id": "fixture-backend",
        "object_id": "object-a",
        "frame_id": "frame-0001",
        "candidate_kind": "gaussian_base",
    }
    payload.update(overrides)
    return payload


def test_external_measurement_schema_canonicalizes_encoded_bytes_alias() -> None:
    canonical = validate_record_payload("external_measurement", _payload())

    assert canonical["size_bytes"] == 4096
    assert "encoded_bytes" not in canonical
    assert canonical["visible_quality"] == 0.92
    assert canonical["provenance"]["command"] == ["fixture", "--measure"]


def test_external_measurement_record_can_be_stamped_and_materialized() -> None:
    stamped = stamp_record({"record_type": "external_measurement", **_payload()})
    materialized = materialize_record(stamped, expected_record_type="external_measurement")

    assert stamped["schema_version"] == DOMAIN_SCHEMA_VERSION
    assert stamped["record_type"] == "external_measurement"
    assert isinstance(materialized, ExternalMeasurementRecord)
    assert materialized.encoded_bytes == 4096
    assert materialized.backend_id == "fixture-backend"


def test_external_measurement_rejects_invalid_timings_quality_and_alias_conflicts() -> None:
    with pytest.raises(SchemaError, match="generation_ms must be non-negative"):
        validate_record_payload("external_measurement", _payload(generation_ms=-0.1))

    with pytest.raises(SchemaError, match="visible_quality must be between 0 and 1"):
        validate_record_payload("external_measurement", _payload(visible_quality=1.1))

    with pytest.raises(SchemaError, match="encoded_bytes must match"):
        validate_record_payload("external_measurement", _payload(size_bytes=10, encoded_bytes=11))

    with pytest.raises(SchemaError, match="size_bytes/encoded_bytes"):
        payload = _payload()
        payload.pop("encoded_bytes")
        validate_record_payload("external_measurement", payload)


def test_external_measurement_helpers_wrap_schema_errors() -> None:
    with pytest.raises(ExternalMeasurementError, match="dropped_frame must be a boolean"):
        validate_external_measurement_payload(_payload(dropped_frame="no"))

    record = materialize_external_measurement_record(_payload())
    stamped = stamp_external_measurement_record(record)

    assert stamped["record_type"] == "external_measurement"
    assert stamped["size_bytes"] == 4096


def test_load_external_measurements_supports_json_jsonl_and_yaml(tmp_path) -> None:
    stamped = stamp_external_measurement_record(_payload())
    json_path = tmp_path / "measurements.json"
    jsonl_path = tmp_path / "measurements.jsonl"
    yaml_path = tmp_path / "measurements.yaml"

    json_path.write_text(json.dumps({"records": [stamped]}), encoding="utf-8")
    jsonl_path.write_text(json.dumps(stamped) + "\n", encoding="utf-8")
    yaml_path.write_text(
        """
records:
  - generation_ms: 1.5
    transfer_ms: 2.0
    decode_ms: 3.0
    restore_ms: 4.0
    render_ms: 5.0
    size_bytes: 4096
    visible_quality: 0.92
    dropped_frame: false
    deadline_hit: true
    provenance:
      backend: fixture
      source_uri: file://trace/sample.mp4
""",
        encoding="utf-8",
    )

    assert load_external_measurement_records(json_path)[0].size_bytes == 4096
    assert load_external_measurement_records(jsonl_path)[0].record_id == "measurement-a"
    assert load_external_measurement_records(yaml_path)[0].provenance["backend"] == "fixture"


def test_dump_external_measurement_records_is_deterministic_and_stamped() -> None:
    dumped = dump_external_measurement_records([_payload()])

    assert dumped == [stamp_external_measurement_record(_payload())]
