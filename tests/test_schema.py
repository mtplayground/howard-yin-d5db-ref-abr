from __future__ import annotations

import pytest

from ref_abr.domain import (
    FrameOutcome,
    MediaObject,
    MediaType,
    MetricRecord,
    WorkloadManifest,
)
from ref_abr.schema import (
    DOMAIN_SCHEMA_VERSION,
    SchemaError,
    materialize_record,
    stamp_record,
    validate_record_payload,
    validate_stamped_record,
)


def test_stamp_record_adds_version_and_record_type() -> None:
    media = MediaObject(
        object_id="obj-a",
        uri="file://obj-a",
        media_type=MediaType.METADATA,
        size_bytes=12,
    )

    stamped = stamp_record(media)

    assert stamped["schema_version"] == DOMAIN_SCHEMA_VERSION
    assert stamped["record_type"] == "media_object"
    assert stamped["object_id"] == "obj-a"
    assert validate_stamped_record(stamped) == stamped


def test_validate_payload_rejects_missing_required_field() -> None:
    with pytest.raises(SchemaError, match="missing required field"):
        validate_record_payload(
            "media_object",
            {
                "object_id": "obj-a",
                "uri": "file://obj-a",
                "media_type": "metadata",
            },
        )


def test_validate_payload_rejects_bad_unit_field_with_path() -> None:
    with pytest.raises(SchemaError, match="record is invalid: size_bytes must be non-negative"):
        validate_record_payload(
            "media_object",
            {
                "object_id": "obj-a",
                "uri": "file://obj-a",
                "media_type": "metadata",
                "size_bytes": -1,
            },
        )


def test_workload_manifest_validates_nested_media_objects() -> None:
    with pytest.raises(SchemaError, match=r"record.media_objects\[0\].*size_bytes"):
        validate_record_payload(
            "workload_manifest",
            {
                "manifest_id": "manifest-a",
                "config_id": "cfg-a",
                "split": "train",
                "seed": 1,
                "media_objects": [
                    {
                        "object_id": "obj-a",
                        "uri": "file://obj-a",
                        "media_type": "metadata",
                        "size_bytes": -1,
                    }
                ],
            },
        )


def test_materialize_record_returns_domain_type() -> None:
    manifest = WorkloadManifest(
        manifest_id="manifest-a",
        config_id="cfg-a",
        split="train",
        seed=1,
        media_objects=(
            MediaObject(
                object_id="obj-a",
                uri="file://obj-a",
                media_type="metadata",
                size_bytes=1,
            ),
        ),
    )

    materialized = materialize_record(stamp_record(manifest), expected_record_type="workload_manifest")

    assert isinstance(materialized, WorkloadManifest)
    assert materialized.media_objects[0].object_id == "obj-a"


def test_version_and_type_mismatches_are_clear_errors() -> None:
    metric = MetricRecord(metric_name="qoe", value=1.0, unit="score")
    stamped = stamp_record(metric)

    with pytest.raises(SchemaError, match="schema_version must be 1"):
        validate_stamped_record({**stamped, "schema_version": 2})

    with pytest.raises(SchemaError, match="record_type must be 'frame_outcome'"):
        validate_stamped_record(stamped, expected_record_type="frame_outcome")


def test_frame_outcome_unit_validation_uses_domain_constraints() -> None:
    with pytest.raises(SchemaError, match="quality_score must be between 0 and 1"):
        stamp_record(
            {
                "record_type": "frame_outcome",
                "frame_id": "frame-a",
                "scheduled_time_ms": 1,
                "rendered_time_ms": 2,
                "deadline_ms": 3,
                "quality_score": 1.5,
            }
        )

    stamped = stamp_record(
        FrameOutcome(
            frame_id="frame-a",
            scheduled_time_ms=1,
            rendered_time_ms=2,
            deadline_ms=3,
            quality_score=1.0,
        )
    )
    assert stamped["record_type"] == "frame_outcome"
