from __future__ import annotations

import pytest

from ref_abr.domain import (
    ControllerState,
    DomainError,
    FrameOutcome,
    FrozenMethodManifest,
    LifecycleStatus,
    MediaObject,
    MediaType,
    MetricRecord,
    ReferenceLifecycleState,
    ScheduleDecision,
    WorkloadManifest,
)


def test_workload_manifest_payload_is_deterministic() -> None:
    media = MediaObject(
        object_id="obj-a",
        uri="s3://bucket/prefix/obj-a.bin",
        media_type=MediaType.GAUSSIAN_SPLAT,
        size_bytes=1024,
        duration_ms=33,
        dependencies=("base",),
        metadata={"z": 2, "a": ["x", "y"]},
    )

    manifest = WorkloadManifest(
        manifest_id="manifest-a",
        config_id="cfg-a",
        split="train",
        seed=7,
        media_objects=(media,),
        source_uri="file://workload.json",
        metadata={"owner": "test"},
    )

    assert manifest.as_payload() == {
        "manifest_id": "manifest-a",
        "config_id": "cfg-a",
        "split": "train",
        "seed": 7,
        "media_objects": [
            {
                "object_id": "obj-a",
                "uri": "s3://bucket/prefix/obj-a.bin",
                "media_type": "gaussian_splat",
                "size_bytes": 1024,
                "duration_ms": 33,
                "dependencies": ["base"],
                "metadata": {"a": ["x", "y"], "z": 2},
            }
        ],
        "source_uri": "file://workload.json",
        "metadata": {"owner": "test"},
    }


def test_duplicate_media_object_ids_are_rejected() -> None:
    media = MediaObject(
        object_id="obj-a",
        uri="file://obj-a",
        media_type="metadata",
        size_bytes=1,
    )

    with pytest.raises(DomainError, match="duplicate object_id"):
        WorkloadManifest(
            manifest_id="manifest-a",
            config_id="cfg-a",
            split="train",
            seed=1,
            media_objects=(media, media),
        )


def test_lifecycle_and_controller_records_round_trip_to_payloads() -> None:
    lifecycle = ReferenceLifecycleState(
        reference_id="ref-a",
        status=LifecycleStatus.IN_FLIGHT,
        updated_at_ms=10,
        deadline_ms=20,
        attempts=1,
    )
    controller = ControllerState(
        controller_id="ctrl-a",
        method_name="baseline",
        step_index=2,
        active_split="calibration",
        state={"queue": ["ref-a"]},
    )

    assert lifecycle.as_payload()["status"] == "in_flight"
    assert controller.as_payload()["state"] == {"queue": ["ref-a"]}


def test_schedule_decision_and_frame_outcome_validate_timing() -> None:
    with pytest.raises(DomainError, match="target_deadline_ms"):
        ScheduleDecision(
            decision_id="decision-a",
            controller_id="ctrl-a",
            frame_id="frame-a",
            selected_object_ids=("obj-a",),
            decision_time_ms=100,
            target_deadline_ms=99,
        )

    outcome = FrameOutcome(
        frame_id="frame-a",
        scheduled_time_ms=100,
        rendered_time_ms=110,
        deadline_ms=120,
        delivered_object_ids=("obj-a",),
        quality_score=0.75,
    )

    assert outcome.as_payload()["deadline_hit"] is True


def test_metric_record_and_frozen_method_manifest_payloads() -> None:
    metric = MetricRecord(
        metric_name="deadline_hit_rate",
        value=0.9,
        unit="ratio",
        tags={"method": "baseline", "split": "final"},
        split="final",
    )
    frozen = FrozenMethodManifest(
        method_id="method-a",
        method_name="baseline",
        version="1.0",
        config_id="cfg-a",
        artifact_uri="file://methods/baseline.json",
        entrypoint="ref_abr.methods.baseline:build",
        parameters={"cadence_ms": 33},
    )

    assert metric.as_payload()["tags"] == {"method": "baseline", "split": "final"}
    assert frozen.as_payload()["parameters"] == {"cadence_ms": 33}
