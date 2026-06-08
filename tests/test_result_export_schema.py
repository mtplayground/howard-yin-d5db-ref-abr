from __future__ import annotations

import hashlib
import json

import pytest

from ref_abr.accounting import CandidateResourceAccount, ComponentTimingAccount, ResourceAccountingSummary, ResourceUtilization
from ref_abr.artifacts import ArtifactExportError, ArtifactProvenance, RawArtifactExportConfig, export_raw_artifacts
from ref_abr.candidates import CandidateObject
from ref_abr.domain import ControllerState, FrameOutcome, MetricRecord, ScheduleDecision
from ref_abr.lifecycle import LifecycleAction, LifecyclePhase, ReferenceLifecycleEvent
from ref_abr.schema import DOMAIN_SCHEMA_VERSION, SchemaError, validate_stamped_record


def test_raw_first_export_records_carry_schema_provenance_and_source_references(tmp_path) -> None:
    provenance = ArtifactProvenance(
        run_id="run-schema-1",
        config_id="cfg-schema",
        split="final",
        method_id="refabr",
        source="schema-test",
        metadata={
            "source_references": {
                "workload_manifest_id": "workload-final",
                "network_trace_id": "trace-7",
                "viewport_trace_id": "viewport-3",
            }
        },
    )
    summary = ResourceAccountingSummary(
        summary_id="summary-1",
        accounts=(_account(),),
        total_timing=ComponentTimingAccount(1, 2, 3, 4, 5, 6),
        peak_memory_mb=64.0,
        total_transfer_bytes=1024,
    )

    manifest = export_raw_artifacts(
        tmp_path,
        provenance=provenance,
        object_candidates=(_candidate(),),
        controller_states=(_controller_state(),),
        decisions=(_decision(),),
        lifecycle_events=(_lifecycle_event(),),
        frame_outcomes=(_frame_outcome(),),
        metric_records=(_metric_record(),),
        timing_records=(_account(), summary, ComponentTimingAccount(1, 2, 3, 4, 5, 6)),
        config=RawArtifactExportConfig(output_format="jsonl", include_empty_files=True),
    )

    manifest_payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload == manifest.as_payload()
    assert manifest_payload["schema_version"] == DOMAIN_SCHEMA_VERSION
    assert manifest_payload["record_type"] == "raw_artifact_manifest"
    assert manifest_payload["provenance"]["run_id"] == "run-schema-1"
    assert manifest_payload["provenance"]["metadata"]["source_references"]["workload_manifest_id"] == "workload-final"
    assert {artifact_file["artifact_name"] for artifact_file in manifest_payload["files"]} == {
        "object_candidates",
        "controller_states",
        "decisions",
        "lifecycle_events",
        "frame_outcomes",
        "metric_records",
        "timing_records",
    }

    rows_by_artifact = {
        artifact_file["artifact_name"]: _read_jsonl_artifact(tmp_path, artifact_file)
        for artifact_file in manifest_payload["files"]
    }
    for artifact_file in manifest_payload["files"]:
        content = (tmp_path / f"{artifact_file['artifact_name']}.jsonl").read_text(encoding="utf-8")
        assert artifact_file["sha256"] == hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert artifact_file["path"].endswith(f"{artifact_file['artifact_name']}.jsonl")
        assert artifact_file["record_count"] == len(rows_by_artifact[artifact_file["artifact_name"]])

    for rows in rows_by_artifact.values():
        for sequence_index, row in enumerate(rows):
            assert row["schema_version"] == DOMAIN_SCHEMA_VERSION
            assert row["sequence_index"] == sequence_index
            assert row["artifact_name"]
            assert row["provenance"]["run_id"] == "run-schema-1"
            assert row["provenance"]["config_id"] == "cfg-schema"
            assert row["provenance"]["split"] == "final"
            assert row["provenance"]["method_id"] == "refabr"
            assert row["provenance"]["source"] == "schema-test"
            assert row["provenance"]["metadata"]["source_references"]["network_trace_id"] == "trace-7"
            assert row["payload"]["schema_version"] == DOMAIN_SCHEMA_VERSION
            assert row["payload"]["record_type"] == row["record_type"]

    assert validate_stamped_record(rows_by_artifact["controller_states"][0]["payload"], expected_record_type="controller_state")["controller_id"] == "controller-1"
    assert validate_stamped_record(rows_by_artifact["decisions"][0]["payload"], expected_record_type="schedule_decision")["decision_id"] == "decision-1"
    assert validate_stamped_record(rows_by_artifact["frame_outcomes"][0]["payload"], expected_record_type="frame_outcome")["frame_id"] == "frame-1"
    assert validate_stamped_record(rows_by_artifact["metric_records"][0]["payload"], expected_record_type="metric_record")["metric_name"] == "deadline_qoe"

    assert rows_by_artifact["object_candidates"][0]["payload"]["candidate_id"] == "candidate-1"
    assert rows_by_artifact["lifecycle_events"][0]["payload"]["event_id"] == "event-1"
    assert [row["record_type"] for row in rows_by_artifact["timing_records"]] == [
        "candidate_resource_account",
        "resource_accounting_summary",
        "component_timing_account",
    ]


def test_json_export_uses_same_envelope_schema(tmp_path) -> None:
    manifest = export_raw_artifacts(
        tmp_path,
        provenance=ArtifactProvenance(run_id="run-json", source="json-schema-test"),
        decisions=(_decision(),),
        config=RawArtifactExportConfig(output_format="json"),
    )

    rows = json.loads((tmp_path / "decisions.json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["schema_version"] == DOMAIN_SCHEMA_VERSION
    assert rows[0]["record_type"] == "schedule_decision"
    assert rows[0]["provenance"]["source"] == "json-schema-test"
    assert validate_stamped_record(rows[0]["payload"], expected_record_type="schedule_decision")["selected_object_ids"] == ["object-1"]
    assert manifest.files[0].format == "json"


def test_missing_export_payload_fields_fail_with_clear_schema_error(tmp_path) -> None:
    export_raw_artifacts(
        tmp_path,
        provenance=ArtifactProvenance(run_id="run-missing"),
        decisions=(_decision(),),
    )
    row = json.loads((tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    malformed_payload = dict(row["payload"])
    malformed_payload.pop("selected_object_ids")

    with pytest.raises(SchemaError, match="record is missing required field\\(s\\): selected_object_ids"):
        validate_stamped_record(malformed_payload, expected_record_type="schedule_decision")


def test_missing_provenance_fields_fail_clearly() -> None:
    with pytest.raises(ArtifactExportError, match="run_id"):
        ArtifactProvenance(run_id="")

    with pytest.raises(ArtifactExportError, match="source"):
        ArtifactProvenance(run_id="run-1", source="")


def _read_jsonl_artifact(tmp_path, artifact_file: dict[str, object]) -> list[dict[str, object]]:
    path = tmp_path / f"{artifact_file['artifact_name']}.jsonl"
    content = path.read_text(encoding="utf-8")
    if not content:
        return []
    return [json.loads(line) for line in content.splitlines()]


def _candidate() -> CandidateObject:
    return CandidateObject(
        candidate_id="candidate-1",
        object_id="object-1",
        candidate_kind="gaussian_base",
        decision_time_ms=10,
        layer=0,
        resolution="720p",
        fov_deg=90.0,
        viewpoint=None,
        lookahead_ms=0,
        expiration_ms=100,
        retransmit_priority=0,
        size_bytes=1024,
    )


def _controller_state() -> ControllerState:
    return ControllerState(
        controller_id="controller-1",
        method_name="refabr",
        step_index=0,
        active_split="final",
        state={"frame_id": "frame-1"},
        metadata={"source_reference": "controller-log-1"},
    )


def _decision() -> ScheduleDecision:
    return ScheduleDecision(
        decision_id="decision-1",
        controller_id="controller-1",
        frame_id="frame-1",
        selected_object_ids=("object-1",),
        decision_time_ms=10,
        target_deadline_ms=20,
        expected_utility=0.5,
        metadata={"source_reference": "decision-log-1"},
    )


def _lifecycle_event() -> ReferenceLifecycleEvent:
    return ReferenceLifecycleEvent(
        event_id="event-1",
        reference_id="object-1",
        action=LifecycleAction.REQUEST,
        from_phase=LifecyclePhase.CANDIDATE,
        to_phase=LifecyclePhase.REQUESTED,
        status="requested",
        event_time_ms=10,
        deadline_ms=20,
        attempts=1,
        metadata={"source_reference": "lifecycle-log-1"},
    )


def _frame_outcome() -> FrameOutcome:
    return FrameOutcome(
        frame_id="frame-1",
        scheduled_time_ms=10,
        rendered_time_ms=18,
        deadline_ms=20,
        delivered_object_ids=("object-1",),
        missing_object_ids=(),
        quality_score=0.9,
        deadline_hit=True,
        metadata={"source_reference": "frame-log-1"},
    )


def _metric_record() -> MetricRecord:
    return MetricRecord(
        metric_name="deadline_qoe",
        value=0.75,
        unit="score",
        frame_id="frame-1",
        split="final",
        tags={"method": "refabr"},
        metadata={"source_reference": "metric-log-1"},
    )


def _account() -> CandidateResourceAccount:
    return CandidateResourceAccount(
        account_id="account-1",
        candidate_id="candidate-1",
        object_id="object-1",
        provider_id="provider-1",
        device_profile_id="device-1",
        timing=ComponentTimingAccount(
            server_generation_ms=1.0,
            queue_ms=1.0,
            transfer_ms=1.0,
            decode_ms=1.0,
            restore_ms=1.0,
            render_ms=1.0,
        ),
        utilization=ResourceUtilization(
            server_generation=0.1,
            queue=0.1,
            transfer_time=0.1,
            decode=0.1,
            restore=0.1,
            render=0.1,
            memory=0.1,
        ),
        memory_mb=64.0,
        bandwidth_bps=None,
        transfer_bytes=1024,
        metadata={"source_reference": "timing-log-1"},
    )
