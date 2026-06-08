from __future__ import annotations

import json

import pytest

from ref_abr.accounting import (
    CandidateResourceAccount,
    ComponentTimingAccount,
    ResourceUtilization,
)
from ref_abr.artifacts import (
    ArtifactExportError,
    ArtifactProvenance,
    RawArtifactExportConfig,
    export_raw_artifacts,
    raw_artifact_envelope,
)
from ref_abr.candidates import CandidateObject
from ref_abr.domain import ControllerState, FrameOutcome, ScheduleDecision
from ref_abr.lifecycle import LifecycleAction, LifecyclePhase, ReferenceLifecycleEvent
from ref_abr.schema import DOMAIN_SCHEMA_VERSION, validate_stamped_record


def test_export_raw_artifacts_writes_jsonl_files_with_versioned_payloads(tmp_path) -> None:
    provenance = ArtifactProvenance(
        run_id="run-1",
        config_id="cfg-1",
        split="final",
        method_id="method-1",
        source="unit-test",
    )

    manifest = export_raw_artifacts(
        tmp_path,
        provenance=provenance,
        object_candidates=(_candidate(),),
        controller_states=(_controller_state(),),
        decisions=(_decision(),),
        lifecycle_events=(_lifecycle_event(),),
        frame_outcomes=(_frame_outcome(),),
        timing_records=(_account(),),
    )

    payload = manifest.as_payload()
    assert payload["schema_version"] == DOMAIN_SCHEMA_VERSION
    assert payload["record_type"] == "raw_artifact_manifest"
    assert len(payload["files"]) == 6
    assert (tmp_path / "manifest.json").exists()

    candidates_file = tmp_path / "object_candidates.jsonl"
    candidate_rows = [json.loads(line) for line in candidates_file.read_text(encoding="utf-8").splitlines()]
    assert candidate_rows[0]["schema_version"] == DOMAIN_SCHEMA_VERSION
    assert candidate_rows[0]["record_type"] == "candidate_object"
    assert candidate_rows[0]["artifact_name"] == "object_candidates"
    assert candidate_rows[0]["sequence_index"] == 0
    assert candidate_rows[0]["provenance"]["run_id"] == "run-1"
    assert candidate_rows[0]["payload"]["schema_version"] == DOMAIN_SCHEMA_VERSION
    assert candidate_rows[0]["payload"]["record_type"] == "candidate_object"
    assert candidate_rows[0]["payload"]["candidate_id"] == "candidate-1"

    outcomes_file = tmp_path / "frame_outcomes.jsonl"
    outcome_row = json.loads(outcomes_file.read_text(encoding="utf-8").splitlines()[0])
    assert validate_stamped_record(outcome_row["payload"], expected_record_type="frame_outcome")["frame_id"] == "frame-1"


def test_export_raw_artifacts_writes_json_arrays_when_requested(tmp_path) -> None:
    provenance = ArtifactProvenance(run_id="run-json")
    manifest = export_raw_artifacts(
        tmp_path,
        provenance=provenance,
        decisions=(_decision(),),
        config=RawArtifactExportConfig(output_format="json"),
    )

    decision_file = tmp_path / "decisions.json"
    rows = json.loads(decision_file.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    assert rows[0]["payload"]["record_type"] == "schedule_decision"
    assert manifest.files[0].format == "json"
    assert manifest.files[0].record_count == 1


def test_raw_artifact_envelope_is_deterministic_and_schema_stamped() -> None:
    provenance = ArtifactProvenance(run_id="run-2")

    first = raw_artifact_envelope(_decision(), provenance=provenance, artifact_name="decisions", sequence_index=0)
    second = raw_artifact_envelope(_decision(), provenance=provenance, artifact_name="decisions", sequence_index=0)

    assert first == second
    assert first["schema_version"] == DOMAIN_SCHEMA_VERSION
    assert validate_stamped_record(first["payload"], expected_record_type="schedule_decision")["decision_id"] == "decision-1"


def test_export_rejects_malformed_record_collections(tmp_path) -> None:
    provenance = ArtifactProvenance(run_id="run-3")
    with pytest.raises(ArtifactExportError, match="object_candidates"):
        export_raw_artifacts(
            tmp_path,
            provenance=provenance,
            object_candidates=(object(),),  # type: ignore[arg-type]
        )

    with pytest.raises(ArtifactExportError, match="output_format"):
        RawArtifactExportConfig(output_format="csv")  # type: ignore[arg-type]


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
        method_name="method",
        step_index=0,
        active_split="final",
        state={"frame_id": "frame-1"},
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
    )


def _frame_outcome() -> FrameOutcome:
    return FrameOutcome(
        frame_id="frame-1",
        scheduled_time_ms=10,
        rendered_time_ms=18,
        deadline_ms=20,
        delivered_object_ids=("object-1",),
        quality_score=0.9,
        deadline_hit=True,
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
    )
