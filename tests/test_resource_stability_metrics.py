from __future__ import annotations

import pytest

from ref_abr.accounting import CandidateResourceAccount, ComponentTimingAccount, ResourceAccountingSummary, ResourceUtilization
from ref_abr.domain import FrameOutcome, ScheduleDecision
from ref_abr.resource_stability_metrics import (
    NO_CONTROL_TRANSITIONS,
    NO_RECOVERY_EVENTS,
    NO_RESOURCE_RECORDS,
    NO_VIEWPORT_SAMPLES,
    ResourceStabilityMetricConfig,
    ResourceStabilityMetricError,
    ViewportPredictionMetricSample,
    compute_resource_stability_viewport_metrics,
    control_stability_metrics,
    resource_cost_metrics,
    viewport_prediction_metrics,
)


def test_resource_cost_metrics_sum_bytes_timing_and_peak_memory() -> None:
    summary = ResourceAccountingSummary(
        summary_id="summary-1",
        accounts=(
            _account("account-a", "candidate-a", "object-a", transfer_bytes=100, memory_mb=64.0, transfer_ms=2.0),
            _account("account-b", "candidate-b", "object-b", transfer_bytes=300, memory_mb=96.0, transfer_ms=4.0),
        ),
        total_timing=ComponentTimingAccount(2.0, 2.0, 6.0, 2.0, 2.0, 2.0),
        peak_memory_mb=96.0,
        total_transfer_bytes=400,
    )

    metrics = resource_cost_metrics(summary, config=ResourceStabilityMetricConfig(split="final", tags={"method": "ref"}))
    by_name = {metric.metric_name: metric for metric in metrics}

    assert by_name["resource_bytes_cost"].value == pytest.approx(400)
    assert by_name["resource_timing_cost_ms"].value == pytest.approx(16.0)
    assert by_name["resource_memory_cost_mb"].value == pytest.approx(96.0)
    assert by_name["resource_bytes_cost"].split == "final"
    assert by_name["resource_bytes_cost"].tags["method"] == "ref"


def test_resource_cost_metrics_zero_when_no_records() -> None:
    metrics = resource_cost_metrics(())

    assert [metric.value for metric in metrics] == [0.0, 0.0, 0.0]
    assert all(metric.metadata["null_rule"] == NO_RESOURCE_RECORDS for metric in metrics)


def test_control_stability_metrics_variance_switch_rate_and_recovery_time() -> None:
    outcomes = (
        _outcome("frame-1", 10, 0.8, deadline_hit=True),
        _outcome("frame-2", 20, 0.6, deadline_hit=False),
        _outcome("frame-3", 30, 0.9, deadline_hit=True),
    )
    decisions = (
        _decision("decision-1", "frame-1", ("object-a",)),
        _decision("decision-2", "frame-2", ("object-a",)),
        _decision("decision-3", "frame-3", ("object-b",)),
    )

    metrics = control_stability_metrics(outcomes, decisions=decisions)
    by_name = {metric.metric_name: metric for metric in metrics}

    assert by_name["control_quality_variance"].value == pytest.approx(0.01555555555555555)
    assert by_name["control_switch_rate"].value == pytest.approx(0.5)
    assert by_name["control_switch_rate"].metadata["switch_count"] == 1
    assert by_name["control_recovery_time_ms"].value == pytest.approx(10.0)
    assert by_name["control_recovery_time_ms"].metadata["recovery_count"] == 1


def test_control_stability_metrics_zero_for_missing_denominators() -> None:
    metrics = control_stability_metrics((_outcome("frame-1", 10, 0.8, deadline_hit=True),), decisions=())
    by_name = {metric.metric_name: metric for metric in metrics}

    assert by_name["control_switch_rate"].value == pytest.approx(0.0)
    assert by_name["control_switch_rate"].metadata["null_rule"] == NO_CONTROL_TRANSITIONS
    assert by_name["control_recovery_time_ms"].value == pytest.approx(0.0)
    assert by_name["control_recovery_time_ms"].metadata["null_rule"] == NO_RECOVERY_EVENTS


def test_viewport_prediction_metrics_average_error_coverage_and_overfetch() -> None:
    samples = (
        ViewportPredictionMetricSample(frame_id="frame-1", angular_error_deg=4.0, coverage=0.75, overfetch_ratio=0.20),
        ViewportPredictionMetricSample(frame_id="frame-2", angular_error_deg=8.0, coverage=0.95, overfetch_ratio=0.40),
    )

    metrics = viewport_prediction_metrics(samples)
    by_name = {metric.metric_name: metric for metric in metrics}

    assert by_name["viewport_error_deg"].value == pytest.approx(6.0)
    assert by_name["viewport_coverage"].value == pytest.approx(0.85)
    assert by_name["viewport_overfetch_ratio"].value == pytest.approx(0.30)


def test_viewport_prediction_metrics_zero_when_no_samples() -> None:
    metrics = viewport_prediction_metrics(())

    assert [metric.value for metric in metrics] == [0.0, 0.0, 0.0]
    assert all(metric.metadata["null_rule"] == NO_VIEWPORT_SAMPLES for metric in metrics)


def test_compute_resource_stability_viewport_metrics_combines_groups() -> None:
    metrics = compute_resource_stability_viewport_metrics(
        resource_records=(_account("account-a", "candidate-a", "object-a", transfer_bytes=100),),
        frame_outcomes=(_outcome("frame-1", 10, 0.8, deadline_hit=True),),
        decisions=(_decision("decision-1", "frame-1", ("object-a",)),),
        viewport_samples=(ViewportPredictionMetricSample(frame_id="frame-1", angular_error_deg=5.0, coverage=0.8, overfetch_ratio=0.1),),
    )

    names = [metric.metric_name for metric in metrics]
    assert "resource_bytes_cost" in names
    assert "control_quality_variance" in names
    assert "viewport_error_deg" in names


def test_resource_stability_metrics_reject_malformed_inputs() -> None:
    with pytest.raises(ResourceStabilityMetricError, match="viewport_samples"):
        viewport_prediction_metrics((object(),))  # type: ignore[arg-type]

    with pytest.raises(ResourceStabilityMetricError, match="decisions"):
        control_stability_metrics((), decisions=(object(),))  # type: ignore[arg-type]


def _account(
    account_id: str,
    candidate_id: str,
    object_id: str,
    *,
    transfer_bytes: int,
    memory_mb: float = 64.0,
    transfer_ms: float = 1.0,
) -> CandidateResourceAccount:
    return CandidateResourceAccount(
        account_id=account_id,
        candidate_id=candidate_id,
        object_id=object_id,
        provider_id="provider-1",
        device_profile_id="device-1",
        timing=ComponentTimingAccount(1.0, 1.0, transfer_ms, 1.0, 1.0, 1.0),
        utilization=ResourceUtilization(
            server_generation=0.1,
            queue=0.1,
            transfer_time=0.1,
            decode=0.1,
            restore=0.1,
            render=0.1,
            memory=0.1,
        ),
        memory_mb=memory_mb,
        bandwidth_bps=None,
        transfer_bytes=transfer_bytes,
    )


def _outcome(frame_id: str, scheduled_time_ms: int, quality_score: float, *, deadline_hit: bool) -> FrameOutcome:
    return FrameOutcome(
        frame_id=frame_id,
        scheduled_time_ms=scheduled_time_ms,
        rendered_time_ms=scheduled_time_ms + 5 if deadline_hit else scheduled_time_ms + 50,
        deadline_ms=scheduled_time_ms + 20,
        delivered_object_ids=("object-1",) if deadline_hit else (),
        missing_object_ids=() if deadline_hit else ("object-1",),
        quality_score=quality_score,
        deadline_hit=deadline_hit,
        metadata={"frame_evaluation": {"quality_score": quality_score, "missing": not deadline_hit}},
    )


def _decision(decision_id: str, frame_id: str, selected_object_ids: tuple[str, ...]) -> ScheduleDecision:
    return ScheduleDecision(
        decision_id=decision_id,
        controller_id="controller-1",
        frame_id=frame_id,
        selected_object_ids=selected_object_ids,
        decision_time_ms=0,
        target_deadline_ms=20,
    )
