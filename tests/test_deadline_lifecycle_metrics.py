from __future__ import annotations

import pytest

from ref_abr.accounting import CandidateResourceAccount, ComponentTimingAccount, ResourceAccountingSummary, ResourceUtilization
from ref_abr.deadline_lifecycle_metrics import (
    DeadlineLifecycleMetricConfig,
    DeadlineLifecycleMetricError,
    DeadlineQoeWeights,
    NO_EVENTS_ZERO_RATE,
    NO_RESOURCE_ZERO_RATIO,
    compute_deadline_lifecycle_metrics,
    deadline_qoe,
    reference_lifecycle_rates,
    useful_resource_ratio,
)
from ref_abr.domain import FrameOutcome
from ref_abr.lifecycle import DropReason, LifecycleAction, LifecyclePhase, ReferenceLifecycleEvent
from ref_abr.quality_metrics import MISSING_RENDER_POLICY


def test_deadline_qoe_uses_explicit_weights_and_penalties() -> None:
    outcome = _outcome(
        "frame-1",
        rendered_time_ms=18,
        deadline_hit=True,
        visible_quality=0.8,
        full_quality=0.6,
        freeze=False,
        missing_count=1,
        required_count=4,
    )
    config = DeadlineLifecycleMetricConfig(
        split="final",
        tags={"method": "ref"},
        deadline_qoe_weights=DeadlineQoeWeights(
            visible_quality=0.5,
            full_quality=0.3,
            deadline_hit=0.2,
            freeze_penalty=0.1,
            missing_penalty=0.2,
        ),
    )

    metric = deadline_qoe(outcome, config=config)

    assert metric.metric_name == "deadline_qoe"
    assert metric.value == pytest.approx(0.73)
    assert metric.split == "final"
    assert metric.tags["method"] == "ref"
    assert metric.metadata["weights"]["visible_quality"] == pytest.approx(0.5)
    assert metric.metadata["missing_ratio"] == pytest.approx(0.25)


def test_deadline_qoe_missing_render_is_zero_and_documented() -> None:
    outcome = _outcome(
        "frame-2",
        rendered_time_ms=None,
        deadline_hit=False,
        visible_quality=0.9,
        full_quality=0.8,
        freeze=True,
        missing_count=2,
        required_count=2,
    )

    metric = deadline_qoe(outcome)

    assert metric.value == pytest.approx(0.0)
    assert metric.metadata["rendered"] is False
    assert metric.metadata["null_rule"] == MISSING_RENDER_POLICY


def test_useful_resource_ratio_uses_transferred_bytes_and_zero_denominator_rule() -> None:
    summary = ResourceAccountingSummary(
        summary_id="summary-1",
        accounts=(
            _account("account-a", "candidate-a", "object-a", transfer_bytes=100),
            _account("account-b", "candidate-b", "object-b", transfer_bytes=300),
        ),
        total_timing=ComponentTimingAccount(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        peak_memory_mb=64.0,
        total_transfer_bytes=400,
    )

    metric = useful_resource_ratio(summary, useful_object_ids=("object-a",))
    empty_metric = useful_resource_ratio((), useful_object_ids=("object-a",))

    assert metric.value == pytest.approx(0.25)
    assert metric.metadata["useful_bytes"] == 100
    assert metric.metadata["total_bytes"] == 400
    assert empty_metric.value == pytest.approx(0.0)
    assert empty_metric.metadata["null_rule"] == NO_RESOURCE_ZERO_RATIO


def test_reference_lifecycle_rates_classify_distinct_references() -> None:
    events = (
        _event("event-use", "ref-use", LifecycleAction.USE, LifecyclePhase.RESTORED, LifecyclePhase.USED),
        _event("event-stale", "ref-stale", LifecycleAction.STALE, LifecyclePhase.USED, LifecyclePhase.STALE, drop_reason=DropReason.STALE),
        _event(
            "event-expire",
            "ref-expire",
            LifecycleAction.EXPIRE,
            LifecyclePhase.TRANSFERRING,
            LifecyclePhase.EXPIRED,
            event_time_ms=30,
            deadline_ms=20,
            drop_reason=DropReason.DEADLINE_MISSED,
        ),
        _event(
            "event-off-view",
            "ref-off-view",
            LifecycleAction.CANCEL,
            LifecyclePhase.REQUESTED,
            LifecyclePhase.CANCELLED,
            metadata={"lifecycle": {"off_view": True}},
        ),
    )

    metrics = reference_lifecycle_rates(events)
    by_name = {metric.metric_name: metric for metric in metrics}

    assert by_name["reference_lifecycle_useful_rate"].value == pytest.approx(0.25)
    assert by_name["reference_lifecycle_stale_rate"].value == pytest.approx(0.25)
    assert by_name["reference_lifecycle_expired_rate"].value == pytest.approx(0.25)
    assert by_name["reference_lifecycle_late_rate"].value == pytest.approx(0.25)
    assert by_name["reference_lifecycle_off_view_rate"].value == pytest.approx(0.25)
    assert by_name["reference_lifecycle_late_rate"].metadata["denominator"] == 4


def test_reference_lifecycle_rates_zero_when_no_events() -> None:
    metrics = reference_lifecycle_rates(())

    assert len(metrics) == 5
    assert all(metric.value == 0.0 for metric in metrics)
    assert all(metric.metadata["null_rule"] == NO_EVENTS_ZERO_RATE for metric in metrics)


def test_compute_deadline_lifecycle_metrics_combines_metric_groups() -> None:
    outcome = _outcome("frame-3", rendered_time_ms=18, deadline_hit=True, visible_quality=0.7, full_quality=0.6)
    metrics = compute_deadline_lifecycle_metrics(
        frame_outcomes=(outcome,),
        resource_records=(_account("account-a", "candidate-a", "object-a", transfer_bytes=10),),
        useful_object_ids=("object-a",),
        lifecycle_events=(_event("event-use", "ref-use", LifecycleAction.USE, LifecyclePhase.RESTORED, LifecyclePhase.USED),),
    )

    names = [metric.metric_name for metric in metrics]
    assert "deadline_qoe" in names
    assert "useful_resource_ratio" in names
    assert "reference_lifecycle_useful_rate" in names


def test_deadline_lifecycle_metrics_reject_malformed_inputs() -> None:
    with pytest.raises(DeadlineLifecycleMetricError, match="useful_object_ids"):
        useful_resource_ratio((), useful_object_ids=("object-a", ""))

    with pytest.raises(DeadlineLifecycleMetricError, match="lifecycle_events"):
        reference_lifecycle_rates((object(),))  # type: ignore[arg-type]


def _outcome(
    frame_id: str,
    *,
    rendered_time_ms: int | None,
    deadline_hit: bool,
    visible_quality: float,
    full_quality: float,
    freeze: bool = False,
    missing_count: int = 0,
    required_count: int = 1,
) -> FrameOutcome:
    delivered_count = max(0, required_count - missing_count)
    return FrameOutcome(
        frame_id=frame_id,
        scheduled_time_ms=10,
        rendered_time_ms=rendered_time_ms,
        deadline_ms=25,
        delivered_object_ids=tuple(f"object-delivered-{index}" for index in range(delivered_count)),
        missing_object_ids=tuple(f"object-missing-{index}" for index in range(missing_count)),
        quality_score=full_quality,
        deadline_hit=deadline_hit,
        metadata={
            "frame_evaluation": {
                "visible_quality": visible_quality,
                "full_quality": full_quality,
                "freeze": freeze,
                "required_object_ids": [f"object-{index}" for index in range(required_count)],
            }
        },
    )


def _account(account_id: str, candidate_id: str, object_id: str, *, transfer_bytes: int) -> CandidateResourceAccount:
    return CandidateResourceAccount(
        account_id=account_id,
        candidate_id=candidate_id,
        object_id=object_id,
        provider_id="provider-1",
        device_profile_id="device-1",
        timing=ComponentTimingAccount(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
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
        transfer_bytes=transfer_bytes,
    )


def _event(
    event_id: str,
    reference_id: str,
    action: LifecycleAction,
    from_phase: LifecyclePhase,
    to_phase: LifecyclePhase,
    *,
    event_time_ms: int = 10,
    deadline_ms: int = 20,
    drop_reason: DropReason | None = None,
    metadata: dict[str, object] | None = None,
) -> ReferenceLifecycleEvent:
    return ReferenceLifecycleEvent(
        event_id=event_id,
        reference_id=reference_id,
        action=action,
        from_phase=from_phase,
        to_phase=to_phase,
        status="available" if to_phase == LifecyclePhase.USED else "expired" if to_phase in {LifecyclePhase.STALE, LifecyclePhase.EXPIRED} else "dropped",
        event_time_ms=event_time_ms,
        deadline_ms=deadline_ms,
        drop_reason=drop_reason,
        metadata=metadata or {},
    )
