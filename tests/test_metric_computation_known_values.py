from __future__ import annotations

import pytest

from ref_abr.accounting import CandidateResourceAccount, ComponentTimingAccount, ResourceAccountingSummary, ResourceUtilization
from ref_abr.deadline_lifecycle_metrics import (
    DeadlineLifecycleMetricConfig,
    DeadlineQoeWeights,
    NO_EVENTS_ZERO_RATE,
    NO_RESOURCE_ZERO_RATIO,
    compute_deadline_lifecycle_metrics,
)
from ref_abr.domain import FrameOutcome, MetricRecord, ScheduleDecision
from ref_abr.lifecycle import DropReason, LifecycleAction, LifecyclePhase, ReferenceLifecycleEvent
from ref_abr.quality_metrics import MISSING_RENDER_POLICY, QualityMetricConfig, compute_quality_metrics
from ref_abr.resource_stability_metrics import (
    NO_CONTROL_TRANSITIONS,
    NO_RECOVERY_EVENTS,
    NO_RESOURCE_RECORDS,
    NO_VIEWPORT_SAMPLES,
    ResourceStabilityMetricConfig,
    ViewportPredictionMetricSample,
    compute_resource_stability_viewport_metrics,
)
from ref_abr.statistical_confidence import StatisticalConfidenceConfig, paired_confidence_metric, paired_metric_samples


def test_quality_metrics_known_values_and_missing_render_null_rule() -> None:
    restored = (
        _frame("frame-a", rendered_time_ms=18, deadline_hit=True, visible_quality=0.80, full_quality=0.60),
        _frame("frame-b", rendered_time_ms=40, deadline_hit=False, visible_quality=0.90, full_quality=0.70),
        _frame("frame-c", rendered_time_ms=None, deadline_hit=False, visible_quality=0.50, full_quality=0.50),
    )
    baseline = (
        _frame("frame-a", rendered_time_ms=17, deadline_hit=True, visible_quality=0.40, full_quality=0.25),
        _frame("frame-c", rendered_time_ms=None, deadline_hit=False, visible_quality=0.20, full_quality=0.20),
    )

    metrics = compute_quality_metrics(
        restored,
        baseline_outcomes=baseline,
        config=QualityMetricConfig(split="final", tags={"method": "refabr"}),
    )
    by_key = {(metric.metric_name, metric.frame_id, metric.tags.get("role", metric.tags.get("paired", ""))): metric for metric in metrics}

    assert by_key[("deadline_hit_visible_quality", "frame-a", "")].value == pytest.approx(0.80)
    assert by_key[("deadline_hit_visible_quality", "frame-b", "")].value == pytest.approx(0.0)
    assert by_key[("deadline_hit_visible_quality", "frame-c", "")].value == pytest.approx(0.0)
    assert by_key[("full_frame_quality", "frame-a", "restored")].value == pytest.approx(0.60)
    assert by_key[("full_frame_quality", "frame-c", "restored")].value == pytest.approx(0.0)
    assert by_key[("restoration_gain", "frame-a", "restored_minus_baseline")].value == pytest.approx(0.35)
    assert by_key[("restoration_gain", "frame-c", "restored_minus_baseline")].value == pytest.approx(0.0)
    assert by_key[("full_frame_quality", "frame-c", "restored")].metadata["missing_render_policy"] == MISSING_RENDER_POLICY
    assert by_key[("deadline_hit_visible_quality", "frame-a", "")].split == "final"
    assert by_key[("deadline_hit_visible_quality", "frame-a", "")].tags["method"] == "refabr"


def test_deadline_lifecycle_metrics_known_numerators_denominators_and_penalties() -> None:
    outcome = _frame(
        "frame-qoe",
        rendered_time_ms=22,
        deadline_hit=True,
        visible_quality=0.75,
        full_quality=0.50,
        freeze=True,
        delivered=("object-a", "object-b"),
        missing=("object-c", "object-d"),
        required=("object-a", "object-b", "object-c", "object-d"),
    )
    accounts = (
        _account("account-useful-a", "object-a", transfer_bytes=100),
        _account("account-useful-b", "object-b", transfer_bytes=200),
        _account("account-waste", "object-z", transfer_bytes=300),
    )
    events = (
        _event("event-use", "ref-use", LifecycleAction.USE, LifecyclePhase.RESTORED, LifecyclePhase.USED),
        _event(
            "event-late",
            "ref-late",
            LifecycleAction.EXPIRE,
            LifecyclePhase.TRANSFERRING,
            LifecyclePhase.EXPIRED,
            event_time_ms=31,
            deadline_ms=30,
            drop_reason=DropReason.DEADLINE_MISSED,
        ),
        _event("event-stale", "ref-stale", LifecycleAction.STALE, LifecyclePhase.RESTORED, LifecyclePhase.STALE, drop_reason=DropReason.STALE),
        _event(
            "event-off-view",
            "ref-off-view",
            LifecycleAction.CANCEL,
            LifecyclePhase.REQUESTED,
            LifecyclePhase.CANCELLED,
            metadata={"viewport": {"off_view": True}},
        ),
    )
    config = DeadlineLifecycleMetricConfig(
        split="final",
        tags={"method": "refabr"},
        deadline_qoe_weights=DeadlineQoeWeights(
            visible_quality=0.40,
            full_quality=0.40,
            deadline_hit=0.20,
            freeze_penalty=0.10,
            missing_penalty=0.20,
        ),
    )

    metrics = compute_deadline_lifecycle_metrics(
        frame_outcomes=(outcome,),
        resource_records=accounts,
        useful_object_ids=("object-a", "object-b"),
        lifecycle_events=events,
        config=config,
    )
    by_name = {metric.metric_name: metric for metric in metrics}

    assert by_name["deadline_qoe"].value == pytest.approx(0.50)
    assert by_name["deadline_qoe"].metadata["missing_ratio"] == pytest.approx(0.50)
    assert by_name["deadline_qoe"].metadata["null_rule"] == "not_null"
    assert by_name["useful_resource_ratio"].value == pytest.approx(0.50)
    assert by_name["useful_resource_ratio"].metadata["useful_bytes"] == 300
    assert by_name["useful_resource_ratio"].metadata["total_bytes"] == 600
    assert by_name["reference_lifecycle_late_rate"].value == pytest.approx(0.25)
    assert by_name["reference_lifecycle_stale_rate"].value == pytest.approx(0.25)
    assert by_name["reference_lifecycle_off_view_rate"].value == pytest.approx(0.25)
    assert by_name["reference_lifecycle_expired_rate"].value == pytest.approx(0.25)
    assert by_name["reference_lifecycle_useful_rate"].value == pytest.approx(0.25)
    assert by_name["reference_lifecycle_late_rate"].metadata["denominator"] == 4


def test_metric_null_rules_for_empty_raw_records_are_explicit() -> None:
    missing = _frame("frame-missing", rendered_time_ms=None, deadline_hit=False, visible_quality=0.60, full_quality=0.40)

    metrics = compute_deadline_lifecycle_metrics(frame_outcomes=(missing,), resource_records=(), useful_object_ids=(), lifecycle_events=())
    by_name = {metric.metric_name: metric for metric in metrics}

    assert by_name["deadline_qoe"].value == pytest.approx(0.0)
    assert by_name["deadline_qoe"].metadata["null_rule"] == MISSING_RENDER_POLICY
    assert by_name["useful_resource_ratio"].value == pytest.approx(0.0)
    assert by_name["useful_resource_ratio"].metadata["null_rule"] == NO_RESOURCE_ZERO_RATIO
    assert by_name["reference_lifecycle_useful_rate"].value == pytest.approx(0.0)
    assert by_name["reference_lifecycle_useful_rate"].metadata["null_rule"] == NO_EVENTS_ZERO_RATE

    resource_metrics = compute_resource_stability_viewport_metrics(resource_records=(), frame_outcomes=(), decisions=(), viewport_samples=())
    resource_by_name = {metric.metric_name: metric for metric in resource_metrics}
    assert resource_by_name["resource_bytes_cost"].metadata["null_rule"] == NO_RESOURCE_RECORDS
    assert resource_by_name["control_switch_rate"].metadata["null_rule"] == NO_CONTROL_TRANSITIONS
    assert resource_by_name["control_recovery_time_ms"].metadata["null_rule"] == NO_RECOVERY_EVENTS
    assert resource_by_name["viewport_error_deg"].metadata["null_rule"] == NO_VIEWPORT_SAMPLES


def test_resource_stability_and_viewport_metrics_known_values() -> None:
    summary = ResourceAccountingSummary(
        summary_id="summary-known",
        accounts=(
            _account("account-a", "object-a", transfer_bytes=100, timing=(1, 2, 3, 4, 5, 6), memory_mb=64.0),
            _account("account-b", "object-b", transfer_bytes=200, timing=(2, 3, 4, 5, 6, 7), memory_mb=80.0),
        ),
        total_timing=ComponentTimingAccount(3, 5, 7, 9, 11, 13),
        peak_memory_mb=80.0,
        total_transfer_bytes=300,
    )
    outcomes = (
        _frame("frame-1", rendered_time_ms=5, deadline_hit=True, visible_quality=0.0, full_quality=0.0, scheduled_time_ms=0),
        _frame("frame-2", rendered_time_ms=35, deadline_hit=False, visible_quality=1.0, full_quality=1.0, scheduled_time_ms=10),
        _frame("frame-3", rendered_time_ms=None, deadline_hit=False, visible_quality=0.5, full_quality=0.5, scheduled_time_ms=20),
        _frame("frame-4", rendered_time_ms=45, deadline_hit=True, visible_quality=0.5, full_quality=0.5, scheduled_time_ms=40),
    )
    decisions = (
        _decision("decision-1", "frame-1", ("object-a",)),
        _decision("decision-2", "frame-2", ("object-b",)),
        _decision("decision-3", "frame-3", ("object-b",)),
        _decision("decision-4", "frame-4", ("object-c",)),
    )
    viewport_samples = (
        ViewportPredictionMetricSample("frame-1", angular_error_deg=2.0, coverage=0.50, overfetch_ratio=0.10),
        ViewportPredictionMetricSample("frame-2", angular_error_deg=10.0, coverage=0.75, overfetch_ratio=0.20),
        ViewportPredictionMetricSample("frame-3", angular_error_deg=6.0, coverage=1.00, overfetch_ratio=0.60),
    )

    metrics = compute_resource_stability_viewport_metrics(
        resource_records=summary,
        frame_outcomes=outcomes,
        decisions=decisions,
        viewport_samples=viewport_samples,
        config=ResourceStabilityMetricConfig(split="final", tags={"method": "refabr"}),
    )
    by_name = {metric.metric_name: metric for metric in metrics}

    assert by_name["resource_bytes_cost"].value == pytest.approx(300)
    assert by_name["resource_timing_cost_ms"].value == pytest.approx(48)
    assert by_name["resource_memory_cost_mb"].value == pytest.approx(80)
    assert by_name["control_quality_variance"].value == pytest.approx(0.125)
    assert by_name["control_switch_rate"].value == pytest.approx(2 / 3)
    assert by_name["control_recovery_time_ms"].value == pytest.approx(30)
    assert by_name["viewport_error_deg"].value == pytest.approx(6)
    assert by_name["viewport_coverage"].value == pytest.approx(0.75)
    assert by_name["viewport_overfetch_ratio"].value == pytest.approx(0.30)
    assert by_name["resource_bytes_cost"].split == "final"
    assert by_name["resource_bytes_cost"].tags["method"] == "refabr"


def test_paired_confidence_known_values_and_missing_pair_validation() -> None:
    treatment = (
        _metric("deadline_qoe", 1.25, "pair-a"),
        _metric("deadline_qoe", 2.25, "pair-b"),
        _metric("deadline_qoe", 3.25, "pair-c"),
        _metric("deadline_qoe", 4.00, "pair-missing-baseline"),
    )
    baseline = (
        _metric("deadline_qoe", 1.00, "pair-a"),
        _metric("deadline_qoe", 2.00, "pair-b"),
        _metric("deadline_qoe", 3.00, "pair-c"),
        _metric("deadline_qoe", 9.00, "pair-missing-treatment"),
    )

    samples, validation = paired_metric_samples(treatment, baseline, metric_name="deadline_qoe", min_pairs=3)
    assert [sample.delta for sample in samples] == pytest.approx([0.25, 0.25, 0.25])
    assert validation.matched_pair_ids == ("pair-a", "pair-b", "pair-c")
    assert validation.missing_baseline_pair_ids == ("pair-missing-baseline",)
    assert validation.missing_treatment_pair_ids == ("pair-missing-treatment",)
    assert validation.promotion_blocked is True

    metric = paired_confidence_metric(
        treatment[:3],
        baseline[:3],
        metric_name="deadline_qoe",
        config=StatisticalConfidenceConfig(confidence_level=0.90, bootstrap_iterations=25, seed=19, min_pairs=3),
    )
    interval = metric.metadata["confidence_interval"]
    assert metric.value == pytest.approx(0.25)
    assert interval["mean_treatment"] == pytest.approx(2.25)
    assert interval["mean_baseline"] == pytest.approx(2.00)
    assert interval["ci_lower"] == pytest.approx(0.25)
    assert interval["ci_upper"] == pytest.approx(0.25)
    assert interval["bootstrap_std_error"] == pytest.approx(0.0)
    assert metric.metadata["promotion_blocked"] is False


def _frame(
    frame_id: str,
    *,
    rendered_time_ms: int | None,
    deadline_hit: bool,
    visible_quality: float,
    full_quality: float,
    freeze: bool = False,
    delivered: tuple[str, ...] = ("object-a",),
    missing: tuple[str, ...] = (),
    required: tuple[str, ...] = ("object-a",),
    scheduled_time_ms: int = 10,
) -> FrameOutcome:
    return FrameOutcome(
        frame_id=frame_id,
        scheduled_time_ms=scheduled_time_ms,
        rendered_time_ms=rendered_time_ms,
        deadline_ms=scheduled_time_ms + 25,
        delivered_object_ids=delivered if rendered_time_ms is not None else (),
        missing_object_ids=missing,
        quality_score=full_quality,
        deadline_hit=deadline_hit,
        metadata={
            "frame_evaluation": {
                "visible_quality": visible_quality,
                "full_quality": full_quality,
                "quality_score": full_quality,
                "freeze": freeze,
                "missing": bool(missing),
                "required_object_ids": list(required),
            }
        },
    )


def _account(
    account_id: str,
    object_id: str,
    *,
    transfer_bytes: int,
    timing: tuple[float, float, float, float, float, float] = (1, 1, 1, 1, 1, 1),
    memory_mb: float = 64.0,
) -> CandidateResourceAccount:
    return CandidateResourceAccount(
        account_id=account_id,
        candidate_id=f"candidate-{account_id}",
        object_id=object_id,
        provider_id="provider-known",
        device_profile_id="device-known",
        timing=ComponentTimingAccount(*timing),
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


def _event(
    event_id: str,
    reference_id: str,
    action: LifecycleAction,
    from_phase: LifecyclePhase,
    to_phase: LifecyclePhase,
    *,
    event_time_ms: int = 10,
    deadline_ms: int = 30,
    drop_reason: DropReason | None = None,
    metadata: dict[str, object] | None = None,
) -> ReferenceLifecycleEvent:
    if to_phase == LifecyclePhase.USED:
        status = "available"
    elif to_phase in {LifecyclePhase.STALE, LifecyclePhase.EXPIRED}:
        status = "expired"
    else:
        status = "dropped"
    return ReferenceLifecycleEvent(
        event_id=event_id,
        reference_id=reference_id,
        action=action,
        from_phase=from_phase,
        to_phase=to_phase,
        status=status,
        event_time_ms=event_time_ms,
        deadline_ms=deadline_ms,
        drop_reason=drop_reason,
        metadata=metadata or {},
    )


def _decision(decision_id: str, frame_id: str, selected_object_ids: tuple[str, ...]) -> ScheduleDecision:
    return ScheduleDecision(
        decision_id=decision_id,
        controller_id="controller-known",
        frame_id=frame_id,
        selected_object_ids=selected_object_ids,
        decision_time_ms=0,
        target_deadline_ms=25,
    )


def _metric(metric_name: str, value: float, frame_id: str) -> MetricRecord:
    return MetricRecord(metric_name=metric_name, value=value, unit="score", frame_id=frame_id, tags={"method": "candidate"})
