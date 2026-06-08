from __future__ import annotations

import pytest

from ref_abr.accounting import (
    CandidateResourceAccount,
    ComponentTimingAccount,
    ResourceAccountingSummary,
    ResourceUtilization,
)
from ref_abr.candidates import CandidateObject
from ref_abr.outcomes import FrameEvaluationConfig, OutcomeEvaluationError, evaluate_frame_outcome
from ref_abr.utility import CandidateUtilityEstimate, ResourceDebt, ResourcePrice, UtilityUncertainty


def test_frame_outcome_computes_coverage_quality_latency_and_deadline_hit() -> None:
    candidates = {
        "candidate-a": _candidate("candidate-a", "object-a"),
        "candidate-b": _candidate("candidate-b", "object-b"),
    }
    estimates = (
        _estimate("candidate-a", 0.8),
        _estimate("candidate-b", 0.6),
    )
    accounting = ResourceAccountingSummary(
        summary_id="summary-1",
        accounts=(
            _account("account-a", "candidate-a", "object-a", transfer_ms=3.0),
            _account("account-b", "candidate-b", "object-b", transfer_ms=3.0),
        ),
        total_timing=ComponentTimingAccount(
            server_generation_ms=4.0,
            queue_ms=0.0,
            transfer_ms=10.0,
            decode_ms=2.0,
            restore_ms=2.0,
            render_ms=2.0,
        ),
        peak_memory_mb=96.0,
        total_transfer_bytes=2000,
    )

    outcome = evaluate_frame_outcome(
        frame_id="frame-1",
        scheduled_time_ms=100,
        deadline_ms=120,
        required_object_ids=("object-a", "object-b", "object-c"),
        visible_object_ids=("object-a", "object-b"),
        delivered_object_ids=("object-a", "object-b"),
        accounting=accounting,
        utility_estimates=estimates,
        candidate_by_id=candidates,
        selected_candidate_ids=("candidate-a", "candidate-b"),
    )

    evaluation = outcome.as_payload()["metadata"]["frame_evaluation"]
    assert outcome.rendered_time_ms == 120
    assert outcome.deadline_hit is True
    assert outcome.delivered_object_ids == ("object-a", "object-b")
    assert outcome.missing_object_ids == ("object-c",)
    assert evaluation["coverage"] == pytest.approx(2 / 3)
    assert evaluation["visible_coverage"] == pytest.approx(1.0)
    assert evaluation["visible_quality"] == pytest.approx(0.7)
    assert evaluation["full_quality"] == pytest.approx((0.8 + 0.6) / 3)
    assert evaluation["latency_ms"] == pytest.approx(20.0)
    assert evaluation["fps"] == pytest.approx(50.0)
    assert outcome.quality_score == pytest.approx(0.5516666666666666)


def test_frame_outcome_miss_deadline_and_freeze_penalties() -> None:
    outcome = evaluate_frame_outcome(
        frame_id="frame-2",
        scheduled_time_ms=100,
        deadline_ms=110,
        required_object_ids=("object-a",),
        delivered_object_ids=(),
        frozen=True,
        previous_frame_id="frame-1",
        accounting=(_account("account-a", "candidate-a", "object-a", transfer_ms=20.0),),
    )

    evaluation = outcome.as_payload()["metadata"]["frame_evaluation"]
    assert outcome.rendered_time_ms == 127
    assert outcome.deadline_hit is False
    assert outcome.missing_object_ids == ("object-a",)
    assert evaluation["missing"] is True
    assert evaluation["freeze"] is True
    assert evaluation["penalties"]["deadline_miss"] == pytest.approx(0.25)
    assert evaluation["penalties"]["missing"] == pytest.approx(0.20)
    assert evaluation["penalties"]["freeze"] == pytest.approx(0.20)
    assert outcome.quality_score == pytest.approx(0.0)


def test_frame_outcome_uses_frame_interval_for_fps() -> None:
    outcome = evaluate_frame_outcome(
        frame_id="frame-3",
        scheduled_time_ms=0,
        deadline_ms=40,
        required_object_ids=(),
        delivered_object_ids=(),
        config=FrameEvaluationConfig(frame_interval_ms=16.6667),
    )

    evaluation = outcome.as_payload()["metadata"]["frame_evaluation"]
    assert outcome.quality_score == pytest.approx(1.0)
    assert evaluation["coverage"] == pytest.approx(1.0)
    assert evaluation["fps"] == pytest.approx(59.99988)


def test_frame_outcome_validation_rejects_malformed_inputs() -> None:
    with pytest.raises(OutcomeEvaluationError, match="scheduled_time_ms"):
        evaluate_frame_outcome(
            frame_id="frame-4",
            scheduled_time_ms=-1,
            deadline_ms=10,
            required_object_ids=("object-a",),
        )

    with pytest.raises(OutcomeEvaluationError, match="required_object_ids"):
        evaluate_frame_outcome(
            frame_id="frame-4",
            scheduled_time_ms=0,
            deadline_ms=10,
            required_object_ids=("object-a", ""),
        )

    with pytest.raises(OutcomeEvaluationError, match="config"):
        evaluate_frame_outcome(
            frame_id="frame-4",
            scheduled_time_ms=0,
            deadline_ms=10,
            required_object_ids=("object-a",),
            config={"deadline_miss_penalty": 0.1},  # type: ignore[arg-type]
        )


def _candidate(candidate_id: str, object_id: str) -> CandidateObject:
    return CandidateObject(
        candidate_id=candidate_id,
        object_id=object_id,
        candidate_kind="gaussian_base",
        decision_time_ms=0,
        layer=0,
        resolution="720p",
        fov_deg=90.0,
        viewpoint=None,
        lookahead_ms=0,
        expiration_ms=100,
        retransmit_priority=0,
        size_bytes=1000,
    )


def _estimate(candidate_id: str, visible_qoe_gain: float) -> CandidateUtilityEstimate:
    return CandidateUtilityEstimate(
        estimate_id=f"estimate-{candidate_id}",
        candidate_id=candidate_id,
        visible_qoe_gain=visible_qoe_gain,
        lifecycle_risk=0.0,
        deadline_miss_probability=0.0,
        resource_price=ResourcePrice(time_price=0.0, transfer_price=0.0, memory_price=0.0),
        resource_debt=ResourceDebt(
            time_debt_ms=0.0,
            transfer_debt_bytes=0,
            memory_debt_mb=0.0,
            carried_queue_debt_ms=0.0,
            carried_transfer_debt_bytes=0,
        ),
        expected_utility=visible_qoe_gain,
        uncertainty=UtilityUncertainty(
            quality_stddev=0.0,
            timing_stddev_ms=0.0,
            deadline_probability_stddev=0.0,
            utility_stddev=0.0,
            confidence=1.0,
        ),
    )


def _account(account_id: str, candidate_id: str, object_id: str, *, transfer_ms: float) -> CandidateResourceAccount:
    return CandidateResourceAccount(
        account_id=account_id,
        candidate_id=candidate_id,
        object_id=object_id,
        provider_id="provider-1",
        device_profile_id="device-1",
        timing=ComponentTimingAccount(
            server_generation_ms=3.0,
            queue_ms=1.0,
            transfer_ms=transfer_ms,
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
        memory_mb=48.0,
        bandwidth_bps=None,
        transfer_bytes=1000,
    )
