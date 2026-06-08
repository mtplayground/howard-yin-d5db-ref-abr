from __future__ import annotations

import pytest

from ref_abr.candidates import CandidateObject, CandidateSet
from ref_abr.domain import ControllerState
from ref_abr.methods import ActionBudget, ObservationBudget, SchedulingObservation, plan_schedule
from ref_abr.virtual_queue import (
    VirtualQueueConfig,
    VirtualQueueDeadlineController,
    VirtualQueueError,
    plan_virtual_queue_deadline,
)


def test_virtual_queue_controller_runs_under_adapter_and_records_debts() -> None:
    observation = _observation(
        (
            _candidate("base", kind="gaussian_base", value=3.0, size_bytes=150, deadline_ms=100),
            _candidate("ref", kind="reference_action", value=7.0, size_bytes=350, deadline_ms=100),
        ),
        target_deadline_ms=100,
        metadata={"virtual_queue": {"queue_debt_ms": 40.0, "viewport_debt": 0.2}},
    )

    decision = plan_schedule(
        VirtualQueueDeadlineController(config=VirtualQueueConfig(overload_threshold=10.0)),
        observation,
        observation_budget=ObservationBudget(max_candidates=10),
        action_budget=ActionBudget(max_selected_objects=2, max_selected_candidates=2, max_selected_bytes=600),
    )

    assert decision.metadata["adapter"]["method_id"] == "virtual-queue-deadline-controller"
    assert decision.metadata["baseline"]["policy"] == "virtual_queue_deadline"
    assert decision.metadata["virtual_queue"]["route"] == "debt_weighted"
    assert decision.metadata["virtual_queue"]["debt_state"]["queue_debt_ms"] == pytest.approx(40.0)
    assert decision.metadata["adapter"]["selected_candidate_ids"] == ("candidate-base", "candidate-ref")


def test_virtual_queue_debts_can_prefer_low_cost_alternate_route() -> None:
    observation = _observation(
        (
            _candidate("base", kind="gaussian_base", value=4.0, size_bytes=100, deadline_ms=100),
            _candidate(
                "ref",
                kind="reference_action",
                value=7.0,
                size_bytes=900_000,
                deadline_ms=100,
                virtual_queue={"viewport_sensitivity": 1.0},
            ),
        ),
        target_deadline_ms=100,
        metadata={"virtual_queue": {"queue_debt_ms": 100.0, "transfer_debt_bytes": 2_000_000, "viewport_debt": 1.0}},
    )

    decision = plan_schedule(
        VirtualQueueDeadlineController(
            config=VirtualQueueConfig(
                queue_debt_weight=1.0,
                transfer_debt_weight=1.0,
                viewport_debt_weight=2.0,
                overload_threshold=10.0,
            )
        ),
        observation,
        observation_budget=ObservationBudget(max_candidates=10),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=1_000_000),
    )

    values = decision.metadata["virtual_queue"]["candidate_values"]
    assert values["candidate-ref"] < values["candidate-base"]
    assert decision.metadata["adapter"]["selected_candidate_ids"] == ("candidate-base",)


def test_virtual_queue_overload_fallback_filters_reference_actions_and_scales_budget() -> None:
    observation = _observation(
        (
            _candidate("base", kind="gaussian_base", value=3.0, size_bytes=180, deadline_ms=100),
            _candidate("tile", kind="tile", value=2.0, size_bytes=120, deadline_ms=100),
            _candidate("ref", kind="reference_action", value=20.0, size_bytes=200, deadline_ms=100),
        ),
        target_deadline_ms=100,
        metadata={"virtual_queue": {"overload_score": 2.0}},
    )

    plan = plan_virtual_queue_deadline(
        observation,
        ActionBudget(max_selected_objects=2, max_selected_candidates=2, max_selected_bytes=400),
        config=VirtualQueueConfig(overload_threshold=1.0, overload_byte_scale=0.75),
    )

    assert plan.overload_fallback is True
    assert plan.route == "overload_fallback"
    assert plan.allocation.budget.max_bytes == 300
    assert "candidate-ref" not in plan.candidate_values
    assert set(plan.selected_candidate_ids).issubset({"candidate-base", "candidate-tile"})


def test_virtual_queue_reads_controller_state_debts_and_validates_config() -> None:
    observation = _observation(
        (_candidate("base", kind="gaussian_base", value=1.0, size_bytes=100, deadline_ms=100),),
        target_deadline_ms=100,
        state={"virtual_queue": {"lifecycle_debt": 0.5, "deadline_debt": 0.25}},
    )

    plan = plan_virtual_queue_deadline(
        observation,
        ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=200),
        config=VirtualQueueConfig(overload_threshold=10.0),
    )

    assert plan.debt_state.lifecycle_debt == pytest.approx(0.5)
    assert plan.debt_state.deadline_debt == pytest.approx(0.25)
    with pytest.raises(VirtualQueueError, match="overload_threshold"):
        VirtualQueueConfig(overload_threshold=0.0)
    with pytest.raises(VirtualQueueError, match="overload_byte_scale"):
        VirtualQueueConfig(overload_byte_scale=0.0)


def _candidate(
    suffix: str,
    *,
    kind: str,
    value: float,
    size_bytes: int,
    deadline_ms: int,
    virtual_queue: dict[str, float] | None = None,
) -> CandidateObject:
    return CandidateObject(
        candidate_id=f"candidate-{suffix}",
        object_id=f"object-{suffix}",
        candidate_kind=kind,
        decision_time_ms=0,
        layer=0 if kind == "gaussian_base" else 1,
        resolution="720p",
        fov_deg=90.0,
        viewpoint=None,
        lookahead_ms=0,
        expiration_ms=deadline_ms,
        retransmit_priority=0,
        size_bytes=size_bytes,
        metadata={
            "allocator": {
                "value": value,
                "render_ms": 1.0,
                "compute_ms": 1.0,
                "deadline_ms": deadline_ms,
            },
            "virtual_queue": virtual_queue or {},
        },
    )


def _observation(
    candidates: tuple[CandidateObject, ...],
    *,
    target_deadline_ms: int,
    metadata: dict | None = None,
    state: dict | None = None,
) -> SchedulingObservation:
    return SchedulingObservation(
        observation_id="obs-virtual-queue",
        controller_state=ControllerState(
            controller_id="ctrl-virtual-queue",
            method_name="virtual-queue",
            step_index=0,
            active_split="calibration",
            state=state or {},
        ),
        frame_id="frame-virtual-queue",
        decision_time_ms=0,
        target_deadline_ms=target_deadline_ms,
        candidate_set=CandidateSet(
            candidate_set_id="candidate-set-virtual-queue",
            decision_time_ms=0,
            candidates=candidates,
        ),
        metadata=metadata or {},
    )
