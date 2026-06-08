from __future__ import annotations

import pytest

from ref_abr.candidates import CandidateObject, CandidateSet
from ref_abr.domain import ControllerState
from ref_abr.methods import ActionBudget, ObservationBudget, SchedulingObservation, plan_schedule
from ref_abr.mpc import (
    MPCConfig,
    MPCError,
    RobustDeadlineAwareMPCController,
    RobustInterval,
    plan_robust_deadline_aware_mpc,
)


def test_robust_deadline_aware_mpc_runs_allocator_inner_loop_under_adapter() -> None:
    observation = _observation(
        (
            _candidate("base", kind="gaussian_base", value=3.0, size_bytes=200, deadline_ms=90),
            _candidate("ref", kind="reference_action", value=7.0, size_bytes=350, deadline_ms=90),
            _candidate("late", kind="tile", value=20.0, size_bytes=100, deadline_ms=160),
        ),
        target_deadline_ms=100,
    )
    controller = RobustDeadlineAwareMPCController(
        config=MPCConfig(
            horizon_steps=2,
            max_scenarios=2,
            bandwidth_interval=(1.0, 1.0),
            viewport_error_interval=(0.0, 0.0),
            deadline_scale_interval=(1.0, 1.0),
            runtime_cap_ms=50.0,
        )
    )

    decision = plan_schedule(
        controller,
        observation,
        observation_budget=ObservationBudget(max_candidates=10),
        action_budget=ActionBudget(
            max_selected_objects=2,
            max_selected_candidates=2,
            max_selected_bytes=600,
            max_deadline_ms=100,
        ),
    )

    assert decision.metadata["adapter"]["method_id"] == "robust-deadline-aware-mpc"
    assert decision.metadata["baseline"]["policy"] == "robust_deadline_aware_mpc"
    assert decision.metadata["adapter"]["selected_candidate_ids"] == ("candidate-base", "candidate-ref")
    assert decision.metadata["mpc"]["selected_step"]["allocation"]["total_value"] == pytest.approx(10.0)
    assert decision.metadata["mpc"]["metadata"]["scenario_count"] == 1
    assert decision.metadata["mpc"]["runtime_capped"] is False


def test_mpc_viewport_interval_can_shift_selection_to_robust_gaussian() -> None:
    reference = _candidate(
        "ref",
        kind="reference_action",
        value=10.0,
        size_bytes=200,
        deadline_ms=100,
        mpc={"viewport_sensitivity": 1.0},
    )
    gaussian = _candidate(
        "gaussian",
        kind="gaussian_base",
        value=4.5,
        size_bytes=200,
        deadline_ms=100,
        mpc={"viewport_sensitivity": 0.0},
    )
    observation = _observation((reference, gaussian), target_deadline_ms=100)
    config = MPCConfig(
        horizon_steps=1,
        max_scenarios=3,
        bandwidth_interval=(1.0, 1.0),
        viewport_error_interval=(90.0, 90.0),
        deadline_scale_interval=(1.0, 1.0),
    )

    decision = plan_schedule(
        RobustDeadlineAwareMPCController(config=config),
        observation,
        observation_budget=ObservationBudget(max_candidates=5),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=500),
    )

    assert decision.metadata["adapter"]["selected_candidate_ids"] == ("candidate-gaussian",)
    assert decision.expected_utility == pytest.approx(4.5)


def test_mpc_runtime_cap_is_recorded_while_returning_a_feasible_plan() -> None:
    observation = _observation(
        (
            _candidate("a", kind="gaussian_base", value=1.0, size_bytes=100, deadline_ms=100),
            _candidate("b", kind="reference_action", value=2.0, size_bytes=100, deadline_ms=100),
        ),
        target_deadline_ms=100,
    )

    plan = plan_robust_deadline_aware_mpc(
        observation,
        ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=250),
        config=MPCConfig(
            horizon_steps=5,
            max_scenarios=8,
            runtime_cap_ms=0.0,
            bandwidth_interval=(0.5, 1.0),
            viewport_error_interval=(0.0, 30.0),
            deadline_scale_interval=(1.0, 1.0),
        ),
    )

    assert plan.runtime_capped is True
    assert plan.step_plans
    assert plan.selected_candidate_ids
    assert plan.metadata["algorithm"] == "short_horizon_robust_allocator_mpc"


def test_mpc_config_rejects_invalid_intervals_and_counts() -> None:
    with pytest.raises(MPCError, match="interval lower"):
        RobustInterval(2.0, 1.0)
    with pytest.raises(MPCError, match="horizon_steps"):
        MPCConfig(horizon_steps=0)


def _candidate(
    suffix: str,
    *,
    kind: str,
    value: float,
    size_bytes: int,
    deadline_ms: int,
    mpc: dict[str, float] | None = None,
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
            "mpc": mpc or {},
        },
    )


def _observation(candidates: tuple[CandidateObject, ...], *, target_deadline_ms: int) -> SchedulingObservation:
    return SchedulingObservation(
        observation_id="obs-mpc",
        controller_state=ControllerState(
            controller_id="ctrl-mpc",
            method_name="mpc",
            step_index=0,
            active_split="calibration",
        ),
        frame_id="frame-mpc",
        decision_time_ms=0,
        target_deadline_ms=target_deadline_ms,
        candidate_set=CandidateSet(
            candidate_set_id="candidate-set-mpc",
            decision_time_ms=0,
            candidates=candidates,
        ),
    )
