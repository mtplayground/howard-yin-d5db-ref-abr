from __future__ import annotations

import pytest

from ref_abr.ablation import (
    AblatedRefABRController,
    AblationError,
    RefABRAblationSwitches,
    apply_refabr_ablation,
    refabr_self_ablation_variants,
)
from ref_abr.candidates import CandidateObject, CandidateSet
from ref_abr.domain import ControllerState, LifecycleStatus, ReferenceLifecycleState
from ref_abr.methods import ActionBudget, ObservationBudget, SchedulingObservation, plan_schedule
from ref_abr.mpc import MPCConfig, RobustDeadlineAwareMPCController
from ref_abr.utility import CandidateUtilityEstimate, ResourceDebt, ResourcePrice, UtilityUncertainty


def test_no_lifecycle_switch_removes_lifecycle_inputs_and_risks() -> None:
    observation = _observation(
        (_candidate("ref", kind="reference_action", value=5.0, size_bytes=100, deadline_ms=100),),
        utility_estimates=(_estimate("candidate-ref", lifecycle_risk=0.8),),
        lifecycle_states=(
            ReferenceLifecycleState("object-ref", LifecycleStatus.EXPIRED, updated_at_ms=90, deadline_ms=100, attempts=2),
        ),
        metadata={"virtual_queue": {"lifecycle_debt": 0.9, "queue_debt_ms": 10.0}},
    )

    ablated = apply_refabr_ablation(observation, RefABRAblationSwitches(no_lifecycle=True))

    assert ablated.lifecycle_states == ()
    assert ablated.utility_estimates[0].lifecycle_risk == 0.0
    assert "lifecycle_debt" not in ablated.metadata["virtual_queue"]
    assert ablated.metadata["refabr_ablation"]["variant_id"] == "no-lifecycle"


def test_no_uncertainty_switch_collapses_mpc_to_nominal_scenario() -> None:
    observation = _observation(
        (
            _candidate("ref", kind="reference_action", value=10.0, size_bytes=200, deadline_ms=100, mpc={"viewport_sensitivity": 1.0}),
            _candidate("base", kind="gaussian_base", value=4.5, size_bytes=200, deadline_ms=100, mpc={"viewport_sensitivity": 0.0}),
        ),
        target_deadline_ms=100,
    )
    controller = RobustDeadlineAwareMPCController(
        config=MPCConfig(
            horizon_steps=1,
            max_scenarios=3,
            bandwidth_interval=(1.0, 1.0),
            viewport_error_interval=(90.0, 90.0),
            deadline_scale_interval=(1.0, 1.0),
        )
    )

    raw_decision = plan_schedule(
        controller,
        observation,
        observation_budget=ObservationBudget(max_candidates=5),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=500),
    )
    ablated_decision = plan_schedule(
        AblatedRefABRController(controller, RefABRAblationSwitches(no_uncertainty=True)),
        observation,
        observation_budget=ObservationBudget(max_candidates=5),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=500),
    )

    assert raw_decision.metadata["adapter"]["selected_candidate_ids"] == ("candidate-base",)
    assert ablated_decision.metadata["adapter"]["selected_candidate_ids"] == ("candidate-ref",)
    assert ablated_decision.metadata["mpc"]["metadata"]["scenario_count"] == 1
    assert ablated_decision.metadata["ablation"]["switches"]["variant_id"] == "no-uncertainty"


def test_no_component_cost_switch_removes_render_compute_costs_without_changing_structure() -> None:
    observation = _observation(
        (
            _candidate("expensive", kind="reference_action", value=9.0, size_bytes=100, deadline_ms=100, render_ms=10.0, compute_ms=10.0),
            _candidate("cheap", kind="gaussian_base", value=1.0, size_bytes=100, deadline_ms=100, render_ms=0.5, compute_ms=0.5),
        ),
        target_deadline_ms=100,
    )
    controller = RobustDeadlineAwareMPCController(
        config=MPCConfig(
            horizon_steps=1,
            max_scenarios=1,
            bandwidth_interval=(1.0, 1.0),
            viewport_error_interval=(0.0, 0.0),
            deadline_scale_interval=(1.0, 1.0),
            max_render_ms=1.0,
            max_compute_ms=1.0,
        )
    )

    raw_decision = plan_schedule(
        controller,
        observation,
        observation_budget=ObservationBudget(max_candidates=5),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=500),
    )
    ablated_decision = plan_schedule(
        AblatedRefABRController(controller, RefABRAblationSwitches(no_component_cost=True)),
        observation,
        observation_budget=ObservationBudget(max_candidates=5),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=500),
    )

    assert raw_decision.metadata["adapter"]["selected_candidate_ids"] == ("candidate-cheap",)
    assert ablated_decision.metadata["adapter"]["selected_candidate_ids"] == ("candidate-expensive",)
    allocation = ablated_decision.metadata["mpc"]["selected_step"]["allocation"]
    assert allocation["total_render_ms"] == pytest.approx(0.0)
    assert allocation["total_compute_ms"] == pytest.approx(0.0)


def test_ablation_variant_factory_and_validation() -> None:
    controller = RobustDeadlineAwareMPCController()
    variants = refabr_self_ablation_variants(controller)

    assert [variant.switches.variant_id for variant in variants] == [
        "no-lifecycle",
        "no-uncertainty",
        "no-component-cost",
    ]
    assert [variant.method_id for variant in variants] == [
        "robust-deadline-aware-mpc-no-lifecycle",
        "robust-deadline-aware-mpc-no-uncertainty",
        "robust-deadline-aware-mpc-no-component-cost",
    ]
    with pytest.raises(AblationError, match="no_lifecycle"):
        RefABRAblationSwitches(no_lifecycle="yes")  # type: ignore[arg-type]


def _candidate(
    suffix: str,
    *,
    kind: str,
    value: float,
    size_bytes: int,
    deadline_ms: int,
    render_ms: float = 1.0,
    compute_ms: float = 1.0,
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
                "render_ms": render_ms,
                "compute_ms": compute_ms,
                "deadline_ms": deadline_ms,
            },
            "mpc": mpc or {},
        },
    )


def _estimate(candidate_id: str, *, lifecycle_risk: float) -> CandidateUtilityEstimate:
    return CandidateUtilityEstimate(
        estimate_id=f"estimate-{candidate_id}",
        candidate_id=candidate_id,
        visible_qoe_gain=1.0,
        lifecycle_risk=lifecycle_risk,
        deadline_miss_probability=0.25,
        resource_price=ResourcePrice(time_price=0.1, transfer_price=0.1, memory_price=0.1),
        resource_debt=ResourceDebt(
            time_debt_ms=1.0,
            transfer_debt_bytes=100,
            memory_debt_mb=1.0,
            carried_queue_debt_ms=1.0,
            carried_transfer_debt_bytes=100,
        ),
        expected_utility=1.0,
        uncertainty=UtilityUncertainty(
            quality_stddev=0.2,
            timing_stddev_ms=1.0,
            deadline_probability_stddev=0.2,
            utility_stddev=0.2,
            confidence=0.5,
        ),
    )


def _observation(
    candidates: tuple[CandidateObject, ...],
    *,
    target_deadline_ms: int = 100,
    utility_estimates: tuple[CandidateUtilityEstimate, ...] = (),
    lifecycle_states: tuple[ReferenceLifecycleState, ...] = (),
    metadata: dict | None = None,
) -> SchedulingObservation:
    return SchedulingObservation(
        observation_id="obs-ablation",
        controller_state=ControllerState(
            controller_id="ctrl-ablation",
            method_name="ablation",
            step_index=0,
            active_split="calibration",
        ),
        frame_id="frame-ablation",
        decision_time_ms=0,
        target_deadline_ms=target_deadline_ms,
        candidate_set=CandidateSet(
            candidate_set_id="candidate-set-ablation",
            decision_time_ms=0,
            candidates=candidates,
        ),
        utility_estimates=utility_estimates,
        lifecycle_states=lifecycle_states,
        metadata=metadata or {},
    )
