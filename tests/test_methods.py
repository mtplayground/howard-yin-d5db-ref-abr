from __future__ import annotations

import pytest

from ref_abr.candidates import CandidateGenerationSpec, DecisionEpoch, generate_candidate_objects
from ref_abr.domain import ControllerState, MediaType, ScheduleDecision
from ref_abr.methods import (
    ActionBudget,
    MethodAdapter,
    MethodError,
    ObservationBudget,
    SchedulingObservation,
    apply_observation_budget,
    plan_schedule,
    validate_schedule_decision,
)
from ref_abr.substrate import ParametricSubstrateValueProvider
from ref_abr.utility import ResourceBudget, estimate_candidate_set_utility
from ref_abr.workloads import assemble_workload_manifest


def test_plan_schedule_applies_equal_observation_budget_before_method_call() -> None:
    observation = _observation()
    method = RecordingMethod()

    decision = plan_schedule(
        method,
        observation,
        observation_budget=ObservationBudget(max_candidates=2, max_utility_estimates=1, max_lifecycle_states=0),
        action_budget=ActionBudget(max_selected_objects=2, max_selected_bytes=1_000_000),
    )

    assert method.seen_candidate_ids == [candidate.candidate_id for candidate in observation.candidates[:2]]
    assert method.seen_utility_count == 1
    assert decision.controller_id == observation.controller_state.controller_id
    assert decision.frame_id == observation.frame_id
    assert decision.selected_object_ids == tuple(candidate.object_id for candidate in observation.candidates[:2])
    assert decision.metadata["adapter"]["method_id"] == "recording-method"


def test_mapping_decision_can_select_candidate_ids_and_records_deterministic_id() -> None:
    observation = _observation()
    selected_candidate_id = observation.candidates[0].candidate_id
    method = lambda obs, budget: {  # noqa: E731
        "selected_candidate_ids": [selected_candidate_id],
        "expected_utility": 0.5,
    }

    first = plan_schedule(
        method,
        observation,
        observation_budget=ObservationBudget(max_candidates=3),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=1_000_000),
        method_id="lambda-method",
        method_name="lambda",
    )
    second = plan_schedule(
        method,
        observation,
        observation_budget=ObservationBudget(max_candidates=3),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=1_000_000),
        method_id="lambda-method",
        method_name="lambda",
    )

    assert first.decision_id == second.decision_id
    assert first.selected_object_ids == (observation.candidates[0].object_id,)
    assert first.expected_utility == 0.5
    assert first.metadata["adapter"]["selected_candidate_ids"] == (selected_candidate_id,)


def test_action_budget_rejects_too_many_objects_and_bytes() -> None:
    observation = _observation(size_bytes=600_000)

    with pytest.raises(MethodError, match="max_selected_objects"):
        plan_schedule(
            SelectTwoMethod(),
            observation,
            observation_budget=ObservationBudget(max_candidates=2),
            action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=2_000_000),
        )

    with pytest.raises(MethodError, match="selected bytes"):
        plan_schedule(
            RecordingMethod(),
            observation,
            observation_budget=ObservationBudget(max_candidates=2),
            action_budget=ActionBudget(max_selected_objects=2, max_selected_bytes=500_000),
        )


def test_validate_schedule_decision_rejects_unknown_objects_and_late_deadlines() -> None:
    observation = apply_observation_budget(_observation(), ObservationBudget(max_candidates=2))
    decision = ScheduleDecision(
        decision_id="decision-x",
        controller_id=observation.controller_state.controller_id,
        frame_id=observation.frame_id,
        selected_object_ids=("unknown",),
        decision_time_ms=observation.decision_time_ms,
        target_deadline_ms=observation.target_deadline_ms,
    )

    with pytest.raises(MethodError, match="unknown object_id"):
        validate_schedule_decision(
            decision,
            observation=observation,
            action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=1_000_000),
        )

    late_decision = ScheduleDecision(
        decision_id="decision-late",
        controller_id=observation.controller_state.controller_id,
        frame_id=observation.frame_id,
        selected_object_ids=(),
        decision_time_ms=observation.decision_time_ms,
        target_deadline_ms=observation.target_deadline_ms,
    )
    with pytest.raises(MethodError, match="max_deadline_ms"):
        validate_schedule_decision(
            late_decision,
            observation=observation,
            action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=1_000_000, max_deadline_ms=1),
        )


def test_adapter_rejects_method_without_contract() -> None:
    with pytest.raises(MethodError, match="method"):
        MethodAdapter(
            method=object(),  # type: ignore[arg-type]
            observation_budget=ObservationBudget(max_candidates=1),
            action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=1),
        )

    with pytest.raises(MethodError, match="max_candidates"):
        ObservationBudget(max_candidates=0)

    with pytest.raises(MethodError, match="max_selected_objects"):
        ActionBudget(max_selected_objects=0, max_selected_bytes=1)


class RecordingMethod:
    method_id = "recording-method"
    method_name = "recording"

    def __init__(self) -> None:
        self.seen_candidate_ids: list[str] = []
        self.seen_utility_count = 0

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, object]:
        self.seen_candidate_ids = [candidate.candidate_id for candidate in observation.candidates]
        self.seen_utility_count = len(observation.utility_estimates)
        selected = observation.candidates[: action_budget.max_selected_objects]
        return {
            "selected_object_ids": [candidate.object_id for candidate in selected],
            "expected_utility": 0.25,
        }


class SelectTwoMethod:
    method_id = "select-two"
    method_name = "select-two"

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, object]:
        selected = observation.candidates[:2]
        return {
            "selected_object_ids": [candidate.object_id for candidate in selected],
            "expected_utility": 0.25,
        }


def _observation(*, size_bytes: int = 100_000) -> SchedulingObservation:
    candidate_set = generate_candidate_objects(
        _workload(size_bytes=size_bytes),
        DecisionEpoch(decision_time_ms=10, frame_id="frame-1"),
        spec=CandidateGenerationSpec(
            resolutions=("720p",),
            fov_degrees=(90,),
            lookahead_ms=(0,),
            expiration_ms=(100,),
            retransmit_priorities=(0,),
            enhancement_layers=(1,),
            include_tiles=False,
            include_reference_actions=False,
        ),
        substrate_provider=ParametricSubstrateValueProvider(),
    )
    utilities = estimate_candidate_set_utility(
        candidate_set,
        budgets=ResourceBudget(available_time_ms=100, available_bytes=1_000_000, available_memory_mb=1024),
    )
    return SchedulingObservation(
        observation_id="obs-a",
        controller_state=ControllerState(
            controller_id="ctrl-a",
            method_name="recording",
            step_index=0,
            active_split="calibration",
        ),
        frame_id="frame-1",
        decision_time_ms=10,
        target_deadline_ms=110,
        candidate_set=candidate_set,
        utility_estimates=utilities.estimates,
        metadata={"source": "test"},
    )


def _workload(*, size_bytes: int):
    return assemble_workload_manifest(
        {
            "dataset": "methods-test",
            "sequences": [
                {
                    "scene": "scene",
                    "name": "seq",
                    "assets": [
                        {
                            "object_id": "splat-a",
                            "path": "splat-a.ply",
                            "size_bytes": size_bytes,
                            "media_type": MediaType.GAUSSIAN_SPLAT.value,
                        },
                        {
                            "object_id": "splat-b",
                            "path": "splat-b.ply",
                            "size_bytes": size_bytes,
                            "media_type": MediaType.GAUSSIAN_SPLAT.value,
                        },
                    ],
                }
            ],
        },
        split="calibration",
        config_id="methods-test-config",
        seed=9,
    )
