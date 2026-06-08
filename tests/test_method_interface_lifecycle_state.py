from __future__ import annotations

import pytest

from ref_abr.candidates import CandidateObject, CandidateSet
from ref_abr.deadline_lifecycle_metrics import reference_lifecycle_rates
from ref_abr.domain import ControllerState, LifecycleStatus, ReferenceLifecycleState, ScheduleDecision
from ref_abr.lifecycle import (
    DropReason,
    LifecycleAction,
    LifecycleError,
    LifecyclePhase,
    ReferenceLifecycleEvent,
    ReferenceLifecycleStateMachine,
)
from ref_abr.methods import ActionBudget, MethodError, ObservationBudget, SchedulingObservation, plan_schedule


def test_mock_planner_returns_feasible_candidate_decision_under_shared_budget() -> None:
    observation = _observation(
        lifecycle_states=(
            ReferenceLifecycleState("ref-requested", LifecycleStatus.REQUESTED, updated_at_ms=11, deadline_ms=90, attempts=1),
            ReferenceLifecycleState("ref-available", LifecycleStatus.AVAILABLE, updated_at_ms=12, deadline_ms=90, attempts=1),
        )
    )
    method = BudgetAwareMockMethod()

    decision = plan_schedule(
        method,
        observation,
        observation_budget=ObservationBudget(max_candidates=3, max_lifecycle_states=2),
        action_budget=ActionBudget(max_selected_objects=2, max_selected_candidates=2, max_selected_bytes=350),
    )

    assert isinstance(decision, ScheduleDecision)
    assert decision.controller_id == observation.controller_state.controller_id
    assert decision.frame_id == observation.frame_id
    assert decision.decision_time_ms == observation.decision_time_ms
    assert decision.target_deadline_ms == observation.target_deadline_ms
    assert decision.selected_object_ids == ("object-a", "object-b")
    assert decision.metadata["adapter"]["selected_candidate_ids"] == ("candidate-a", "candidate-b")
    assert decision.metadata["adapter"]["method_id"] == "budget-aware-mock"
    assert method.seen_lifecycle_statuses == (LifecycleStatus.REQUESTED, LifecycleStatus.AVAILABLE)


def test_callable_mock_can_return_schedule_decision_record_directly() -> None:
    observation = _observation()

    def mock_method(obs: SchedulingObservation, action_budget: ActionBudget) -> ScheduleDecision:
        assert action_budget.max_selected_objects == 1
        return ScheduleDecision(
            decision_id="direct-mock-decision",
            controller_id=obs.controller_state.controller_id,
            frame_id=obs.frame_id,
            selected_object_ids=(obs.candidates[0].object_id,),
            decision_time_ms=obs.decision_time_ms,
            target_deadline_ms=obs.target_deadline_ms,
            expected_utility=0.75,
            metadata={"mock": {"direct_record": True}},
        )

    decision = plan_schedule(
        mock_method,
        observation,
        observation_budget=ObservationBudget(max_candidates=2),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=150),
        method_id="callable-direct-mock",
        method_name="callable-direct-mock",
    )

    assert decision.decision_id == "direct-mock-decision"
    assert decision.selected_object_ids == ("object-a",)
    assert decision.expected_utility == pytest.approx(0.75)
    assert decision.metadata["mock"]["direct_record"] is True


def test_adapter_rejects_mock_decision_that_escapes_visible_candidates() -> None:
    observation = _observation()
    hidden_candidate_id = observation.candidates[1].candidate_id

    def hidden_candidate_mock(obs: SchedulingObservation, action_budget: ActionBudget) -> dict[str, object]:
        assert [candidate.candidate_id for candidate in obs.candidates] == ["candidate-a"]
        return {"selected_candidate_ids": [hidden_candidate_id]}

    with pytest.raises(MethodError, match="unknown candidate_id"):
        plan_schedule(
            hidden_candidate_mock,
            observation,
            observation_budget=ObservationBudget(max_candidates=1),
            action_budget=ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=1_000),
            method_id="hidden-candidate-mock",
            method_name="hidden-candidate-mock",
        )


def test_lifecycle_states_and_metric_flags_distinguish_useful_late_and_stale() -> None:
    useful_machine, useful_events = _run_lifecycle(
        "ref-useful",
        deadline_ms=50,
        actions=(
            (LifecycleAction.REQUEST, 1),
            (LifecycleAction.GENERATE, 2),
            (LifecycleAction.TRANSFER, 3),
            (LifecycleAction.ARRIVE, 4),
            (LifecycleAction.RESTORE, 5),
            (LifecycleAction.USE, 6),
        ),
    )
    late_machine, late_events = _run_lifecycle(
        "ref-late",
        deadline_ms=10,
        actions=(
            (LifecycleAction.REQUEST, 1),
            (LifecycleAction.GENERATE, 2),
            (LifecycleAction.EXPIRE, 11),
        ),
        expire_reason=DropReason.DEADLINE_MISSED,
    )
    stale_machine, stale_events = _run_lifecycle(
        "ref-stale",
        deadline_ms=50,
        actions=(
            (LifecycleAction.REQUEST, 1),
            (LifecycleAction.GENERATE, 2),
            (LifecycleAction.TRANSFER, 3),
            (LifecycleAction.ARRIVE, 4),
            (LifecycleAction.RESTORE, 5),
            (LifecycleAction.STALE, 6),
        ),
    )

    assert useful_machine.phase == LifecyclePhase.USED
    assert useful_machine.state.status == LifecycleStatus.AVAILABLE
    assert useful_machine.state.metadata["lifecycle"]["terminal"] is False
    assert late_machine.phase == LifecyclePhase.EXPIRED
    assert late_machine.state.metadata["lifecycle"]["drop_reason"] == "deadline_missed"
    assert stale_machine.phase == LifecyclePhase.STALE
    assert stale_machine.state.metadata["lifecycle"]["drop_reason"] == "stale"

    metrics = reference_lifecycle_rates((*useful_events, *late_events, *stale_events))
    by_name = {metric.metric_name: metric for metric in metrics}
    assert by_name["reference_lifecycle_useful_rate"].value == pytest.approx(1 / 3)
    assert by_name["reference_lifecycle_late_rate"].value == pytest.approx(1 / 3)
    assert by_name["reference_lifecycle_stale_rate"].value == pytest.approx(1 / 3)
    assert by_name["reference_lifecycle_expired_rate"].value == pytest.approx(1 / 3)
    assert by_name["reference_lifecycle_off_view_rate"].value == pytest.approx(0.0)

    with pytest.raises(LifecycleError, match="terminal"):
        stale_machine.use(at_ms=7)


def test_lifecycle_rejects_use_before_restore() -> None:
    requested, _ = ReferenceLifecycleStateMachine("ref-illegal").request(at_ms=1)

    with pytest.raises(LifecycleError, match="Illegal lifecycle transition"):
        requested.use(at_ms=2)


class BudgetAwareMockMethod:
    method_id = "budget-aware-mock"
    method_name = "budget-aware-mock"

    def __init__(self) -> None:
        self.seen_lifecycle_statuses: tuple[LifecycleStatus, ...] = ()

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, object]:
        self.seen_lifecycle_statuses = tuple(state.status for state in observation.lifecycle_states)
        selected: list[str] = []
        selected_bytes = 0
        max_candidates = action_budget.max_selected_candidates or action_budget.max_selected_objects
        for candidate in observation.candidates:
            if len(selected) >= min(action_budget.max_selected_objects, max_candidates):
                break
            if selected_bytes + candidate.size_bytes > action_budget.max_selected_bytes:
                continue
            selected.append(candidate.candidate_id)
            selected_bytes += candidate.size_bytes
        return {
            "selected_candidate_ids": selected,
            "expected_utility": 0.5,
            "metadata": {"mock": {"selected_bytes": selected_bytes}},
        }


def _observation(*, lifecycle_states: tuple[ReferenceLifecycleState, ...] = ()) -> SchedulingObservation:
    return SchedulingObservation(
        observation_id="method-lifecycle-obs",
        controller_state=ControllerState(
            controller_id="controller-test",
            method_name="mock",
            step_index=0,
            active_split="calibration",
        ),
        frame_id="frame-test",
        decision_time_ms=10,
        target_deadline_ms=80,
        candidate_set=CandidateSet(
            candidate_set_id="method-lifecycle-candidates",
            decision_time_ms=10,
            candidates=(
                _candidate("candidate-a", "object-a", size_bytes=100),
                _candidate("candidate-b", "object-b", size_bytes=200),
                _candidate("candidate-c", "object-c", size_bytes=400),
            ),
        ),
        lifecycle_states=lifecycle_states,
        metadata={"fixture": "method-interface-lifecycle-state"},
    )


def _candidate(candidate_id: str, object_id: str, *, size_bytes: int) -> CandidateObject:
    return CandidateObject(
        candidate_id=candidate_id,
        object_id=object_id,
        candidate_kind="gaussian_base",
        decision_time_ms=10,
        layer=0,
        resolution="720p",
        fov_deg=90.0,
        viewpoint=None,
        lookahead_ms=0,
        expiration_ms=70,
        retransmit_priority=0,
        size_bytes=size_bytes,
    )


def _run_lifecycle(
    reference_id: str,
    *,
    deadline_ms: int,
    actions: tuple[tuple[LifecycleAction, int], ...],
    expire_reason: DropReason = DropReason.EXPIRED,
) -> tuple[ReferenceLifecycleStateMachine, tuple[ReferenceLifecycleEvent, ...]]:
    machine = ReferenceLifecycleStateMachine(reference_id, deadline_ms=deadline_ms)
    events = []
    for action, at_ms in actions:
        if action == LifecycleAction.EXPIRE:
            machine, emission = machine.expire(at_ms=at_ms, drop_reason=expire_reason)
        else:
            machine, emission = machine.transition(action, at_ms=at_ms)
        events.append(emission.event)
    return machine, tuple(events)
