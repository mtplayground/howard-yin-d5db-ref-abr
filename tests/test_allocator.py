from __future__ import annotations

import pytest

from ref_abr.allocator import (
    AllocatorError,
    KnapsackBudget,
    allocate_deadline_aware_knapsack,
)
from ref_abr.baselines import DeadlineAwareKnapsackAllocatorBaseline, deadline_aware_allocators
from ref_abr.candidates import CandidateObject, CandidateSet
from ref_abr.domain import ControllerState
from ref_abr.methods import ActionBudget, ObservationBudget, SchedulingObservation, plan_schedule


def test_allocator_selects_best_value_subset_under_heterogeneous_budgets() -> None:
    candidates = (
        _candidate("a", value=6.0, size_bytes=400, render_ms=2.0, compute_ms=3.0),
        _candidate("b", value=5.0, size_bytes=300, render_ms=2.0, compute_ms=2.0),
        _candidate("c", value=9.0, size_bytes=800, render_ms=6.0, compute_ms=8.0),
    )

    allocation = allocate_deadline_aware_knapsack(
        candidates,
        budget=KnapsackBudget(max_bytes=700, max_render_ms=5.0, max_compute_ms=6.0, max_selected_candidates=2),
    )

    assert allocation.selected_candidate_ids == ("candidate-a", "candidate-b")
    assert allocation.total_value == pytest.approx(11.0)
    assert allocation.total_bytes == 700
    assert "candidate-c" in allocation.infeasible_reasons


def test_allocator_enforces_dependency_closure() -> None:
    base = _candidate("base", value=1.0, size_bytes=200)
    enhancement = _candidate("enh", value=10.0, size_bytes=200, dependencies=("candidate-base",))
    fallback = _candidate("fallback", value=5.0, size_bytes=300)

    tight = allocate_deadline_aware_knapsack(
        (base, enhancement, fallback),
        budget=KnapsackBudget(max_bytes=300, max_selected_candidates=2),
    )
    roomy = allocate_deadline_aware_knapsack(
        (base, enhancement, fallback),
        budget=KnapsackBudget(max_bytes=450, max_selected_candidates=2),
    )

    assert tight.selected_candidate_ids == ("candidate-fallback",)
    assert roomy.selected_candidate_ids == ("candidate-base", "candidate-enh")
    assert roomy.total_value == pytest.approx(11.0)


def test_deadline_aware_knapsack_allocator_baseline_runs_under_adapter() -> None:
    candidates = (
        _candidate("a", object_id="obj-a", value=2.0, size_bytes=200, deadline_ms=80),
        _candidate("b", object_id="obj-b", value=8.0, size_bytes=200, deadline_ms=80),
        _candidate("late", object_id="obj-late", value=20.0, size_bytes=100, deadline_ms=140),
    )
    observation = _observation(candidates, target_deadline_ms=100)

    decision = plan_schedule(
        DeadlineAwareKnapsackAllocatorBaseline(max_render_ms=10.0, max_compute_ms=10.0),
        observation,
        observation_budget=ObservationBudget(max_candidates=10),
        action_budget=ActionBudget(max_selected_objects=2, max_selected_candidates=2, max_selected_bytes=500, max_deadline_ms=100),
    )

    assert decision.metadata["adapter"]["method_id"] == "deadline-aware-knapsack-allocator"
    assert decision.metadata["adapter"]["selected_candidate_ids"] == ("candidate-a", "candidate-b")
    assert decision.metadata["baseline"]["policy"] == "deadline_aware_knapsack"
    assert decision.metadata["allocation"]["total_value"] == pytest.approx(10.0)
    assert [method.method_id for method in deadline_aware_allocators()] == ["deadline-aware-knapsack-allocator"]


def test_allocator_rejects_unknown_utility_estimate_inputs() -> None:
    with pytest.raises(AllocatorError, match="candidates"):
        allocate_deadline_aware_knapsack((), budget=KnapsackBudget(max_bytes=1))

    with pytest.raises(AllocatorError, match="max_bytes"):
        KnapsackBudget(max_bytes=-1)


def _candidate(
    suffix: str,
    *,
    object_id: str | None = None,
    value: float,
    size_bytes: int,
    render_ms: float = 1.0,
    compute_ms: float = 1.0,
    deadline_ms: int = 100,
    dependencies: tuple[str, ...] = (),
) -> CandidateObject:
    return CandidateObject(
        candidate_id=f"candidate-{suffix}",
        object_id=object_id or f"object-{suffix}",
        candidate_kind="gaussian_enhancement",
        decision_time_ms=0,
        layer=1,
        resolution="720p",
        fov_deg=90.0,
        viewpoint=None,
        lookahead_ms=0,
        expiration_ms=deadline_ms,
        retransmit_priority=0,
        size_bytes=size_bytes,
        dependencies=dependencies,
        metadata={"allocator": {"value": value, "render_ms": render_ms, "compute_ms": compute_ms, "deadline_ms": deadline_ms}},
    )


def _observation(candidates: tuple[CandidateObject, ...], *, target_deadline_ms: int) -> SchedulingObservation:
    return SchedulingObservation(
        observation_id="obs-allocator",
        controller_state=ControllerState(
            controller_id="ctrl-allocator",
            method_name="allocator",
            step_index=0,
            active_split="calibration",
        ),
        frame_id="frame-allocator",
        decision_time_ms=0,
        target_deadline_ms=target_deadline_ms,
        candidate_set=CandidateSet(
            candidate_set_id="candidate-set-allocator",
            decision_time_ms=0,
            candidates=candidates,
        ),
        metadata={"allocator_budget": {"max_render_ms": 10.0, "max_compute_ms": 10.0}},
    )
