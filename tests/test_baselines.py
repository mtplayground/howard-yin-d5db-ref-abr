from __future__ import annotations

import pytest

from ref_abr.baselines import (
    BaselineError,
    CAGSFixedReferenceBaseline,
    ReferenceOnlyAfterBaseBaseline,
    SVQGaussianOnlyABRBaseline,
    minimum_baselines,
)
from ref_abr.candidates import CandidateGenerationSpec, DecisionEpoch, generate_candidate_objects
from ref_abr.domain import ControllerState, LifecycleStatus, MediaType, ReferenceLifecycleState
from ref_abr.methods import ActionBudget, ObservationBudget, SchedulingObservation, plan_schedule
from ref_abr.substrate import ParametricSubstrateValueProvider
from ref_abr.utility import ResourceBudget, estimate_candidate_set_utility
from ref_abr.workloads import assemble_workload_manifest


def test_cags_fixed_reference_selects_reference_actions_under_adapter() -> None:
    observation = _observation()
    method = CAGSFixedReferenceBaseline(fixed_resolution="720p")

    decision = plan_schedule(
        method,
        observation,
        observation_budget=ObservationBudget(max_candidates=20),
        action_budget=ActionBudget(max_selected_objects=2, max_selected_bytes=1_000_000),
    )

    selected = _selected_candidates(observation, decision.metadata["adapter"]["selected_candidate_ids"])
    assert {candidate.candidate_kind for candidate in selected} == {"reference_action"}
    assert decision.metadata["baseline"]["policy"] == "fixed_reference"
    assert decision.metadata["adapter"]["method_id"] == "cags-fixed-reference"


def test_svq_gaussian_only_abr_excludes_reference_actions_and_uses_utility_order() -> None:
    observation = _observation()
    method = SVQGaussianOnlyABRBaseline()

    decision = plan_schedule(
        method,
        observation,
        observation_budget=ObservationBudget(max_candidates=20),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=1_000_000),
    )

    selected = _selected_candidates(observation, decision.metadata["adapter"]["selected_candidate_ids"])
    assert len(selected) == 1
    assert selected[0].candidate_kind in {"gaussian_base", "gaussian_enhancement", "tile"}
    assert selected[0].candidate_kind != "reference_action"
    assert decision.expected_utility is not None
    assert decision.metadata["baseline"]["policy"] == "gaussian_only_abr"


def test_reference_only_after_base_switches_from_base_to_reference() -> None:
    observation = _observation()
    method = ReferenceOnlyAfterBaseBaseline()

    before_base = plan_schedule(
        method,
        observation,
        observation_budget=ObservationBudget(max_candidates=20),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=1_000_000),
    )
    before_selected = _selected_candidates(observation, before_base.metadata["adapter"]["selected_candidate_ids"])
    assert {candidate.candidate_kind for candidate in before_selected} == {"gaussian_base"}
    assert before_base.metadata["baseline"]["policy"] == "base_before_reference"

    after_observation = _observation(
        lifecycle_states=(
            ReferenceLifecycleState("splat-a", LifecycleStatus.AVAILABLE, updated_at_ms=5),
            ReferenceLifecycleState("splat-b", LifecycleStatus.AVAILABLE, updated_at_ms=5),
        )
    )
    after_base = plan_schedule(
        method,
        after_observation,
        observation_budget=ObservationBudget(max_candidates=20),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=1_000_000),
    )
    after_selected = _selected_candidates(after_observation, after_base.metadata["adapter"]["selected_candidate_ids"])
    assert {candidate.candidate_kind for candidate in after_selected} == {"reference_action"}
    assert after_base.metadata["baseline"]["policy"] == "reference_after_base"


def test_baselines_respect_action_byte_budget_by_selecting_nothing_when_needed() -> None:
    observation = _observation(size_bytes=900_000)

    decision = plan_schedule(
        CAGSFixedReferenceBaseline(),
        observation,
        observation_budget=ObservationBudget(max_candidates=20),
        action_budget=ActionBudget(max_selected_objects=2, max_selected_bytes=10),
    )

    assert decision.selected_object_ids == ()
    assert decision.metadata["adapter"]["selected_candidate_ids"] == ()


def test_minimum_baseline_set_and_invalid_config() -> None:
    baselines = minimum_baselines()

    assert [baseline.method_id for baseline in baselines] == [
        "cags-fixed-reference",
        "svq-gaussian-only-abr",
        "reference-only-after-base",
    ]
    with pytest.raises(BaselineError, match="fixed_resolution"):
        CAGSFixedReferenceBaseline(fixed_resolution="")


def _selected_candidates(observation: SchedulingObservation, selected_candidate_ids) -> tuple:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in observation.candidates}
    return tuple(candidate_by_id[candidate_id] for candidate_id in selected_candidate_ids)


def _observation(
    *,
    size_bytes: int = 100_000,
    lifecycle_states: tuple[ReferenceLifecycleState, ...] = (),
) -> SchedulingObservation:
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
            tile_rows=1,
            tile_columns=1,
        ),
        substrate_provider=ParametricSubstrateValueProvider(),
    )
    utilities = estimate_candidate_set_utility(
        candidate_set,
        budgets=ResourceBudget(available_time_ms=100, available_bytes=1_000_000, available_memory_mb=1024),
    )
    return SchedulingObservation(
        observation_id="obs-baseline",
        controller_state=ControllerState(
            controller_id="ctrl-baseline",
            method_name="baseline",
            step_index=0,
            active_split="calibration",
        ),
        frame_id="frame-1",
        decision_time_ms=10,
        target_deadline_ms=110,
        candidate_set=candidate_set,
        utility_estimates=utilities.estimates,
        lifecycle_states=lifecycle_states,
    )


def _workload(*, size_bytes: int):
    return assemble_workload_manifest(
        {
            "dataset": "baseline-test",
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
        config_id="baseline-test-config",
        seed=11,
    )
