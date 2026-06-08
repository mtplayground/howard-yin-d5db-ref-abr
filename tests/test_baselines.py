from __future__ import annotations

import pytest

from ref_abr.baselines import (
    BaselineError,
    BandwidthGreedyBaseline,
    BOLASlackAdaptedBaseline,
    CAGSFixedReferenceBaseline,
    DeadlineGreedyBaseline,
    FixedReferenceCadenceBaseline,
    IndependentGaussianSchedulerBaseline,
    IndependentReferenceSchedulerBaseline,
    PerfectInformationOracle,
    QualityMaxDeadlineUnawareBaseline,
    ReferenceOnlyAfterBaseBaseline,
    RobustMPCJointSpaceBaseline,
    SVQGaussianOnlyABRBaseline,
    canonical_abr_baselines,
    minimum_baselines,
    perfect_information_oracles,
    simple_baselines,
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


def test_fixed_reference_cadence_selects_only_on_aligned_epochs() -> None:
    observation = _observation(decision_time_ms=10)
    method = FixedReferenceCadenceBaseline(cadence_ms=10)

    aligned = plan_schedule(
        method,
        observation,
        observation_budget=ObservationBudget(max_candidates=20),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=1_000_000),
    )
    selected = _selected_candidates(observation, aligned.metadata["adapter"]["selected_candidate_ids"])
    assert {candidate.candidate_kind for candidate in selected} == {"reference_action"}
    assert aligned.metadata["baseline"]["policy"] == "fixed_reference_cadence"

    skipped = plan_schedule(
        FixedReferenceCadenceBaseline(cadence_ms=100, phase_ms=0),
        observation,
        observation_budget=ObservationBudget(max_candidates=20),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=1_000_000),
    )
    assert skipped.selected_object_ids == ()
    assert skipped.metadata["baseline"]["policy"] == "fixed_reference_cadence_skip"


def test_independent_gaussian_and_reference_schedulers_partition_candidates() -> None:
    observation = _observation()

    gaussian_decision = plan_schedule(
        IndependentGaussianSchedulerBaseline(),
        observation,
        observation_budget=ObservationBudget(max_candidates=20),
        action_budget=ActionBudget(max_selected_objects=2, max_selected_bytes=1_000_000),
    )
    gaussian_selected = _selected_candidates(observation, gaussian_decision.metadata["adapter"]["selected_candidate_ids"])
    assert gaussian_selected
    assert all(candidate.candidate_kind in {"gaussian_base", "gaussian_enhancement", "tile"} for candidate in gaussian_selected)

    reference_decision = plan_schedule(
        IndependentReferenceSchedulerBaseline(),
        observation,
        observation_budget=ObservationBudget(max_candidates=20),
        action_budget=ActionBudget(max_selected_objects=2, max_selected_bytes=1_000_000),
    )
    reference_selected = _selected_candidates(observation, reference_decision.metadata["adapter"]["selected_candidate_ids"])
    assert reference_selected
    assert {candidate.candidate_kind for candidate in reference_selected} == {"reference_action"}


def test_bandwidth_greedy_prefers_smallest_candidates() -> None:
    observation = _observation(size_bytes=100_001, tile_columns=2)

    decision = plan_schedule(
        BandwidthGreedyBaseline(),
        observation,
        observation_budget=ObservationBudget(max_candidates=30),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=1_000_000),
    )

    selected = _selected_candidates(observation, decision.metadata["adapter"]["selected_candidate_ids"])
    assert len(selected) == 1
    assert selected[0].candidate_kind == "tile"
    assert selected[0].size_bytes == 50_001
    assert decision.metadata["baseline"]["policy"] == "bandwidth_greedy"


def test_deadline_greedy_prefers_nearest_deadline() -> None:
    observation = _observation(expiration_ms=(50, 100))

    decision = plan_schedule(
        DeadlineGreedyBaseline(),
        observation,
        observation_budget=ObservationBudget(max_candidates=40),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=1_000_000),
    )

    selected = _selected_candidates(observation, decision.metadata["adapter"]["selected_candidate_ids"])
    assert len(selected) == 1
    assert selected[0].expiration_ms == 50
    assert selected[0].deadline_ms == 60
    assert decision.metadata["baseline"]["policy"] == "deadline_greedy"


def test_quality_max_deadline_unaware_prefers_highest_quality_resolution() -> None:
    observation = _observation(resolutions=("720p", "1080p"), expiration_ms=(50, 200))

    decision = plan_schedule(
        QualityMaxDeadlineUnawareBaseline(),
        observation,
        observation_budget=ObservationBudget(max_candidates=80),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=1_000_000),
    )

    selected = _selected_candidates(observation, decision.metadata["adapter"]["selected_candidate_ids"])
    assert len(selected) == 1
    assert selected[0].resolution.height_px == 1080
    assert decision.metadata["baseline"]["policy"] == "quality_max_deadline_unaware"


def test_simple_baseline_set_and_cadence_validation() -> None:
    baselines = simple_baselines()

    assert [baseline.method_id for baseline in baselines] == [
        "fixed-reference-cadence",
        "independent-gaussian",
        "independent-reference",
        "bandwidth-greedy",
        "deadline-greedy",
        "quality-max-deadline-unaware",
    ]
    with pytest.raises(BaselineError, match="cadence_ms"):
        FixedReferenceCadenceBaseline(cadence_ms=0)


def test_robust_mpc_joint_space_selects_utility_adjusted_candidate() -> None:
    observation = _observation(resolutions=("720p", "1080p"), expiration_ms=(50, 200))
    method = RobustMPCJointSpaceBaseline(uncertainty_penalty=0.25, deadline_penalty=0.25, resource_penalty=0.1)

    decision = plan_schedule(
        method,
        observation,
        observation_budget=ObservationBudget(max_candidates=80),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=1_000_000),
    )

    selected = _selected_candidates(observation, decision.metadata["adapter"]["selected_candidate_ids"])
    assert len(selected) == 1
    assert decision.expected_utility is not None
    assert decision.metadata["baseline"]["policy"] == "robust_mpc_joint_space"
    assert decision.metadata["baseline"]["freeze_eligible"] is True


def test_bola_slack_adapted_uses_slack_and_records_freeze_eligibility() -> None:
    observation = _observation(expiration_ms=(50, 200))
    method = BOLASlackAdaptedBaseline(slack_weight=2.0, size_penalty=0.0)

    decision = plan_schedule(
        method,
        observation,
        observation_budget=ObservationBudget(max_candidates=80),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=1_000_000),
    )

    selected = _selected_candidates(observation, decision.metadata["adapter"]["selected_candidate_ids"])
    assert len(selected) == 1
    assert selected[0].expiration_ms == 200
    assert decision.metadata["baseline"]["policy"] == "bola_slack_adapted"
    assert decision.metadata["baseline"]["freeze_eligible"] is True


def test_perfect_information_oracle_selects_best_feasible_candidate_and_is_not_freeze_eligible() -> None:
    observation = _observation(resolutions=("720p", "1080p"), expiration_ms=(50, 200))
    utility_by_candidate = {estimate.candidate_id: estimate for estimate in observation.utility_estimates}
    feasible = [
        candidate
        for candidate in observation.candidates
        if candidate.size_bytes <= 1_000_000
    ]
    expected_best = max(feasible, key=lambda candidate: utility_by_candidate[candidate.candidate_id].expected_utility)

    decision = plan_schedule(
        PerfectInformationOracle(),
        observation,
        observation_budget=ObservationBudget(max_candidates=80),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=1_000_000),
    )

    selected = _selected_candidates(observation, decision.metadata["adapter"]["selected_candidate_ids"])
    assert selected == (expected_best,)
    assert decision.metadata["baseline"]["policy"] == "perfect_information_oracle"
    assert decision.metadata["baseline"]["oracle"] is True
    assert decision.metadata["baseline"]["freeze_eligible"] is False
    with pytest.raises(BaselineError, match="freeze_eligible"):
        PerfectInformationOracle(freeze_eligible=True)


def test_canonical_baseline_and_oracle_sets() -> None:
    canonical = canonical_abr_baselines()
    oracles = perfect_information_oracles()

    assert [method.method_id for method in canonical] == ["robust-mpc-joint-space", "bola-slack-adapted"]
    assert all(method.freeze_eligible for method in canonical)
    assert [method.method_id for method in oracles] == ["perfect-information-oracle"]
    assert all(not oracle.freeze_eligible for oracle in oracles)


def _selected_candidates(observation: SchedulingObservation, selected_candidate_ids) -> tuple:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in observation.candidates}
    return tuple(candidate_by_id[candidate_id] for candidate_id in selected_candidate_ids)


def _observation(
    *,
    size_bytes: int = 100_000,
    decision_time_ms: int = 10,
    resolutions: tuple[str, ...] = ("720p",),
    expiration_ms: tuple[int, ...] = (100,),
    tile_columns: int = 1,
    lifecycle_states: tuple[ReferenceLifecycleState, ...] = (),
) -> SchedulingObservation:
    candidate_set = generate_candidate_objects(
        _workload(size_bytes=size_bytes),
        DecisionEpoch(decision_time_ms=decision_time_ms, frame_id="frame-1"),
        spec=CandidateGenerationSpec(
            resolutions=resolutions,
            fov_degrees=(90,),
            lookahead_ms=(0,),
            expiration_ms=expiration_ms,
            retransmit_priorities=(0,),
            enhancement_layers=(1,),
            tile_rows=1,
            tile_columns=tile_columns,
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
        decision_time_ms=decision_time_ms,
        target_deadline_ms=decision_time_ms + max(expiration_ms),
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
