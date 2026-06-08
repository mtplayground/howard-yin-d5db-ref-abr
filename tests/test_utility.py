from __future__ import annotations

import pytest

from ref_abr.candidates import CandidateGenerationSpec, DecisionEpoch, generate_candidate_objects
from ref_abr.devices import DeviceBudgets, DeviceProfile
from ref_abr.domain import MediaType
from ref_abr.substrate import ParametricSubstrateValueProvider
from ref_abr.utility import (
    ResourceBudget,
    UtilityError,
    UtilityModelWeights,
    ViewportRisk,
    coerce_resource_budget,
    coerce_utility_model_weights,
    estimate_candidate_set_utility,
    estimate_candidate_utility,
)
from ref_abr.workloads import assemble_workload_manifest


def test_candidate_utility_uses_embedded_substrate_and_records_breakdowns() -> None:
    candidate = _candidate_with_substrate(expiration_ms=120, size_bytes=100_000)
    budgets = ResourceBudget(available_time_ms=100, available_bytes=200_000, available_memory_mb=512)

    estimate = estimate_candidate_utility(candidate, viewport_risk=ViewportRisk(0.1), budgets=budgets)

    assert estimate.estimate_id.startswith("utility-")
    assert estimate.candidate_id == candidate.candidate_id
    assert 0 <= estimate.visible_qoe_gain <= 1
    assert 0 <= estimate.lifecycle_risk <= 1
    assert 0 <= estimate.deadline_miss_probability <= 1
    assert estimate.deadline_miss_probability < 0.01
    assert estimate.resource_price.total > 0
    assert estimate.resource_debt.time_debt_ms == 0
    assert estimate.uncertainty.utility_stddev > 0
    assert estimate.metadata["model"]["kind"] == "utility_deadline"
    assert estimate.metadata["inputs"]["substrate_provider_id"] == "parametric-substrate-default"


def test_deadline_pressure_and_debt_increase_under_tight_budgets() -> None:
    candidate = _candidate_with_substrate(expiration_ms=15, size_bytes=500_000)
    loose = estimate_candidate_utility(
        candidate,
        budgets=ResourceBudget(available_time_ms=200, available_bytes=1_000_000, available_memory_mb=512),
    )
    tight = estimate_candidate_utility(
        candidate,
        budgets=ResourceBudget(
            available_time_ms=10,
            available_bytes=100_000,
            available_memory_mb=1,
            queue_debt_ms=3,
            transfer_debt_bytes=50_000,
        ),
    )

    assert tight.deadline_miss_probability > loose.deadline_miss_probability
    assert tight.resource_debt.time_debt_ms > 0
    assert tight.resource_debt.transfer_debt_bytes == 400_000
    assert tight.resource_debt.carried_queue_debt_ms == 3
    assert tight.expected_utility < loose.expected_utility


def test_viewport_risk_reduces_qoe_and_increases_lifecycle_risk() -> None:
    candidate = _candidate_with_substrate(expiration_ms=100)
    budgets = ResourceBudget(available_time_ms=100, available_bytes=1_000_000, available_memory_mb=1024)
    stable = estimate_candidate_utility(candidate, viewport_risk=ViewportRisk(), budgets=budgets)
    risky = estimate_candidate_utility(
        candidate,
        viewport_risk={"mismatch_probability": 0.8, "occlusion_probability": 0.4, "motion_instability": 0.6},
        budgets=budgets,
    )

    assert risky.visible_qoe_gain < stable.visible_qoe_gain
    assert risky.lifecycle_risk > stable.lifecycle_risk
    assert risky.uncertainty.confidence < stable.uncertainty.confidence


def test_candidate_set_utility_is_deterministic_and_can_use_provider_fallback() -> None:
    workload = _workload()
    spec = CandidateGenerationSpec(
        resolutions=("720p",),
        fov_degrees=(90,),
        lookahead_ms=(0,),
        expiration_ms=(80,),
        retransmit_priorities=(0,),
        enhancement_layers=(1,),
        include_tiles=False,
        include_reference_actions=False,
    )
    candidate_set = generate_candidate_objects(workload, DecisionEpoch(0), spec=spec)
    provider = ParametricSubstrateValueProvider()
    budgets = {"available_time_ms": 100, "available_bytes": 1_000_000, "available_memory_mb": 1024}

    first = estimate_candidate_set_utility(candidate_set, substrate_provider=provider, budgets=budgets)
    second = estimate_candidate_set_utility(candidate_set, substrate_provider=provider, budgets=budgets)

    assert first.estimate_set_id == second.estimate_set_id
    assert first.as_payload() == second.as_payload()
    assert first.candidate_set_id == candidate_set.candidate_set_id
    assert len(first.estimates) == len(candidate_set.candidates)


def test_resource_budget_can_be_derived_from_device_profile_and_budgets() -> None:
    profile = DeviceProfile(
        profile_id="edge-a",
        device_class="edge",
        budgets=DeviceBudgets(5, 10, 3, 8, 2048, 60),
        metadata={"available_bytes": 250_000},
    )

    from_profile = coerce_resource_budget(profile)
    from_budgets = ResourceBudget.from_device_budgets(profile.budgets, available_bytes=125_000)

    assert from_profile.available_time_ms == 26
    assert from_profile.available_bytes == 250_000
    assert from_profile.metadata["device_profile_id"] == "edge-a"
    assert from_budgets.available_time_ms == 26
    assert from_budgets.available_bytes == 125_000


def test_missing_substrate_or_invalid_inputs_raise_clear_errors() -> None:
    candidate = _candidate_without_substrate()

    with pytest.raises(UtilityError, match="substrate"):
        estimate_candidate_utility(candidate)

    with pytest.raises(UtilityError, match="mismatch_probability"):
        ViewportRisk(mismatch_probability=1.5)

    with pytest.raises(UtilityError, match="available_time_ms"):
        ResourceBudget(available_time_ms=0, available_bytes=1, available_memory_mb=1)

    with pytest.raises(UtilityError, match="unknown field"):
        coerce_utility_model_weights({"bad": 1})


def _candidate_with_substrate(*, expiration_ms: int, size_bytes: int = 100_000):
    spec = CandidateGenerationSpec(
        resolutions=("720p",),
        fov_degrees=(90,),
        lookahead_ms=(0,),
        expiration_ms=(expiration_ms,),
        retransmit_priorities=(0,),
        enhancement_layers=(1,),
        include_tiles=False,
        include_reference_actions=False,
        max_candidates=1,
    )
    return generate_candidate_objects(
        _workload(size_bytes=size_bytes),
        DecisionEpoch(0),
        spec=spec,
        substrate_provider=ParametricSubstrateValueProvider(),
    ).candidates[0]


def _candidate_without_substrate():
    spec = CandidateGenerationSpec(
        resolutions=("720p",),
        fov_degrees=(90,),
        lookahead_ms=(0,),
        expiration_ms=(100,),
        retransmit_priorities=(0,),
        include_gaussian_enhancement=False,
        include_tiles=False,
        include_reference_actions=False,
        max_candidates=1,
    )
    return generate_candidate_objects(_workload(), DecisionEpoch(0), spec=spec).candidates[0]


def _workload(*, size_bytes: int = 100_000):
    return assemble_workload_manifest(
        {
            "dataset": "utility-test",
            "sequences": [
                {
                    "scene": "scene",
                    "name": "seq",
                    "assets": [
                        {
                            "object_id": "splat-a",
                            "path": "splat.ply",
                            "size_bytes": size_bytes,
                            "media_type": MediaType.GAUSSIAN_SPLAT.value,
                        }
                    ],
                }
            ],
        },
        split="calibration",
        config_id="utility-test-config",
        seed=7,
    )
