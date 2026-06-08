from __future__ import annotations

import json

import pytest

from ref_abr.calibration import (
    CALIBRATION_RECORD_TYPE,
    CalibrationError,
    UtilityLifecycleCalibration,
    build_utility_lifecycle_calibration,
    export_utility_lifecycle_calibration,
    load_calibrated_substrate_provider,
    load_calibrated_utility_model_weights,
    load_utility_lifecycle_calibration,
)
from ref_abr.candidates import CandidateGenerationSpec, DecisionEpoch, generate_candidate_objects
from ref_abr.domain import MediaType
from ref_abr.lifecycle_deadline_harness import LifecycleDeadlineConfig, run_reference_lifecycle_deadline_harness
from ref_abr.schema import DOMAIN_SCHEMA_VERSION, RECORD_TYPE_FIELD, SCHEMA_VERSION_FIELD
from ref_abr.substitution_surface import SubstitutionSurfaceConfig, run_gaussian_reference_substitution_surface
from ref_abr.utility import ResourceBudget, estimate_candidate_utility
from ref_abr.workloads import assemble_workload_manifest


def test_build_calibration_persists_schema_stamped_manifest(tmp_path) -> None:
    calibration = _calibration()

    path = export_utility_lifecycle_calibration(tmp_path, calibration)
    loaded = load_utility_lifecycle_calibration(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "utility_lifecycle_calibration.json"
    assert payload[SCHEMA_VERSION_FIELD] == DOMAIN_SCHEMA_VERSION
    assert payload[RECORD_TYPE_FIELD] == CALIBRATION_RECORD_TYPE
    assert loaded.as_payload() == calibration.as_payload()
    assert loaded.calibration_id.startswith("utility-lifecycle-calibration-")
    assert loaded.source_ids == calibration.source_ids
    assert loaded.parameter_map()["utility.visible_qoe_weight"].value == pytest.approx(
        loaded.utility_model_weights.visible_qoe_weight
    )


def test_loaded_calibration_feeds_candidate_utility_and_substrate_provider(tmp_path) -> None:
    calibration = _calibration()
    export_utility_lifecycle_calibration(tmp_path / "calibration.json", calibration)

    weights = load_calibrated_utility_model_weights(tmp_path / "calibration.json")
    provider = load_calibrated_substrate_provider(tmp_path / "calibration.json", provider_id="calibrated-test")
    candidate = _candidate(provider=provider)
    estimate = estimate_candidate_utility(
        candidate,
        substrate_provider=provider,
        budgets=ResourceBudget(available_time_ms=120, available_bytes=1_000_000, available_memory_mb=1024),
        model_weights=weights,
    )

    assert provider.provider_id == "calibrated-test"
    assert provider.metadata["calibration_id"] == calibration.calibration_id
    assert weights.lifecycle_risk_weight == pytest.approx(calibration.utility_model_weights.lifecycle_risk_weight)
    assert estimate.metadata["model"]["weights"] == weights.as_payload()
    assert estimate.metadata["inputs"]["substrate_provider_id"] == "calibrated-test"


def test_calibration_id_is_deterministic_for_same_inputs() -> None:
    first = _calibration()
    second = _calibration()

    assert first.calibration_id == second.calibration_id
    assert first.as_payload() == second.as_payload()


def test_invalid_calibration_payload_raises_clear_errors(tmp_path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(
        json.dumps(
            {
                SCHEMA_VERSION_FIELD: DOMAIN_SCHEMA_VERSION,
                RECORD_TYPE_FIELD: "wrong",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CalibrationError, match="record_type"):
        load_utility_lifecycle_calibration(bad_path)

    with pytest.raises(CalibrationError, match="utility_model_weights"):
        UtilityLifecycleCalibration.from_parts(
            split="calibration",
            utility_model_weights="bad",  # type: ignore[arg-type]
            substrate_coefficients=_calibration().substrate_coefficients,
            lifecycle_coefficients={"risk_intercept": 0.1},
            uncertainty=_calibration().uncertainty,
            parameters=(),
            source_ids=("source",),
        )


def _calibration() -> UtilityLifecycleCalibration:
    substitution = run_gaussian_reference_substitution_surface(
        SubstitutionSurfaceConfig(
            layers=(0, 2),
            ref_resolutions=("720p",),
            fov_degrees=(90.0,),
            view_mismatches=(0.0, 0.4),
            budget_bytes=(1_000_000,),
            slack_ms=(40.0, 100.0),
            seeds=(1,),
        )
    )
    lifecycle = run_reference_lifecycle_deadline_harness(
        LifecycleDeadlineConfig(
            latency_ms=(20.0, 60.0),
            queue_ms=(0.0, 20.0),
            restore_ms=(10.0,),
            viewport_errors=(0.1, 0.6),
            deadline_slack_ms=(50.0, 100.0),
            seeds=(1,),
        )
    )
    return build_utility_lifecycle_calibration(
        substitution_result=substitution,
        lifecycle_result=lifecycle,
        split="calibration",
        metadata={"test": "calibration"},
    )


def _candidate(*, provider):
    spec = CandidateGenerationSpec(
        resolutions=("720p",),
        fov_degrees=(90.0,),
        lookahead_ms=(0,),
        expiration_ms=(100,),
        retransmit_priorities=(0,),
        enhancement_layers=(1,),
        include_tiles=False,
        include_reference_actions=False,
        max_candidates=1,
    )
    return generate_candidate_objects(
        _workload(),
        DecisionEpoch(0),
        spec=spec,
        substrate_provider=provider,
    ).candidates[0]


def _workload():
    return assemble_workload_manifest(
        {
            "dataset": "calibration-test",
            "sequences": [
                {
                    "scene": "scene",
                    "name": "seq",
                    "assets": [
                        {
                            "object_id": "splat-a",
                            "path": "splat.ply",
                            "size_bytes": 100_000,
                            "media_type": MediaType.GAUSSIAN_SPLAT.value,
                        }
                    ],
                }
            ],
        },
        split="calibration",
        config_id="calibration-test-config",
        seed=9,
    )
