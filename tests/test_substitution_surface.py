from __future__ import annotations

import json

import pytest

from ref_abr.substitution_surface import (
    SUBSTITUTION_METHOD_IDS,
    SubstitutionSurfaceConfig,
    SubstitutionSurfacePoint,
    evaluate_substitution_point,
    run_gaussian_reference_substitution_surface,
)


def test_surface_config_expands_full_cross_product() -> None:
    config = SubstitutionSurfaceConfig(
        layers=(0, 1),
        ref_resolutions=("720p", "1080p"),
        fov_degrees=(70.0,),
        view_mismatches=(0.0, 0.4),
        budget_bytes=(900_000,),
        slack_ms=(20.0, 60.0),
        seeds=(3,),
    )

    points = config.surface_points()

    assert len(points) == 16
    assert points[0].point_id.startswith("substitution-point-")
    assert config.surface_id.startswith("gaussian-reference-substitution-surface-")


def test_evaluate_substitution_point_selects_reference_when_budget_and_match_are_good() -> None:
    point = SubstitutionSurfacePoint(
        layer=2,
        ref_resolution="1080p",
        fov_deg=100.0,
        view_mismatch=0.0,
        budget_bytes=4_000_000,
        slack_ms=120.0,
    )

    svq = evaluate_substitution_point(point, method_id="svq-gaussian-only-abr", seed=1)
    allocator = evaluate_substitution_point(point, method_id="deadline-aware-knapsack-allocator", seed=1)
    oracle = evaluate_substitution_point(point, method_id="perfect-information-oracle", seed=1)

    assert svq.selected_action == "gaussian"
    assert svq.substitution_gain == pytest.approx(0.0)
    assert allocator.selected_action in {"reference", "mixed"}
    assert allocator.substitution_gain > 0.0
    assert oracle.utility_score >= allocator.utility_score


def test_evaluate_substitution_point_penalizes_mismatch_and_tight_budget() -> None:
    point = SubstitutionSurfacePoint(
        layer=0,
        ref_resolution="1080p",
        fov_deg=70.0,
        view_mismatch=0.8,
        budget_bytes=400_000,
        slack_ms=5.0,
    )

    allocator = evaluate_substitution_point(point, method_id="deadline-aware-knapsack-allocator", seed=2)

    assert allocator.selected_action == "gaussian"
    assert allocator.budget_pressure > 1.0
    assert allocator.deadline_slack_ratio < 1.0


def test_run_gaussian_reference_substitution_surface_emits_harness_and_outputs(tmp_path) -> None:
    config = SubstitutionSurfaceConfig(
        layers=(1,),
        ref_resolutions=("720p",),
        fov_degrees=(90.0,),
        view_mismatches=(0.0,),
        budget_bytes=(2_000_000,),
        slack_ms=(80.0,),
        seeds=(4,),
        output_root=tmp_path,
    )

    result = run_gaussian_reference_substitution_surface(config)

    assert len(result.surface_points) == 1
    assert len(result.outcomes) == len(SUBSTITUTION_METHOD_IDS)
    assert len(result.harness_result.run_results) == len(SUBSTITUTION_METHOD_IDS)
    assert result.harness_result.comparison_summary.rows
    assert (tmp_path / "substitution_surface.jsonl").exists()
    assert (tmp_path / "substitution_surface_summary.json").exists()
    row = json.loads((tmp_path / "substitution_surface.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["outcome_id"].startswith("substitution-outcome-")
    assert row["point"]["point_id"] == result.surface_points[0].point_id
