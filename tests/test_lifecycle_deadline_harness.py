from __future__ import annotations

import json

import pytest

from ref_abr.lifecycle_deadline_harness import (
    LIFECYCLE_DEADLINE_METHOD_IDS,
    LifecycleDeadlineConfig,
    LifecycleDeadlinePoint,
    evaluate_lifecycle_deadline_point,
    run_reference_lifecycle_deadline_harness,
)


def test_lifecycle_deadline_config_expands_full_cross_product() -> None:
    config = LifecycleDeadlineConfig(
        latency_ms=(20.0, 40.0),
        queue_ms=(0.0,),
        restore_ms=(10.0,),
        viewport_errors=(0.0, 0.5),
        deadline_slack_ms=(50.0, 100.0),
        seeds=(1,),
    )

    points = config.surface_points()

    assert len(points) == 8
    assert points[0].point_id.startswith("lifecycle-deadline-point-")
    assert config.matrix_id.startswith("reference-lifecycle-deadline-matrix-")


def test_evaluate_lifecycle_deadline_point_rewards_oracle_and_blocks_no_lifecycle() -> None:
    point = LifecycleDeadlinePoint(
        latency_ms=10.0,
        queue_ms=5.0,
        restore_ms=8.0,
        viewport_error=0.05,
        deadline_slack_ms=80.0,
    )

    deadline_greedy = evaluate_lifecycle_deadline_point(point, method_id="deadline-greedy", seed=2)
    no_lifecycle = evaluate_lifecycle_deadline_point(point, method_id="no-lifecycle", seed=2)
    oracle = evaluate_lifecycle_deadline_point(point, method_id="perfect-information-oracle", seed=2)

    assert deadline_greedy.useful
    assert no_lifecycle.selected_lifecycle_decision == "skip"
    assert no_lifecycle.lifecycle_deadline_success_rate == pytest.approx(0.0)
    assert oracle.lifecycle_deadline_risk <= deadline_greedy.lifecycle_deadline_risk
    assert oracle.lifecycle_deadline_success_rate >= deadline_greedy.lifecycle_deadline_success_rate


def test_evaluate_lifecycle_deadline_point_marks_tight_high_error_cases_risky() -> None:
    point = LifecycleDeadlinePoint(
        latency_ms=80.0,
        queue_ms=40.0,
        restore_ms=30.0,
        viewport_error=0.8,
        deadline_slack_ms=50.0,
    )

    deadline_greedy = evaluate_lifecycle_deadline_point(point, method_id="deadline-greedy", seed=3)
    quality_max = evaluate_lifecycle_deadline_point(point, method_id="quality-max-deadline-unaware", seed=3)

    assert deadline_greedy.late
    assert deadline_greedy.lifecycle_deadline_risk > 0.5
    assert quality_max.lifecycle_deadline_risk >= deadline_greedy.lifecycle_deadline_risk
    assert quality_max.selected_lifecycle_decision in {"late", "expire"}


def test_run_reference_lifecycle_deadline_harness_emits_matrix_curves_and_summary(tmp_path) -> None:
    config = LifecycleDeadlineConfig(
        latency_ms=(20.0,),
        queue_ms=(5.0,),
        restore_ms=(10.0,),
        viewport_errors=(0.1,),
        deadline_slack_ms=(70.0,),
        seeds=(4,),
        output_root=tmp_path,
    )

    result = run_reference_lifecycle_deadline_harness(config)

    assert len(result.surface_points) == 1
    assert len(result.outcomes) == len(LIFECYCLE_DEADLINE_METHOD_IDS)
    assert len(result.harness_result.run_results) == len(LIFECYCLE_DEADLINE_METHOD_IDS)
    assert result.harness_result.comparison_summary.rows
    assert (tmp_path / "harness" / "harness_result.json").exists()
    assert (tmp_path / "lifecycle_matrix.jsonl").exists()
    assert (tmp_path / "lifecycle_risk_curves.json").exists()
    assert (tmp_path / "lifecycle_deadline_summary.json").exists()

    row = json.loads((tmp_path / "lifecycle_matrix.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["outcome_id"].startswith("lifecycle-deadline-outcome-")
    assert row["point"]["point_id"] == result.surface_points[0].point_id

    curves = json.loads((tmp_path / "lifecycle_risk_curves.json").read_text(encoding="utf-8"))
    assert {curve["method_id"] for curve in curves["curves"]} == set(LIFECYCLE_DEADLINE_METHOD_IDS)
