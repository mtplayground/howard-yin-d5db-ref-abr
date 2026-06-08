from __future__ import annotations

import json

import pytest

from ref_abr.candidate_method_selection import (
    CANDIDATE_METHOD_SELECTION_METHOD_IDS,
    CANDIDATE_METHOD_SELECTION_METRIC_NAMES,
    CandidateMethodSelectionConfig,
    CandidateMethodSelectionError,
    CandidateMethodSelectionPoint,
    evaluate_candidate_method_selection_point,
    export_candidate_method_selection_outputs,
    run_candidate_method_selection_harness,
)


def test_candidate_method_selection_config_expands_reduced_workload_points() -> None:
    config = CandidateMethodSelectionConfig(
        scene_complexities=(0.25, 0.75),
        budget_bytes=(450_000,),
        viewport_risks=(0.1,),
        queue_debt_ms=(0.0,),
        deadline_slack_ms=(80.0, 140.0),
        methods=(
            "robust-deadline-aware-mpc",
            "deadline-aware-knapsack-allocator",
            "virtual-queue-deadline-controller",
            "learned-diagnostic-selector",
        ),
    )

    points = config.selection_points()

    assert len(points) == 4
    assert config.matrix_id.startswith("candidate-method-selection-matrix-")
    assert {
        "robust-deadline-aware-mpc",
        "deadline-aware-knapsack-allocator",
        "virtual-queue-deadline-controller",
        "robust-mpc-joint-space",
        "bola-slack-adapted",
        "bandwidth-greedy",
        "deadline-greedy",
        "quality-max-deadline-unaware",
        "diagnostic-layered-3dgs",
        "diagnostic-viewport-tile",
        "learned-diagnostic-selector",
    }.issubset(set(CANDIDATE_METHOD_SELECTION_METHOD_IDS))


def test_candidate_method_selection_evaluator_records_runtime_and_interpretability_traces() -> None:
    point = CandidateMethodSelectionPoint(
        scene_complexity=0.65,
        budget_bytes=500_000,
        viewport_risk=0.45,
        queue_debt_ms=25.0,
        deadline_slack_ms=90.0,
    )

    virtual_queue = evaluate_candidate_method_selection_point(
        point,
        method_id="virtual-queue-deadline-controller",
        seed=3,
    )
    mpc = evaluate_candidate_method_selection_point(point, method_id="robust-deadline-aware-mpc", seed=3)
    greedy = evaluate_candidate_method_selection_point(point, method_id="bandwidth-greedy", seed=3)

    assert {record.metric_name for record in virtual_queue.metric_records(run_id="run-a")} == set(CANDIDATE_METHOD_SELECTION_METRIC_NAMES)
    assert "runtime_trace" in virtual_queue.metadata
    assert "interpretability_trace" in virtual_queue.metadata
    assert virtual_queue.metadata["interpretability_trace"]["uses_deadline_signal"] is True
    assert virtual_queue.interpretability_score > mpc.interpretability_score
    assert mpc.runtime_ms > greedy.runtime_ms


def test_run_candidate_method_selection_harness_emits_comparison_rows() -> None:
    config = CandidateMethodSelectionConfig(
        scene_complexities=(0.3,),
        budget_bytes=(500_000,),
        viewport_risks=(0.2,),
        queue_debt_ms=(10.0,),
        deadline_slack_ms=(100.0,),
        seeds=(0,),
        methods=(
            "robust-deadline-aware-mpc",
            "deadline-aware-knapsack-allocator",
            "virtual-queue-deadline-controller",
        ),
        baseline_method_id="robust-deadline-aware-mpc",
    )

    result = run_candidate_method_selection_harness(config)

    assert len(result.selection_points) == 1
    assert len(result.outcomes) == 3
    assert result.harness_result.comparison_summary.baseline_method_id == "robust-deadline-aware-mpc"
    assert result.harness_result.comparison_summary.rows
    assert result.as_payload()["runtime_traces"]["robust-deadline-aware-mpc"]["count"] == 1


def test_candidate_method_selection_export_writes_jsonl_and_summary(tmp_path) -> None:
    config = CandidateMethodSelectionConfig(
        scene_complexities=(0.3,),
        budget_bytes=(500_000,),
        viewport_risks=(0.2,),
        queue_debt_ms=(10.0,),
        deadline_slack_ms=(100.0,),
        methods=("robust-deadline-aware-mpc", "deadline-aware-knapsack-allocator"),
        baseline_method_id="robust-deadline-aware-mpc",
    )
    result = run_candidate_method_selection_harness(config)

    outcomes_path, summary_path = export_candidate_method_selection_outputs(tmp_path, result)

    outcome_lines = outcomes_path.read_text(encoding="utf-8").splitlines()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(outcome_lines) == 2
    assert json.loads(outcome_lines[0])["method_id"] in config.methods
    assert summary["matrix_id"] == result.matrix_id
    assert summary["interpretability_traces"]


def test_candidate_method_selection_validation_errors_are_clear() -> None:
    with pytest.raises(CandidateMethodSelectionError, match="scene_complexity"):
        CandidateMethodSelectionPoint(
            scene_complexity=1.2,
            budget_bytes=500_000,
            viewport_risk=0.2,
            queue_debt_ms=0.0,
            deadline_slack_ms=100.0,
        )

    point = CandidateMethodSelectionPoint(
        scene_complexity=0.2,
        budget_bytes=500_000,
        viewport_risk=0.2,
        queue_debt_ms=0.0,
        deadline_slack_ms=100.0,
    )
    with pytest.raises(CandidateMethodSelectionError, match="Unknown candidate selection method_id"):
        evaluate_candidate_method_selection_point(point, method_id="unknown-method")

    with pytest.raises(CandidateMethodSelectionError, match="baseline_method_id"):
        CandidateMethodSelectionConfig(
            scene_complexities=(0.2,),
            budget_bytes=(500_000,),
            viewport_risks=(0.2,),
            queue_debt_ms=(0.0,),
            deadline_slack_ms=(100.0,),
            methods=("deadline-aware-knapsack-allocator",),
            baseline_method_id="robust-deadline-aware-mpc",
        )
