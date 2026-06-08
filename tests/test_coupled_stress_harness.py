from __future__ import annotations

import json

import pytest

from ref_abr.coupled_stress_harness import (
    COUPLED_STRESS_AXES,
    COUPLED_STRESS_BASELINE_METHOD_IDS,
    COUPLED_STRESS_METRIC_NAMES,
    CoupledStressConfig,
    CoupledStressHarnessError,
    CoupledStressPoint,
    coupled_stress_matrix,
    degradation_slopes,
    evaluate_coupled_stress_point,
    export_coupled_stress_outputs,
    recovery_timelines,
    run_coupled_stress_harness,
)
from ref_abr.domain import FrozenMethodManifest


def test_coupled_stress_config_expands_stress_points_and_methods() -> None:
    config = CoupledStressConfig(
        scenes=("scene-a",),
        traces=("trace-a",),
        viewports=("viewport-a",),
        devices=("desktop", "mobile"),
        stress_matrix=_stress_matrix(),
        frozen_method_manifest=_frozen_manifest("frozen-refabr"),
        baseline_methods=("deadline-aware-knapsack-allocator", "deadline-greedy"),
    )

    points = config.stress_points()

    assert config.frozen_method_id == "frozen-refabr"
    assert config.methods == ("frozen-refabr", "deadline-aware-knapsack-allocator", "deadline-greedy")
    assert len(points) == 4
    assert set(COUPLED_STRESS_AXES) == {"bandwidth", "viewport", "server", "client", "deadline"}
    assert set(COUPLED_STRESS_BASELINE_METHOD_IDS) >= {"virtual-queue-deadline-controller", "bola-slack-adapted"}


def test_coupled_stress_evaluator_emits_metrics_and_recovery_timeline() -> None:
    point = CoupledStressPoint(
        scene_id="scene-a",
        trace_id="trace-a",
        viewport_id="viewport-a",
        device_profile_id="desktop",
        stress_id="high",
        stress_levels={"bandwidth": 0.9, "viewport": 0.7, "server": 0.8, "client": 0.6, "deadline": 0.9},
    )

    frozen = evaluate_coupled_stress_point(point, method_id="frozen-refabr", frozen_method_id="frozen-refabr", seed=0)
    greedy = evaluate_coupled_stress_point(point, method_id="deadline-greedy", frozen_method_id="frozen-refabr", seed=0)

    assert frozen.method_role == "frozen_refabr"
    assert greedy.method_role == "baseline"
    assert frozen.stress_deadline_hit_qoe >= greedy.stress_deadline_hit_qoe
    assert frozen.stress_recovery_ms <= greedy.stress_recovery_ms
    assert {metric.metric_name for metric in frozen.metric_records(run_id="run-a")} == set(COUPLED_STRESS_METRIC_NAMES)
    assert len(frozen.metadata["recovery_timeline"]) == 5


def test_run_coupled_stress_harness_builds_matrix_timeline_and_slopes() -> None:
    config = CoupledStressConfig(
        scenes=("scene-a", "scene-b"),
        traces=("trace-a",),
        viewports=("viewport-a",),
        devices=("desktop",),
        stress_matrix=_stress_matrix(),
        frozen_method_manifest=_frozen_manifest("frozen-refabr"),
        baseline_methods=("deadline-greedy",),
        seeds=(0,),
    )

    result = run_coupled_stress_harness(config)
    matrix = coupled_stress_matrix(result)
    timelines = recovery_timelines(result)
    slopes = degradation_slopes(result)

    assert len(result.outcomes) == 8
    assert result.harness_result.comparison_summary.baseline_method_id == "frozen-refabr"
    assert result.harness_result.comparison_summary.rows
    assert {row["stress_id"] for row in matrix} == {"moderate", "high"}
    assert any(row["method_role"] == "frozen_refabr" for row in matrix)
    assert len(timelines) == len(result.outcomes) * 5
    assert all(0.0 <= row["recovered_fraction"] <= 1.0 for row in timelines)
    assert len(slopes) == len(result.outcomes)


def test_export_coupled_stress_outputs_writes_required_payloads(tmp_path) -> None:
    result = run_coupled_stress_harness(
        CoupledStressConfig(
            scenes=("scene-a",),
            traces=("trace-a",),
            viewports=("viewport-a",),
            devices=("desktop",),
            stress_matrix=_stress_matrix(),
            frozen_method_manifest=_frozen_manifest("frozen-refabr"),
            baseline_methods=("deadline-greedy",),
            output_root=tmp_path,
        )
    )

    paths = export_coupled_stress_outputs(tmp_path, result)

    assert {path.name for path in paths} == {
        "coupled_stress_outcomes.jsonl",
        "coupled_stress_matrix.json",
        "recovery_timelines.json",
        "degradation_slopes.json",
        "coupled_stress_summary.json",
    }
    matrix = json.loads((tmp_path / "coupled_stress_matrix.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "coupled_stress_summary.json").read_text(encoding="utf-8"))
    assert matrix
    assert summary["recovery_timelines"]
    assert (tmp_path / "harness" / "harness_result.json").exists()


def test_coupled_stress_config_validation_errors_are_clear() -> None:
    with pytest.raises(CoupledStressHarnessError, match="final split"):
        CoupledStressConfig(
            scenes=("scene-a",),
            traces=("trace-a",),
            viewports=("viewport-a",),
            devices=("desktop",),
            stress_matrix=_stress_matrix(),
            split="calibration",
        )

    with pytest.raises(CoupledStressHarnessError, match="stress axes"):
        CoupledStressConfig(
            scenes=("scene-a",),
            traces=("trace-a",),
            viewports=("viewport-a",),
            devices=("desktop",),
            stress_matrix=({"stress_id": "bad", "stress_levels": {"bandwidth": 0.5}},),
        )

    with pytest.raises(CoupledStressHarnessError, match="duplicated"):
        CoupledStressConfig(
            scenes=("scene-a",),
            traces=("trace-a",),
            viewports=("viewport-a",),
            devices=("desktop",),
            stress_matrix=_stress_matrix(),
            frozen_method_id="deadline-greedy",
            baseline_methods=("deadline-greedy",),
        )


def _stress_matrix() -> tuple[dict[str, object], ...]:
    return (
        {
            "stress_id": "moderate",
            "stress_levels": {"bandwidth": 0.35, "viewport": 0.30, "server": 0.25, "client": 0.20, "deadline": 0.40},
        },
        {
            "stress_id": "high",
            "stress_levels": {"bandwidth": 0.80, "viewport": 0.70, "server": 0.75, "client": 0.65, "deadline": 0.85},
        },
    )


def _frozen_manifest(method_id: str) -> FrozenMethodManifest:
    return FrozenMethodManifest(
        method_id=method_id,
        method_name="Frozen RefABR",
        version="1.0",
        config_id=f"{method_id}-config",
        artifact_uri=f"frozen-method://{method_id}",
        entrypoint="ref_abr.methods:plan_schedule",
        parameters={"frozen_method_id": method_id},
        metadata={"freeze_method": {"primary_metric": "deadline_hit_qoe"}},
    )
