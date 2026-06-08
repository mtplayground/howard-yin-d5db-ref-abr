from __future__ import annotations

import json

import pytest

from ref_abr.domain import MetricRecord
from ref_abr.harness import (
    HarnessConfig,
    HarnessRunResult,
    build_harness_run_specs,
    run_harness,
    run_key,
)
from ref_abr.metrics import ComputeMetricsConfig, export_metric_records


def test_build_harness_run_specs_crosses_methods_workloads_and_seeds() -> None:
    config = HarnessConfig(
        harness_name="candidate-selection",
        methods=("baseline", "candidate"),
        workloads=("scene-a", "scene-b"),
        seeds=(3, 5),
        fixed_variables={"device": "mobile", "network": "lte"},
        tags={"split": "final"},
    )

    specs = build_harness_run_specs(config)

    assert len(specs) == 8
    assert specs[0].method_id == "baseline"
    assert specs[0].workload_id == "scene-a"
    assert specs[0].seed == 3
    assert specs[0].fixed_variables["device"] == "mobile"
    assert specs[0].run_key == "baseline__scene-a__seed-3"


def test_run_harness_uses_executor_and_builds_comparison_summary() -> None:
    config = HarnessConfig(
        harness_name="deadline-hit",
        methods=("baseline", "candidate"),
        workloads=("scene-a",),
        seeds=(1, 2),
        comparison_metric_names=("deadline_qoe",),
        fixed_variables={"network_trace": "trace-a"},
    )

    def executor(spec):
        value = 0.5 + 0.1 * spec.seed
        if spec.method_id == "candidate":
            value += 0.2
        return HarnessRunResult(
            spec=spec,
            status="executed",
            metrics=(
                MetricRecord(
                    metric_name="deadline_qoe",
                    value=value,
                    unit="score",
                    tags={"method": spec.method_id},
                    split="final",
                    metadata={"metric_id": f"metric-{spec.run_key}"},
                ),
            ),
        )

    result = run_harness(config, executor=executor)

    rows = result.comparison_summary.rows
    assert len(rows) == 2
    assert {row["seed"] for row in rows} == {"1", "2"}
    assert all(row["baseline_method_id"] == "baseline" for row in rows)
    assert all(row["method_id"] == "candidate" for row in rows)
    assert all(row["delta"] == pytest.approx(0.2) for row in rows)
    assert result.comparison_summary.missing_pairs == ()


def test_run_harness_plan_only_emits_planned_runs_without_metrics(tmp_path) -> None:
    config = HarnessConfig(
        harness_name="plan",
        methods=("baseline",),
        workloads=("scene-a",),
        seeds=(0,),
        run_mode="plan_only",
        output_root=tmp_path,
    )

    result = run_harness(config)

    assert result.run_results[0].status == "planned"
    assert result.run_results[0].metrics == ()
    assert result.comparison_summary.rows == ()
    assert (tmp_path / "harness_result.json").exists()


def test_run_harness_loads_metric_artifacts_and_compares(tmp_path) -> None:
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    export_metric_records(
        baseline_root,
        (
            MetricRecord(
                metric_name="deadline_qoe",
                value=0.6,
                unit="score",
                tags={"method": "baseline"},
                split="final",
                metadata={"metric_id": "baseline-metric"},
            ),
        ),
        config=ComputeMetricsConfig(run_id="baseline-run", method_id="baseline", split="final", metric_set=("quality",)),
    )
    export_metric_records(
        candidate_root,
        (
            MetricRecord(
                metric_name="deadline_qoe",
                value=0.85,
                unit="score",
                tags={"method": "candidate"},
                split="final",
                metadata={"metric_id": "candidate-metric"},
            ),
        ),
        config=ComputeMetricsConfig(run_id="candidate-run", method_id="candidate", split="final", metric_set=("quality",)),
    )

    config = HarnessConfig(
        harness_name="loaded",
        methods=("baseline", "candidate"),
        workloads=("scene-a",),
        seeds=(9,),
        comparison_metric_names=("deadline_qoe",),
        metric_artifact_roots={
            run_key("baseline", "scene-a", 9): baseline_root,
            run_key("candidate", "scene-a", 9): candidate_root,
        },
    )

    result = run_harness(config)

    assert [run.status for run in result.run_results] == ["loaded", "loaded"]
    assert result.comparison_summary.rows[0]["delta"] == 0.25
    payload = json.loads((candidate_root / "manifest.json").read_text(encoding="utf-8"))
    assert payload["record_type"] == "raw_artifact_manifest"
