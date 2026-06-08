from __future__ import annotations

import json

import pytest

from ref_abr.domain import FrozenMethodManifest
from ref_abr.full_system_qoe_harness import (
    FULL_SYSTEM_BASELINE_METHOD_IDS,
    FULL_SYSTEM_QOE_METRIC_NAMES,
    FullSystemQoeConfig,
    FullSystemQoeHarnessError,
    FullSystemQoePoint,
    deadline_hit_qoe_cdf,
    evaluate_full_system_qoe_point,
    export_full_system_qoe_outputs,
    main_qoe_table,
    quality_deadline_pareto,
    run_full_system_qoe_harness,
)


def test_full_system_qoe_config_uses_frozen_manifest_and_final_split() -> None:
    manifest = _frozen_manifest("frozen-refabr")
    config = FullSystemQoeConfig(
        scenes=("scene-a",),
        traces=("trace-a",),
        viewports=("viewport-a",),
        devices=("desktop", "mobile"),
        frozen_method_manifest=manifest,
        baseline_methods=("deadline-greedy", "quality-max-deadline-unaware"),
    )

    points = config.workload_points()

    assert config.frozen_method_id == "frozen-refabr"
    assert config.methods == ("frozen-refabr", "deadline-greedy", "quality-max-deadline-unaware")
    assert config.split == "final"
    assert len(points) == 2
    assert set(FULL_SYSTEM_BASELINE_METHOD_IDS) >= {"deadline-greedy", "quality-max-deadline-unaware", "virtual-queue-deadline-controller"}


def test_full_system_qoe_evaluator_emits_all_metrics_and_refabr_role() -> None:
    point = FullSystemQoePoint(scene_id="scene-a", trace_id="trace-a", viewport_id="viewport-a", device_profile_id="desktop")

    frozen = evaluate_full_system_qoe_point(point, method_id="frozen-refabr", frozen_method_id="frozen-refabr", seed=0)
    greedy = evaluate_full_system_qoe_point(point, method_id="deadline-greedy", frozen_method_id="frozen-refabr", seed=0)

    assert frozen.method_role == "frozen_refabr"
    assert greedy.method_role == "baseline"
    assert frozen.deadline_hit_qoe >= greedy.deadline_hit_qoe
    assert {metric.metric_name for metric in frozen.metric_records(run_id="run-a")} == set(FULL_SYSTEM_QOE_METRIC_NAMES)


def test_run_full_system_qoe_harness_builds_paper_inputs_and_comparisons() -> None:
    config = FullSystemQoeConfig(
        scenes=("scene-a", "scene-b"),
        traces=("trace-a",),
        viewports=("viewport-a",),
        devices=("desktop",),
        frozen_method_manifest=_frozen_manifest("frozen-refabr"),
        baseline_methods=("deadline-greedy", "quality-max-deadline-unaware"),
        seeds=(0,),
    )

    result = run_full_system_qoe_harness(config)
    table = main_qoe_table(result)
    pareto = quality_deadline_pareto(result)
    cdf = deadline_hit_qoe_cdf(result)

    assert len(result.outcomes) == 6
    assert result.harness_result.comparison_summary.baseline_method_id == "frozen-refabr"
    assert result.harness_result.comparison_summary.rows
    assert table[0]["method_role"] == "frozen_refabr"
    assert {row["method_id"] for row in pareto} == {"frozen-refabr", "deadline-greedy", "quality-max-deadline-unaware"}
    assert all("pareto_frontier" in row for row in pareto)
    assert len(cdf) == len(result.outcomes)
    assert all(0.0 < row["cumulative_probability"] <= 1.0 for row in cdf)


def test_export_full_system_qoe_outputs_writes_required_payloads(tmp_path) -> None:
    result = run_full_system_qoe_harness(
        FullSystemQoeConfig(
            scenes=("scene-a",),
            traces=("trace-a",),
            viewports=("viewport-a",),
            devices=("desktop",),
            frozen_method_manifest=_frozen_manifest("frozen-refabr"),
            baseline_methods=("deadline-greedy",),
            output_root=tmp_path,
        )
    )

    paths = export_full_system_qoe_outputs(tmp_path, result)

    assert {path.name for path in paths} == {
        "full_system_qoe_outcomes.jsonl",
        "main_qoe_table.json",
        "quality_deadline_pareto.json",
        "deadline_hit_qoe_cdf.json",
        "full_system_qoe_summary.json",
    }
    main_table = json.loads((tmp_path / "main_qoe_table.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "full_system_qoe_summary.json").read_text(encoding="utf-8"))
    assert main_table[0]["method_id"] == "frozen-refabr"
    assert summary["main_qoe_table"]
    assert (tmp_path / "harness" / "harness_result.json").exists()


def test_full_system_qoe_config_rejects_non_final_split_and_duplicate_frozen_method() -> None:
    with pytest.raises(FullSystemQoeHarnessError, match="final split"):
        FullSystemQoeConfig(
            scenes=("scene-a",),
            traces=("trace-a",),
            viewports=("viewport-a",),
            devices=("desktop",),
            split="calibration",
        )

    with pytest.raises(FullSystemQoeHarnessError, match="duplicated"):
        FullSystemQoeConfig(
            scenes=("scene-a",),
            traces=("trace-a",),
            viewports=("viewport-a",),
            devices=("desktop",),
            frozen_method_id="deadline-greedy",
            baseline_methods=("deadline-greedy",),
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
        metadata={"freeze_method": {"primary_metric": "method_selection_quality"}},
    )
