from __future__ import annotations

import json

import pytest

from ref_abr.domain import FrozenMethodManifest
from ref_abr.mechanism_attribution_harness import (
    MECHANISM_ABLATION_METRIC_NAMES,
    MECHANISM_ABLATION_VARIANTS,
    MechanismAttributionConfig,
    MechanismAttributionHarnessError,
    MechanismAttributionPoint,
    decision_trace_cases,
    evaluate_mechanism_attribution_point,
    export_mechanism_attribution_outputs,
    oracle_gap_cases,
    paired_ablation_table,
    run_mechanism_attribution_harness,
)


def test_mechanism_attribution_config_expands_paired_cases_and_methods() -> None:
    config = MechanismAttributionConfig(
        scenes=("scene-a",),
        traces=("trace-a",),
        viewports=("viewport-a",),
        devices=("desktop",),
        decision_cases=("nominal", "congested"),
        frozen_method_manifest=_frozen_manifest("frozen-refabr"),
        variants=("full", "no-lifecycle", "no-uncertainty", "oracle"),
    )

    assert len(config.attribution_points()) == 2
    assert config.methods == (
        "frozen-refabr",
        "frozen-refabr-no-lifecycle",
        "frozen-refabr-no-uncertainty",
        "perfect-information-oracle",
    )
    assert {row["variant_id"] for row in MECHANISM_ABLATION_VARIANTS} >= {
        "full",
        "no-lifecycle",
        "no-uncertainty",
        "no-component-cost",
        "no-cancellation",
        "no-fov",
        "no-lead-time",
        "oracle",
    }


def test_mechanism_attribution_evaluator_records_oracle_gap_and_decision_trace() -> None:
    point = MechanismAttributionPoint("scene-a", "trace-a", "viewport-a", "desktop", "congested")

    full = evaluate_mechanism_attribution_point(point, variant_id="full", frozen_method_id="frozen-refabr", seed=0)
    no_lifecycle = evaluate_mechanism_attribution_point(point, variant_id="no-lifecycle", frozen_method_id="frozen-refabr", seed=0)
    oracle = evaluate_mechanism_attribution_point(point, variant_id="oracle", frozen_method_id="frozen-refabr", seed=0)

    assert full.method_id == "frozen-refabr"
    assert no_lifecycle.method_id == "frozen-refabr-no-lifecycle"
    assert no_lifecycle.qoe_delta_from_full < 0.0
    assert no_lifecycle.oracle_gap > full.oracle_gap
    assert oracle.oracle_gap == 0.0
    assert no_lifecycle.metadata["decision_trace"]["lifecycle_weight"] == 0.0
    assert {metric.metric_name for metric in no_lifecycle.metric_records(run_id="run-a")} == set(MECHANISM_ABLATION_METRIC_NAMES)


def test_run_mechanism_attribution_harness_builds_tables_and_comparisons() -> None:
    config = MechanismAttributionConfig(
        scenes=("scene-a",),
        traces=("trace-a",),
        viewports=("viewport-a",),
        devices=("desktop",),
        decision_cases=("nominal", "congested"),
        frozen_method_manifest=_frozen_manifest("frozen-refabr"),
        variants=("full", "no-lifecycle", "no-component-cost", "oracle"),
        seeds=(0,),
    )

    result = run_mechanism_attribution_harness(config)
    table = paired_ablation_table(result)
    oracle_rows = oracle_gap_cases(result)
    trace_rows = decision_trace_cases(result)

    assert len(result.outcomes) == 8
    assert result.harness_result.comparison_summary.baseline_method_id == "frozen-refabr"
    assert result.harness_result.comparison_summary.rows
    assert table[0]["variant_id"] == "full"
    assert {row["variant_id"] for row in table} == {"full", "no-lifecycle", "no-component-cost", "oracle"}
    assert len(oracle_rows) == len(result.outcomes)
    assert len(trace_rows) == len(result.outcomes)
    assert any(row["lifecycle_weight"] == 0.0 for row in trace_rows if row["variant_id"] == "no-lifecycle")


def test_export_mechanism_attribution_outputs_writes_required_payloads(tmp_path) -> None:
    result = run_mechanism_attribution_harness(
        MechanismAttributionConfig(
            scenes=("scene-a",),
            traces=("trace-a",),
            viewports=("viewport-a",),
            devices=("desktop",),
            decision_cases=("nominal",),
            frozen_method_manifest=_frozen_manifest("frozen-refabr"),
            variants=("full", "no-fov", "no-lead-time", "oracle"),
            output_root=tmp_path,
        )
    )

    paths = export_mechanism_attribution_outputs(tmp_path, result)

    assert {path.name for path in paths} == {
        "mechanism_attribution_outcomes.jsonl",
        "paired_ablation_table.json",
        "oracle_gap_cases.json",
        "decision_trace_cases.json",
        "mechanism_attribution_summary.json",
    }
    table = json.loads((tmp_path / "paired_ablation_table.json").read_text(encoding="utf-8"))
    trace_rows = json.loads((tmp_path / "decision_trace_cases.json").read_text(encoding="utf-8"))
    assert table[0]["variant_id"] == "full"
    assert any(row["fov_weight"] == 0.0 for row in trace_rows if row["variant_id"] == "no-fov")
    assert (tmp_path / "harness" / "harness_result.json").exists()


def test_mechanism_attribution_config_validation_errors_are_clear() -> None:
    with pytest.raises(MechanismAttributionHarnessError, match="full"):
        MechanismAttributionConfig(
            scenes=("scene-a",),
            traces=("trace-a",),
            viewports=("viewport-a",),
            devices=("desktop",),
            decision_cases=("nominal",),
            variants=("no-lifecycle",),
        )

    with pytest.raises(MechanismAttributionHarnessError, match="final split"):
        MechanismAttributionConfig(
            scenes=("scene-a",),
            traces=("trace-a",),
            viewports=("viewport-a",),
            devices=("desktop",),
            decision_cases=("nominal",),
            split="calibration",
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
    )
