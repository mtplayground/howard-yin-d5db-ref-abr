from __future__ import annotations

import json

from click.testing import CliRunner

from ref_abr.cli import main
from ref_abr.domain import FrozenMethodManifest, MetricRecord
from ref_abr.freeze import (
    FreezeMethodConfig,
    apply_freeze_decision,
    export_frozen_method_manifest,
    freeze_method,
)
from ref_abr.schema import DOMAIN_SCHEMA_VERSION, materialize_record, stamp_record


def test_apply_freeze_decision_excludes_diagnostics_and_selects_primary_metric_winner() -> None:
    metrics = (
        _metric("robust-deadline-aware-mpc", "method_selection_quality", 0.82),
        _metric("deadline-aware-knapsack-allocator", "method_selection_quality", 0.78),
        _metric("diagnostic-layered-3dgs", "method_selection_quality", 0.99),
        _metric("perfect-information-oracle", "method_selection_quality", 1.0),
    )
    config = FreezeMethodConfig(primary_metric="method_selection_quality")

    decision = apply_freeze_decision(metrics, config=config)

    assert decision.selected_method_id == "robust-deadline-aware-mpc"
    assert decision.ranked_methods[0]["metrics"]["method_selection_quality"] == 0.82
    assert {row["method_id"] for row in decision.excluded_methods} == {
        "diagnostic-layered-3dgs",
        "perfect-information-oracle",
    }


def test_freeze_method_manifest_records_config_splits_metric_and_provenance() -> None:
    result = freeze_method(
        (
            _metric("robust-deadline-aware-mpc", "method_selection_quality", 0.8),
            _metric("deadline-aware-knapsack-allocator", "method_selection_quality", 0.7),
        ),
        config=FreezeMethodConfig(
            method_name="RefABR-MPC",
            version="2026.06",
            calibration_splits=("calibration", "train"),
            excluded_splits=("final",),
            method_entrypoints={"robust-deadline-aware-mpc": "ref_abr.mpc:RobustDeadlineAwareMpcController"},
            method_parameters={"horizon_epochs": 3},
            provenance={"candidate_selection_run": "selection-a"},
        ),
    )

    manifest = result.manifest
    metadata = manifest.as_payload()["metadata"]["freeze_method"]

    assert manifest.method_id == "robust-deadline-aware-mpc"
    assert manifest.method_name == "RefABR-MPC"
    assert manifest.entrypoint == "ref_abr.mpc:RobustDeadlineAwareMpcController"
    assert manifest.parameters["horizon_epochs"] == 3
    assert metadata["calibration_splits"] == ["calibration", "train"]
    assert metadata["excluded_splits"] == ["final"]
    assert metadata["primary_metric"] == "method_selection_quality"
    assert metadata["provenance"]["candidate_selection_run"] == "selection-a"


def test_export_frozen_method_manifest_writes_schema_stamped_json(tmp_path) -> None:
    result = freeze_method((_metric("robust-deadline-aware-mpc", "method_selection_quality", 0.8),))

    path = export_frozen_method_manifest(tmp_path, result)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == DOMAIN_SCHEMA_VERSION
    assert payload["record_type"] == "frozen_method_manifest"
    record = materialize_record(payload, expected_record_type="frozen_method_manifest")
    assert isinstance(record, FrozenMethodManifest)
    assert record.method_id == "robust-deadline-aware-mpc"


def test_freeze_method_cli_loads_metric_records_and_exports_manifest(tmp_path) -> None:
    metric_path = tmp_path / "metric_records.jsonl"
    output_root = tmp_path / "frozen"
    rows = [
        stamp_record(_metric("robust-deadline-aware-mpc", "method_selection_quality", 0.75), "metric_record"),
        stamp_record(_metric("deadline-aware-knapsack-allocator", "method_selection_quality", 0.7), "metric_record"),
    ]
    metric_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        f"""
seed: 5
split: calibration
freeze_method:
  metric_records: {metric_path.as_posix()}
  output_root: {output_root.as_posix()}
  method_name: RefABR-MPC
  version: frozen-test
  calibration_splits:
    - calibration
  excluded_splits:
    - final
  method_entrypoints:
    robust-deadline-aware-mpc: ref_abr.mpc:RobustDeadlineAwareMpcController
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["freeze_method", "--config", str(config_path), "--split", "calibration", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["payload"]["manifest"]["method_id"] == "robust-deadline-aware-mpc"
    assert (output_root / "frozen_method_manifest.json").exists()


def _metric(method_id: str, metric_name: str, value: float) -> MetricRecord:
    unit = "ms" if metric_name.endswith("_ms") else "score"
    return MetricRecord(metric_name=metric_name, value=value, unit=unit, tags={"method": method_id}, split="calibration")
