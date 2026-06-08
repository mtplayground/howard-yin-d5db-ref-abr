from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from ref_abr.artifacts import ArtifactProvenance, export_raw_artifacts
from ref_abr.cli import main
from ref_abr.domain import FrameOutcome, MetricRecord, ScheduleDecision
from ref_abr.harness import HarnessConfig, HarnessRunResult, run_harness, run_key
from ref_abr.paper_outputs import PaperOutputConfig, PaperOutputSpec, derive_paper_outputs
from ref_abr.schema import materialize_record


def test_cli_smoke_end_to_end_deterministic_mini_run(tmp_path) -> None:
    first = _run_mini_pipeline(tmp_path / "first")
    second = _run_mini_pipeline(tmp_path / "second")

    assert first["cli_statuses"] == ["ok", "ok"]
    assert first["cli_metric_names"] == second["cli_metric_names"]
    assert first["comparison_rows"] == second["comparison_rows"]
    assert first["paper_outputs"] == second["paper_outputs"]
    assert first["metric_records_by_method"] == second["metric_records_by_method"]
    assert all(row["delta"] == pytest.approx(0.2) for row in first["comparison_rows"])
    assert {row["frame_id"] for row in first["comparison_rows"]} == {"frame-1", "frame-2"}


def _run_mini_pipeline(root: Path) -> dict[str, Any]:
    raw_roots = _write_raw_method_artifacts(root / "raw")
    metric_roots: dict[str, Path] = {}
    cli_payloads: list[dict[str, Any]] = []

    config = HarnessConfig(
        harness_name="deterministic-mini",
        methods=("baseline-simple", "candidate-simple"),
        workloads=("tiny-workload",),
        seeds=(7,),
        run_mode="full",
        baseline_method_id="baseline-simple",
        comparison_metric_names=("full_frame_quality",),
        comparison_group_keys=("workload_id", "seed", "metric_name", "frame_id"),
        fixed_variables={"device": "toy-device", "network": "toy-network"},
        output_root=root / "harness",
        tags={"split": "final"},
    )

    def executor(spec) -> HarnessRunResult:
        metric_root = root / "metrics" / spec.method_id
        metric_roots[spec.method_id] = metric_root
        payload = _invoke_compute_metrics_cli(
            root=root,
            method_id=spec.method_id,
            raw_root=raw_roots[spec.method_id],
            metric_root=metric_root,
            run_id=spec.run_id,
        )
        cli_payloads.append(payload)
        return HarnessRunResult(
            spec=spec,
            status="executed",
            metrics=_load_metric_records(metric_root / "metric_records.jsonl"),
            metadata={"cli_metric_count": payload["payload"]["metric_count"]},
        )

    harness_result = run_harness(config, executor=executor)
    comparison_rows = [dict(row) for row in harness_result.comparison_summary.rows]
    assert (root / "harness" / "harness_result.json").exists()

    paper_source_root = root / "paper_sources"
    paper_source_root.mkdir(parents=True)
    _write_json(
        paper_source_root / "main_qoe_table.json",
        [
            {
                "method_id": row["method_id"],
                "baseline_method_id": row["baseline_method_id"],
                "frame_id": row["frame_id"],
                "deadline_hit_qoe": row["method_value"],
                "baseline_deadline_hit_qoe": row["baseline_value"],
                "delta": row["delta"],
            }
            for row in comparison_rows
        ],
    )
    _write_json(
        paper_source_root / "claim_artifact_traceability.json",
        [
            {
                "claim_id": "mini-run-deterministic-agreement",
                "artifact_ids": [
                    "harness_result.json",
                    *[f"metrics/{method_id}/manifest.json" for method_id in sorted(metric_roots)],
                ],
                "traceable": True,
            }
        ],
    )
    paper_result = derive_paper_outputs(
        PaperOutputConfig(
            artifact_roots=(paper_source_root,),
            output_root=root / "paper_outputs",
            output_specs=(
                PaperOutputSpec(output_name="main_qoe_table", source_filenames=("main_qoe_table.json",)),
                PaperOutputSpec(output_name="traceability", source_filenames=("claim_artifact_traceability.json",)),
            ),
        )
    )
    assert (root / "paper_outputs" / "paper_outputs_manifest.json").exists()

    return {
        "cli_statuses": [payload["status"] for payload in sorted(cli_payloads, key=lambda item: item["payload"]["manifest"]["provenance"]["method_id"])],
        "cli_metric_names": [payload["payload"]["metric_names"] for payload in sorted(cli_payloads, key=lambda item: item["payload"]["manifest"]["provenance"]["method_id"])],
        "comparison_rows": sorted(comparison_rows, key=lambda row: row["frame_id"]),
        "metric_records_by_method": {
            method_id: _metric_digest(metric_roots[method_id] / "metric_records.jsonl")
            for method_id in sorted(metric_roots)
        },
        "paper_outputs": {
            record.output_name: json.loads(Path(record.output_path).read_text(encoding="utf-8"))
            for record in sorted(paper_result.outputs, key=lambda item: item.output_name)
        },
    }


def _write_raw_method_artifacts(root: Path) -> dict[str, Path]:
    methods = {
        "baseline-simple": (0.50, 0.60),
        "candidate-simple": (0.70, 0.80),
    }
    raw_roots: dict[str, Path] = {}
    for method_id, qualities in methods.items():
        method_root = root / method_id
        raw_roots[method_id] = method_root
        export_raw_artifacts(
            method_root,
            provenance=ArtifactProvenance(
                run_id=f"raw-{method_id}",
                config_id="mini-config",
                split="final",
                method_id=method_id,
                source="deterministic-mini-run",
            ),
            frame_outcomes=tuple(
                _outcome(f"frame-{index}", quality=quality)
                for index, quality in enumerate(qualities, start=1)
            ),
            decisions=tuple(
                _decision(f"decision-{method_id}-{index}", f"frame-{index}", method_id=method_id)
                for index in range(1, 3)
            ),
        )
    return raw_roots


def _invoke_compute_metrics_cli(
    *,
    root: Path,
    method_id: str,
    raw_root: Path,
    metric_root: Path,
    run_id: str,
) -> dict[str, Any]:
    config_path = root / "configs" / f"{method_id}.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""
seed: 7
split: final
compute_metrics:
  input_root: {raw_root.as_posix()}
  output_root: {metric_root.as_posix()}
  run_id: {run_id}
  config_id: mini-config
  method_id: {method_id}
  metric_set:
    - quality
    - deadline_lifecycle
  grouping_keys:
    - frame_id
  paired_metric_names:
    - full_frame_quality
  bootstrap_iterations: 25
  seed: 7
""",
        encoding="utf-8",
    )
    result = CliRunner().invoke(main, ["compute_metrics", "--config", str(config_path), "--split", "final", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert (metric_root / "manifest.json").exists()
    assert (metric_root / "metric_records.jsonl").exists()
    return payload


def _load_metric_records(path: Path) -> tuple[MetricRecord, ...]:
    records: list[MetricRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        record = materialize_record(row["payload"], expected_record_type="metric_record")
        assert isinstance(record, MetricRecord)
        records.append(record)
    return tuple(records)


def _metric_digest(path: Path) -> tuple[tuple[str, str | None, float, str], ...]:
    return tuple(
        (metric.metric_name, metric.frame_id, float(metric.value), metric.unit)
        for metric in _load_metric_records(path)
    )


def _outcome(frame_id: str, *, quality: float) -> FrameOutcome:
    return FrameOutcome(
        frame_id=frame_id,
        scheduled_time_ms=0,
        rendered_time_ms=10,
        deadline_ms=20,
        delivered_object_ids=("object-1",),
        missing_object_ids=(),
        quality_score=quality,
        deadline_hit=True,
    )


def _decision(decision_id: str, frame_id: str, *, method_id: str) -> ScheduleDecision:
    return ScheduleDecision(
        decision_id=decision_id,
        controller_id=method_id,
        frame_id=frame_id,
        selected_object_ids=("object-1",),
        decision_time_ms=0,
        target_deadline_ms=20,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
