from __future__ import annotations

import json

from click.testing import CliRunner

from ref_abr.artifacts import ArtifactProvenance, export_raw_artifacts
from ref_abr.cli import main
from ref_abr.domain import FrameOutcome
from ref_abr.metrics import (
    ComputeMetricsConfig,
    ComputeMetricsInput,
    compute_metric_records,
    export_metric_records,
)
from ref_abr.schema import DOMAIN_SCHEMA_VERSION, validate_stamped_record


def test_compute_metric_records_wires_metric_sets_and_paired_baseline() -> None:
    records = ComputeMetricsInput(
        frame_outcomes=(
            _outcome("frame-1", quality=0.8),
            _outcome("frame-2", quality=0.6),
        ),
        baseline_frame_outcomes=(
            _outcome("frame-1", quality=0.5),
            _outcome("frame-2", quality=0.4),
        ),
    )
    config = ComputeMetricsConfig(
        run_id="run-1",
        config_id="cfg-1",
        split="final",
        method_id="refabr",
        baseline_method_id="baseline",
        metric_set=("quality", "deadline_lifecycle", "paired_baselines"),
        paired_metric_names=("deadline_hit_visible_quality", "deadline_qoe"),
        bootstrap_iterations=100,
        seed=5,
    )

    metrics = compute_metric_records(records, config=config)
    by_name = {metric.metric_name: metric for metric in metrics}
    paired = [metric for metric in metrics if metric.metric_name == "paired_mean_delta"]

    assert "deadline_hit_visible_quality" in by_name
    assert "deadline_qoe" in by_name
    assert len(paired) == 2
    assert all(metric.tags["run_id"] == "run-1" for metric in metrics)
    assert all(metric.tags["config_id"] == "cfg-1" for metric in metrics)
    assert all(metric.split == "final" for metric in metrics)
    assert paired[0].metadata["promotion_blocked"] is False


def test_export_metric_records_writes_schema_stamped_metric_file(tmp_path) -> None:
    config = ComputeMetricsConfig(run_id="run-export", split="final", metric_set=("quality",))
    metrics = compute_metric_records(
        ComputeMetricsInput(frame_outcomes=(_outcome("frame-1", quality=0.7),)),
        config=config,
    )

    manifest = export_metric_records(tmp_path, metrics, config=config)

    assert manifest.files[0].artifact_name == "metric_records"
    rows = [json.loads(line) for line in (tmp_path / "metric_records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["schema_version"] == DOMAIN_SCHEMA_VERSION
    assert rows[0]["record_type"] == "metric_record"
    assert rows[0]["provenance"]["source"] == "compute_metrics"
    stamped = validate_stamped_record(rows[0]["payload"], expected_record_type="metric_record")
    assert stamped["metric_name"] == "deadline_hit_visible_quality"


def test_compute_metrics_cli_loads_artifacts_and_exports_metric_records(tmp_path) -> None:
    input_root = tmp_path / "raw"
    output_root = tmp_path / "metrics"
    export_raw_artifacts(
        input_root,
        provenance=ArtifactProvenance(run_id="raw-run"),
        frame_outcomes=(_outcome("frame-1", quality=0.9),),
    )
    _write_baseline_outcomes(input_root / "baseline_frame_outcomes.jsonl")
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        f"""
seed: 7
split: final
compute_metrics:
  input_root: {input_root.as_posix()}
  output_root: {output_root.as_posix()}
  run_id: cli-run
  method_id: refabr
  baseline_method_id: baseline
  metric_set:
    - quality
    - paired_baselines
  paired_metric_names:
    - deadline_hit_visible_quality
  bootstrap_iterations: 50
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["compute_metrics", "--config", str(config_path), "--split", "final", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["payload"]["metric_count"] == 4
    assert (output_root / "metric_records.jsonl").exists()


def _write_baseline_outcomes(path) -> None:
    row = {
        "schema_version": DOMAIN_SCHEMA_VERSION,
        "record_type": "frame_outcome",
        "artifact_name": "baseline_frame_outcomes",
        "sequence_index": 0,
        "provenance": {"run_id": "raw-run"},
        "payload": {
            "schema_version": DOMAIN_SCHEMA_VERSION,
            "record_type": "frame_outcome",
            **_outcome("frame-1", quality=0.6).as_payload(),
        },
    }
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def _outcome(frame_id: str, *, quality: float) -> FrameOutcome:
    return FrameOutcome(
        frame_id=frame_id,
        scheduled_time_ms=0,
        rendered_time_ms=10,
        deadline_ms=20,
        delivered_object_ids=("object-1",),
        quality_score=quality,
        deadline_hit=True,
    )
