from __future__ import annotations

import json
from pathlib import Path

from ref_abr.artifacts import ArtifactProvenance, export_raw_artifacts
from ref_abr.domain import FrameOutcome
from ref_abr.external_measurements import load_external_measurement_records
from ref_abr.harness import HarnessConfig, HarnessRunResult, run_harness
from ref_abr.metrics import ComputeMetricsConfig, ComputeMetricsInput, compute_metric_records, export_metric_records
from ref_abr.providers.base import ExternalTraceSubstrateProvider
from ref_abr.schema import materialize_record


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "external_measurements" / "offline_trace.jsonl"


def test_offline_external_trace_runs_through_harness_and_compute_metrics(tmp_path) -> None:
    """Load recorded external measurements, run a harness executor, and export metrics."""

    loaded_fixture = load_external_measurement_records(FIXTURE)
    output_root = tmp_path / "harness"

    def executor(spec):
        method_records = tuple(record for record in loaded_fixture if record.metadata["method_id"] == spec.method_id)
        provider = ExternalTraceSubstrateProvider(records=method_records)
        frame_outcomes = []
        for record in method_records:
            value = provider.evaluate(record.metadata["query"])
            rendered_time_ms = int(round(value.component_timing.total_ms))
            frame_outcomes.append(
                FrameOutcome(
                    frame_id=record.frame_id or "frame",
                    scheduled_time_ms=0,
                    rendered_time_ms=rendered_time_ms,
                    deadline_ms=20,
                    delivered_object_ids=(record.object_id or "object",),
                    quality_score=value.visible_quality,
                    deadline_hit=record.deadline_hit,
                    metadata={
                        "external_record_id": record.record_id,
                        "provider_id": value.provider_id,
                        "size_bytes": record.size_bytes,
                    },
                )
            )
        raw_root = output_root / spec.run_key / "raw"
        metric_root = output_root / spec.run_key / "metrics"
        raw_manifest = export_raw_artifacts(
            raw_root,
            provenance=ArtifactProvenance(
                run_id=spec.run_id,
                split="final",
                method_id=spec.method_id,
                source="external_measurement_fixture",
            ),
            frame_outcomes=tuple(frame_outcomes),
        )
        config = ComputeMetricsConfig(
            run_id=spec.run_id,
            split="final",
            method_id=spec.method_id,
            metric_set=("quality",),
            tags={"workload_id": spec.workload_id},
        )
        first_metrics = compute_metric_records(ComputeMetricsInput(frame_outcomes=tuple(frame_outcomes)), config=config)
        second_metrics = compute_metric_records(ComputeMetricsInput(frame_outcomes=tuple(frame_outcomes)), config=config)
        assert [metric.as_payload() for metric in first_metrics] == [metric.as_payload() for metric in second_metrics]
        metric_manifest = export_metric_records(metric_root, first_metrics, config=config)
        return HarnessRunResult(
            spec=spec,
            status="executed",
            raw_artifacts=raw_manifest,
            metric_artifacts=metric_manifest,
            metrics=first_metrics,
            metadata={"external_trace_fixture": FIXTURE.as_posix()},
        )

    result = run_harness(
        HarnessConfig(
            harness_name="external-backend-offline-trace",
            methods=("baseline", "candidate"),
            workloads=("offline-scene",),
            seeds=(0,),
            run_mode="full",
            comparison_metric_names=("deadline_hit_visible_quality",),
            output_root=output_root,
        ),
        executor=executor,
    )

    assert len(result.run_results) == 2
    assert result.comparison_summary.rows[0]["baseline_value"] == 0.6
    assert result.comparison_summary.rows[0]["method_value"] == 0.9
    assert result.comparison_summary.rows[0]["delta"] == 0.30000000000000004
    harness_payload = json.loads((output_root / "harness_result.json").read_text(encoding="utf-8"))
    assert harness_payload["comparison_summary"]["rows"][0]["method_id"] == "candidate"

    candidate_metrics = output_root / "candidate__offline-scene__seed-0" / "metrics" / "metric_records.jsonl"
    first_lines = candidate_metrics.read_text(encoding="utf-8").splitlines()
    second_lines = candidate_metrics.read_text(encoding="utf-8").splitlines()
    assert first_lines == second_lines
    parsed_metric = json.loads(first_lines[0])["payload"]
    assert materialize_record(parsed_metric, expected_record_type="metric_record").metric_name == "deadline_hit_visible_quality"
