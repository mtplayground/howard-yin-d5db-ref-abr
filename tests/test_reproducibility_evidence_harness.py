from __future__ import annotations

import json

import pytest

from ref_abr.replay import ReplayCase, ReplaySourceRef, ReplaySubsetManifest
from ref_abr.reproducibility_evidence_harness import (
    REPRODUCIBILITY_METRIC_NAMES,
    ClaimEvidenceSpec,
    ReproducibilityEvidenceConfig,
    ReproducibilityEvidenceHarnessError,
    claim_artifact_traceability,
    deterministic_replay_checks,
    evaluate_reproducibility_case,
    export_reproducibility_evidence_outputs,
    run_reproducibility_evidence_harness,
    tolerance_checks,
    workload_config_coverage,
)


def test_reproducibility_config_uses_replay_subset_and_methods() -> None:
    manifest = _replay_manifest()
    config = ReproducibilityEvidenceConfig(
        replay_subset_manifest=manifest,
        methods=("frozen-refabr", "deadline-greedy"),
        claim_specs=_claim_specs(),
    )

    assert config.replay_subset_manifest.subset_id == "subset-a"
    assert config.methods == ("frozen-refabr", "deadline-greedy")
    assert config.split == "final"
    assert config.harness_id.startswith("reproducibility-artifact-evidence-")


def test_reproducibility_evaluator_records_determinism_traceability_and_tolerance() -> None:
    manifest = _replay_manifest()
    outcome = evaluate_reproducibility_case(
        manifest.cases[0],
        method_id="frozen-refabr",
        seed=0,
        replay_subset_manifest=manifest,
        claim_specs=_claim_specs(),
        default_tolerance=1.0,
    )

    assert outcome.replay_determinism_pass is True
    assert outcome.replay_digest_a == outcome.replay_digest_b
    assert outcome.workload_config_coverage_ratio == 1.0
    assert outcome.claim_traceability_ratio == 1.0
    assert outcome.tolerance_pass_rate == 1.0
    assert {metric.metric_name for metric in outcome.metric_records(run_id="run-a")} == set(REPRODUCIBILITY_METRIC_NAMES)


def test_run_reproducibility_harness_builds_evidence_tables_and_comparisons() -> None:
    config = ReproducibilityEvidenceConfig(
        replay_subset_manifest=_replay_manifest(),
        methods=("frozen-refabr", "deadline-greedy"),
        claim_specs=_claim_specs(),
        seeds=(0,),
        tolerance=1.0,
    )

    result = run_reproducibility_evidence_harness(config)
    replay_rows = deterministic_replay_checks(result)
    coverage_rows = workload_config_coverage(result)
    traceability_rows = claim_artifact_traceability(result)
    tolerance_rows = tolerance_checks(result)

    assert len(result.outcomes) == 4
    assert result.harness_result.comparison_summary.baseline_method_id == "frozen-refabr"
    assert result.harness_result.comparison_summary.rows
    assert all(row["replay_determinism_pass"] for row in replay_rows)
    assert all(row["workload_config_coverage_ratio"] == 1.0 for row in coverage_rows)
    assert len(traceability_rows) == len(result.outcomes) * 2
    assert all(row["traceable"] for row in traceability_rows)
    assert all(row["tolerance_pass"] for row in tolerance_rows)


def test_export_reproducibility_outputs_writes_required_payloads(tmp_path) -> None:
    result = run_reproducibility_evidence_harness(
        ReproducibilityEvidenceConfig(
            replay_subset_manifest=_replay_manifest(),
            methods=("frozen-refabr",),
            claim_specs=_claim_specs(),
            tolerance=1.0,
            output_root=tmp_path,
        )
    )

    paths = export_reproducibility_evidence_outputs(tmp_path, result)

    assert {path.name for path in paths} == {
        "reproducibility_evidence_outcomes.jsonl",
        "deterministic_replay_checks.json",
        "workload_config_coverage.json",
        "claim_artifact_traceability.json",
        "tolerance_checks.json",
        "reproducibility_evidence_summary.json",
    }
    summary = json.loads((tmp_path / "reproducibility_evidence_summary.json").read_text(encoding="utf-8"))
    traceability = json.loads((tmp_path / "claim_artifact_traceability.json").read_text(encoding="utf-8"))
    assert summary["deterministic_replay_checks"]
    assert traceability[0]["artifact_ids"]
    assert (tmp_path / "harness" / "harness_result.json").exists()


def test_reproducibility_config_validation_errors_are_clear() -> None:
    with pytest.raises(ReproducibilityEvidenceHarnessError, match="split"):
        ReproducibilityEvidenceConfig(
            replay_subset_manifest=_replay_manifest(),
            methods=("frozen-refabr",),
            claim_specs=_claim_specs(),
            split="calibration",
        )

    with pytest.raises(ReproducibilityEvidenceHarnessError, match="claim_specs"):
        ReproducibilityEvidenceConfig(
            replay_subset_manifest=_replay_manifest(),
            methods=("frozen-refabr",),
            claim_specs=(),
        )

    with pytest.raises(ReproducibilityEvidenceHarnessError, match="duplicate"):
        ReproducibilityEvidenceConfig(
            replay_subset_manifest=_replay_manifest(),
            methods=("frozen-refabr",),
            claim_specs=(
                {"claim_id": "claim-a", "metric_name": "coverage_ratio", "expected_value": 1.0, "tolerance": 0.0, "artifact_ids": ("artifact-a",)},
                {"claim_id": "claim-a", "metric_name": "coverage_ratio", "expected_value": 1.0, "tolerance": 0.0, "artifact_ids": ("artifact-b",)},
            ),
        )


def _claim_specs() -> tuple[ClaimEvidenceSpec, ...]:
    return (
        ClaimEvidenceSpec(
            claim_id="claim-deterministic-replay",
            metric_name="coverage_ratio",
            expected_value=1.0,
            tolerance=0.0,
            artifact_ids=("subset-a", "artifact-replay-json"),
        ),
        ClaimEvidenceSpec(
            claim_id="claim-traceability",
            metric_name="artifact_traceability",
            expected_value=1.0,
            tolerance=0.0,
            artifact_ids=("artifact-manifest-json",),
        ),
    )


def _replay_manifest() -> ReplaySubsetManifest:
    cases = (
        ReplayCase(
            case_id="case-a",
            workload_manifest_id="workload-a",
            viewport_trace_id="viewport-a",
            network_trace_id="network-a",
            device_profile_id="device-a",
        ),
        ReplayCase(
            case_id="case-b",
            workload_manifest_id="workload-b",
            viewport_trace_id="viewport-a",
            network_trace_id="network-a",
            device_profile_id="device-a",
        ),
    )
    return ReplaySubsetManifest(
        subset_id="subset-a",
        config_id="config-a",
        split="final",
        seed=0,
        cases=cases,
        sources={
            "workloads": (
                ReplaySourceRef("workloads", "workload-a", "workload_manifest", source_uri="workload://a"),
                ReplaySourceRef("workloads", "workload-b", "workload_manifest", source_uri="workload://b"),
            ),
            "viewports": (ReplaySourceRef("viewports", "viewport-a", "viewport_trace", source_uri="viewport://a"),),
            "networks": (ReplaySourceRef("networks", "network-a", "network_trace", source_uri="network://a"),),
            "devices": (ReplaySourceRef("devices", "device-a", "device_profile", source_uri="device://a"),),
        },
        metadata={"provenance": {"assembly": "test"}},
    )
