from __future__ import annotations

import json

import pytest

from ref_abr.paper_outputs import (
    DEFAULT_PAPER_OUTPUT_SPECS,
    PaperOutputConfig,
    PaperOutputError,
    PaperOutputSpec,
    derive_paper_outputs,
    load_paper_output_manifest,
)


def test_derive_paper_outputs_maps_existing_artifacts_without_rerun(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    output_root = tmp_path / "paper"
    _write_required_artifacts(artifacts)

    result = derive_paper_outputs(PaperOutputConfig(artifact_roots=(artifacts,), output_root=output_root))

    output_names = {record.output_name for record in result.outputs}
    assert output_names >= {
        "substitution_surface",
        "lifecycle_matrix",
        "screening_table",
        "main_qoe_table",
        "quality_deadline_pareto",
        "deadline_hit_qoe_cdf",
        "ablation_table",
        "stress_matrix",
        "traceability",
        "tolerance_checks",
    }
    assert result.validation["derived_without_rerun"] is True
    assert (output_root / "paper_outputs_manifest.json").exists()
    assert json.loads((output_root / "screening_table.json").read_text(encoding="utf-8"))[0]["method_id"] == "frozen-refabr"


def test_load_paper_output_manifest_round_trips_records(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    output_root = tmp_path / "paper"
    _write_required_artifacts(artifacts)
    result = derive_paper_outputs(PaperOutputConfig(artifact_roots=(artifacts,), output_root=output_root))

    loaded = load_paper_output_manifest(output_root)

    assert loaded.derivation_id == result.derivation_id
    assert {record.output_name for record in loaded.outputs} == {record.output_name for record in result.outputs}
    assert loaded.config.artifact_roots == (artifacts.as_posix(),)


def test_derive_paper_outputs_can_use_subset_specs_and_jsonl_sources(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    output_root = tmp_path / "paper"
    artifacts.mkdir()
    (artifacts / "substitution_surface.jsonl").write_text(
        json.dumps(
            {
                "method_id": "svq-gaussian-only-abr",
                "substitution_gain": 0.1,
                "utility_score": 0.7,
                "budget_pressure": 0.3,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = derive_paper_outputs(
        PaperOutputConfig(
            artifact_roots=(artifacts,),
            output_root=output_root,
            output_specs=(PaperOutputSpec(output_name="substitution_surface", source_filenames=("substitution_surface.jsonl",)),),
        )
    )

    rows = json.loads((output_root / "substitution_surface.json").read_text(encoding="utf-8"))
    assert result.outputs[0].row_count == 1
    assert rows[0]["substitution_gain"] == 0.1


def test_derive_paper_outputs_rejects_missing_required_outputs(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    with pytest.raises(PaperOutputError, match="Missing required"):
        derive_paper_outputs(
            PaperOutputConfig(
                artifact_roots=(artifacts,),
                output_root=tmp_path / "paper",
                output_specs=(PaperOutputSpec(output_name="main_qoe_table", source_filenames=("main_qoe_table.json",)),),
            )
        )


def test_derive_paper_outputs_enforces_tolerance_and_traceability(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_json(artifacts / "tolerance_checks.json", [{"claim_id": "claim-a", "tolerance_pass": False}])
    _write_json(artifacts / "claim_artifact_traceability.json", [{"claim_id": "claim-a", "artifact_ids": (), "traceable": False}])

    with pytest.raises(PaperOutputError, match="Tolerance"):
        derive_paper_outputs(
            PaperOutputConfig(
                artifact_roots=(artifacts,),
                output_root=tmp_path / "paper-tolerance",
                output_specs=(PaperOutputSpec(output_name="tolerance_checks", source_filenames=("tolerance_checks.json",)),),
            )
        )

    with pytest.raises(PaperOutputError, match="Traceability"):
        derive_paper_outputs(
            PaperOutputConfig(
                artifact_roots=(artifacts,),
                output_root=tmp_path / "paper-traceability",
                output_specs=(PaperOutputSpec(output_name="traceability", source_filenames=("claim_artifact_traceability.json",)),),
                enforce_tolerances=False,
            )
        )


def _write_required_artifacts(root) -> None:
    root.mkdir(parents=True)
    _write_json(
        root / "substitution_surface_summary.json",
        {
            "outcomes": [
                {"method_id": "svq-gaussian-only-abr", "substitution_gain": 0.1, "utility_score": 0.7, "budget_pressure": 0.3},
                {"method_id": "svq-gaussian-only-abr", "substitution_gain": 0.3, "utility_score": 0.9, "budget_pressure": 0.5},
            ]
        },
    )
    (root / "lifecycle_matrix.jsonl").write_text(json.dumps({"method_id": "fixed-cadence", "late": 1}) + "\n", encoding="utf-8")
    _write_json(
        root / "candidate_method_selection_summary.json",
        {
            "outcomes": [
                {
                    "method_id": "frozen-refabr",
                    "quality_score": 0.9,
                    "deadline_score": 0.8,
                    "resource_efficiency": 0.7,
                    "runtime_ms": 12.0,
                    "interpretability_score": 0.6,
                }
            ]
        },
    )
    _write_json(root / "main_qoe_table.json", [{"method_id": "frozen-refabr", "deadline_hit_qoe": 0.9}])
    _write_json(root / "quality_deadline_pareto.json", [{"method_id": "frozen-refabr", "pareto_frontier": True}])
    _write_json(root / "deadline_hit_qoe_cdf.json", [{"method_id": "frozen-refabr", "cumulative_probability": 1.0}])
    _write_json(root / "paired_ablation_table.json", [{"variant_id": "full", "deadline_hit_qoe": 0.9}])
    _write_json(root / "coupled_stress_matrix.json", [{"method_id": "frozen-refabr", "stress_id": "high"}])
    _write_json(root / "claim_artifact_traceability.json", [{"claim_id": "claim-a", "artifact_ids": ("artifact-a",), "traceable": True}])
    _write_json(root / "tolerance_checks.json", [{"claim_id": "claim-a", "tolerance_pass": True}])


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def test_default_specs_include_issue_required_named_outputs() -> None:
    names = {str(spec["output_name"]) for spec in DEFAULT_PAPER_OUTPUT_SPECS}
    assert {
        "substitution_surface",
        "lifecycle_matrix",
        "screening_table",
        "main_qoe_table",
        "quality_deadline_pareto",
        "deadline_hit_qoe_cdf",
        "ablation_table",
        "stress_matrix",
        "traceability",
    } <= names
