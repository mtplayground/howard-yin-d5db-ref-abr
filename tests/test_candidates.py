from __future__ import annotations

import pytest

from ref_abr.candidates import (
    CandidateError,
    CandidateGenerationSpec,
    CandidateObject,
    DecisionEpoch,
    TileSpec,
    Viewpoint,
    candidate_generation_spec_from_mapping,
    generate_candidate_objects,
)
from ref_abr.domain import MediaType
from ref_abr.substrate import ParametricSubstrateValueProvider, ReferenceResolution
from ref_abr.viewport import ViewportPose
from ref_abr.workloads import assemble_workload_manifest


def test_generate_gaussian_base_enhancement_tile_and_reference_candidates() -> None:
    workload = _workload()
    spec = CandidateGenerationSpec(
        resolutions=("720p",),
        fov_degrees=(90,),
        lookahead_ms=(0, 100),
        expiration_ms=(100,),
        retransmit_priorities=(1,),
        enhancement_layers=(1,),
        tile_rows=1,
        tile_columns=2,
    )
    epoch = DecisionEpoch(decision_time_ms=1000, frame_id="frame-1", viewpoint=Viewpoint(yaw_deg=30))

    candidate_set = generate_candidate_objects(workload, epoch, spec=spec)

    assert candidate_set.decision_time_ms == 1000
    assert candidate_set.candidate_set_id.startswith("candidate-set-")
    assert len(candidate_set.candidates) == 12
    assert _kind_counts(candidate_set.candidates) == {
        "gaussian_base": 2,
        "gaussian_enhancement": 2,
        "tile": 4,
        "reference_action": 4,
    }
    gaussian_base = [candidate for candidate in candidate_set.candidates if candidate.candidate_kind == "gaussian_base"]
    assert {candidate.layer for candidate in gaussian_base} == {0}
    enhancement = [candidate for candidate in candidate_set.candidates if candidate.candidate_kind == "gaussian_enhancement"]
    assert {candidate.layer for candidate in enhancement} == {1}
    tiles = [candidate for candidate in candidate_set.candidates if candidate.candidate_kind == "tile"]
    assert {candidate.tile.column for candidate in tiles if candidate.tile is not None} == {0, 1}
    assert {candidate.size_bytes for candidate in tiles} == {50}
    assert all(candidate.resolution == ReferenceResolution(1280, 720) for candidate in candidate_set.candidates)
    assert all(candidate.deadline_ms in {1100} for candidate in candidate_set.candidates)
    assert candidate_set.metadata["provenance"]["candidate_count"] == 12


def test_candidates_include_optional_substrate_estimates() -> None:
    workload = _workload()
    spec = CandidateGenerationSpec(
        resolutions=("1080p",),
        fov_degrees=(90,),
        lookahead_ms=(50,),
        expiration_ms=(100,),
        retransmit_priorities=(0,),
        enhancement_layers=(1,),
        tile_rows=1,
        tile_columns=1,
        include_tiles=False,
        include_reference_actions=False,
    )

    candidate_set = generate_candidate_objects(
        workload,
        {"decision_time_ms": 0, "viewpoint": {"yaw": 10}},
        spec=spec,
        substrate_provider=ParametricSubstrateValueProvider(),
    )

    assert len(candidate_set.candidates) == 2
    assert all("substrate" in candidate.metadata for candidate in candidate_set.candidates)
    assert candidate_set.candidates[0].metadata["substrate"]["provider_id"] == "parametric-substrate-default"
    assert candidate_set.candidates[0].metadata["substrate"]["query"]["freshness_ms"] == 50.0


def test_candidate_generation_is_deterministic_for_equivalent_inputs() -> None:
    spec = CandidateGenerationSpec(
        resolutions=("720p",),
        fov_degrees=(90,),
        lookahead_ms=(0,),
        expiration_ms=(100,),
        retransmit_priorities=(0,),
        enhancement_layers=(1,),
        include_tiles=False,
    )
    first = generate_candidate_objects(_workload(order=("mesh", "splat")), DecisionEpoch(0), spec=spec)
    second = generate_candidate_objects(_workload(order=("splat", "mesh")), DecisionEpoch(0), spec=spec)

    assert first.candidate_set_id == second.candidate_set_id
    assert [candidate.as_payload() for candidate in first.candidates] == [
        candidate.as_payload() for candidate in second.candidates
    ]


def test_candidate_generation_spec_from_mapping_and_viewpoint_coercion() -> None:
    spec = candidate_generation_spec_from_mapping(
        {
            "resolutions": ["480p"],
            "fov_degrees": 80,
            "lookahead_ms": [0],
            "expiration_ms": [100],
            "retransmit_priorities": [2],
            "enhancement_layers": [3],
            "tile_rows": "2",
            "tile_columns": "2",
            "include_tiles": "false",
            "max_candidates": "4",
        }
    )
    epoch = DecisionEpoch(
        decision_time_ms="5",
        viewpoint=ViewportPose(timestamp_ms=0, x_m=1, y_m=2, z_m=3, yaw_deg=4, pitch_deg=5, roll_deg=6),
    )

    assert spec.resolutions == (ReferenceResolution(854, 480),)
    assert spec.include_tiles is False
    assert spec.max_candidates == 4
    assert epoch.viewpoint.as_payload() == {
        "x_m": 1.0,
        "y_m": 2.0,
        "z_m": 3.0,
        "yaw_deg": 4.0,
        "pitch_deg": 5.0,
        "roll_deg": 6.0,
    }


def test_candidate_generator_respects_max_candidates() -> None:
    spec = CandidateGenerationSpec(
        resolutions=("720p",),
        fov_degrees=(90,),
        lookahead_ms=(0, 100),
        expiration_ms=(100,),
        retransmit_priorities=(0,),
        max_candidates=3,
    )

    candidate_set = generate_candidate_objects(_workload(), DecisionEpoch(0), spec=spec)

    assert len(candidate_set.candidates) == 3


def test_invalid_candidate_inputs_raise_clear_errors() -> None:
    with pytest.raises(CandidateError, match="expiration_ms"):
        CandidateGenerationSpec(lookahead_ms=(200,), expiration_ms=(100,))

    with pytest.raises(CandidateError, match="tile.row"):
        TileSpec(row=2, column=0, rows=2, columns=1)

    with pytest.raises(CandidateError, match="candidate_kind"):
        CandidateObject(
            candidate_id="c",
            object_id="o",
            candidate_kind="unknown",
            decision_time_ms=0,
            layer=0,
            resolution="720p",
            fov_deg=90,
            viewpoint=None,
            lookahead_ms=0,
            expiration_ms=1,
            retransmit_priority=0,
            size_bytes=0,
        )

    with pytest.raises(CandidateError, match="No feasible candidates"):
        generate_candidate_objects(
            _workload(),
            DecisionEpoch(0),
            spec=CandidateGenerationSpec(
                include_gaussian_base=False,
                include_gaussian_enhancement=False,
                include_tiles=False,
                include_reference_actions=False,
            ),
        )


def _kind_counts(candidates) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.candidate_kind] = counts.get(candidate.candidate_kind, 0) + 1
    return counts


def _workload(order: tuple[str, ...] = ("splat", "mesh")):
    asset_by_name = {
        "splat": {
            "object_id": "splat-a",
            "path": "splat.ply",
            "size_bytes": 100,
            "media_type": MediaType.GAUSSIAN_SPLAT.value,
        },
        "mesh": {
            "object_id": "mesh-a",
            "path": "mesh.obj",
            "size_bytes": 50,
            "media_type": MediaType.MESH.value,
            "dependencies": ["splat-a"],
        },
    }
    return assemble_workload_manifest(
        {
            "dataset": "candidate-test",
            "sequences": [
                {
                    "scene": "scene",
                    "name": "seq",
                    "assets": [asset_by_name[name] for name in order],
                }
            ],
        },
        config_id="cfg-a",
        split="train",
        seed=1,
        source_uri="memory://candidate-workload",
    )
