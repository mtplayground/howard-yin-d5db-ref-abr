from __future__ import annotations

import pytest

from ref_abr.candidates import CandidateGenerationSpec, DecisionEpoch, generate_candidate_objects
from ref_abr.diagnostics import (
    DiagnosticError,
    Layered3DGSComparator,
    ViewportTileComparator,
    diagnostic_comparators,
)
from ref_abr.domain import ControllerState, MediaType
from ref_abr.methods import ActionBudget, ObservationBudget, SchedulingObservation, plan_schedule
from ref_abr.substrate import ParametricSubstrateValueProvider
from ref_abr.utility import ResourceBudget, estimate_candidate_set_utility
from ref_abr.workloads import assemble_workload_manifest


def test_layered_3dgs_comparator_selects_base_layers_first_under_adapter() -> None:
    observation = _observation(tile_rows=2, tile_columns=2)

    decision = plan_schedule(
        Layered3DGSComparator(metadata={"study": "layers"}),
        observation,
        observation_budget=ObservationBudget(max_candidates=80),
        action_budget=ActionBudget(max_selected_objects=2, max_selected_candidates=2, max_selected_bytes=1_000_000),
    )

    selected = _selected_candidates(observation, decision.metadata["adapter"]["selected_candidate_ids"])
    assert selected
    assert {candidate.candidate_kind for candidate in selected} == {"gaussian_base"}
    assert [candidate.layer for candidate in selected] == [0, 0]
    assert decision.metadata["diagnostic"]["diagnostic"] is True
    assert decision.metadata["diagnostic"]["comparator"] == "layered_3dgs"
    assert decision.metadata["diagnostic"]["freeze_eligible"] is False
    assert decision.metadata["diagnostic"]["parameters"]["study"] == "layers"
    assert decision.metadata["adapter"]["method_id"] == "diagnostic-layered-3dgs"


def test_layered_3dgs_comparator_respects_action_budget_bytes() -> None:
    observation = _observation(size_bytes=900_000)

    decision = plan_schedule(
        Layered3DGSComparator(),
        observation,
        observation_budget=ObservationBudget(max_candidates=40),
        action_budget=ActionBudget(max_selected_objects=2, max_selected_bytes=10),
    )

    assert decision.selected_object_ids == ()
    assert decision.metadata["diagnostic"]["selected_candidate_kinds"] == ()


def test_viewport_tile_comparator_selects_nearest_target_tile() -> None:
    observation = _observation(tile_rows=3, tile_columns=3)
    method = ViewportTileComparator(target_row=2, target_column=1)

    decision = plan_schedule(
        method,
        observation,
        observation_budget=ObservationBudget(max_candidates=100),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=1_000_000),
    )

    selected = _selected_candidates(observation, decision.metadata["adapter"]["selected_candidate_ids"])
    assert len(selected) == 1
    assert selected[0].candidate_kind == "tile"
    assert selected[0].tile is not None
    assert selected[0].tile.row == 2
    assert selected[0].tile.column == 1
    assert decision.metadata["diagnostic"]["comparator"] == "viewport_tile"
    assert decision.metadata["diagnostic"]["parameters"]["target_row"] == 2
    assert decision.metadata["diagnostic"]["parameters"]["target_column"] == 1


def test_viewport_tile_comparator_returns_empty_when_no_tiles_are_visible() -> None:
    observation = _observation(include_tiles=False)

    decision = plan_schedule(
        ViewportTileComparator(),
        observation,
        observation_budget=ObservationBudget(max_candidates=20),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_bytes=1_000_000),
    )

    assert decision.selected_object_ids == ()
    assert decision.metadata["diagnostic"]["selected_candidate_kinds"] == ()
    assert decision.metadata["diagnostic"]["parameters"]["target_row"] == 0
    assert decision.metadata["diagnostic"]["parameters"]["target_column"] == 0


def test_diagnostic_comparator_set_and_validation() -> None:
    comparators = diagnostic_comparators()

    assert [comparator.method_id for comparator in comparators] == [
        "diagnostic-layered-3dgs",
        "diagnostic-viewport-tile",
    ]
    assert all(not comparator.freeze_eligible for comparator in comparators)
    with pytest.raises(DiagnosticError, match="freeze_eligible"):
        Layered3DGSComparator(freeze_eligible=True)
    with pytest.raises(DiagnosticError, match="target_row"):
        ViewportTileComparator(target_row=-1)


def test_viewport_tile_comparator_rejects_out_of_grid_target() -> None:
    observation = _observation(tile_rows=2, tile_columns=2)

    with pytest.raises(DiagnosticError, match="target_row"):
        ViewportTileComparator(target_row=3).plan_schedule(
            observation,
            ActionBudget(max_selected_objects=1, max_selected_bytes=1_000_000),
        )


def _selected_candidates(observation: SchedulingObservation, selected_candidate_ids) -> tuple:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in observation.candidates}
    return tuple(candidate_by_id[candidate_id] for candidate_id in selected_candidate_ids)


def _observation(
    *,
    size_bytes: int = 120_000,
    tile_rows: int = 2,
    tile_columns: int = 2,
    include_tiles: bool = True,
) -> SchedulingObservation:
    candidate_set = generate_candidate_objects(
        _workload(size_bytes=size_bytes),
        DecisionEpoch(decision_time_ms=20, frame_id="frame-diagnostic"),
        spec=CandidateGenerationSpec(
            resolutions=("720p",),
            fov_degrees=(90,),
            lookahead_ms=(0,),
            expiration_ms=(100,),
            retransmit_priorities=(0,),
            enhancement_layers=(1, 2),
            tile_rows=tile_rows,
            tile_columns=tile_columns,
            include_tiles=include_tiles,
        ),
        substrate_provider=ParametricSubstrateValueProvider(),
    )
    utilities = estimate_candidate_set_utility(
        candidate_set,
        budgets=ResourceBudget(available_time_ms=100, available_bytes=1_000_000, available_memory_mb=1024),
    )
    return SchedulingObservation(
        observation_id="obs-diagnostic",
        controller_state=ControllerState(
            controller_id="ctrl-diagnostic",
            method_name="diagnostic",
            step_index=0,
            active_split="calibration",
        ),
        frame_id="frame-diagnostic",
        decision_time_ms=20,
        target_deadline_ms=120,
        candidate_set=candidate_set,
        utility_estimates=utilities.estimates,
    )


def _workload(*, size_bytes: int):
    return assemble_workload_manifest(
        {
            "dataset": "diagnostic-test",
            "sequences": [
                {
                    "scene": "scene",
                    "name": "seq",
                    "assets": [
                        {
                            "object_id": "splat-a",
                            "path": "splat-a.ply",
                            "size_bytes": size_bytes,
                            "media_type": MediaType.GAUSSIAN_SPLAT.value,
                        },
                        {
                            "object_id": "splat-b",
                            "path": "splat-b.ply",
                            "size_bytes": size_bytes,
                            "media_type": MediaType.GAUSSIAN_SPLAT.value,
                        },
                    ],
                }
            ],
        },
        split="calibration",
        config_id="diagnostic-test-config",
        seed=19,
    )
