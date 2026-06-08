"""Diagnostic scheduling comparators for controlled 3DGS/viewport studies."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.candidates import CandidateObject, TileSpec
from ref_abr.methods import ActionBudget, SchedulingObservation
from ref_abr.utility import CandidateUtilityEstimate


class DiagnosticError(ValueError):
    """Raised when a diagnostic comparator is configured with invalid parameters."""


LAYERED_3DGS_KINDS: tuple[str, ...] = ("gaussian_base", "gaussian_enhancement")
VIEWPORT_TILE_KINDS: tuple[str, ...] = ("tile",)


@dataclass(frozen=True)
class Layered3DGSComparator:
    """Diagnostic comparator that exposes layered Gaussian-splat scheduling behavior.

    The comparator is intentionally not a production controller. It selects base
    layers before enhancement layers, keeping object identity and action budgets
    enforced by the shared method adapter.
    """

    method_id: str = "diagnostic-layered-3dgs"
    method_name: str = "Diagnostic layered 3DGS comparator"
    freeze_eligible: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.method_name, "method_name")
        if self.freeze_eligible is not False:
            raise DiagnosticError("Layered3DGSComparator.freeze_eligible must be False.")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, Any]:
        utility_by_candidate = _utility_by_candidate(observation.utility_estimates)
        ranked = tuple(
            sorted(
                (candidate for candidate in observation.candidates if candidate.candidate_kind in LAYERED_3DGS_KINDS),
                key=_layered_3dgs_sort_key,
            )
        )
        selected = _select_with_budget(ranked, action_budget, allowed_kinds=LAYERED_3DGS_KINDS)
        return _diagnostic_payload(
            self,
            selected,
            comparator="layered_3dgs",
            utility_by_candidate=utility_by_candidate,
            parameters={
                "allowed_candidate_kinds": LAYERED_3DGS_KINDS,
                "rank_order": ("layer", "object_id", "resolution", "lookahead", "candidate_id"),
                **self.metadata,
            },
        )


@dataclass(frozen=True)
class ViewportTileComparator:
    """Diagnostic comparator that isolates viewport-tile selection behavior."""

    target_row: int | None = None
    target_column: int | None = None
    method_id: str = "diagnostic-viewport-tile"
    method_name: str = "Diagnostic viewport-tile comparator"
    freeze_eligible: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_row is not None:
            object.__setattr__(self, "target_row", _non_negative_int(self.target_row, "target_row"))
        if self.target_column is not None:
            object.__setattr__(self, "target_column", _non_negative_int(self.target_column, "target_column"))
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.method_name, "method_name")
        if self.freeze_eligible is not False:
            raise DiagnosticError("ViewportTileComparator.freeze_eligible must be False.")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, Any]:
        tile_candidates = tuple(candidate for candidate in observation.candidates if candidate.candidate_kind == "tile")
        utility_by_candidate = _utility_by_candidate(observation.utility_estimates)
        target = _resolve_tile_target(tile_candidates, self.target_row, self.target_column)
        ranked = tuple(sorted(tile_candidates, key=lambda candidate: _viewport_tile_sort_key(candidate, target)))
        selected = _select_with_budget(ranked, action_budget, allowed_kinds=VIEWPORT_TILE_KINDS)
        return _diagnostic_payload(
            self,
            selected,
            comparator="viewport_tile",
            utility_by_candidate=utility_by_candidate,
            parameters={
                "target_row": target[0],
                "target_column": target[1],
                "allowed_candidate_kinds": VIEWPORT_TILE_KINDS,
                "rank_order": ("tile_distance", "row", "column", "object_id", "candidate_id"),
                **self.metadata,
            },
        )


def diagnostic_comparators() -> tuple[Layered3DGSComparator, ViewportTileComparator]:
    """Return all diagnostic comparators, explicitly marked freeze-ineligible."""

    return (Layered3DGSComparator(), ViewportTileComparator())


def _select_with_budget(
    candidates: tuple[CandidateObject, ...],
    action_budget: ActionBudget,
    *,
    allowed_kinds: tuple[str, ...],
) -> tuple[CandidateObject, ...]:
    selected: list[CandidateObject] = []
    selected_object_ids: set[str] = set()
    selected_bytes = 0
    max_selected_candidates = action_budget.max_selected_candidates or action_budget.max_selected_objects
    for candidate in candidates:
        if candidate.candidate_kind not in allowed_kinds:
            continue
        if candidate.object_id in selected_object_ids:
            continue
        if len(selected) >= max_selected_candidates:
            break
        if len(selected_object_ids) >= action_budget.max_selected_objects:
            break
        if selected_bytes + candidate.size_bytes > action_budget.max_selected_bytes:
            continue
        selected.append(candidate)
        selected_object_ids.add(candidate.object_id)
        selected_bytes += candidate.size_bytes
    return tuple(selected)


def _diagnostic_payload(
    method: Any,
    selected: tuple[CandidateObject, ...],
    *,
    comparator: str,
    utility_by_candidate: Mapping[str, CandidateUtilityEstimate],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    expected_utility = sum(
        utility_by_candidate[candidate.candidate_id].expected_utility
        for candidate in selected
        if candidate.candidate_id in utility_by_candidate
    )
    return {
        "selected_candidate_ids": [candidate.candidate_id for candidate in selected],
        "expected_utility": expected_utility if utility_by_candidate else None,
        "metadata": {
            "diagnostic": {
                "method_id": method.method_id,
                "method_name": method.method_name,
                "comparator": comparator,
                "diagnostic": True,
                "freeze_eligible": False,
                "selected_candidate_kinds": [candidate.candidate_kind for candidate in selected],
                "parameters": _to_payload(parameters),
            }
        },
    }


def _layered_3dgs_sort_key(candidate: CandidateObject) -> tuple[int, str, int, float, int, int, str]:
    return (
        candidate.layer,
        candidate.object_id,
        candidate.resolution.pixel_count,
        candidate.fov_deg,
        candidate.lookahead_ms,
        candidate.expiration_ms,
        candidate.candidate_id,
    )


def _viewport_tile_sort_key(candidate: CandidateObject, target: tuple[int, int]) -> tuple[int, int, int, str, int, str]:
    tile = _require_tile(candidate)
    tile_distance = abs(tile.row - target[0]) + abs(tile.column - target[1])
    return (
        tile_distance,
        tile.row,
        tile.column,
        candidate.object_id,
        candidate.size_bytes,
        candidate.candidate_id,
    )


def _resolve_tile_target(
    candidates: tuple[CandidateObject, ...],
    target_row: int | None,
    target_column: int | None,
) -> tuple[int, int]:
    tiles = tuple(_require_tile(candidate) for candidate in candidates)
    if not tiles:
        return (target_row or 0, target_column or 0)
    max_row = max(tile.rows - 1 for tile in tiles)
    max_column = max(tile.columns - 1 for tile in tiles)
    row = target_row if target_row is not None else max_row // 2
    column = target_column if target_column is not None else max_column // 2
    if row > max_row:
        raise DiagnosticError("target_row must be within the visible tile grid.")
    if column > max_column:
        raise DiagnosticError("target_column must be within the visible tile grid.")
    return (row, column)


def _require_tile(candidate: CandidateObject) -> TileSpec:
    if candidate.tile is None:
        raise DiagnosticError("tile candidates must include tile coordinates.")
    return candidate.tile


def _utility_by_candidate(
    estimates: tuple[CandidateUtilityEstimate, ...],
) -> dict[str, CandidateUtilityEstimate]:
    return {estimate.candidate_id: estimate for estimate in estimates}


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise DiagnosticError(f"{field_name} must be a mapping.")
    return {str(key): _to_payload(item) for key, item in value.items()}


def _to_payload(value: Any) -> Any:
    if hasattr(value, "as_payload"):
        return value.as_payload()
    if isinstance(value, MappingABC):
        return {str(key): _to_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_payload(item) for item in value]
    if isinstance(value, list):
        return [_to_payload(item) for item in value]
    return value


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DiagnosticError(f"{field_name} must be a non-empty string.")


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise DiagnosticError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DiagnosticError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise DiagnosticError(f"{field_name} must be a non-negative integer.")
    return parsed


__all__ = [
    "DiagnosticError",
    "Layered3DGSComparator",
    "ViewportTileComparator",
    "diagnostic_comparators",
]
