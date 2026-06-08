"""Deadline-aware knapsack allocation for heterogeneous candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.candidates import CandidateObject
from ref_abr.config import stable_config_id
from ref_abr.methods import ActionBudget, SchedulingObservation
from ref_abr.utility import CandidateUtilityEstimate


class AllocatorError(ValueError):
    """Raised when allocator inputs are malformed or infeasible."""


@dataclass(frozen=True)
class KnapsackBudget:
    """Budgets enforced by the deadline-aware allocator."""

    max_bytes: int
    max_render_ms: float | None = None
    max_compute_ms: float | None = None
    max_deadline_ms: int | None = None
    max_selected_objects: int | None = None
    max_selected_candidates: int | None = None
    satisfied_dependencies: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_bytes", _non_negative_int(self.max_bytes, "max_bytes"))
        if self.max_render_ms is not None:
            object.__setattr__(self, "max_render_ms", _non_negative_float(self.max_render_ms, "max_render_ms"))
        if self.max_compute_ms is not None:
            object.__setattr__(self, "max_compute_ms", _non_negative_float(self.max_compute_ms, "max_compute_ms"))
        if self.max_deadline_ms is not None:
            object.__setattr__(self, "max_deadline_ms", _non_negative_int(self.max_deadline_ms, "max_deadline_ms"))
        if self.max_selected_objects is not None:
            object.__setattr__(self, "max_selected_objects", _positive_int(self.max_selected_objects, "max_selected_objects"))
        if self.max_selected_candidates is not None:
            object.__setattr__(self, "max_selected_candidates", _positive_int(self.max_selected_candidates, "max_selected_candidates"))
        object.__setattr__(self, "satisfied_dependencies", _string_tuple(self.satisfied_dependencies, "satisfied_dependencies", allow_empty=True))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @classmethod
    def from_action_budget(
        cls,
        action_budget: ActionBudget,
        observation: SchedulingObservation,
        *,
        max_render_ms: float | None = None,
        max_compute_ms: float | None = None,
        satisfied_dependencies: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "KnapsackBudget":
        """Build allocator budgets from the shared method action budget."""

        if not isinstance(action_budget, ActionBudget):
            raise AllocatorError("action_budget must be an ActionBudget record.")
        if not isinstance(observation, SchedulingObservation):
            raise AllocatorError("observation must be a SchedulingObservation record.")
        metadata_budget = _allocator_budget_metadata(observation.metadata)
        render_budget = max_render_ms if max_render_ms is not None else _first_present(metadata_budget, ("max_render_ms", "render_ms"))
        compute_budget = max_compute_ms if max_compute_ms is not None else _first_present(metadata_budget, ("max_compute_ms", "compute_ms"))
        max_deadline = action_budget.max_deadline_ms if action_budget.max_deadline_ms is not None else observation.target_deadline_ms
        return cls(
            max_bytes=action_budget.max_selected_bytes,
            max_render_ms=render_budget,
            max_compute_ms=compute_budget,
            max_deadline_ms=max_deadline,
            max_selected_objects=action_budget.max_selected_objects,
            max_selected_candidates=action_budget.max_selected_candidates or action_budget.max_selected_objects,
            satisfied_dependencies=tuple(satisfied_dependencies),
            metadata={
                "source": "action_budget",
                "action_budget": action_budget.as_payload(),
                "observation_id": observation.observation_id,
                **_plain_json_mapping(metadata, "metadata"),
            },
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "max_bytes": self.max_bytes,
            "max_render_ms": self.max_render_ms,
            "max_compute_ms": self.max_compute_ms,
            "max_deadline_ms": self.max_deadline_ms,
            "max_selected_objects": self.max_selected_objects,
            "max_selected_candidates": self.max_selected_candidates,
            "satisfied_dependencies": list(self.satisfied_dependencies),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class CandidateResourceCost:
    """Allocator resource costs for one candidate."""

    bytes: int
    render_ms: float
    compute_ms: float
    deadline_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bytes", _non_negative_int(self.bytes, "bytes"))
        object.__setattr__(self, "render_ms", _non_negative_float(self.render_ms, "render_ms"))
        object.__setattr__(self, "compute_ms", _non_negative_float(self.compute_ms, "compute_ms"))
        object.__setattr__(self, "deadline_ms", _non_negative_int(self.deadline_ms, "deadline_ms"))

    def as_payload(self) -> dict[str, float | int]:
        return {
            "bytes": self.bytes,
            "render_ms": self.render_ms,
            "compute_ms": self.compute_ms,
            "deadline_ms": self.deadline_ms,
        }


@dataclass(frozen=True)
class AllocationItem:
    """Candidate plus scalar value and resource cost used by the allocator."""

    candidate: CandidateObject
    value: float
    cost: CandidateResourceCost
    dependency_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, CandidateObject):
            raise AllocatorError("candidate must be a CandidateObject record.")
        object.__setattr__(self, "value", _finite_float(self.value, "value"))
        if not isinstance(self.cost, CandidateResourceCost):
            raise AllocatorError("cost must be a CandidateResourceCost record.")
        object.__setattr__(self, "dependency_ids", _string_tuple(self.dependency_ids, "dependency_ids", allow_empty=True))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def object_id(self) -> str:
        return self.candidate.object_id

    def as_payload(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.as_payload(),
            "value": self.value,
            "cost": self.cost.as_payload(),
            "dependency_ids": list(self.dependency_ids),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class KnapsackAllocation:
    """Allocator result with selected items and feasibility diagnostics."""

    allocation_id: str
    selected_items: tuple[AllocationItem, ...]
    budget: KnapsackBudget
    total_value: float
    total_bytes: int
    total_render_ms: float
    total_compute_ms: float
    infeasible_reasons: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.allocation_id, "allocation_id")
        selected_items = tuple(self.selected_items)
        for item in selected_items:
            if not isinstance(item, AllocationItem):
                raise AllocatorError("selected_items must contain AllocationItem records.")
        if not isinstance(self.budget, KnapsackBudget):
            raise AllocatorError("budget must be a KnapsackBudget record.")
        object.__setattr__(self, "selected_items", selected_items)
        object.__setattr__(self, "total_value", _finite_float(self.total_value, "total_value"))
        object.__setattr__(self, "total_bytes", _non_negative_int(self.total_bytes, "total_bytes"))
        object.__setattr__(self, "total_render_ms", _non_negative_float(self.total_render_ms, "total_render_ms"))
        object.__setattr__(self, "total_compute_ms", _non_negative_float(self.total_compute_ms, "total_compute_ms"))
        object.__setattr__(self, "infeasible_reasons", _reason_mapping(self.infeasible_reasons))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def selected_candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.selected_items)

    @property
    def selected_object_ids(self) -> tuple[str, ...]:
        return tuple(item.object_id for item in self.selected_items)

    def as_payload(self) -> dict[str, Any]:
        return {
            "allocation_id": self.allocation_id,
            "selected_items": [item.as_payload() for item in self.selected_items],
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selected_object_ids": list(self.selected_object_ids),
            "budget": self.budget.as_payload(),
            "total_value": self.total_value,
            "total_bytes": self.total_bytes,
            "total_render_ms": self.total_render_ms,
            "total_compute_ms": self.total_compute_ms,
            "infeasible_reasons": {key: list(value) for key, value in self.infeasible_reasons.items()},
            "metadata": _to_payload(self.metadata),
        }


def allocate_deadline_aware_knapsack(
    candidates: Sequence[CandidateObject],
    *,
    budget: KnapsackBudget,
    utility_estimates: Sequence[CandidateUtilityEstimate] = (),
    values: Mapping[str, float] | None = None,
) -> KnapsackAllocation:
    """Select a deterministic best-value feasible candidate subset."""

    candidate_tuple = _candidate_tuple(candidates)
    if not isinstance(budget, KnapsackBudget):
        raise AllocatorError("budget must be a KnapsackBudget record.")
    utility_by_candidate = _utility_by_candidate(utility_estimates, candidate_tuple)
    value_overrides = _float_mapping(values, "values") if values is not None else {}
    items = tuple(_allocation_item(candidate, utility_by_candidate, value_overrides) for candidate in candidate_tuple)
    infeasible_reasons = {item.candidate_id: _single_item_infeasible_reasons(item, budget) for item in items}
    selected = _search_best_selection(items, budget)
    totals = _totals(selected)
    payload = {
        "selected_candidate_ids": [item.candidate_id for item in selected],
        "budget": budget.as_payload(),
        "totals": totals,
        "candidate_count": len(items),
    }
    return KnapsackAllocation(
        allocation_id=f"knapsack-allocation-{stable_config_id(payload)}",
        selected_items=selected,
        budget=budget,
        total_value=totals["value"],
        total_bytes=int(totals["bytes"]),
        total_render_ms=totals["render_ms"],
        total_compute_ms=totals["compute_ms"],
        infeasible_reasons={key: value for key, value in infeasible_reasons.items() if value},
        metadata={
            "algorithm": "deadline_aware_dependency_closed_enumeration",
            "candidate_count": len(items),
            "considered_item_count": sum(1 for reasons in infeasible_reasons.values() if not reasons),
        },
    )


def allocate_from_observation(
    observation: SchedulingObservation,
    action_budget: ActionBudget,
    *,
    max_render_ms: float | None = None,
    max_compute_ms: float | None = None,
    satisfied_dependencies: Sequence[str] = (),
) -> KnapsackAllocation:
    """Run the standalone allocator from a scheduling observation and action budget."""

    budget = KnapsackBudget.from_action_budget(
        action_budget,
        observation,
        max_render_ms=max_render_ms,
        max_compute_ms=max_compute_ms,
        satisfied_dependencies=satisfied_dependencies,
    )
    return allocate_deadline_aware_knapsack(
        observation.candidates,
        budget=budget,
        utility_estimates=observation.utility_estimates,
    )


def decision_payload_from_allocation(
    allocation: KnapsackAllocation,
    *,
    method_id: str,
    method_name: str,
    policy: str = "deadline_aware_knapsack",
) -> dict[str, Any]:
    """Return a method-adapter-compatible decision payload."""

    if not isinstance(allocation, KnapsackAllocation):
        raise AllocatorError("allocation must be a KnapsackAllocation record.")
    _require_non_empty(method_id, "method_id")
    _require_non_empty(method_name, "method_name")
    return {
        "selected_candidate_ids": list(allocation.selected_candidate_ids),
        "expected_utility": allocation.total_value,
        "metadata": {
            "baseline": {
                "method_id": method_id,
                "method_name": method_name,
                "policy": policy,
                "selected_candidate_kinds": [item.candidate.candidate_kind for item in allocation.selected_items],
            },
            "allocation": allocation.as_payload(),
        },
    }


def _search_best_selection(items: tuple[AllocationItem, ...], budget: KnapsackBudget) -> tuple[AllocationItem, ...]:
    ranked = tuple(sorted(items, key=_item_rank_key))
    max_selected_candidates = budget.max_selected_candidates or len(ranked)
    best: tuple[AllocationItem, ...] = ()
    best_key = _selection_key(best)

    def visit(index: int, selected: tuple[AllocationItem, ...]) -> None:
        nonlocal best, best_key
        if _selection_feasible(selected, budget):
            key = _selection_key(selected)
            if key < best_key:
                best = selected
                best_key = key
        if index >= len(ranked) or len(selected) >= max_selected_candidates:
            return
        remaining_positive = sum(max(0.0, item.value) for item in ranked[index:])
        if -_selection_value(selected) - remaining_positive > best_key[0]:
            return
        for next_index in range(index, len(ranked)):
            candidate = ranked[next_index]
            next_selected = (*selected, candidate)
            if _partial_budget_exceeded(next_selected, budget):
                continue
            visit(next_index + 1, next_selected)

    visit(0, ())
    return tuple(sorted(best, key=lambda item: item.candidate_id))


def _selection_feasible(selected: tuple[AllocationItem, ...], budget: KnapsackBudget) -> bool:
    if _partial_budget_exceeded(selected, budget):
        return False
    selected_candidate_ids = {item.candidate_id for item in selected}
    selected_object_ids = {item.object_id for item in selected}
    satisfied = set(budget.satisfied_dependencies)
    available = selected_candidate_ids | selected_object_ids | satisfied
    for item in selected:
        if any(dependency not in available for dependency in item.dependency_ids):
            return False
    return True


def _partial_budget_exceeded(selected: tuple[AllocationItem, ...], budget: KnapsackBudget) -> bool:
    totals = _totals(selected)
    if totals["bytes"] > budget.max_bytes:
        return True
    if budget.max_render_ms is not None and totals["render_ms"] > budget.max_render_ms:
        return True
    if budget.max_compute_ms is not None and totals["compute_ms"] > budget.max_compute_ms:
        return True
    if budget.max_selected_objects is not None and len({item.object_id for item in selected}) > budget.max_selected_objects:
        return True
    if budget.max_selected_candidates is not None and len(selected) > budget.max_selected_candidates:
        return True
    if budget.max_deadline_ms is not None and any(item.cost.deadline_ms > budget.max_deadline_ms for item in selected):
        return True
    return False


def _single_item_infeasible_reasons(item: AllocationItem, budget: KnapsackBudget) -> tuple[str, ...]:
    reasons: list[str] = []
    if item.cost.bytes > budget.max_bytes:
        reasons.append("byte_budget")
    if budget.max_render_ms is not None and item.cost.render_ms > budget.max_render_ms:
        reasons.append("render_budget")
    if budget.max_compute_ms is not None and item.cost.compute_ms > budget.max_compute_ms:
        reasons.append("compute_budget")
    if budget.max_deadline_ms is not None and item.cost.deadline_ms > budget.max_deadline_ms:
        reasons.append("deadline_budget")
    if item.dependency_ids and not set(item.dependency_ids).issubset(set(budget.satisfied_dependencies)):
        reasons.append("dependency_closure_required")
    return tuple(reasons)


def _allocation_item(
    candidate: CandidateObject,
    utility_by_candidate: Mapping[str, CandidateUtilityEstimate],
    value_overrides: Mapping[str, float],
) -> AllocationItem:
    estimate = utility_by_candidate.get(candidate.candidate_id)
    value = value_overrides.get(candidate.candidate_id)
    value_source = "override"
    if value is None and estimate is not None:
        value = estimate.expected_utility - estimate.deadline_miss_probability - 0.25 * estimate.lifecycle_risk
        value_source = "utility_estimate"
    if value is None:
        value = _metadata_value(candidate)
        value_source = "candidate_metadata"
    cost = _candidate_cost(candidate)
    return AllocationItem(
        candidate=candidate,
        value=value,
        cost=cost,
        dependency_ids=candidate.dependencies,
        metadata={"value_source": value_source},
    )


def _candidate_cost(candidate: CandidateObject) -> CandidateResourceCost:
    allocator_metadata = _allocator_cost_metadata(candidate.metadata)
    substrate = candidate.metadata.get("substrate")
    timing = substrate.get("component_timing") if isinstance(substrate, MappingABC) else None
    timing = timing if isinstance(timing, MappingABC) else {}
    generation_ms = _first_present(timing, ("generation_ms", "generate_ms")) or 0.0
    restoration_ms = _first_present(timing, ("restoration_ms", "restore_ms")) or 0.0
    render_ms = _first_present(allocator_metadata, ("render_ms",)) if allocator_metadata else None
    compute_ms = _first_present(allocator_metadata, ("compute_ms",)) if allocator_metadata else None
    if render_ms is None:
        render_ms = _first_present(timing, ("render_ms", "rendering_ms")) or 0.0
    if compute_ms is None:
        compute_ms = float(generation_ms) + float(restoration_ms)
    deadline_ms = _first_present(allocator_metadata, ("deadline_ms",)) if allocator_metadata else None
    return CandidateResourceCost(
        bytes=candidate.size_bytes,
        render_ms=render_ms,
        compute_ms=compute_ms,
        deadline_ms=deadline_ms if deadline_ms is not None else candidate.deadline_ms,
    )


def _metadata_value(candidate: CandidateObject) -> float:
    allocator_metadata = _allocator_cost_metadata(candidate.metadata)
    raw_value = _first_present(allocator_metadata, ("value", "utility", "expected_utility")) if allocator_metadata else None
    if raw_value is not None:
        return _finite_float(raw_value, "candidate.metadata.allocator.value")
    substrate = candidate.metadata.get("substrate")
    if isinstance(substrate, MappingABC):
        visible_quality = substrate.get("visible_quality")
        if visible_quality is not None:
            return _finite_float(visible_quality, "candidate.metadata.substrate.visible_quality")
    return max(0.0, 0.05 + 0.05 * candidate.layer + candidate.resolution.megapixels / 20.0)


def _totals(selected: tuple[AllocationItem, ...]) -> dict[str, float]:
    return {
        "value": _selection_value(selected),
        "bytes": float(sum(item.cost.bytes for item in selected)),
        "render_ms": sum(item.cost.render_ms for item in selected),
        "compute_ms": sum(item.cost.compute_ms for item in selected),
    }


def _selection_value(selected: tuple[AllocationItem, ...]) -> float:
    return sum(item.value for item in selected)


def _selection_key(selected: tuple[AllocationItem, ...]) -> tuple[float, int, float, float, tuple[str, ...]]:
    totals = _totals(selected)
    return (
        -totals["value"],
        int(totals["bytes"]),
        totals["render_ms"],
        totals["compute_ms"],
        tuple(item.candidate_id for item in sorted(selected, key=lambda item: item.candidate_id)),
    )


def _item_rank_key(item: AllocationItem) -> tuple[float, int, int, str]:
    density = item.value / max(1.0, item.cost.bytes + 1000.0 * item.cost.render_ms + 1000.0 * item.cost.compute_ms)
    return (-density, item.cost.deadline_ms, item.cost.bytes, item.candidate_id)


def _candidate_tuple(values: Sequence[CandidateObject]) -> tuple[CandidateObject, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise AllocatorError("candidates must be a sequence of CandidateObject records.")
    parsed = tuple(values)
    if not parsed:
        raise AllocatorError("candidates must not be empty.")
    ids = [candidate.candidate_id for candidate in parsed]
    duplicates = sorted({candidate_id for candidate_id in ids if ids.count(candidate_id) > 1})
    if duplicates:
        raise AllocatorError(f"candidates must not contain duplicate candidate_id values: {', '.join(duplicates)}.")
    for candidate in parsed:
        if not isinstance(candidate, CandidateObject):
            raise AllocatorError("candidates must contain CandidateObject records.")
    return parsed


def _utility_by_candidate(
    estimates: Sequence[CandidateUtilityEstimate],
    candidates: tuple[CandidateObject, ...],
) -> dict[str, CandidateUtilityEstimate]:
    if isinstance(estimates, (str, bytes)) or not isinstance(estimates, Sequence):
        raise AllocatorError("utility_estimates must be a sequence of CandidateUtilityEstimate records.")
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    parsed: dict[str, CandidateUtilityEstimate] = {}
    for estimate in estimates:
        if not isinstance(estimate, CandidateUtilityEstimate):
            raise AllocatorError("utility_estimates must contain CandidateUtilityEstimate records.")
        if estimate.candidate_id not in candidate_ids:
            raise AllocatorError(f"utility_estimates references unknown candidate_id {estimate.candidate_id!r}.")
        parsed[estimate.candidate_id] = estimate
    return parsed


def _allocator_cost_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    allocator = metadata.get("allocator")
    if isinstance(allocator, MappingABC):
        return allocator
    allocator_cost = metadata.get("allocator_cost")
    if isinstance(allocator_cost, MappingABC):
        return allocator_cost
    return {}


def _allocator_budget_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    allocator = metadata.get("allocator_budget")
    if isinstance(allocator, MappingABC):
        return allocator
    return {}


def _reason_mapping(value: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, MappingABC):
        raise AllocatorError("infeasible_reasons must be a mapping.")
    return {str(key): _string_tuple(reasons, f"infeasible_reasons.{key}", allow_empty=True) for key, reasons in value.items()}


def _float_mapping(value: Mapping[str, float], field_name: str) -> dict[str, float]:
    if not isinstance(value, MappingABC):
        raise AllocatorError(f"{field_name} must be a mapping.")
    return {str(key): _finite_float(item, f"{field_name}.{key}") for key, item in value.items()}


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise AllocatorError(f"{field_name} must be a mapping.")
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


def _first_present(mapping: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _string_tuple(values: Sequence[str], field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise AllocatorError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    if not parsed and not allow_empty:
        raise AllocatorError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise AllocatorError(f"{field_name} must be a non-empty string.")


def _positive_int(value: Any, field_name: str) -> int:
    parsed = _non_negative_int(value, field_name)
    if parsed <= 0:
        raise AllocatorError(f"{field_name} must be positive.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise AllocatorError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise AllocatorError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise AllocatorError(f"{field_name} must be non-negative.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0:
        raise AllocatorError(f"{field_name} must be non-negative.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise AllocatorError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AllocatorError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise AllocatorError(f"{field_name} must be finite.")
    return parsed


__all__ = [
    "AllocationItem",
    "AllocatorError",
    "CandidateResourceCost",
    "KnapsackAllocation",
    "KnapsackBudget",
    "allocate_deadline_aware_knapsack",
    "allocate_from_observation",
    "decision_payload_from_allocation",
]
