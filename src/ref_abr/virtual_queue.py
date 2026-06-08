"""Virtual-queue deadline controller with debt-weighted fallback routing."""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.allocator import KnapsackAllocation, KnapsackBudget, allocate_deadline_aware_knapsack
from ref_abr.candidates import CandidateObject
from ref_abr.config import stable_config_id
from ref_abr.domain import LifecycleStatus
from ref_abr.methods import ActionBudget, SchedulingObservation
from ref_abr.utility import CandidateUtilityEstimate


class VirtualQueueError(ValueError):
    """Raised when virtual-queue controller inputs are invalid."""


@dataclass(frozen=True)
class VirtualQueueConfig:
    """Debt weights and fallback thresholds for the virtual-queue controller."""

    queue_debt_weight: float = 0.35
    transfer_debt_weight: float = 0.20
    lifecycle_debt_weight: float = 0.75
    viewport_debt_weight: float = 0.60
    deadline_debt_weight: float = 0.80
    overload_threshold: float = 1.0
    overload_byte_scale: float = 0.65
    overload_render_scale: float = 0.75
    max_render_ms: float | None = None
    max_compute_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "queue_debt_weight",
            "transfer_debt_weight",
            "lifecycle_debt_weight",
            "viewport_debt_weight",
            "deadline_debt_weight",
        ):
            object.__setattr__(self, field_name, _non_negative_float(getattr(self, field_name), field_name))
        object.__setattr__(self, "overload_threshold", _positive_float(self.overload_threshold, "overload_threshold"))
        object.__setattr__(self, "overload_byte_scale", _unit_interval(self.overload_byte_scale, "overload_byte_scale"))
        object.__setattr__(self, "overload_render_scale", _unit_interval(self.overload_render_scale, "overload_render_scale"))
        if self.overload_byte_scale <= 0.0:
            raise VirtualQueueError("overload_byte_scale must be greater than zero.")
        if self.overload_render_scale <= 0.0:
            raise VirtualQueueError("overload_render_scale must be greater than zero.")
        if self.max_render_ms is not None:
            object.__setattr__(self, "max_render_ms", _non_negative_float(self.max_render_ms, "max_render_ms"))
        if self.max_compute_ms is not None:
            object.__setattr__(self, "max_compute_ms", _non_negative_float(self.max_compute_ms, "max_compute_ms"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "queue_debt_weight": self.queue_debt_weight,
            "transfer_debt_weight": self.transfer_debt_weight,
            "lifecycle_debt_weight": self.lifecycle_debt_weight,
            "viewport_debt_weight": self.viewport_debt_weight,
            "deadline_debt_weight": self.deadline_debt_weight,
            "overload_threshold": self.overload_threshold,
            "overload_byte_scale": self.overload_byte_scale,
            "overload_render_scale": self.overload_render_scale,
            "max_render_ms": self.max_render_ms,
            "max_compute_ms": self.max_compute_ms,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class VirtualQueueDebtState:
    """Virtual debts visible to one scheduling decision."""

    queue_debt_ms: float = 0.0
    transfer_debt_bytes: int = 0
    lifecycle_debt: float = 0.0
    viewport_debt: float = 0.0
    deadline_debt: float = 0.0
    overload_score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "queue_debt_ms", _non_negative_float(self.queue_debt_ms, "queue_debt_ms"))
        object.__setattr__(self, "transfer_debt_bytes", _non_negative_int(self.transfer_debt_bytes, "transfer_debt_bytes"))
        object.__setattr__(self, "lifecycle_debt", _non_negative_float(self.lifecycle_debt, "lifecycle_debt"))
        object.__setattr__(self, "viewport_debt", _non_negative_float(self.viewport_debt, "viewport_debt"))
        object.__setattr__(self, "deadline_debt", _non_negative_float(self.deadline_debt, "deadline_debt"))
        object.__setattr__(self, "overload_score", _non_negative_float(self.overload_score, "overload_score"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def normalized_transfer_debt(self) -> float:
        return self.transfer_debt_bytes / 1_000_000.0

    def weighted_total(self, config: VirtualQueueConfig) -> float:
        return (
            config.queue_debt_weight * (self.queue_debt_ms / 100.0)
            + config.transfer_debt_weight * self.normalized_transfer_debt
            + config.lifecycle_debt_weight * self.lifecycle_debt
            + config.viewport_debt_weight * self.viewport_debt
            + config.deadline_debt_weight * self.deadline_debt
            + self.overload_score
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "queue_debt_ms": self.queue_debt_ms,
            "transfer_debt_bytes": self.transfer_debt_bytes,
            "normalized_transfer_debt": self.normalized_transfer_debt,
            "lifecycle_debt": self.lifecycle_debt,
            "viewport_debt": self.viewport_debt,
            "deadline_debt": self.deadline_debt,
            "overload_score": self.overload_score,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class VirtualQueuePlan:
    """Allocator-backed virtual-queue decision before adapter coercion."""

    plan_id: str
    allocation: KnapsackAllocation
    debt_state: VirtualQueueDebtState
    overload_fallback: bool
    route: str
    candidate_values: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.plan_id, "plan_id")
        if not isinstance(self.allocation, KnapsackAllocation):
            raise VirtualQueueError("allocation must be a KnapsackAllocation record.")
        if not isinstance(self.debt_state, VirtualQueueDebtState):
            raise VirtualQueueError("debt_state must be a VirtualQueueDebtState record.")
        if not isinstance(self.overload_fallback, bool):
            raise VirtualQueueError("overload_fallback must be boolean.")
        _require_non_empty(self.route, "route")
        object.__setattr__(self, "candidate_values", _float_mapping(self.candidate_values, "candidate_values"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def selected_candidate_ids(self) -> tuple[str, ...]:
        return self.allocation.selected_candidate_ids

    def as_payload(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "allocation": self.allocation.as_payload(),
            "debt_state": self.debt_state.as_payload(),
            "overload_fallback": self.overload_fallback,
            "route": self.route,
            "candidate_values": dict(self.candidate_values),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class VirtualQueueDeadlineController:
    """Queue/debt-weighted deadline controller with overload fallback."""

    config: VirtualQueueConfig = field(default_factory=VirtualQueueConfig)
    method_id: str = "virtual-queue-deadline-controller"
    method_name: str = "Virtual-queue deadline controller"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.config, VirtualQueueConfig):
            raise VirtualQueueError("config must be a VirtualQueueConfig record.")
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.method_name, "method_name")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, Any]:
        plan = plan_virtual_queue_deadline(observation, action_budget, config=self.config)
        selected_kinds = [item.candidate.candidate_kind for item in plan.allocation.selected_items]
        return {
            "selected_candidate_ids": list(plan.selected_candidate_ids),
            "expected_utility": plan.allocation.total_value,
            "metadata": {
                "baseline": {
                    "method_id": self.method_id,
                    "method_name": self.method_name,
                    "policy": "virtual_queue_deadline",
                    "selected_candidate_kinds": selected_kinds,
                    "parameters": {
                        **self.config.as_payload(),
                        **_to_payload(self.metadata),
                    },
                },
                "virtual_queue": plan.as_payload(),
            },
        }


def plan_virtual_queue_deadline(
    observation: SchedulingObservation,
    action_budget: ActionBudget,
    *,
    config: VirtualQueueConfig | None = None,
) -> VirtualQueuePlan:
    """Plan one queue/debt-weighted deadline decision."""

    if not isinstance(observation, SchedulingObservation):
        raise VirtualQueueError("observation must be a SchedulingObservation record.")
    if not isinstance(action_budget, ActionBudget):
        raise VirtualQueueError("action_budget must be an ActionBudget record.")
    config = config or VirtualQueueConfig()
    if not isinstance(config, VirtualQueueConfig):
        raise VirtualQueueError("config must be a VirtualQueueConfig record.")

    debt_state = _debt_state(observation)
    overload_fallback = debt_state.weighted_total(config) >= config.overload_threshold
    candidates = _fallback_candidates(observation.candidates) if overload_fallback else observation.candidates
    if not candidates:
        candidates = observation.candidates
    budget = _budget(observation, action_budget, config, debt_state, overload_fallback)
    utility_by_candidate = {estimate.candidate_id: estimate for estimate in observation.utility_estimates}
    values = _candidate_values(candidates, utility_by_candidate, observation, config, debt_state, overload_fallback)
    allocation = allocate_deadline_aware_knapsack(
        candidates,
        budget=budget,
        utility_estimates=tuple(estimate for estimate in observation.utility_estimates if estimate.candidate_id in values),
        values=values,
    )
    route = "overload_fallback" if overload_fallback else "debt_weighted"
    payload = {
        "observation_id": observation.observation_id,
        "candidate_set_id": observation.candidate_set.candidate_set_id,
        "selected_candidate_ids": list(allocation.selected_candidate_ids),
        "debt_state": debt_state.as_payload(),
        "route": route,
        "budget": budget.as_payload(),
    }
    return VirtualQueuePlan(
        plan_id=f"virtual-queue-plan-{stable_config_id(payload)}",
        allocation=allocation,
        debt_state=debt_state,
        overload_fallback=overload_fallback,
        route=route,
        candidate_values=values,
        metadata={
            "algorithm": "virtual_queue_deadline_allocator",
            "budget": budget.as_payload(),
            "candidate_count": len(candidates),
            "input_candidate_count": len(observation.candidates),
            "weighted_debt_total": debt_state.weighted_total(config),
        },
    )


def _debt_state(observation: SchedulingObservation) -> VirtualQueueDebtState:
    metadata = observation.metadata.get("virtual_queue")
    metadata = metadata if isinstance(metadata, MappingABC) else {}
    controller_state = observation.controller_state.state.get("virtual_queue")
    controller_state = controller_state if isinstance(controller_state, MappingABC) else {}
    queue_debt_ms = _first_present(metadata, controller_state, ("queue_debt_ms", "queue_ms")) or 0.0
    transfer_debt_bytes = _first_present(metadata, controller_state, ("transfer_debt_bytes", "queued_bytes")) or 0
    viewport_debt = _first_present(metadata, controller_state, ("viewport_debt", "viewport_risk")) or 0.0
    overload_score = _first_present(metadata, controller_state, ("overload_score", "overload")) or 0.0
    lifecycle_debt = _first_present(metadata, controller_state, ("lifecycle_debt", "lifecycle_risk"))
    if lifecycle_debt is None:
        lifecycle_debt = _lifecycle_debt(observation)
    deadline_debt = _first_present(metadata, controller_state, ("deadline_debt", "deadline_pressure"))
    if deadline_debt is None:
        deadline_debt = _deadline_debt(observation)
    return VirtualQueueDebtState(
        queue_debt_ms=queue_debt_ms,
        transfer_debt_bytes=transfer_debt_bytes,
        lifecycle_debt=lifecycle_debt,
        viewport_debt=viewport_debt,
        deadline_debt=deadline_debt,
        overload_score=overload_score,
        metadata={
            "source": "observation_metadata_and_controller_state",
            "observation_id": observation.observation_id,
        },
    )


def _budget(
    observation: SchedulingObservation,
    action_budget: ActionBudget,
    config: VirtualQueueConfig,
    debt_state: VirtualQueueDebtState,
    overload_fallback: bool,
) -> KnapsackBudget:
    max_bytes = action_budget.max_selected_bytes
    max_render_ms = config.max_render_ms
    if overload_fallback:
        max_bytes = max(0, int(math.floor(max_bytes * config.overload_byte_scale)))
        if max_render_ms is not None:
            max_render_ms *= config.overload_render_scale
    return KnapsackBudget(
        max_bytes=max_bytes,
        max_render_ms=max_render_ms,
        max_compute_ms=config.max_compute_ms,
        max_deadline_ms=action_budget.max_deadline_ms or observation.target_deadline_ms,
        max_selected_objects=action_budget.max_selected_objects,
        max_selected_candidates=action_budget.max_selected_candidates or action_budget.max_selected_objects,
        satisfied_dependencies=_satisfied_dependencies(observation),
        metadata={
            "source": "virtual_queue_deadline",
            "observation_id": observation.observation_id,
            "overload_fallback": overload_fallback,
            "debt_state": debt_state.as_payload(),
        },
    )


def _candidate_values(
    candidates: Sequence[CandidateObject],
    utility_by_candidate: Mapping[str, CandidateUtilityEstimate],
    observation: SchedulingObservation,
    config: VirtualQueueConfig,
    debt_state: VirtualQueueDebtState,
    overload_fallback: bool,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for candidate in candidates:
        base_value = _candidate_base_value(candidate, utility_by_candidate)
        debt_cost = _candidate_debt_cost(candidate, utility_by_candidate, observation, config, debt_state)
        fallback_bonus = _fallback_bonus(candidate, overload_fallback)
        values[candidate.candidate_id] = base_value - debt_cost + fallback_bonus
    return values


def _candidate_debt_cost(
    candidate: CandidateObject,
    utility_by_candidate: Mapping[str, CandidateUtilityEstimate],
    observation: SchedulingObservation,
    config: VirtualQueueConfig,
    debt_state: VirtualQueueDebtState,
) -> float:
    estimate = utility_by_candidate.get(candidate.candidate_id)
    lifecycle_risk = estimate.lifecycle_risk if estimate is not None else _candidate_virtual_queue_float(candidate, "lifecycle_debt", 0.0)
    viewport_sensitivity = _candidate_virtual_queue_float(candidate, "viewport_sensitivity", _default_viewport_sensitivity(candidate))
    bytes_ratio = candidate.size_bytes / _byte_normalizer(observation)
    queue_pressure = debt_state.queue_debt_ms / max(1.0, observation.target_deadline_ms - observation.decision_time_ms)
    transfer_pressure = debt_state.normalized_transfer_debt * bytes_ratio
    deadline_pressure = _candidate_deadline_pressure(candidate, observation) + debt_state.deadline_debt
    return (
        config.queue_debt_weight * queue_pressure * (bytes_ratio + 0.25)
        + config.transfer_debt_weight * transfer_pressure
        + config.lifecycle_debt_weight * (debt_state.lifecycle_debt + lifecycle_risk)
        + config.viewport_debt_weight * debt_state.viewport_debt * viewport_sensitivity
        + config.deadline_debt_weight * deadline_pressure
    )


def _candidate_base_value(
    candidate: CandidateObject,
    utility_by_candidate: Mapping[str, CandidateUtilityEstimate],
) -> float:
    estimate = utility_by_candidate.get(candidate.candidate_id)
    if estimate is not None:
        return estimate.expected_utility
    allocator = _candidate_allocator_metadata(candidate)
    for key in ("value", "utility", "expected_utility"):
        raw_value = allocator.get(key)
        if raw_value is not None:
            return _finite_float(raw_value, f"candidate.{candidate.candidate_id}.{key}")
    substrate = candidate.metadata.get("substrate")
    if isinstance(substrate, MappingABC) and substrate.get("visible_quality") is not None:
        return _finite_float(substrate["visible_quality"], f"candidate.{candidate.candidate_id}.visible_quality")
    return max(0.0, 0.05 + 0.05 * candidate.layer + candidate.resolution.megapixels / 20.0)


def _candidate_deadline_pressure(candidate: CandidateObject, observation: SchedulingObservation) -> float:
    horizon_ms = max(1.0, float(observation.target_deadline_ms - observation.decision_time_ms))
    slack_ms = max(0.0, float(candidate.deadline_ms - observation.decision_time_ms))
    return max(0.0, 1.0 - min(1.0, slack_ms / horizon_ms))


def _fallback_candidates(candidates: Sequence[CandidateObject]) -> tuple[CandidateObject, ...]:
    preferred = tuple(candidate for candidate in candidates if candidate.candidate_kind in {"gaussian_base", "tile"})
    if preferred:
        return preferred
    return tuple(sorted(candidates, key=lambda candidate: (candidate.size_bytes, candidate.deadline_ms, candidate.candidate_id))[:1])


def _fallback_bonus(candidate: CandidateObject, overload_fallback: bool) -> float:
    if not overload_fallback:
        return 0.0
    if candidate.candidate_kind == "gaussian_base":
        return 0.35
    if candidate.candidate_kind == "tile":
        return 0.20
    return -0.50


def _lifecycle_debt(observation: SchedulingObservation) -> float:
    if not observation.lifecycle_states:
        return 0.0
    debt = 0.0
    for state in observation.lifecycle_states:
        if state.status in {LifecycleStatus.EXPIRED, LifecycleStatus.DROPPED}:
            debt += 1.0
        elif state.deadline_ms is not None and state.deadline_ms <= observation.target_deadline_ms:
            debt += 0.5
        debt += min(1.0, state.attempts / 4.0)
    return min(1.0, debt / max(1, len(observation.lifecycle_states)))


def _deadline_debt(observation: SchedulingObservation) -> float:
    horizon_ms = max(1.0, float(observation.target_deadline_ms - observation.decision_time_ms))
    urgent_count = sum(1 for candidate in observation.candidates if candidate.deadline_ms <= observation.decision_time_ms + 0.5 * horizon_ms)
    return min(1.0, urgent_count / max(1, len(observation.candidates)))


def _satisfied_dependencies(observation: SchedulingObservation) -> tuple[str, ...]:
    active_statuses = {LifecycleStatus.REQUESTED, LifecycleStatus.IN_FLIGHT, LifecycleStatus.AVAILABLE}
    satisfied: list[str] = []
    seen: set[str] = set()
    for state in observation.lifecycle_states:
        if state.status in active_statuses and state.reference_id not in seen:
            satisfied.append(state.reference_id)
            seen.add(state.reference_id)
    return tuple(satisfied)


def _candidate_allocator_metadata(candidate: CandidateObject) -> Mapping[str, Any]:
    allocator = candidate.metadata.get("allocator")
    if isinstance(allocator, MappingABC):
        return allocator
    allocator_cost = candidate.metadata.get("allocator_cost")
    if isinstance(allocator_cost, MappingABC):
        return allocator_cost
    return {}


def _candidate_virtual_queue_float(candidate: CandidateObject, key: str, default: float) -> float:
    metadata = candidate.metadata.get("virtual_queue")
    if isinstance(metadata, MappingABC) and key in metadata:
        return _non_negative_float(metadata[key], f"candidate.metadata.virtual_queue.{key}")
    mpc_metadata = candidate.metadata.get("mpc")
    if isinstance(mpc_metadata, MappingABC) and key in mpc_metadata:
        return _non_negative_float(mpc_metadata[key], f"candidate.metadata.mpc.{key}")
    return default


def _default_viewport_sensitivity(candidate: CandidateObject) -> float:
    if candidate.candidate_kind == "reference_action":
        return 0.70
    if candidate.candidate_kind == "tile":
        return 0.45
    if candidate.candidate_kind == "gaussian_enhancement":
        return 0.30
    return 0.10


def _byte_normalizer(observation: SchedulingObservation) -> float:
    raw_value = observation.metadata.get("byte_normalizer", 1_000_000)
    parsed = _positive_float(raw_value, "metadata.byte_normalizer")
    return max(1.0, parsed)


def _first_present(first: Mapping[str, Any], second: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in first:
            return first[name]
        if name in second:
            return second[name]
    return None


def _float_mapping(value: Mapping[str, float], field_name: str) -> dict[str, float]:
    if not isinstance(value, MappingABC):
        raise VirtualQueueError(f"{field_name} must be a mapping.")
    return {str(key): _finite_float(item, f"{field_name}.{key}") for key, item in value.items()}


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise VirtualQueueError(f"{field_name} must be a mapping.")
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
        raise VirtualQueueError(f"{field_name} must be a non-empty string.")


def _positive_float(value: Any, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name)
    if parsed <= 0.0:
        raise VirtualQueueError(f"{field_name} must be positive.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name)
    if parsed > 1.0:
        raise VirtualQueueError(f"{field_name} must be in [0, 1].")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise VirtualQueueError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise VirtualQueueError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise VirtualQueueError(f"{field_name} must be non-negative.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0:
        raise VirtualQueueError(f"{field_name} must be non-negative.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise VirtualQueueError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise VirtualQueueError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise VirtualQueueError(f"{field_name} must be finite.")
    return parsed


__all__ = [
    "VirtualQueueConfig",
    "VirtualQueueDeadlineController",
    "VirtualQueueDebtState",
    "VirtualQueueError",
    "VirtualQueuePlan",
    "plan_virtual_queue_deadline",
]
