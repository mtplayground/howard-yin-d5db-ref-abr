"""Robust deadline-aware MPC scheduling backbone."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.allocator import (
    KnapsackAllocation,
    KnapsackBudget,
    allocate_deadline_aware_knapsack,
)
from ref_abr.candidates import CandidateObject
from ref_abr.config import stable_config_id
from ref_abr.domain import LifecycleStatus
from ref_abr.methods import ActionBudget, SchedulingObservation
from ref_abr.utility import CandidateUtilityEstimate


class MPCError(ValueError):
    """Raised when robust MPC inputs or configuration are invalid."""


@dataclass(frozen=True)
class RobustInterval:
    """Closed interval used for robust bandwidth, viewport, and deadline sweeps."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = _finite_float(self.lower, "lower")
        upper = _finite_float(self.upper, "upper")
        if lower > upper:
            raise MPCError("interval lower must be less than or equal to upper.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2.0

    def samples(self) -> tuple[float, ...]:
        values = (self.lower, self.midpoint, self.upper)
        deduplicated: list[float] = []
        for value in values:
            if value not in deduplicated:
                deduplicated.append(value)
        return tuple(deduplicated)

    def as_payload(self) -> dict[str, float]:
        return {"lower": self.lower, "upper": self.upper}


@dataclass(frozen=True)
class MPCScenario:
    """One deterministic robust-MPC scenario sampled from uncertainty intervals."""

    scenario_id: str
    bandwidth_scale: float
    viewport_error: float
    deadline_scale: float
    probability: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.scenario_id, "scenario_id")
        object.__setattr__(self, "bandwidth_scale", _non_negative_float(self.bandwidth_scale, "bandwidth_scale"))
        object.__setattr__(self, "viewport_error", _non_negative_float(self.viewport_error, "viewport_error"))
        object.__setattr__(self, "deadline_scale", _non_negative_float(self.deadline_scale, "deadline_scale"))
        object.__setattr__(self, "probability", _non_negative_float(self.probability, "probability"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "bandwidth_scale": self.bandwidth_scale,
            "viewport_error": self.viewport_error,
            "deadline_scale": self.deadline_scale,
            "probability": self.probability,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class MPCConfig:
    """Configuration for robust deadline-aware MPC over candidate actions."""

    horizon_steps: int = 3
    step_duration_ms: int = 33
    runtime_cap_ms: float = 25.0
    max_scenarios: int = 8
    bandwidth_interval: RobustInterval | tuple[float, float] = (0.75, 1.0)
    viewport_error_interval: RobustInterval | tuple[float, float] = (0.0, 15.0)
    deadline_scale_interval: RobustInterval | tuple[float, float] = (0.75, 1.0)
    robustness_weight: float = 0.5
    drop_penalty: float = 0.25
    prefetch_bonus: float = 0.05
    max_render_ms: float | None = None
    max_compute_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizon_steps", _positive_int(self.horizon_steps, "horizon_steps"))
        object.__setattr__(self, "step_duration_ms", _positive_int(self.step_duration_ms, "step_duration_ms"))
        object.__setattr__(self, "runtime_cap_ms", _non_negative_float(self.runtime_cap_ms, "runtime_cap_ms"))
        object.__setattr__(self, "max_scenarios", _positive_int(self.max_scenarios, "max_scenarios"))
        object.__setattr__(self, "bandwidth_interval", _coerce_interval(self.bandwidth_interval, "bandwidth_interval"))
        object.__setattr__(self, "viewport_error_interval", _coerce_interval(self.viewport_error_interval, "viewport_error_interval"))
        object.__setattr__(self, "deadline_scale_interval", _coerce_interval(self.deadline_scale_interval, "deadline_scale_interval"))
        object.__setattr__(self, "robustness_weight", _non_negative_float(self.robustness_weight, "robustness_weight"))
        object.__setattr__(self, "drop_penalty", _non_negative_float(self.drop_penalty, "drop_penalty"))
        object.__setattr__(self, "prefetch_bonus", _non_negative_float(self.prefetch_bonus, "prefetch_bonus"))
        if self.max_render_ms is not None:
            object.__setattr__(self, "max_render_ms", _non_negative_float(self.max_render_ms, "max_render_ms"))
        if self.max_compute_ms is not None:
            object.__setattr__(self, "max_compute_ms", _non_negative_float(self.max_compute_ms, "max_compute_ms"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "horizon_steps": self.horizon_steps,
            "step_duration_ms": self.step_duration_ms,
            "runtime_cap_ms": self.runtime_cap_ms,
            "max_scenarios": self.max_scenarios,
            "bandwidth_interval": self.bandwidth_interval.as_payload(),
            "viewport_error_interval": self.viewport_error_interval.as_payload(),
            "deadline_scale_interval": self.deadline_scale_interval.as_payload(),
            "robustness_weight": self.robustness_weight,
            "drop_penalty": self.drop_penalty,
            "prefetch_bonus": self.prefetch_bonus,
            "max_render_ms": self.max_render_ms,
            "max_compute_ms": self.max_compute_ms,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class MPCStepPlan:
    """Allocator result for one horizon step and robust scenario."""

    step_index: int
    scenario: MPCScenario
    allocation: KnapsackAllocation
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_index", _non_negative_int(self.step_index, "step_index"))
        if not isinstance(self.scenario, MPCScenario):
            raise MPCError("scenario must be an MPCScenario record.")
        if not isinstance(self.allocation, KnapsackAllocation):
            raise MPCError("allocation must be a KnapsackAllocation record.")
        object.__setattr__(self, "score", _finite_float(self.score, "score"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "scenario": self.scenario.as_payload(),
            "allocation": self.allocation.as_payload(),
            "score": self.score,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class MPCPlan:
    """Robust-MPC decision record before adapter coercion."""

    plan_id: str
    selected_step: MPCStepPlan
    step_plans: tuple[MPCStepPlan, ...]
    runtime_ms: float
    runtime_capped: bool
    robust_score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.plan_id, "plan_id")
        if not isinstance(self.selected_step, MPCStepPlan):
            raise MPCError("selected_step must be an MPCStepPlan record.")
        step_plans = tuple(self.step_plans)
        if not step_plans:
            raise MPCError("step_plans must not be empty.")
        for step_plan in step_plans:
            if not isinstance(step_plan, MPCStepPlan):
                raise MPCError("step_plans must contain MPCStepPlan records.")
        object.__setattr__(self, "step_plans", step_plans)
        object.__setattr__(self, "runtime_ms", _non_negative_float(self.runtime_ms, "runtime_ms"))
        if not isinstance(self.runtime_capped, bool):
            raise MPCError("runtime_capped must be boolean.")
        object.__setattr__(self, "robust_score", _finite_float(self.robust_score, "robust_score"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def selected_candidate_ids(self) -> tuple[str, ...]:
        return self.selected_step.allocation.selected_candidate_ids

    def as_payload(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selected_step": self.selected_step.as_payload(),
            "step_plans": [step_plan.as_payload() for step_plan in self.step_plans],
            "runtime_ms": self.runtime_ms,
            "runtime_capped": self.runtime_capped,
            "robust_score": self.robust_score,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class RobustDeadlineAwareMPCController:
    """Short-horizon robust controller using the deadline-aware allocator as its inner loop."""

    config: MPCConfig = field(default_factory=MPCConfig)
    method_id: str = "robust-deadline-aware-mpc"
    method_name: str = "Robust deadline-aware MPC"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.config, MPCConfig):
            raise MPCError("config must be an MPCConfig record.")
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.method_name, "method_name")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, Any]:
        plan = plan_robust_deadline_aware_mpc(observation, action_budget, config=self.config)
        selected_kinds = [item.candidate.candidate_kind for item in plan.selected_step.allocation.selected_items]
        return {
            "selected_candidate_ids": list(plan.selected_candidate_ids),
            "expected_utility": plan.robust_score,
            "metadata": {
                "baseline": {
                    "method_id": self.method_id,
                    "method_name": self.method_name,
                    "policy": "robust_deadline_aware_mpc",
                    "selected_candidate_kinds": selected_kinds,
                    "parameters": {
                        **self.config.as_payload(),
                        **_to_payload(self.metadata),
                    },
                },
                "mpc": plan.as_payload(),
            },
        }


def plan_robust_deadline_aware_mpc(
    observation: SchedulingObservation,
    action_budget: ActionBudget,
    *,
    config: MPCConfig | None = None,
) -> MPCPlan:
    """Plan one robust deadline-aware MPC action under the shared action budget."""

    if not isinstance(observation, SchedulingObservation):
        raise MPCError("observation must be a SchedulingObservation record.")
    if not isinstance(action_budget, ActionBudget):
        raise MPCError("action_budget must be an ActionBudget record.")
    config = config or MPCConfig()
    if not isinstance(config, MPCConfig):
        raise MPCError("config must be an MPCConfig record.")

    started_at = time.perf_counter()
    scenarios = _scenarios(config, observation)
    utility_by_candidate = {estimate.candidate_id: estimate for estimate in observation.utility_estimates}
    satisfied_dependencies = _satisfied_dependencies(observation)
    step_plans: list[MPCStepPlan] = []
    runtime_capped = False

    for step_index in range(config.horizon_steps):
        for scenario in scenarios:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            if step_plans and elapsed_ms >= config.runtime_cap_ms:
                runtime_capped = True
                break
            budget = _scenario_budget(observation, action_budget, config, scenario)
            values = _scenario_values(observation.candidates, utility_by_candidate, step_index, scenario, config)
            allocation = allocate_deadline_aware_knapsack(
                observation.candidates,
                budget=budget,
                utility_estimates=observation.utility_estimates,
                values=values,
            )
            score_metadata = _score_allocation(allocation, observation, values, step_index, scenario, config)
            step_plans.append(
                MPCStepPlan(
                    step_index=step_index,
                    scenario=scenario,
                    allocation=allocation,
                    score=score_metadata["score"],
                    metadata={
                        "decision_time_ms": observation.decision_time_ms + step_index * config.step_duration_ms,
                        "budget": budget.as_payload(),
                        "satisfied_dependencies": list(satisfied_dependencies),
                        **score_metadata,
                    },
                )
            )
        if runtime_capped:
            break

    if not step_plans:
        scenario = scenarios[0]
        budget = _scenario_budget(observation, action_budget, config, scenario)
        allocation = allocate_deadline_aware_knapsack(
            observation.candidates,
            budget=budget,
            utility_estimates=observation.utility_estimates,
            values=_scenario_values(observation.candidates, utility_by_candidate, 0, scenario, config),
        )
        step_plans.append(MPCStepPlan(step_index=0, scenario=scenario, allocation=allocation, score=allocation.total_value))
        runtime_capped = True

    selected_step, robust_score, group_metadata = _select_robust_step(tuple(step_plans), config)
    runtime_ms = (time.perf_counter() - started_at) * 1000.0
    payload = {
        "observation_id": observation.observation_id,
        "candidate_set_id": observation.candidate_set.candidate_set_id,
        "selected_candidate_ids": list(selected_step.allocation.selected_candidate_ids),
        "runtime_capped": runtime_capped,
        "robust_score": robust_score,
        "config": config.as_payload(),
    }
    return MPCPlan(
        plan_id=f"robust-mpc-plan-{stable_config_id(payload)}",
        selected_step=selected_step,
        step_plans=tuple(step_plans),
        runtime_ms=runtime_ms,
        runtime_capped=runtime_capped,
        robust_score=robust_score,
        metadata={
            "algorithm": "short_horizon_robust_allocator_mpc",
            "scenario_count": len(scenarios),
            "horizon_steps": config.horizon_steps,
            "runtime_cap_ms": config.runtime_cap_ms,
            "selection_group": group_metadata,
        },
    )


def _scenarios(config: MPCConfig, observation: SchedulingObservation) -> tuple[MPCScenario, ...]:
    metadata_scenarios = _metadata_scenarios(observation.metadata)
    if metadata_scenarios:
        return metadata_scenarios[: config.max_scenarios]

    candidates: list[MPCScenario] = []
    for bandwidth_scale in config.bandwidth_interval.samples():
        for viewport_error in config.viewport_error_interval.samples():
            for deadline_scale in config.deadline_scale_interval.samples():
                payload = {
                    "bandwidth_scale": bandwidth_scale,
                    "viewport_error": viewport_error,
                    "deadline_scale": deadline_scale,
                }
                candidates.append(
                    MPCScenario(
                        scenario_id=f"mpc-scenario-{stable_config_id(payload)}",
                        bandwidth_scale=bandwidth_scale,
                        viewport_error=viewport_error,
                        deadline_scale=deadline_scale,
                        metadata={"source": "config_intervals"},
                    )
                )
    return tuple(candidates[: config.max_scenarios])


def _metadata_scenarios(metadata: Mapping[str, Any]) -> tuple[MPCScenario, ...]:
    raw_scenarios = metadata.get("mpc_scenarios")
    if not isinstance(raw_scenarios, Sequence) or isinstance(raw_scenarios, (str, bytes)):
        return ()
    scenarios: list[MPCScenario] = []
    for index, raw_scenario in enumerate(raw_scenarios):
        if not isinstance(raw_scenario, MappingABC):
            raise MPCError("metadata.mpc_scenarios entries must be mappings.")
        bandwidth_scale = raw_scenario.get("bandwidth_scale", 1.0)
        viewport_error = raw_scenario.get("viewport_error", 0.0)
        deadline_scale = raw_scenario.get("deadline_scale", 1.0)
        scenario_id = raw_scenario.get("scenario_id", f"metadata-scenario-{index}")
        scenarios.append(
            MPCScenario(
                scenario_id=str(scenario_id),
                bandwidth_scale=bandwidth_scale,
                viewport_error=viewport_error,
                deadline_scale=deadline_scale,
                probability=raw_scenario.get("probability", 1.0),
                metadata={"source": "observation_metadata"},
            )
        )
    return tuple(scenarios)


def _scenario_budget(
    observation: SchedulingObservation,
    action_budget: ActionBudget,
    config: MPCConfig,
    scenario: MPCScenario,
) -> KnapsackBudget:
    horizon_ms = max(0, observation.target_deadline_ms - observation.decision_time_ms)
    scenario_deadline = observation.decision_time_ms + int(round(horizon_ms * scenario.deadline_scale))
    max_deadline = action_budget.max_deadline_ms if action_budget.max_deadline_ms is not None else scenario_deadline
    max_deadline = min(max_deadline, scenario_deadline)
    return KnapsackBudget(
        max_bytes=int(math.floor(action_budget.max_selected_bytes * scenario.bandwidth_scale)),
        max_render_ms=config.max_render_ms,
        max_compute_ms=config.max_compute_ms,
        max_deadline_ms=max_deadline,
        max_selected_objects=action_budget.max_selected_objects,
        max_selected_candidates=action_budget.max_selected_candidates or action_budget.max_selected_objects,
        satisfied_dependencies=_satisfied_dependencies(observation),
        metadata={
            "source": "robust_mpc_scenario",
            "observation_id": observation.observation_id,
            "scenario_id": scenario.scenario_id,
            "action_budget": action_budget.as_payload(),
        },
    )


def _scenario_values(
    candidates: Sequence[CandidateObject],
    utility_by_candidate: Mapping[str, CandidateUtilityEstimate],
    step_index: int,
    scenario: MPCScenario,
    config: MPCConfig,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for candidate in candidates:
        value = _candidate_base_value(candidate, utility_by_candidate)
        value -= _viewport_penalty(candidate, value, scenario)
        value -= _deadline_penalty(candidate, step_index, config, scenario)
        if candidate.lookahead_ms > 0 or step_index > 0:
            value += config.prefetch_bonus * (step_index + 1)
        values[candidate.candidate_id] = value
    return values


def _score_allocation(
    allocation: KnapsackAllocation,
    observation: SchedulingObservation,
    values: Mapping[str, float],
    step_index: int,
    scenario: MPCScenario,
    config: MPCConfig,
) -> dict[str, Any]:
    selected_ids = set(allocation.selected_candidate_ids)
    unselected_urgent_value = 0.0
    expired_unselected = 0
    horizon_time_ms = observation.decision_time_ms + step_index * config.step_duration_ms
    for candidate in observation.candidates:
        if candidate.candidate_id in selected_ids:
            continue
        if candidate.deadline_ms <= horizon_time_ms + config.step_duration_ms:
            candidate_value = max(0.0, values.get(candidate.candidate_id, 0.0))
            unselected_urgent_value += candidate_value
            if candidate_value > 0.0:
                expired_unselected += 1
    drop_cost = config.drop_penalty * unselected_urgent_value
    score = allocation.total_value - drop_cost
    action_counts = _action_counts(allocation, observation)
    return {
        "score": score,
        "drop_cost": drop_cost,
        "expired_unselected_count": expired_unselected,
        "action_counts": action_counts,
        "scenario_id": scenario.scenario_id,
    }


def _select_robust_step(step_plans: tuple[MPCStepPlan, ...], config: MPCConfig) -> tuple[MPCStepPlan, float, dict[str, Any]]:
    by_selection: dict[tuple[str, ...], list[MPCStepPlan]] = {}
    for step_plan in step_plans:
        key = step_plan.allocation.selected_candidate_ids
        by_selection.setdefault(key, []).append(step_plan)

    best_key: tuple[float, int, float, tuple[str, ...]] | None = None
    best_selection: tuple[str, ...] = ()
    best_score = 0.0
    for selected_ids, plans in by_selection.items():
        scores = [plan.score for plan in plans]
        mean_score = sum(scores) / len(scores)
        spread = max(scores) - min(scores)
        robust_score = mean_score - config.robustness_weight * spread
        rank_key = (-robust_score, len(selected_ids), sum(plan.allocation.total_bytes for plan in plans) / len(plans), selected_ids)
        if best_key is None or rank_key < best_key:
            best_key = rank_key
            best_selection = selected_ids
            best_score = robust_score

    selected_plans = by_selection[best_selection]
    selected_step = min(selected_plans, key=lambda plan: (plan.step_index, plan.scenario.scenario_id))
    return (
        selected_step,
        best_score,
        {
            "selected_candidate_ids": list(best_selection),
            "mean_score": sum(plan.score for plan in selected_plans) / len(selected_plans),
            "min_score": min(plan.score for plan in selected_plans),
            "max_score": max(plan.score for plan in selected_plans),
            "support_count": len(selected_plans),
        },
    )


def _candidate_base_value(
    candidate: CandidateObject,
    utility_by_candidate: Mapping[str, CandidateUtilityEstimate],
) -> float:
    estimate = utility_by_candidate.get(candidate.candidate_id)
    if estimate is not None:
        return estimate.expected_utility - estimate.deadline_miss_probability - 0.25 * estimate.lifecycle_risk
    allocator = _candidate_allocator_metadata(candidate)
    for key in ("value", "utility", "expected_utility"):
        raw_value = allocator.get(key)
        if raw_value is not None:
            return _finite_float(raw_value, f"candidate.{candidate.candidate_id}.{key}")
    substrate = candidate.metadata.get("substrate")
    if isinstance(substrate, MappingABC) and substrate.get("visible_quality") is not None:
        return _finite_float(substrate["visible_quality"], f"candidate.{candidate.candidate_id}.visible_quality")
    return max(0.0, 0.05 + 0.05 * candidate.layer + candidate.resolution.megapixels / 20.0)


def _viewport_penalty(candidate: CandidateObject, base_value: float, scenario: MPCScenario) -> float:
    sensitivity = _candidate_mpc_float(candidate, "viewport_sensitivity", default=_default_viewport_sensitivity(candidate))
    normalized_error = scenario.viewport_error if scenario.viewport_error <= 1.0 else min(1.0, scenario.viewport_error / 90.0)
    return max(0.0, base_value) * sensitivity * normalized_error


def _deadline_penalty(candidate: CandidateObject, step_index: int, config: MPCConfig, scenario: MPCScenario) -> float:
    projected_time = candidate.decision_time_ms + step_index * config.step_duration_ms
    remaining_ms = candidate.deadline_ms - projected_time
    if remaining_ms <= 0:
        return 1.0 + abs(remaining_ms) / max(1.0, config.step_duration_ms)
    slack_ratio = remaining_ms / max(1.0, config.step_duration_ms * config.horizon_steps)
    scenario_pressure = max(0.0, 1.0 - scenario.deadline_scale)
    return scenario_pressure / max(0.25, slack_ratio)


def _action_counts(allocation: KnapsackAllocation, observation: SchedulingObservation) -> dict[str, int]:
    counts = {
        "gaussian": 0,
        "reference": 0,
        "tile": 0,
        "prefetch": 0,
        "drop": 0,
    }
    selected_ids = set(allocation.selected_candidate_ids)
    for item in allocation.selected_items:
        kind = item.candidate.candidate_kind
        if kind.startswith("gaussian"):
            counts["gaussian"] += 1
        elif kind == "reference_action":
            counts["reference"] += 1
        elif kind == "tile":
            counts["tile"] += 1
        if item.candidate.lookahead_ms > 0:
            counts["prefetch"] += 1
    for candidate in observation.candidates:
        if candidate.candidate_id not in selected_ids and candidate.deadline_ms <= observation.target_deadline_ms:
            counts["drop"] += 1
    return counts


def _satisfied_dependencies(observation: SchedulingObservation) -> tuple[str, ...]:
    active_statuses = {
        LifecycleStatus.REQUESTED.value,
        LifecycleStatus.IN_FLIGHT.value,
        LifecycleStatus.AVAILABLE.value,
    }
    satisfied: list[str] = []
    seen: set[str] = set()
    for state in observation.lifecycle_states:
        status = state.status.value if hasattr(state.status, "value") else str(state.status)
        if status in active_statuses and state.reference_id not in seen:
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


def _candidate_mpc_float(candidate: CandidateObject, key: str, *, default: float) -> float:
    metadata = candidate.metadata.get("mpc")
    if isinstance(metadata, MappingABC) and key in metadata:
        return _non_negative_float(metadata[key], f"candidate.metadata.mpc.{key}")
    return default


def _default_viewport_sensitivity(candidate: CandidateObject) -> float:
    if candidate.candidate_kind == "reference_action":
        return 0.65
    if candidate.candidate_kind == "tile":
        return 0.45
    if candidate.candidate_kind == "gaussian_enhancement":
        return 0.25
    return 0.10


def _coerce_interval(value: RobustInterval | tuple[float, float], field_name: str) -> RobustInterval:
    if isinstance(value, RobustInterval):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return RobustInterval(value[0], value[1])
    raise MPCError(f"{field_name} must be a RobustInterval or two-value tuple.")


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise MPCError(f"{field_name} must be a mapping.")
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
        raise MPCError(f"{field_name} must be a non-empty string.")


def _positive_int(value: Any, field_name: str) -> int:
    parsed = _non_negative_int(value, field_name)
    if parsed <= 0:
        raise MPCError(f"{field_name} must be positive.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise MPCError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MPCError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise MPCError(f"{field_name} must be non-negative.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0:
        raise MPCError(f"{field_name} must be non-negative.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise MPCError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MPCError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise MPCError(f"{field_name} must be finite.")
    return parsed


__all__ = [
    "MPCConfig",
    "MPCError",
    "MPCPlan",
    "MPCScenario",
    "MPCStepPlan",
    "RobustDeadlineAwareMPCController",
    "RobustInterval",
    "plan_robust_deadline_aware_mpc",
]
