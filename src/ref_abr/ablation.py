"""RefABR self-ablation switches over frozen controller structures."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, runtime_checkable

from ref_abr.candidates import CandidateObject, CandidateSet
from ref_abr.config import stable_config_id
from ref_abr.methods import ActionBudget, SchedulingObservation
from ref_abr.mpc import MPCConfig, RobustDeadlineAwareMPCController, RobustInterval
from ref_abr.utility import CandidateUtilityEstimate, ResourceDebt, ResourcePrice, UtilityUncertainty
from ref_abr.virtual_queue import VirtualQueueDeadlineController


class AblationError(ValueError):
    """Raised when ablation switches or wrapped controllers are invalid."""


@runtime_checkable
class ScheduleController(Protocol):
    """Minimal controller protocol for ablation wrappers."""

    method_id: str
    method_name: str

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> Mapping[str, Any]:
        """Return a method-adapter-compatible decision mapping."""


@dataclass(frozen=True)
class RefABRAblationSwitches:
    """Mechanism switches for RefABR self-ablation variants."""

    no_lifecycle: bool = False
    no_uncertainty: bool = False
    no_component_cost: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("no_lifecycle", "no_uncertainty", "no_component_cost"):
            if not isinstance(getattr(self, field_name), bool):
                raise AblationError(f"{field_name} must be boolean.")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def enabled_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.no_lifecycle:
            names.append("no-lifecycle")
        if self.no_uncertainty:
            names.append("no-uncertainty")
        if self.no_component_cost:
            names.append("no-component-cost")
        return tuple(names)

    @property
    def variant_id(self) -> str:
        if not self.enabled_names:
            return "full"
        return "-".join(self.enabled_names)

    def as_payload(self) -> dict[str, Any]:
        return {
            "no_lifecycle": self.no_lifecycle,
            "no_uncertainty": self.no_uncertainty,
            "no_component_cost": self.no_component_cost,
            "variant_id": self.variant_id,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class AblatedRefABRController:
    """Controller wrapper that applies RefABR ablation switches before planning."""

    controller: ScheduleController
    switches: RefABRAblationSwitches
    method_id: str | None = None
    method_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(getattr(self.controller, "plan_schedule", None)):
            raise AblationError("controller must expose plan_schedule(observation, action_budget).")
        if not isinstance(self.switches, RefABRAblationSwitches):
            raise AblationError("switches must be a RefABRAblationSwitches record.")
        resolved_method_id = self.method_id or f"{self.controller.method_id}-{self.switches.variant_id}"
        resolved_method_name = self.method_name or f"{self.controller.method_name} ({self.switches.variant_id})"
        _require_non_empty(resolved_method_id, "method_id")
        _require_non_empty(resolved_method_name, "method_name")
        object.__setattr__(self, "method_id", resolved_method_id)
        object.__setattr__(self, "method_name", resolved_method_name)
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, Any]:
        if not isinstance(observation, SchedulingObservation):
            raise AblationError("observation must be a SchedulingObservation record.")
        if not isinstance(action_budget, ActionBudget):
            raise AblationError("action_budget must be an ActionBudget record.")
        ablated_observation = apply_refabr_ablation(observation, self.switches)
        ablated_controller = _ablate_controller_config(self.controller, self.switches)
        decision = dict(ablated_controller.plan_schedule(ablated_observation, action_budget))
        metadata = _plain_json_mapping(decision.get("metadata", {}), "decision.metadata")
        metadata["ablation"] = {
            "switches": self.switches.as_payload(),
            "wrapped_method_id": self.controller.method_id,
            "wrapped_method_name": self.controller.method_name,
            "method_id": self.method_id,
            "method_name": self.method_name,
            "original_observation_id": observation.observation_id,
            "ablated_observation_id": ablated_observation.observation_id,
            **_to_payload(self.metadata),
        }
        decision["metadata"] = metadata
        return decision


def refabr_self_ablation_variants(controller: ScheduleController) -> tuple[AblatedRefABRController, ...]:
    """Return the standard single-mechanism RefABR ablation variants."""

    return (
        AblatedRefABRController(controller, RefABRAblationSwitches(no_lifecycle=True)),
        AblatedRefABRController(controller, RefABRAblationSwitches(no_uncertainty=True)),
        AblatedRefABRController(controller, RefABRAblationSwitches(no_component_cost=True)),
    )


def apply_refabr_ablation(
    observation: SchedulingObservation,
    switches: RefABRAblationSwitches,
) -> SchedulingObservation:
    """Return an observation with disabled mechanisms removed or neutralized."""

    if not isinstance(observation, SchedulingObservation):
        raise AblationError("observation must be a SchedulingObservation record.")
    if not isinstance(switches, RefABRAblationSwitches):
        raise AblationError("switches must be a RefABRAblationSwitches record.")
    candidates = tuple(_ablate_candidate(candidate, switches) for candidate in observation.candidates)
    candidate_set = CandidateSet(
        candidate_set_id=f"{observation.candidate_set.candidate_set_id}-{switches.variant_id}",
        decision_time_ms=observation.candidate_set.decision_time_ms,
        candidates=candidates,
        metadata={
            **_to_payload(observation.candidate_set.metadata),
            "refabr_ablation": switches.as_payload(),
        },
    )
    metadata = _ablate_observation_metadata(observation.metadata, switches)
    utility_estimates = tuple(_ablate_utility_estimate(estimate, switches) for estimate in observation.utility_estimates)
    lifecycle_states = () if switches.no_lifecycle else observation.lifecycle_states
    payload = {
        "observation_id": observation.observation_id,
        "candidate_set_id": candidate_set.candidate_set_id,
        "switches": switches.as_payload(),
    }
    return SchedulingObservation(
        observation_id=f"{observation.observation_id}-ablation-{stable_config_id(payload)}",
        controller_state=observation.controller_state,
        frame_id=observation.frame_id,
        decision_time_ms=observation.decision_time_ms,
        target_deadline_ms=observation.target_deadline_ms,
        candidate_set=candidate_set,
        utility_estimates=utility_estimates,
        lifecycle_states=lifecycle_states,
        metadata=metadata,
    )


def _ablate_controller_config(controller: ScheduleController, switches: RefABRAblationSwitches) -> ScheduleController:
    if switches.no_uncertainty and isinstance(controller, RobustDeadlineAwareMPCController):
        config = controller.config
        ablated_config = MPCConfig(
            horizon_steps=config.horizon_steps,
            step_duration_ms=config.step_duration_ms,
            runtime_cap_ms=config.runtime_cap_ms,
            max_scenarios=1,
            bandwidth_interval=RobustInterval(1.0, 1.0),
            viewport_error_interval=RobustInterval(0.0, 0.0),
            deadline_scale_interval=RobustInterval(1.0, 1.0),
            robustness_weight=0.0,
            drop_penalty=config.drop_penalty,
            prefetch_bonus=config.prefetch_bonus,
            max_render_ms=config.max_render_ms,
            max_compute_ms=config.max_compute_ms,
            metadata={
                **_to_payload(config.metadata),
                "refabr_ablation": switches.as_payload(),
            },
        )
        return replace(controller, config=ablated_config)
    if isinstance(controller, VirtualQueueDeadlineController):
        return replace(
            controller,
            metadata={
                **_to_payload(controller.metadata),
                "refabr_ablation": switches.as_payload(),
            },
        )
    return controller


def _ablate_candidate(candidate: CandidateObject, switches: RefABRAblationSwitches) -> CandidateObject:
    metadata = _plain_json_mapping(candidate.metadata, "candidate.metadata")
    if switches.no_lifecycle:
        metadata = _without_lifecycle_metadata(metadata)
    if switches.no_uncertainty:
        metadata = _without_uncertainty_metadata(metadata)
    if switches.no_component_cost:
        metadata = _without_component_cost_metadata(metadata)
    return replace(candidate, metadata=metadata)


def _ablate_utility_estimate(
    estimate: CandidateUtilityEstimate,
    switches: RefABRAblationSwitches,
) -> CandidateUtilityEstimate:
    lifecycle_risk = 0.0 if switches.no_lifecycle else estimate.lifecycle_risk
    uncertainty = estimate.uncertainty
    if switches.no_uncertainty:
        uncertainty = UtilityUncertainty(
            quality_stddev=0.0,
            timing_stddev_ms=0.0,
            deadline_probability_stddev=0.0,
            utility_stddev=0.0,
            confidence=1.0,
        )
    resource_price = estimate.resource_price
    resource_debt = estimate.resource_debt
    if switches.no_component_cost:
        resource_price = ResourcePrice(time_price=0.0, transfer_price=0.0, memory_price=0.0)
        resource_debt = ResourceDebt(
            time_debt_ms=0.0,
            transfer_debt_bytes=0,
            memory_debt_mb=0.0,
            carried_queue_debt_ms=0.0,
            carried_transfer_debt_bytes=0,
        )
    metadata = _plain_json_mapping(estimate.metadata, "estimate.metadata")
    metadata["refabr_ablation"] = switches.as_payload()
    return replace(
        estimate,
        lifecycle_risk=lifecycle_risk,
        resource_price=resource_price,
        resource_debt=resource_debt,
        uncertainty=uncertainty,
        metadata=metadata,
    )


def _ablate_observation_metadata(metadata: Mapping[str, Any], switches: RefABRAblationSwitches) -> dict[str, Any]:
    ablated = _plain_json_mapping(metadata, "metadata")
    if switches.no_lifecycle:
        ablated = _without_lifecycle_metadata(ablated)
    if switches.no_uncertainty:
        ablated = _without_uncertainty_metadata(ablated)
        ablated["mpc_scenarios"] = [
            {
                "scenario_id": "no-uncertainty-nominal",
                "bandwidth_scale": 1.0,
                "viewport_error": 0.0,
                "deadline_scale": 1.0,
                "probability": 1.0,
            }
        ]
    if switches.no_component_cost:
        ablated = _without_component_cost_metadata(ablated)
    ablated["refabr_ablation"] = switches.as_payload()
    return ablated


def _without_lifecycle_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = _plain_json_mapping(metadata, "metadata")
    for key in ("lifecycle", "lifecycle_state", "lifecycle_states", "lifecycle_risk", "lifecycle_debt"):
        cleaned.pop(key, None)
    virtual_queue = cleaned.get("virtual_queue")
    if isinstance(virtual_queue, MappingABC):
        cleaned["virtual_queue"] = {
            key: value
            for key, value in virtual_queue.items()
            if key not in {"lifecycle_debt", "lifecycle_risk"}
        }
    return cleaned


def _without_uncertainty_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = _plain_json_mapping(metadata, "metadata")
    for key in ("uncertainty", "viewport_risk", "viewport_debt"):
        cleaned.pop(key, None)
    virtual_queue = cleaned.get("virtual_queue")
    if isinstance(virtual_queue, MappingABC):
        next_queue = dict(virtual_queue)
        next_queue["viewport_debt"] = 0.0
        next_queue["viewport_risk"] = 0.0
        cleaned["virtual_queue"] = next_queue
    for key in ("mpc", "virtual_queue"):
        section = cleaned.get(key)
        if isinstance(section, MappingABC):
            next_section = dict(section)
            next_section["viewport_sensitivity"] = 0.0
            cleaned[key] = next_section
    return cleaned


def _without_component_cost_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = _plain_json_mapping(metadata, "metadata")
    allocator = cleaned.get("allocator")
    if isinstance(allocator, MappingABC):
        next_allocator = dict(allocator)
        for key in ("render_ms", "compute_ms", "generation_ms", "restoration_ms", "decode_ms"):
            if key in next_allocator:
                next_allocator[key] = 0.0
        cleaned["allocator"] = next_allocator
    allocator_cost = cleaned.get("allocator_cost")
    if isinstance(allocator_cost, MappingABC):
        next_allocator_cost = dict(allocator_cost)
        for key in ("render_ms", "compute_ms", "generation_ms", "restoration_ms", "decode_ms"):
            if key in next_allocator_cost:
                next_allocator_cost[key] = 0.0
        cleaned["allocator_cost"] = next_allocator_cost
    substrate = cleaned.get("substrate")
    if isinstance(substrate, MappingABC):
        next_substrate = dict(substrate)
        timing = next_substrate.get("component_timing")
        if isinstance(timing, MappingABC):
            next_substrate["component_timing"] = {str(key): 0.0 for key in timing}
        cleaned["substrate"] = next_substrate
    return cleaned


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise AblationError(f"{field_name} must be a mapping.")
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
        raise AblationError(f"{field_name} must be a non-empty string.")


__all__ = [
    "AblatedRefABRController",
    "AblationError",
    "RefABRAblationSwitches",
    "apply_refabr_ablation",
    "refabr_self_ablation_variants",
]
