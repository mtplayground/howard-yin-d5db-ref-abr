"""Method adapter contract for schedule planning."""

from __future__ import annotations

from collections.abc import Callable, Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from ref_abr.candidates import CandidateObject, CandidateSet
from ref_abr.config import stable_config_id
from ref_abr.domain import ControllerState, ReferenceLifecycleState, ScheduleDecision
from ref_abr.utility import CandidateUtilityEstimate


class MethodError(ValueError):
    """Raised when a scheduling method violates the adapter contract."""


@runtime_checkable
class SchedulePlanner(Protocol):
    """Contract implemented by scheduling methods."""

    method_id: str
    method_name: str

    def plan_schedule(
        self,
        observation: "SchedulingObservation",
        action_budget: "ActionBudget",
    ) -> ScheduleDecision | Mapping[str, Any]:
        """Return one schedule decision for the supplied observation and budget."""


@dataclass(frozen=True)
class ObservationBudget:
    """Fairness budget controlling what each method can observe."""

    max_candidates: int
    max_utility_estimates: int | None = None
    max_lifecycle_states: int | None = None
    include_metadata: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_candidates", _positive_int(self.max_candidates, "max_candidates"))
        if self.max_utility_estimates is not None:
            object.__setattr__(
                self,
                "max_utility_estimates",
                _non_negative_int(self.max_utility_estimates, "max_utility_estimates"),
            )
        if self.max_lifecycle_states is not None:
            object.__setattr__(
                self,
                "max_lifecycle_states",
                _non_negative_int(self.max_lifecycle_states, "max_lifecycle_states"),
            )
        if not isinstance(self.include_metadata, bool):
            raise MethodError("include_metadata must be boolean.")

    def as_payload(self) -> dict[str, Any]:
        return {
            "max_candidates": self.max_candidates,
            "max_utility_estimates": self.max_utility_estimates,
            "max_lifecycle_states": self.max_lifecycle_states,
            "include_metadata": self.include_metadata,
        }


@dataclass(frozen=True)
class ActionBudget:
    """Fairness budget controlling what each method may select."""

    max_selected_objects: int
    max_selected_bytes: int
    max_selected_candidates: int | None = None
    max_deadline_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_selected_objects", _positive_int(self.max_selected_objects, "max_selected_objects"))
        object.__setattr__(self, "max_selected_bytes", _non_negative_int(self.max_selected_bytes, "max_selected_bytes"))
        if self.max_selected_candidates is not None:
            object.__setattr__(
                self,
                "max_selected_candidates",
                _positive_int(self.max_selected_candidates, "max_selected_candidates"),
            )
        if self.max_deadline_ms is not None:
            object.__setattr__(self, "max_deadline_ms", _non_negative_int(self.max_deadline_ms, "max_deadline_ms"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "max_selected_objects": self.max_selected_objects,
            "max_selected_bytes": self.max_selected_bytes,
            "max_selected_candidates": self.max_selected_candidates,
            "max_deadline_ms": self.max_deadline_ms,
        }


@dataclass(frozen=True)
class SchedulingObservation:
    """Inputs visible to a scheduling method at one decision point."""

    observation_id: str
    controller_state: ControllerState
    frame_id: str
    decision_time_ms: int
    target_deadline_ms: int
    candidate_set: CandidateSet
    utility_estimates: tuple[CandidateUtilityEstimate, ...] = field(default_factory=tuple)
    lifecycle_states: tuple[ReferenceLifecycleState, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.observation_id, "observation_id")
        if not isinstance(self.controller_state, ControllerState):
            raise MethodError("controller_state must be a ControllerState record.")
        _require_non_empty(self.frame_id, "frame_id")
        object.__setattr__(self, "decision_time_ms", _non_negative_int(self.decision_time_ms, "decision_time_ms"))
        object.__setattr__(self, "target_deadline_ms", _non_negative_int(self.target_deadline_ms, "target_deadline_ms"))
        if self.target_deadline_ms < self.decision_time_ms:
            raise MethodError("target_deadline_ms must be greater than or equal to decision_time_ms.")
        if not isinstance(self.candidate_set, CandidateSet):
            raise MethodError("candidate_set must be a CandidateSet record.")
        if self.candidate_set.decision_time_ms != self.decision_time_ms:
            raise MethodError("candidate_set.decision_time_ms must match observation.decision_time_ms.")
        utility_estimates = tuple(self.utility_estimates)
        for estimate in utility_estimates:
            if not isinstance(estimate, CandidateUtilityEstimate):
                raise MethodError("utility_estimates must contain CandidateUtilityEstimate records.")
        unknown_estimates = sorted(
            estimate.candidate_id for estimate in utility_estimates if estimate.candidate_id not in self.candidate_ids
        )
        if unknown_estimates:
            raise MethodError(f"utility_estimates reference unknown candidate_id values: {', '.join(unknown_estimates)}.")
        lifecycle_states = tuple(self.lifecycle_states)
        for lifecycle_state in lifecycle_states:
            if not isinstance(lifecycle_state, ReferenceLifecycleState):
                raise MethodError("lifecycle_states must contain ReferenceLifecycleState records.")
        object.__setattr__(self, "utility_estimates", utility_estimates)
        object.__setattr__(self, "lifecycle_states", lifecycle_states)
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def candidates(self) -> tuple[CandidateObject, ...]:
        return self.candidate_set.candidates

    @property
    def candidate_ids(self) -> set[str]:
        return {candidate.candidate_id for candidate in self.candidates}

    @property
    def object_ids(self) -> set[str]:
        return {candidate.object_id for candidate in self.candidates}

    def as_payload(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "controller_state": self.controller_state.as_payload(),
            "frame_id": self.frame_id,
            "decision_time_ms": self.decision_time_ms,
            "target_deadline_ms": self.target_deadline_ms,
            "candidate_set": self.candidate_set.as_payload(),
            "utility_estimates": [estimate.as_payload() for estimate in self.utility_estimates],
            "lifecycle_states": [state.as_payload() for state in self.lifecycle_states],
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class MethodAdapter:
    """Adapter that enforces common observation and action budgets."""

    method: SchedulePlanner | Callable[[SchedulingObservation, ActionBudget], ScheduleDecision | Mapping[str, Any]]
    observation_budget: ObservationBudget
    action_budget: ActionBudget
    controller_id: str | None = None
    method_id: str | None = None
    method_name: str | None = None

    def __post_init__(self) -> None:
        if not callable(self.method) and not callable(getattr(self.method, "plan_schedule", None)):
            raise MethodError("method must be callable or expose plan_schedule(observation, action_budget).")
        if not isinstance(self.observation_budget, ObservationBudget):
            raise MethodError("observation_budget must be an ObservationBudget record.")
        if not isinstance(self.action_budget, ActionBudget):
            raise MethodError("action_budget must be an ActionBudget record.")
        if self.controller_id is not None:
            _require_non_empty(self.controller_id, "controller_id")
        if self.method_id is not None:
            _require_non_empty(self.method_id, "method_id")
        if self.method_name is not None:
            _require_non_empty(self.method_name, "method_name")

    def plan_schedule(self, observation: SchedulingObservation) -> ScheduleDecision:
        """Plan and validate one schedule decision."""

        budgeted_observation = apply_observation_budget(observation, self.observation_budget)
        raw_decision = _invoke_method(self.method, budgeted_observation, self.action_budget)
        decision = coerce_schedule_decision(
            raw_decision,
            observation=budgeted_observation,
            controller_id=self.controller_id or budgeted_observation.controller_state.controller_id,
            method_id=self.method_id or _method_attr(self.method, "method_id") or _callable_name(self.method),
            method_name=self.method_name or _method_attr(self.method, "method_name") or _callable_name(self.method),
        )
        validate_schedule_decision(
            decision,
            observation=budgeted_observation,
            action_budget=self.action_budget,
        )
        return decision


def plan_schedule(
    method: SchedulePlanner | Callable[[SchedulingObservation, ActionBudget], ScheduleDecision | Mapping[str, Any]],
    observation: SchedulingObservation,
    *,
    observation_budget: ObservationBudget,
    action_budget: ActionBudget,
    controller_id: str | None = None,
    method_id: str | None = None,
    method_name: str | None = None,
) -> ScheduleDecision:
    """Run a method through the common schedule-planning adapter."""

    return MethodAdapter(
        method=method,
        observation_budget=observation_budget,
        action_budget=action_budget,
        controller_id=controller_id,
        method_id=method_id,
        method_name=method_name,
    ).plan_schedule(observation)


def apply_observation_budget(
    observation: SchedulingObservation,
    budget: ObservationBudget,
) -> SchedulingObservation:
    """Return the deterministic subset of an observation visible to a method."""

    if not isinstance(observation, SchedulingObservation):
        raise MethodError("observation must be a SchedulingObservation record.")
    if not isinstance(budget, ObservationBudget):
        raise MethodError("budget must be an ObservationBudget record.")
    candidates = observation.candidates[: budget.max_candidates]
    if not candidates:
        raise MethodError("observation_budget leaves no visible candidates.")
    visible_candidate_ids = {candidate.candidate_id for candidate in candidates}
    max_utility_estimates = budget.max_utility_estimates
    if max_utility_estimates is None:
        max_utility_estimates = len(observation.utility_estimates)
    utility_estimates = tuple(
        estimate for estimate in observation.utility_estimates if estimate.candidate_id in visible_candidate_ids
    )[:max_utility_estimates]
    max_lifecycle_states = budget.max_lifecycle_states
    if max_lifecycle_states is None:
        max_lifecycle_states = len(observation.lifecycle_states)
    lifecycle_states = observation.lifecycle_states[:max_lifecycle_states]
    metadata = observation.metadata if budget.include_metadata else {}
    candidate_set = CandidateSet(
        candidate_set_id=f"{observation.candidate_set.candidate_set_id}-obs-{stable_config_id({'candidate_ids': sorted(visible_candidate_ids)})}",
        decision_time_ms=observation.candidate_set.decision_time_ms,
        candidates=candidates,
        metadata={
            "observation_budget": budget.as_payload(),
            "source_candidate_set_id": observation.candidate_set.candidate_set_id,
        },
    )
    payload = {
        "source_observation_id": observation.observation_id,
        "candidate_set_id": candidate_set.candidate_set_id,
        "utility_estimate_ids": [estimate.estimate_id for estimate in utility_estimates],
        "lifecycle_reference_ids": [state.reference_id for state in lifecycle_states],
        "budget": budget.as_payload(),
    }
    return SchedulingObservation(
        observation_id=f"observation-{stable_config_id(payload)}",
        controller_state=observation.controller_state,
        frame_id=observation.frame_id,
        decision_time_ms=observation.decision_time_ms,
        target_deadline_ms=observation.target_deadline_ms,
        candidate_set=candidate_set,
        utility_estimates=utility_estimates,
        lifecycle_states=lifecycle_states,
        metadata={
            **metadata,
            "observation_budget": budget.as_payload(),
            "source_observation_id": observation.observation_id,
        },
    )


def coerce_schedule_decision(
    value: ScheduleDecision | Mapping[str, Any],
    *,
    observation: SchedulingObservation,
    controller_id: str,
    method_id: str,
    method_name: str,
) -> ScheduleDecision:
    """Normalize a method result into a ScheduleDecision."""

    if isinstance(value, ScheduleDecision):
        return value
    mapping = _require_mapping(value, "decision")
    selected_candidate_ids = _string_tuple(_first_present(mapping, ("selected_candidate_ids", "candidate_ids")), "selected_candidate_ids")
    selected_object_ids = _string_tuple(_first_present(mapping, ("selected_object_ids", "object_ids")), "selected_object_ids")
    if selected_candidate_ids:
        candidate_by_id = {candidate.candidate_id: candidate for candidate in observation.candidates}
        missing = sorted(candidate_id for candidate_id in selected_candidate_ids if candidate_id not in candidate_by_id)
        if missing:
            raise MethodError(f"selected_candidate_ids contains unknown candidate_id values: {', '.join(missing)}.")
        selected_object_ids = tuple(candidate_by_id[candidate_id].object_id for candidate_id in selected_candidate_ids)
    if not selected_object_ids:
        selected_object_ids = ()
    decision_time_ms = _non_negative_int(mapping.get("decision_time_ms", observation.decision_time_ms), "decision_time_ms")
    target_deadline_ms = _non_negative_int(mapping.get("target_deadline_ms", observation.target_deadline_ms), "target_deadline_ms")
    expected_utility = _first_present(mapping, ("expected_utility", "utility"))
    metadata = _plain_json_mapping(mapping.get("metadata"), "decision.metadata") if mapping.get("metadata") is not None else {}
    metadata["adapter"] = {
        "method_id": method_id,
        "method_name": method_name,
        "observation_id": observation.observation_id,
        "selected_candidate_ids": list(selected_candidate_ids),
    }
    payload = {
        "controller_id": controller_id,
        "frame_id": str(mapping.get("frame_id", observation.frame_id)),
        "selected_object_ids": list(selected_object_ids),
        "decision_time_ms": decision_time_ms,
        "target_deadline_ms": target_deadline_ms,
        "expected_utility": expected_utility,
        "adapter": metadata["adapter"],
    }
    return ScheduleDecision(
        decision_id=str(mapping.get("decision_id") or f"decision-{stable_config_id(payload)}"),
        controller_id=str(mapping.get("controller_id") or controller_id),
        frame_id=str(mapping.get("frame_id") or observation.frame_id),
        selected_object_ids=selected_object_ids,
        decision_time_ms=decision_time_ms,
        target_deadline_ms=target_deadline_ms,
        expected_utility=None if expected_utility is None else _finite_float(expected_utility, "expected_utility"),
        metadata=metadata,
    )


def validate_schedule_decision(
    decision: ScheduleDecision,
    *,
    observation: SchedulingObservation,
    action_budget: ActionBudget,
) -> None:
    """Validate decision feasibility against the visible observation and action budget."""

    if not isinstance(decision, ScheduleDecision):
        raise MethodError("decision must be a ScheduleDecision record.")
    if decision.controller_id != observation.controller_state.controller_id:
        raise MethodError("decision.controller_id must match observation.controller_state.controller_id.")
    if decision.frame_id != observation.frame_id:
        raise MethodError("decision.frame_id must match observation.frame_id.")
    if decision.decision_time_ms != observation.decision_time_ms:
        raise MethodError("decision.decision_time_ms must match observation.decision_time_ms.")
    if decision.target_deadline_ms > observation.target_deadline_ms:
        raise MethodError("decision.target_deadline_ms exceeds observation target_deadline_ms.")
    if action_budget.max_deadline_ms is not None and decision.target_deadline_ms > action_budget.max_deadline_ms:
        raise MethodError("decision.target_deadline_ms exceeds action_budget.max_deadline_ms.")
    selected_object_ids = tuple(decision.selected_object_ids)
    if len(selected_object_ids) > action_budget.max_selected_objects:
        raise MethodError("decision selects more objects than action_budget.max_selected_objects.")
    unknown_objects = sorted(object_id for object_id in selected_object_ids if object_id not in observation.object_ids)
    if unknown_objects:
        raise MethodError(f"decision selected unknown object_id values: {', '.join(unknown_objects)}.")
    selected_candidate_ids = _decision_selected_candidate_ids(decision)
    if selected_candidate_ids:
        if action_budget.max_selected_candidates is not None and len(selected_candidate_ids) > action_budget.max_selected_candidates:
            raise MethodError("decision selects more candidates than action_budget.max_selected_candidates.")
        unknown_candidates = sorted(candidate_id for candidate_id in selected_candidate_ids if candidate_id not in observation.candidate_ids)
        if unknown_candidates:
            raise MethodError(f"decision selected unknown candidate_id values: {', '.join(unknown_candidates)}.")
    selected_bytes = _selected_bytes(selected_object_ids, selected_candidate_ids, observation)
    if selected_bytes > action_budget.max_selected_bytes:
        raise MethodError("decision selected bytes exceed action_budget.max_selected_bytes.")


def _invoke_method(
    method: SchedulePlanner | Callable[[SchedulingObservation, ActionBudget], ScheduleDecision | Mapping[str, Any]],
    observation: SchedulingObservation,
    action_budget: ActionBudget,
) -> ScheduleDecision | Mapping[str, Any]:
    planner = getattr(method, "plan_schedule", None)
    if callable(planner):
        return planner(observation, action_budget)
    if callable(method):
        return method(observation, action_budget)
    raise MethodError("method must be callable or expose plan_schedule(observation, action_budget).")


def _selected_bytes(
    selected_object_ids: tuple[str, ...],
    selected_candidate_ids: tuple[str, ...],
    observation: SchedulingObservation,
) -> int:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in observation.candidates}
    if selected_candidate_ids:
        return sum(candidate_by_id[candidate_id].size_bytes for candidate_id in selected_candidate_ids)
    size_by_object: dict[str, int] = {}
    for candidate in observation.candidates:
        size_by_object[candidate.object_id] = min(candidate.size_bytes, size_by_object.get(candidate.object_id, candidate.size_bytes))
    return sum(size_by_object[object_id] for object_id in selected_object_ids)


def _decision_selected_candidate_ids(decision: ScheduleDecision) -> tuple[str, ...]:
    adapter_metadata = decision.metadata.get("adapter")
    if not isinstance(adapter_metadata, MappingABC):
        return ()
    return _string_tuple(adapter_metadata.get("selected_candidate_ids"), "selected_candidate_ids")


def _method_attr(method: Any, attr_name: str) -> str | None:
    value = getattr(method, attr_name, None)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise MethodError(f"method.{attr_name} must be a non-empty string when provided.")
    return value.strip()


def _callable_name(method: Any) -> str:
    return getattr(method, "__name__", method.__class__.__name__)


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise MethodError(f"{field_name} must be a mapping.")
    return value


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    mapping = _require_mapping(value, field_name)
    return {str(key): _to_payload(item) for key, item in mapping.items()}


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


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        _require_non_empty(value, field_name)
        return (value,)
    try:
        result = tuple(value)
    except TypeError as exc:
        raise MethodError(f"{field_name} must be a string or iterable of strings.") from exc
    for item in result:
        _require_non_empty(item, field_name)
    return result


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MethodError(f"{field_name} must be a non-empty string.")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise MethodError(f"{field_name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MethodError(f"{field_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise MethodError(f"{field_name} must be positive.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise MethodError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MethodError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise MethodError(f"{field_name} must be non-negative.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise MethodError(f"{field_name} must be a finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodError(f"{field_name} must be a finite number.") from exc
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise MethodError(f"{field_name} must be finite.")
    return parsed
