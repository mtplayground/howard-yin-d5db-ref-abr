"""Discrete-event scheduler core for driving method decisions."""

from __future__ import annotations

from collections.abc import Callable, Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.candidates import CandidateGenerationSpec, DecisionEpoch, generate_candidate_objects
from ref_abr.config import stable_config_id
from ref_abr.domain import ControllerState, ReferenceLifecycleState, ScheduleDecision, WorkloadManifest
from ref_abr.methods import ActionBudget, ObservationBudget, SchedulePlanner, SchedulingObservation, plan_schedule
from ref_abr.substrate import SubstrateValueProvider
from ref_abr.utility import (
    ResourceBudget,
    UtilityModelWeights,
    ViewportRisk,
    estimate_candidate_set_utility,
)


class SchedulerError(ValueError):
    """Raised when scheduler inputs are invalid."""


@dataclass(frozen=True)
class SchedulerClock:
    """Display and motion-to-photon timing model for decision epochs."""

    display_interval_ms: int
    motion_to_photon_latency_ms: int
    start_time_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "display_interval_ms", _positive_int(self.display_interval_ms, "display_interval_ms"))
        object.__setattr__(
            self,
            "motion_to_photon_latency_ms",
            _non_negative_int(self.motion_to_photon_latency_ms, "motion_to_photon_latency_ms"),
        )
        object.__setattr__(self, "start_time_ms", _non_negative_int(self.start_time_ms, "start_time_ms"))

    @classmethod
    def from_fps(
        cls,
        fps: int | float,
        *,
        motion_to_photon_latency_ms: int,
        start_time_ms: int = 0,
    ) -> "SchedulerClock":
        """Build an integer millisecond display clock from a target FPS."""

        parsed_fps = _positive_float(fps, "fps")
        interval_ms = max(1, int(round(1000.0 / parsed_fps)))
        return cls(
            display_interval_ms=interval_ms,
            motion_to_photon_latency_ms=motion_to_photon_latency_ms,
            start_time_ms=start_time_ms,
        )

    def display_time_ms(self, step_index: int) -> int:
        step = _non_negative_int(step_index, "step_index")
        return self.start_time_ms + step * self.display_interval_ms

    def target_deadline_ms(self, step_index: int) -> int:
        return self.display_time_ms(step_index) + self.motion_to_photon_latency_ms

    def as_payload(self) -> dict[str, int]:
        return {
            "display_interval_ms": self.display_interval_ms,
            "motion_to_photon_latency_ms": self.motion_to_photon_latency_ms,
            "start_time_ms": self.start_time_ms,
        }


@dataclass(frozen=True)
class DeadlineWindow:
    """Per-frame decision and display deadline window."""

    frame_id: str
    decision_time_ms: int
    display_time_ms: int
    target_deadline_ms: int
    motion_to_photon_latency_ms: int

    def __post_init__(self) -> None:
        _require_non_empty(self.frame_id, "frame_id")
        object.__setattr__(self, "decision_time_ms", _non_negative_int(self.decision_time_ms, "decision_time_ms"))
        object.__setattr__(self, "display_time_ms", _non_negative_int(self.display_time_ms, "display_time_ms"))
        object.__setattr__(self, "target_deadline_ms", _non_negative_int(self.target_deadline_ms, "target_deadline_ms"))
        object.__setattr__(
            self,
            "motion_to_photon_latency_ms",
            _non_negative_int(self.motion_to_photon_latency_ms, "motion_to_photon_latency_ms"),
        )
        if self.target_deadline_ms < self.decision_time_ms:
            raise SchedulerError("target_deadline_ms must be greater than or equal to decision_time_ms.")

    @property
    def slack_ms(self) -> int:
        return self.target_deadline_ms - self.decision_time_ms

    @property
    def decision_deadline_hit(self) -> bool:
        return self.decision_time_ms <= self.target_deadline_ms

    def as_payload(self) -> dict[str, int | str | bool]:
        return {
            "frame_id": self.frame_id,
            "decision_time_ms": self.decision_time_ms,
            "display_time_ms": self.display_time_ms,
            "target_deadline_ms": self.target_deadline_ms,
            "motion_to_photon_latency_ms": self.motion_to_photon_latency_ms,
            "slack_ms": self.slack_ms,
            "decision_deadline_hit": self.decision_deadline_hit,
        }


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for one deterministic scheduler run."""

    frame_count: int
    clock: SchedulerClock
    observation_budget: ObservationBudget
    action_budget: ActionBudget
    candidate_generation_spec: CandidateGenerationSpec = field(default_factory=CandidateGenerationSpec)
    resource_budget: ResourceBudget | Mapping[str, Any] | None = None
    viewport_risk: ViewportRisk | Mapping[str, Any] | None = None
    utility_model_weights: UtilityModelWeights | Mapping[str, Any] | None = None
    controller_id: str = "scheduler"
    frame_id_prefix: str = "frame"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_count", _positive_int(self.frame_count, "frame_count"))
        if not isinstance(self.clock, SchedulerClock):
            raise SchedulerError("clock must be a SchedulerClock record.")
        if not isinstance(self.observation_budget, ObservationBudget):
            raise SchedulerError("observation_budget must be an ObservationBudget record.")
        if not isinstance(self.action_budget, ActionBudget):
            raise SchedulerError("action_budget must be an ActionBudget record.")
        if not isinstance(self.candidate_generation_spec, CandidateGenerationSpec):
            raise SchedulerError("candidate_generation_spec must be a CandidateGenerationSpec record.")
        _require_non_empty(self.controller_id, "controller_id")
        _require_non_empty(self.frame_id_prefix, "frame_id_prefix")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def frame_id(self, step_index: int) -> str:
        step = _non_negative_int(step_index, "step_index")
        return f"{self.frame_id_prefix}-{step:06d}"

    def as_payload(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "clock": self.clock.as_payload(),
            "observation_budget": self.observation_budget.as_payload(),
            "action_budget": self.action_budget.as_payload(),
            "candidate_generation_spec": self.candidate_generation_spec.as_payload(),
            "resource_budget": _to_payload(self.resource_budget),
            "viewport_risk": _to_payload(self.viewport_risk),
            "utility_model_weights": _to_payload(self.utility_model_weights),
            "controller_id": self.controller_id,
            "frame_id_prefix": self.frame_id_prefix,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class SchedulerEpochResult:
    """Result of one scheduler decision epoch."""

    step_index: int
    deadline: DeadlineWindow
    controller_state: ControllerState
    observation: SchedulingObservation
    decision: ScheduleDecision

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_index", _non_negative_int(self.step_index, "step_index"))
        if not isinstance(self.deadline, DeadlineWindow):
            raise SchedulerError("deadline must be a DeadlineWindow record.")
        if not isinstance(self.controller_state, ControllerState):
            raise SchedulerError("controller_state must be a ControllerState record.")
        if not isinstance(self.observation, SchedulingObservation):
            raise SchedulerError("observation must be a SchedulingObservation record.")
        if not isinstance(self.decision, ScheduleDecision):
            raise SchedulerError("decision must be a ScheduleDecision record.")

    def as_payload(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "deadline": self.deadline.as_payload(),
            "controller_state": self.controller_state.as_payload(),
            "observation": self.observation.as_payload(),
            "decision": self.decision.as_payload(),
        }


@dataclass(frozen=True)
class SchedulerRunResult:
    """Complete deterministic scheduler run result."""

    run_id: str
    config: SchedulerConfig
    epochs: tuple[SchedulerEpochResult, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, "run_id")
        if not isinstance(self.config, SchedulerConfig):
            raise SchedulerError("config must be a SchedulerConfig record.")
        epochs = tuple(self.epochs)
        if len(epochs) != self.config.frame_count:
            raise SchedulerError("epochs length must match config.frame_count.")
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def decisions(self) -> tuple[ScheduleDecision, ...]:
        return tuple(epoch.decision for epoch in self.epochs)

    def as_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config.as_payload(),
            "epochs": [epoch.as_payload() for epoch in self.epochs],
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class DiscreteEventScheduler:
    """Core loop that generates decision epochs and invokes a scheduling method."""

    method: SchedulePlanner | Callable[[SchedulingObservation, ActionBudget], ScheduleDecision | Mapping[str, Any]]
    workload: WorkloadManifest
    config: SchedulerConfig
    substrate_provider: SubstrateValueProvider | None = None
    lifecycle_states: tuple[ReferenceLifecycleState, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not callable(self.method) and not callable(getattr(self.method, "plan_schedule", None)):
            raise SchedulerError("method must be callable or expose plan_schedule(observation, action_budget).")
        if not isinstance(self.workload, WorkloadManifest):
            raise SchedulerError("workload must be a WorkloadManifest record.")
        if not isinstance(self.config, SchedulerConfig):
            raise SchedulerError("config must be a SchedulerConfig record.")
        lifecycle_states = tuple(self.lifecycle_states)
        for lifecycle_state in lifecycle_states:
            if not isinstance(lifecycle_state, ReferenceLifecycleState):
                raise SchedulerError("lifecycle_states must contain ReferenceLifecycleState records.")
        object.__setattr__(self, "lifecycle_states", lifecycle_states)

    def run(self) -> SchedulerRunResult:
        """Execute the deterministic decision-epoch loop."""

        epochs: list[SchedulerEpochResult] = []
        previous_decision: ScheduleDecision | None = None
        method_id = _method_attr(self.method, "method_id") or _callable_name(self.method)
        method_name = _method_attr(self.method, "method_name") or _callable_name(self.method)

        for step_index in range(self.config.frame_count):
            frame_id = self.config.frame_id(step_index)
            display_time_ms = self.config.clock.display_time_ms(step_index)
            target_deadline_ms = self.config.clock.target_deadline_ms(step_index)
            deadline = DeadlineWindow(
                frame_id=frame_id,
                decision_time_ms=display_time_ms,
                display_time_ms=display_time_ms,
                target_deadline_ms=target_deadline_ms,
                motion_to_photon_latency_ms=self.config.clock.motion_to_photon_latency_ms,
            )
            controller_state = self._controller_state(
                step_index=step_index,
                method_name=method_name,
                deadline=deadline,
                previous_decision=previous_decision,
            )
            candidate_set = generate_candidate_objects(
                self.workload,
                DecisionEpoch(decision_time_ms=deadline.decision_time_ms, frame_id=frame_id),
                spec=self.config.candidate_generation_spec,
                substrate_provider=self.substrate_provider,
            )
            utility_estimates = estimate_candidate_set_utility(
                candidate_set,
                substrate_provider=self.substrate_provider,
                viewport_risk=self.config.viewport_risk,
                budgets=self.config.resource_budget,
                model_weights=self.config.utility_model_weights,
            )
            observation = SchedulingObservation(
                observation_id=_observation_id(controller_state, candidate_set.candidate_set_id, utility_estimates.estimate_set_id),
                controller_state=controller_state,
                frame_id=frame_id,
                decision_time_ms=deadline.decision_time_ms,
                target_deadline_ms=deadline.target_deadline_ms,
                candidate_set=candidate_set,
                utility_estimates=utility_estimates.estimates,
                lifecycle_states=self.lifecycle_states,
                metadata={
                    "scheduler": {
                        "clock": self.config.clock.as_payload(),
                        "deadline": deadline.as_payload(),
                        "utility_estimate_set_id": utility_estimates.estimate_set_id,
                    }
                },
            )
            decision = plan_schedule(
                self.method,
                observation,
                observation_budget=self.config.observation_budget,
                action_budget=self.config.action_budget,
                controller_id=self.config.controller_id,
                method_id=method_id,
                method_name=method_name,
            )
            epochs.append(
                SchedulerEpochResult(
                    step_index=step_index,
                    deadline=deadline,
                    controller_state=controller_state,
                    observation=observation,
                    decision=decision,
                )
            )
            previous_decision = decision

        payload = {
            "config": self.config.as_payload(),
            "workload_manifest_id": self.workload.manifest_id,
            "decision_ids": [epoch.decision.decision_id for epoch in epochs],
        }
        return SchedulerRunResult(
            run_id=f"scheduler-run-{stable_config_id(payload)}",
            config=self.config,
            epochs=tuple(epochs),
            metadata={
                "scheduler": {
                    "workload_manifest_id": self.workload.manifest_id,
                    "config_id": self.workload.config_id,
                    "split": self.workload.split,
                    "method_id": method_id,
                    "method_name": method_name,
                    "decision_count": len(epochs),
                }
            },
        )

    def _controller_state(
        self,
        *,
        step_index: int,
        method_name: str,
        deadline: DeadlineWindow,
        previous_decision: ScheduleDecision | None,
    ) -> ControllerState:
        return ControllerState(
            controller_id=self.config.controller_id,
            method_name=method_name,
            step_index=step_index,
            active_split=self.workload.split,
            state={
                "clock": {
                    "display_interval_ms": self.config.clock.display_interval_ms,
                    "display_time_ms": deadline.display_time_ms,
                    "motion_to_photon_latency_ms": deadline.motion_to_photon_latency_ms,
                },
                "deadline": deadline.as_payload(),
                "previous_decision_id": previous_decision.decision_id if previous_decision is not None else None,
                "previous_selected_object_ids": list(previous_decision.selected_object_ids) if previous_decision is not None else [],
            },
            metadata={
                "scheduler": {
                    "frame_count": self.config.frame_count,
                    "frame_id_prefix": self.config.frame_id_prefix,
                    "workload_manifest_id": self.workload.manifest_id,
                }
            },
        )


def run_discrete_event_schedule(
    method: SchedulePlanner | Callable[[SchedulingObservation, ActionBudget], ScheduleDecision | Mapping[str, Any]],
    workload: WorkloadManifest,
    *,
    config: SchedulerConfig,
    substrate_provider: SubstrateValueProvider | None = None,
    lifecycle_states: tuple[ReferenceLifecycleState, ...] = (),
) -> SchedulerRunResult:
    """Run the discrete-event scheduler for one method and workload."""

    return DiscreteEventScheduler(
        method=method,
        workload=workload,
        config=config,
        substrate_provider=substrate_provider,
        lifecycle_states=lifecycle_states,
    ).run()


def _observation_id(controller_state: ControllerState, candidate_set_id: str, utility_estimate_set_id: str) -> str:
    payload = {
        "controller_state": controller_state.as_payload(),
        "candidate_set_id": candidate_set_id,
        "utility_estimate_set_id": utility_estimate_set_id,
    }
    return f"observation-{stable_config_id(payload)}"


def _method_attr(method: Any, attr_name: str) -> str | None:
    value = getattr(method, attr_name, None)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SchedulerError(f"method.{attr_name} must be a non-empty string when provided.")
    return value.strip()


def _callable_name(method: Any) -> str:
    return getattr(method, "__name__", method.__class__.__name__)


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise SchedulerError(f"{field_name} must be a mapping.")
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
        raise SchedulerError(f"{field_name} must be a non-empty string.")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise SchedulerError(f"{field_name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SchedulerError(f"{field_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise SchedulerError(f"{field_name} must be a positive integer.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise SchedulerError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SchedulerError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise SchedulerError(f"{field_name} must be a non-negative integer.")
    return parsed


def _positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise SchedulerError(f"{field_name} must be positive.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SchedulerError(f"{field_name} must be positive.") from exc
    if parsed <= 0 or parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise SchedulerError(f"{field_name} must be positive and finite.")
    return parsed


__all__ = [
    "DeadlineWindow",
    "DiscreteEventScheduler",
    "SchedulerClock",
    "SchedulerConfig",
    "SchedulerEpochResult",
    "SchedulerError",
    "SchedulerRunResult",
    "run_discrete_event_schedule",
]
