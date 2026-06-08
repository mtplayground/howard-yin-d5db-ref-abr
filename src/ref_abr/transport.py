"""Transport-aware expiration and retransmission prioritization."""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ref_abr.candidates import CandidateObject, CandidateSet
from ref_abr.config import stable_config_id
from ref_abr.domain import LifecycleStatus, ReferenceLifecycleState
from ref_abr.lifecycle import (
    DropReason,
    LifecycleError,
    LifecyclePhase,
    ReferenceLifecycleEvent,
    ReferenceLifecycleStateMachine,
)
from ref_abr.network import NetworkSample, NetworkTrace


class TransportError(ValueError):
    """Raised when transport prioritization inputs are invalid."""


class TransportPriorityClass(str, Enum):
    """Transport priority class labels."""

    BASE = "base"
    VISIBLE_TILE = "visible_tile"
    REFERENCE = "reference"
    ENHANCEMENT = "enhancement"
    TILE = "tile"
    RETRANSMIT = "retransmit"


@dataclass(frozen=True)
class TransportPriorityWeights:
    """Tunable weights for transport scheduling priority scores."""

    base: float = 100.0
    visible_tile: float = 90.0
    reference: float = 75.0
    tile: float = 55.0
    enhancement: float = 40.0
    retransmit_bonus: float = 35.0
    deadline_urgency: float = 120.0
    transfer_time_penalty: float = 0.05
    latency_penalty: float = 0.05
    packet_loss_penalty: float = 50.0

    def __post_init__(self) -> None:
        for field_name in (
            "base",
            "visible_tile",
            "reference",
            "tile",
            "enhancement",
            "retransmit_bonus",
            "deadline_urgency",
            "transfer_time_penalty",
            "latency_penalty",
            "packet_loss_penalty",
        ):
            object.__setattr__(self, field_name, _non_negative_float(getattr(self, field_name), field_name))

    def as_payload(self) -> dict[str, float]:
        return {
            "base": self.base,
            "visible_tile": self.visible_tile,
            "reference": self.reference,
            "tile": self.tile,
            "enhancement": self.enhancement,
            "retransmit_bonus": self.retransmit_bonus,
            "deadline_urgency": self.deadline_urgency,
            "transfer_time_penalty": self.transfer_time_penalty,
            "latency_penalty": self.latency_penalty,
            "packet_loss_penalty": self.packet_loss_penalty,
        }


@dataclass(frozen=True)
class TransportCandidatePriority:
    """Scored transport candidate with deadline and network annotations."""

    candidate: CandidateObject
    priority_class: TransportPriorityClass | str
    score: float
    now_ms: int
    deadline_ms: int
    deadline_slack_ms: float
    estimated_transfer_ms: float
    retransmit: bool = False
    expired: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, CandidateObject):
            raise TransportError("candidate must be a CandidateObject record.")
        object.__setattr__(self, "priority_class", _coerce_enum(TransportPriorityClass, self.priority_class, "priority_class"))
        object.__setattr__(self, "score", _finite_float(self.score, "score"))
        object.__setattr__(self, "now_ms", _non_negative_int(self.now_ms, "now_ms"))
        object.__setattr__(self, "deadline_ms", _non_negative_int(self.deadline_ms, "deadline_ms"))
        object.__setattr__(self, "deadline_slack_ms", _finite_float(self.deadline_slack_ms, "deadline_slack_ms"))
        object.__setattr__(self, "estimated_transfer_ms", _non_negative_float(self.estimated_transfer_ms, "estimated_transfer_ms"))
        if not isinstance(self.retransmit, bool):
            raise TransportError("retransmit must be boolean.")
        if not isinstance(self.expired, bool):
            raise TransportError("expired must be boolean.")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    def as_payload(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.as_payload(),
            "priority_class": self.priority_class.value,
            "score": self.score,
            "now_ms": self.now_ms,
            "deadline_ms": self.deadline_ms,
            "deadline_slack_ms": self.deadline_slack_ms,
            "estimated_transfer_ms": self.estimated_transfer_ms,
            "retransmit": self.retransmit,
            "expired": self.expired,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class TransportPlan:
    """Prioritized transport worklist and expired candidate summary."""

    plan_id: str
    prioritized: tuple[TransportCandidatePriority, ...]
    expired_candidate_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.plan_id, "plan_id")
        prioritized = tuple(self.prioritized)
        for priority in prioritized:
            if not isinstance(priority, TransportCandidatePriority):
                raise TransportError("prioritized must contain TransportCandidatePriority records.")
        object.__setattr__(self, "prioritized", prioritized)
        object.__setattr__(self, "expired_candidate_ids", _string_tuple(self.expired_candidate_ids, "expired_candidate_ids"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def prioritized_candidate_ids(self) -> tuple[str, ...]:
        return tuple(priority.candidate_id for priority in self.prioritized)

    def as_payload(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "prioritized": [priority.as_payload() for priority in self.prioritized],
            "expired_candidate_ids": list(self.expired_candidate_ids),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class TransportExpirationResult:
    """Reference states and events emitted by transport expiration cleanup."""

    active_states: tuple[ReferenceLifecycleState, ...]
    dropped_states: tuple[ReferenceLifecycleState, ...]
    events: tuple[ReferenceLifecycleEvent, ...]

    def __post_init__(self) -> None:
        for state in self.active_states:
            if not isinstance(state, ReferenceLifecycleState):
                raise TransportError("active_states must contain ReferenceLifecycleState records.")
        for state in self.dropped_states:
            if not isinstance(state, ReferenceLifecycleState):
                raise TransportError("dropped_states must contain ReferenceLifecycleState records.")
        for event in self.events:
            if not isinstance(event, ReferenceLifecycleEvent):
                raise TransportError("events must contain ReferenceLifecycleEvent records.")

    def as_payload(self) -> dict[str, Any]:
        return {
            "active_states": [state.as_payload() for state in self.active_states],
            "dropped_states": [state.as_payload() for state in self.dropped_states],
            "events": [event.as_payload() for event in self.events],
        }


def prioritize_transport_candidates(
    candidates: CandidateSet | tuple[CandidateObject, ...] | list[CandidateObject],
    *,
    now_ms: int,
    network: NetworkSample | NetworkTrace | None = None,
    visible_tile_candidate_ids: tuple[str, ...] | list[str] | None = None,
    retransmit_candidate_ids: tuple[str, ...] | list[str] = (),
    weights: TransportPriorityWeights | None = None,
    max_candidates: int | None = None,
) -> TransportPlan:
    """Prioritize transport candidates by type, deadline, network, and retransmit state."""

    current_ms = _non_negative_int(now_ms, "now_ms")
    candidate_tuple = _candidate_tuple(candidates)
    priority_weights = weights or TransportPriorityWeights()
    if not isinstance(priority_weights, TransportPriorityWeights):
        raise TransportError("weights must be a TransportPriorityWeights record.")
    visible_tiles = None if visible_tile_candidate_ids is None else set(_string_tuple(visible_tile_candidate_ids, "visible_tile_candidate_ids"))
    retransmit_ids = set(_string_tuple(retransmit_candidate_ids, "retransmit_candidate_ids"))
    sample = _network_sample_at(network, current_ms)
    scored: list[TransportCandidatePriority] = []
    expired_candidate_ids: list[str] = []

    for candidate in candidate_tuple:
        expired = candidate.deadline_ms <= current_ms
        if expired:
            expired_candidate_ids.append(candidate.candidate_id)
            continue
        retransmit = candidate.candidate_id in retransmit_ids
        priority_class = _priority_class(candidate, visible_tiles, retransmit)
        estimated_transfer_ms = _estimated_transfer_ms(candidate, sample)
        deadline_slack_ms = float(candidate.deadline_ms - current_ms) - estimated_transfer_ms
        score = _priority_score(
            candidate,
            priority_class,
            priority_weights,
            estimated_transfer_ms=estimated_transfer_ms,
            deadline_slack_ms=deadline_slack_ms,
            sample=sample,
            retransmit=retransmit,
        )
        scored.append(
            TransportCandidatePriority(
                candidate=candidate,
                priority_class=priority_class,
                score=score,
                now_ms=current_ms,
                deadline_ms=candidate.deadline_ms,
                deadline_slack_ms=deadline_slack_ms,
                estimated_transfer_ms=estimated_transfer_ms,
                retransmit=retransmit,
                expired=False,
                metadata={
                    "transport": {
                        "network_sample": sample.as_payload() if sample is not None else None,
                        "retransmit_priority": candidate.retransmit_priority,
                    }
                },
            )
        )

    prioritized = tuple(sorted(scored, key=_priority_sort_key))
    if max_candidates is not None:
        max_count = _positive_int(max_candidates, "max_candidates")
        prioritized = prioritized[:max_count]
    payload = {
        "now_ms": current_ms,
        "candidate_ids": [candidate.candidate_id for candidate in candidate_tuple],
        "prioritized": [priority.as_payload() for priority in prioritized],
        "expired_candidate_ids": expired_candidate_ids,
        "weights": priority_weights.as_payload(),
        "network_sample": sample.as_payload() if sample is not None else None,
    }
    return TransportPlan(
        plan_id=f"transport-plan-{stable_config_id(payload)}",
        prioritized=prioritized,
        expired_candidate_ids=tuple(expired_candidate_ids),
        metadata={
            "transport": {
                "now_ms": current_ms,
                "candidate_count": len(candidate_tuple),
                "prioritized_count": len(prioritized),
                "expired_candidate_count": len(expired_candidate_ids),
                "weights": priority_weights.as_payload(),
                "network_sample": sample.as_payload() if sample is not None else None,
            }
        },
    )


def drop_expired_references(
    machines: tuple[ReferenceLifecycleStateMachine, ...] | list[ReferenceLifecycleStateMachine],
    *,
    now_ms: int,
) -> TransportExpirationResult:
    """Expire or drop lifecycle machines whose deadlines have passed."""

    current_ms = _non_negative_int(now_ms, "now_ms")
    active_states: list[ReferenceLifecycleState] = []
    dropped_states: list[ReferenceLifecycleState] = []
    events: list[ReferenceLifecycleEvent] = []

    for machine in tuple(machines):
        if not isinstance(machine, ReferenceLifecycleStateMachine):
            raise TransportError("machines must contain ReferenceLifecycleStateMachine records.")
        if machine.deadline_ms is None or machine.deadline_ms > current_ms:
            active_states.append(machine.state)
            continue
        if machine.state.status in {LifecycleStatus.EXPIRED, LifecycleStatus.DROPPED}:
            dropped_states.append(machine.state)
            continue
        next_machine, event = _drop_expired_machine(machine, current_ms)
        dropped_states.append(next_machine.state)
        events.append(event)

    return TransportExpirationResult(
        active_states=tuple(active_states),
        dropped_states=tuple(dropped_states),
        events=tuple(events),
    )


def _drop_expired_machine(
    machine: ReferenceLifecycleStateMachine,
    now_ms: int,
) -> tuple[ReferenceLifecycleStateMachine, ReferenceLifecycleEvent]:
    try:
        next_machine, emission = machine.expire(at_ms=now_ms, drop_reason=DropReason.DEADLINE_MISSED)
    except LifecycleError:
        next_machine, emission = machine.cancel(at_ms=now_ms, drop_reason=DropReason.EXPIRED)
    return next_machine, emission.event


def _candidate_tuple(candidates: CandidateSet | tuple[CandidateObject, ...] | list[CandidateObject]) -> tuple[CandidateObject, ...]:
    if isinstance(candidates, CandidateSet):
        candidate_tuple = candidates.candidates
    else:
        candidate_tuple = tuple(candidates)
    for candidate in candidate_tuple:
        if not isinstance(candidate, CandidateObject):
            raise TransportError("candidates must contain CandidateObject records.")
    return candidate_tuple


def _priority_class(
    candidate: CandidateObject,
    visible_tile_candidate_ids: set[str] | None,
    retransmit: bool,
) -> TransportPriorityClass:
    if retransmit:
        return TransportPriorityClass.RETRANSMIT
    if candidate.candidate_kind == "gaussian_base":
        return TransportPriorityClass.BASE
    if candidate.candidate_kind == "tile":
        if visible_tile_candidate_ids is None or candidate.candidate_id in visible_tile_candidate_ids:
            return TransportPriorityClass.VISIBLE_TILE
        return TransportPriorityClass.TILE
    if candidate.candidate_kind == "reference_action":
        return TransportPriorityClass.REFERENCE
    if candidate.candidate_kind == "gaussian_enhancement":
        return TransportPriorityClass.ENHANCEMENT
    raise TransportError(f"Unsupported candidate_kind {candidate.candidate_kind}.")


def _priority_score(
    candidate: CandidateObject,
    priority_class: TransportPriorityClass,
    weights: TransportPriorityWeights,
    *,
    estimated_transfer_ms: float,
    deadline_slack_ms: float,
    sample: NetworkSample | None,
    retransmit: bool,
) -> float:
    class_score = {
        TransportPriorityClass.BASE: weights.base,
        TransportPriorityClass.VISIBLE_TILE: weights.visible_tile,
        TransportPriorityClass.REFERENCE: weights.reference,
        TransportPriorityClass.TILE: weights.tile,
        TransportPriorityClass.ENHANCEMENT: weights.enhancement,
        TransportPriorityClass.RETRANSMIT: weights.reference,
    }[priority_class]
    positive_slack_ms = max(1.0, deadline_slack_ms)
    urgency = weights.deadline_urgency / positive_slack_ms
    retransmit_score = weights.retransmit_bonus if retransmit else 0.0
    priority_score = min(10.0, float(candidate.retransmit_priority)) * 2.0
    latency_ms = sample.latency_ms if sample is not None else 0.0
    packet_loss = sample.packet_loss if sample is not None else 0.0
    return (
        class_score
        + urgency
        + retransmit_score
        + priority_score
        - weights.transfer_time_penalty * estimated_transfer_ms
        - weights.latency_penalty * latency_ms
        - weights.packet_loss_penalty * packet_loss
    )


def _priority_sort_key(priority: TransportCandidatePriority) -> tuple[float, int, int, int, str]:
    return (
        -priority.score,
        _class_rank(priority.priority_class),
        priority.deadline_ms,
        -priority.candidate.retransmit_priority,
        priority.candidate_id,
    )


def _class_rank(priority_class: TransportPriorityClass) -> int:
    order = {
        TransportPriorityClass.BASE: 0,
        TransportPriorityClass.VISIBLE_TILE: 1,
        TransportPriorityClass.REFERENCE: 2,
        TransportPriorityClass.RETRANSMIT: 3,
        TransportPriorityClass.TILE: 4,
        TransportPriorityClass.ENHANCEMENT: 5,
    }
    return order[priority_class]


def _estimated_transfer_ms(candidate: CandidateObject, sample: NetworkSample | None) -> float:
    if sample is None:
        return 0.0
    if sample.throughput_bps <= 0:
        return 1_000_000_000.0
    transfer_ms = candidate.size_bytes * 8_000.0 / sample.throughput_bps
    return transfer_ms + sample.latency_ms + sample.jitter_ms


def _network_sample_at(network: NetworkSample | NetworkTrace | None, now_ms: int) -> NetworkSample | None:
    if network is None:
        return None
    if isinstance(network, NetworkSample):
        return network
    if not isinstance(network, NetworkTrace):
        raise TransportError("network must be a NetworkSample, NetworkTrace, or None.")
    samples_at_or_before = tuple(sample for sample in network.samples if sample.timestamp_ms <= now_ms)
    if samples_at_or_before:
        return samples_at_or_before[-1]
    return network.samples[0]


def _coerce_enum(enum_type: type[Enum], value: Enum | str, field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            valid = ", ".join(member.value for member in enum_type)
            raise TransportError(f"{field_name} must be one of: {valid}.") from exc
    raise TransportError(f"{field_name} must be a string or {enum_type.__name__}.")


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise TransportError(f"{field_name} must be a mapping.")
    return {str(key): _to_payload(item) for key, item in value.items()}


def _to_payload(value: Any) -> Any:
    if hasattr(value, "as_payload"):
        return value.as_payload()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, MappingABC):
        return {str(key): _to_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_payload(item) for item in value]
    if isinstance(value, list):
        return [_to_payload(item) for item in value]
    return value


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TransportError(f"{field_name} must be a non-empty string.")


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        _require_non_empty(value, field_name)
        return (value,)
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TransportError(f"{field_name} must be a string or iterable of strings.") from exc
    for item in result:
        _require_non_empty(item, field_name)
    return result


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TransportError(f"{field_name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TransportError(f"{field_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise TransportError(f"{field_name} must be a positive integer.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TransportError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TransportError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise TransportError(f"{field_name} must be a non-negative integer.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise TransportError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TransportError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise TransportError(f"{field_name} must be finite.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0:
        raise TransportError(f"{field_name} must be non-negative.")
    return parsed


__all__ = [
    "TransportCandidatePriority",
    "TransportError",
    "TransportExpirationResult",
    "TransportPlan",
    "TransportPriorityClass",
    "TransportPriorityWeights",
    "drop_expired_references",
    "prioritize_transport_candidates",
]
