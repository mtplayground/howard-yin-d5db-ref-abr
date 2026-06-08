"""Reference lifecycle state machine and events."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ref_abr.config import stable_config_id
from ref_abr.domain import LifecycleStatus, ReferenceLifecycleState


class LifecycleError(ValueError):
    """Raised when a reference lifecycle transition is invalid."""


class LifecyclePhase(str, Enum):
    """Detailed lifecycle phases stored in ReferenceLifecycleState metadata."""

    CANDIDATE = "candidate"
    REQUESTED = "requested"
    GENERATING = "generating"
    TRANSFERRING = "transferring"
    ARRIVED = "arrived"
    RESTORED = "restored"
    USED = "used"
    STALE = "stale"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class LifecycleAction(str, Enum):
    """Legal lifecycle transition actions."""

    REQUEST = "request"
    GENERATE = "generate"
    TRANSFER = "transfer"
    ARRIVE = "arrive"
    RESTORE = "restore"
    USE = "use"
    STALE = "stale"
    EXPIRE = "expire"
    CANCEL = "cancel"


class DropReason(str, Enum):
    """Standard reasons for terminal dropped or expired references."""

    CANCELLED = "cancelled"
    EXPIRED = "expired"
    STALE = "stale"
    DEADLINE_MISSED = "deadline_missed"
    SUPERSEDED = "superseded"
    ERROR = "error"


TERMINAL_PHASES: tuple[LifecyclePhase, ...] = (
    LifecyclePhase.STALE,
    LifecyclePhase.EXPIRED,
    LifecyclePhase.CANCELLED,
)


PHASE_STATUS: Mapping[LifecyclePhase, LifecycleStatus] = {
    LifecyclePhase.CANDIDATE: LifecycleStatus.CANDIDATE,
    LifecyclePhase.REQUESTED: LifecycleStatus.REQUESTED,
    LifecyclePhase.GENERATING: LifecycleStatus.IN_FLIGHT,
    LifecyclePhase.TRANSFERRING: LifecycleStatus.IN_FLIGHT,
    LifecyclePhase.ARRIVED: LifecycleStatus.IN_FLIGHT,
    LifecyclePhase.RESTORED: LifecycleStatus.AVAILABLE,
    LifecyclePhase.USED: LifecycleStatus.AVAILABLE,
    LifecyclePhase.STALE: LifecycleStatus.EXPIRED,
    LifecyclePhase.EXPIRED: LifecycleStatus.EXPIRED,
    LifecyclePhase.CANCELLED: LifecycleStatus.DROPPED,
}


LEGAL_TRANSITIONS: Mapping[LifecyclePhase, Mapping[LifecycleAction, LifecyclePhase]] = {
    LifecyclePhase.CANDIDATE: {
        LifecycleAction.REQUEST: LifecyclePhase.REQUESTED,
        LifecycleAction.CANCEL: LifecyclePhase.CANCELLED,
    },
    LifecyclePhase.REQUESTED: {
        LifecycleAction.GENERATE: LifecyclePhase.GENERATING,
        LifecycleAction.EXPIRE: LifecyclePhase.EXPIRED,
        LifecycleAction.CANCEL: LifecyclePhase.CANCELLED,
    },
    LifecyclePhase.GENERATING: {
        LifecycleAction.TRANSFER: LifecyclePhase.TRANSFERRING,
        LifecycleAction.EXPIRE: LifecyclePhase.EXPIRED,
        LifecycleAction.CANCEL: LifecyclePhase.CANCELLED,
    },
    LifecyclePhase.TRANSFERRING: {
        LifecycleAction.ARRIVE: LifecyclePhase.ARRIVED,
        LifecycleAction.EXPIRE: LifecyclePhase.EXPIRED,
        LifecycleAction.CANCEL: LifecyclePhase.CANCELLED,
    },
    LifecyclePhase.ARRIVED: {
        LifecycleAction.RESTORE: LifecyclePhase.RESTORED,
        LifecycleAction.EXPIRE: LifecyclePhase.EXPIRED,
        LifecycleAction.CANCEL: LifecyclePhase.CANCELLED,
    },
    LifecyclePhase.RESTORED: {
        LifecycleAction.USE: LifecyclePhase.USED,
        LifecycleAction.STALE: LifecyclePhase.STALE,
        LifecycleAction.EXPIRE: LifecyclePhase.EXPIRED,
        LifecycleAction.CANCEL: LifecyclePhase.CANCELLED,
    },
    LifecyclePhase.USED: {
        LifecycleAction.USE: LifecyclePhase.USED,
        LifecycleAction.STALE: LifecyclePhase.STALE,
        LifecycleAction.EXPIRE: LifecyclePhase.EXPIRED,
        LifecycleAction.CANCEL: LifecyclePhase.CANCELLED,
    },
}


@dataclass(frozen=True)
class ReferenceLifecycleEvent:
    """One emitted lifecycle transition event."""

    event_id: str
    reference_id: str
    action: LifecycleAction | str
    from_phase: LifecyclePhase | str
    to_phase: LifecyclePhase | str
    status: LifecycleStatus | str
    event_time_ms: int
    deadline_ms: int | None = None
    attempts: int = 0
    drop_reason: DropReason | str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.event_id, "event_id")
        _require_non_empty(self.reference_id, "reference_id")
        object.__setattr__(self, "action", _coerce_enum(LifecycleAction, self.action, "action"))
        object.__setattr__(self, "from_phase", _coerce_enum(LifecyclePhase, self.from_phase, "from_phase"))
        object.__setattr__(self, "to_phase", _coerce_enum(LifecyclePhase, self.to_phase, "to_phase"))
        object.__setattr__(self, "status", _coerce_enum(LifecycleStatus, self.status, "status"))
        object.__setattr__(self, "event_time_ms", _non_negative_int(self.event_time_ms, "event_time_ms"))
        if self.deadline_ms is not None:
            object.__setattr__(self, "deadline_ms", _non_negative_int(self.deadline_ms, "deadline_ms"))
        object.__setattr__(self, "attempts", _non_negative_int(self.attempts, "attempts"))
        if self.drop_reason is not None:
            object.__setattr__(self, "drop_reason", _coerce_enum(DropReason, self.drop_reason, "drop_reason"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "reference_id": self.reference_id,
            "action": self.action.value,
            "from_phase": self.from_phase.value,
            "to_phase": self.to_phase.value,
            "status": self.status.value,
            "event_time_ms": self.event_time_ms,
            "deadline_ms": self.deadline_ms,
            "attempts": self.attempts,
            "drop_reason": self.drop_reason.value if self.drop_reason is not None else None,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class LifecycleTransition:
    """State and event emitted by one lifecycle transition."""

    state: ReferenceLifecycleState
    event: ReferenceLifecycleEvent

    def __post_init__(self) -> None:
        if not isinstance(self.state, ReferenceLifecycleState):
            raise LifecycleError("state must be a ReferenceLifecycleState record.")
        if not isinstance(self.event, ReferenceLifecycleEvent):
            raise LifecycleError("event must be a ReferenceLifecycleEvent record.")

    def as_payload(self) -> dict[str, Any]:
        return {"state": self.state.as_payload(), "event": self.event.as_payload()}


@dataclass(frozen=True)
class ReferenceLifecycleStateMachine:
    """Immutable reference lifecycle state machine."""

    reference_id: str
    phase: LifecyclePhase | str = LifecyclePhase.CANDIDATE
    updated_at_ms: int = 0
    deadline_ms: int | None = None
    attempts: int = 0
    drop_reason: DropReason | str | None = None
    events: tuple[ReferenceLifecycleEvent, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.reference_id, "reference_id")
        object.__setattr__(self, "phase", _coerce_enum(LifecyclePhase, self.phase, "phase"))
        object.__setattr__(self, "updated_at_ms", _non_negative_int(self.updated_at_ms, "updated_at_ms"))
        if self.deadline_ms is not None:
            object.__setattr__(self, "deadline_ms", _non_negative_int(self.deadline_ms, "deadline_ms"))
        object.__setattr__(self, "attempts", _non_negative_int(self.attempts, "attempts"))
        if self.drop_reason is not None:
            object.__setattr__(self, "drop_reason", _coerce_enum(DropReason, self.drop_reason, "drop_reason"))
        events = tuple(self.events)
        for event in events:
            if not isinstance(event, ReferenceLifecycleEvent):
                raise LifecycleError("events must contain ReferenceLifecycleEvent records.")
            if event.reference_id != self.reference_id:
                raise LifecycleError("events reference_id must match state machine reference_id.")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def status(self) -> LifecycleStatus:
        return PHASE_STATUS[self.phase]

    @property
    def state(self) -> ReferenceLifecycleState:
        last_event_id = self.events[-1].event_id if self.events else None
        metadata = {
            **self.metadata,
            "lifecycle": {
                "phase": self.phase.value,
                "terminal": self.phase in TERMINAL_PHASES,
                "drop_reason": self.drop_reason.value if self.drop_reason is not None else None,
                "last_event_id": last_event_id,
            },
        }
        return ReferenceLifecycleState(
            reference_id=self.reference_id,
            status=self.status,
            updated_at_ms=self.updated_at_ms,
            deadline_ms=self.deadline_ms,
            attempts=self.attempts,
            metadata=metadata,
        )

    def transition(
        self,
        action: LifecycleAction | str,
        *,
        at_ms: int,
        drop_reason: DropReason | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple["ReferenceLifecycleStateMachine", LifecycleTransition]:
        """Apply one legal transition and return the updated machine and emission."""

        lifecycle_action = _coerce_enum(LifecycleAction, action, "action")
        event_time_ms = _non_negative_int(at_ms, "at_ms")
        if event_time_ms < self.updated_at_ms:
            raise LifecycleError("at_ms must be greater than or equal to the current updated_at_ms.")
        if self.phase in TERMINAL_PHASES:
            raise LifecycleError(f"Cannot transition terminal lifecycle phase {self.phase.value}.")
        legal = LEGAL_TRANSITIONS.get(self.phase, {})
        if lifecycle_action not in legal:
            allowed = ", ".join(action.value for action in sorted(legal, key=lambda item: item.value))
            raise LifecycleError(f"Illegal lifecycle transition {self.phase.value}->{lifecycle_action.value}; allowed: {allowed}.")
        next_phase = legal[lifecycle_action]
        parsed_drop_reason = _drop_reason_for(lifecycle_action, drop_reason)
        next_attempts = self.attempts + 1 if lifecycle_action == LifecycleAction.REQUEST else self.attempts
        event_payload = {
            "reference_id": self.reference_id,
            "action": lifecycle_action.value,
            "from_phase": self.phase.value,
            "to_phase": next_phase.value,
            "event_time_ms": event_time_ms,
            "deadline_ms": self.deadline_ms,
            "attempts": next_attempts,
            "drop_reason": parsed_drop_reason.value if parsed_drop_reason is not None else None,
            "event_index": len(self.events),
        }
        event = ReferenceLifecycleEvent(
            event_id=f"lifecycle-event-{stable_config_id(event_payload)}",
            reference_id=self.reference_id,
            action=lifecycle_action,
            from_phase=self.phase,
            to_phase=next_phase,
            status=PHASE_STATUS[next_phase],
            event_time_ms=event_time_ms,
            deadline_ms=self.deadline_ms,
            attempts=next_attempts,
            drop_reason=parsed_drop_reason,
            metadata={
                "transition": {
                    "event_index": len(self.events),
                    "legal": True,
                },
                **_plain_json_mapping(metadata, "metadata"),
            },
        )
        next_machine = ReferenceLifecycleStateMachine(
            reference_id=self.reference_id,
            phase=next_phase,
            updated_at_ms=event_time_ms,
            deadline_ms=self.deadline_ms,
            attempts=next_attempts,
            drop_reason=parsed_drop_reason,
            events=(*self.events, event),
            metadata=self.metadata,
        )
        return next_machine, LifecycleTransition(state=next_machine.state, event=event)

    def request(self, *, at_ms: int, metadata: Mapping[str, Any] | None = None) -> tuple["ReferenceLifecycleStateMachine", LifecycleTransition]:
        return self.transition(LifecycleAction.REQUEST, at_ms=at_ms, metadata=metadata)

    def generate(self, *, at_ms: int, metadata: Mapping[str, Any] | None = None) -> tuple["ReferenceLifecycleStateMachine", LifecycleTransition]:
        return self.transition(LifecycleAction.GENERATE, at_ms=at_ms, metadata=metadata)

    def transfer(self, *, at_ms: int, metadata: Mapping[str, Any] | None = None) -> tuple["ReferenceLifecycleStateMachine", LifecycleTransition]:
        return self.transition(LifecycleAction.TRANSFER, at_ms=at_ms, metadata=metadata)

    def arrive(self, *, at_ms: int, metadata: Mapping[str, Any] | None = None) -> tuple["ReferenceLifecycleStateMachine", LifecycleTransition]:
        return self.transition(LifecycleAction.ARRIVE, at_ms=at_ms, metadata=metadata)

    def restore(self, *, at_ms: int, metadata: Mapping[str, Any] | None = None) -> tuple["ReferenceLifecycleStateMachine", LifecycleTransition]:
        return self.transition(LifecycleAction.RESTORE, at_ms=at_ms, metadata=metadata)

    def use(self, *, at_ms: int, metadata: Mapping[str, Any] | None = None) -> tuple["ReferenceLifecycleStateMachine", LifecycleTransition]:
        return self.transition(LifecycleAction.USE, at_ms=at_ms, metadata=metadata)

    def stale(self, *, at_ms: int, metadata: Mapping[str, Any] | None = None) -> tuple["ReferenceLifecycleStateMachine", LifecycleTransition]:
        return self.transition(LifecycleAction.STALE, at_ms=at_ms, metadata=metadata)

    def expire(
        self,
        *,
        at_ms: int,
        drop_reason: DropReason | str = DropReason.EXPIRED,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple["ReferenceLifecycleStateMachine", LifecycleTransition]:
        return self.transition(LifecycleAction.EXPIRE, at_ms=at_ms, drop_reason=drop_reason, metadata=metadata)

    def cancel(
        self,
        *,
        at_ms: int,
        drop_reason: DropReason | str = DropReason.CANCELLED,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple["ReferenceLifecycleStateMachine", LifecycleTransition]:
        return self.transition(LifecycleAction.CANCEL, at_ms=at_ms, drop_reason=drop_reason, metadata=metadata)

    def as_payload(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "phase": self.phase.value,
            "status": self.status.value,
            "updated_at_ms": self.updated_at_ms,
            "deadline_ms": self.deadline_ms,
            "attempts": self.attempts,
            "drop_reason": self.drop_reason.value if self.drop_reason is not None else None,
            "state": self.state.as_payload(),
            "events": [event.as_payload() for event in self.events],
            "metadata": _to_payload(self.metadata),
        }


def lifecycle_state_from_machine(machine: ReferenceLifecycleStateMachine) -> ReferenceLifecycleState:
    """Return the current domain lifecycle state from a machine."""

    if not isinstance(machine, ReferenceLifecycleStateMachine):
        raise LifecycleError("machine must be a ReferenceLifecycleStateMachine record.")
    return machine.state


def _drop_reason_for(action: LifecycleAction, drop_reason: DropReason | str | None) -> DropReason | None:
    if action == LifecycleAction.STALE:
        return DropReason.STALE
    if action == LifecycleAction.EXPIRE:
        return _coerce_enum(DropReason, drop_reason or DropReason.EXPIRED, "drop_reason")
    if action == LifecycleAction.CANCEL:
        return _coerce_enum(DropReason, drop_reason or DropReason.CANCELLED, "drop_reason")
    if drop_reason is not None:
        raise LifecycleError("drop_reason is only valid for stale, expire, and cancel transitions.")
    return None


def _coerce_enum(enum_type: type[Enum], value: Enum | str, field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            valid = ", ".join(member.value for member in enum_type)
            raise LifecycleError(f"{field_name} must be one of: {valid}.") from exc
    raise LifecycleError(f"{field_name} must be a string or {enum_type.__name__}.")


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise LifecycleError(f"{field_name} must be a mapping.")
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
        raise LifecycleError(f"{field_name} must be a non-empty string.")


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise LifecycleError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LifecycleError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise LifecycleError(f"{field_name} must be a non-negative integer.")
    return parsed


__all__ = [
    "DropReason",
    "LifecycleAction",
    "LifecycleError",
    "LifecyclePhase",
    "LifecycleTransition",
    "ReferenceLifecycleEvent",
    "ReferenceLifecycleStateMachine",
    "lifecycle_state_from_machine",
]
