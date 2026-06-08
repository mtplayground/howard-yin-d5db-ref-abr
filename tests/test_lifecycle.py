from __future__ import annotations

import pytest

from ref_abr.domain import LifecycleStatus, ReferenceLifecycleState
from ref_abr.lifecycle import (
    DropReason,
    LifecycleAction,
    LifecycleError,
    LifecyclePhase,
    ReferenceLifecycleEvent,
    ReferenceLifecycleStateMachine,
    lifecycle_state_from_machine,
)


def test_reference_lifecycle_full_path_emits_states_and_events() -> None:
    machine = ReferenceLifecycleStateMachine("ref-a", deadline_ms=100, metadata={"source": "test"})
    emissions = []
    for action, at_ms in (
        (LifecycleAction.REQUEST, 1),
        (LifecycleAction.GENERATE, 2),
        (LifecycleAction.TRANSFER, 3),
        (LifecycleAction.ARRIVE, 4),
        (LifecycleAction.RESTORE, 5),
        (LifecycleAction.USE, 6),
    ):
        machine, emission = machine.transition(action, at_ms=at_ms)
        emissions.append(emission)

    assert machine.phase == LifecyclePhase.USED
    assert machine.status == LifecycleStatus.AVAILABLE
    assert machine.attempts == 1
    assert len(machine.events) == 6
    assert [event.action for event in machine.events] == [
        LifecycleAction.REQUEST,
        LifecycleAction.GENERATE,
        LifecycleAction.TRANSFER,
        LifecycleAction.ARRIVE,
        LifecycleAction.RESTORE,
        LifecycleAction.USE,
    ]
    assert [emission.state.status for emission in emissions] == [
        LifecycleStatus.REQUESTED,
        LifecycleStatus.IN_FLIGHT,
        LifecycleStatus.IN_FLIGHT,
        LifecycleStatus.IN_FLIGHT,
        LifecycleStatus.AVAILABLE,
        LifecycleStatus.AVAILABLE,
    ]
    assert lifecycle_state_from_machine(machine).metadata["lifecycle"]["phase"] == "used"
    assert lifecycle_state_from_machine(machine).metadata["source"] == "test"


def test_lifecycle_rejects_illegal_and_non_monotonic_transitions() -> None:
    machine = ReferenceLifecycleStateMachine("ref-b", updated_at_ms=10)

    with pytest.raises(LifecycleError, match="Illegal lifecycle transition"):
        machine.generate(at_ms=11)
    with pytest.raises(LifecycleError, match="at_ms"):
        machine.request(at_ms=9)


def test_stale_transition_records_drop_reason_and_blocks_future_transitions() -> None:
    machine = _available_machine("ref-stale")

    machine, emission = machine.stale(at_ms=20)

    assert machine.phase == LifecyclePhase.STALE
    assert machine.status == LifecycleStatus.EXPIRED
    assert machine.drop_reason == DropReason.STALE
    assert emission.event.drop_reason == DropReason.STALE
    assert emission.state.metadata["lifecycle"]["terminal"] is True
    assert emission.state.metadata["lifecycle"]["drop_reason"] == "stale"
    with pytest.raises(LifecycleError, match="terminal"):
        machine.use(at_ms=21)


def test_expire_and_cancel_emit_drop_reasons() -> None:
    requested, _ = ReferenceLifecycleStateMachine("ref-expire").request(at_ms=1)
    expired, expire_emission = requested.expire(at_ms=2, drop_reason=DropReason.DEADLINE_MISSED)
    cancelled, cancel_emission = ReferenceLifecycleStateMachine("ref-cancel").cancel(
        at_ms=3,
        drop_reason=DropReason.SUPERSEDED,
    )

    assert expired.state.status == LifecycleStatus.EXPIRED
    assert expired.state.metadata["lifecycle"]["drop_reason"] == "deadline_missed"
    assert expire_emission.event.drop_reason == DropReason.DEADLINE_MISSED
    assert cancelled.state.status == LifecycleStatus.DROPPED
    assert cancelled.state.metadata["lifecycle"]["drop_reason"] == "superseded"
    assert cancel_emission.event.drop_reason == DropReason.SUPERSEDED


def test_event_ids_are_deterministic_for_same_transition_sequence() -> None:
    first, first_emission = ReferenceLifecycleStateMachine("ref-deterministic").request(at_ms=1)
    second, second_emission = ReferenceLifecycleStateMachine("ref-deterministic").request(at_ms=1)

    assert first.events[0].event_id == second.events[0].event_id
    assert first_emission.event.event_id == second_emission.event.event_id


def test_lifecycle_event_and_machine_payloads_round_trip_to_plain_values() -> None:
    machine, emission = ReferenceLifecycleStateMachine("ref-payload", deadline_ms=9).request(
        at_ms=1,
        metadata={"nested": {"ok": True}},
    )
    payload = machine.as_payload()

    assert isinstance(emission.event, ReferenceLifecycleEvent)
    assert isinstance(emission.state, ReferenceLifecycleState)
    assert payload["phase"] == "requested"
    assert payload["status"] == "requested"
    assert payload["state"]["metadata"]["lifecycle"]["last_event_id"] == emission.event.event_id
    assert payload["events"][0]["metadata"]["nested"]["ok"] is True


def test_lifecycle_validation_rejects_bad_values() -> None:
    with pytest.raises(LifecycleError, match="reference_id"):
        ReferenceLifecycleStateMachine("")
    with pytest.raises(LifecycleError, match="drop_reason"):
        ReferenceLifecycleStateMachine("ref-x").transition(LifecycleAction.REQUEST, at_ms=1, drop_reason=DropReason.ERROR)
    with pytest.raises(LifecycleError, match="drop_reason"):
        ReferenceLifecycleEvent(
            event_id="event-x",
            reference_id="ref-x",
            action=LifecycleAction.CANCEL,
            from_phase=LifecyclePhase.CANDIDATE,
            to_phase=LifecyclePhase.CANCELLED,
            status=LifecycleStatus.DROPPED,
            event_time_ms=1,
            drop_reason="unknown",
        )


def _available_machine(reference_id: str) -> ReferenceLifecycleStateMachine:
    machine = ReferenceLifecycleStateMachine(reference_id)
    for action, at_ms in (
        (LifecycleAction.REQUEST, 1),
        (LifecycleAction.GENERATE, 2),
        (LifecycleAction.TRANSFER, 3),
        (LifecycleAction.ARRIVE, 4),
        (LifecycleAction.RESTORE, 5),
    ):
        machine, _ = machine.transition(action, at_ms=at_ms)
    return machine
