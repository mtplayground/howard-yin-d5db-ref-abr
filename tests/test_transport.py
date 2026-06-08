from __future__ import annotations

import pytest

from ref_abr.candidates import CandidateObject, TileSpec
from ref_abr.domain import LifecycleStatus
from ref_abr.lifecycle import DropReason, LifecyclePhase, ReferenceLifecycleStateMachine
from ref_abr.network import NetworkSample, NetworkTrace
from ref_abr.transport import (
    TransportError,
    TransportPriorityClass,
    TransportPriorityWeights,
    drop_expired_references,
    prioritize_transport_candidates,
)


def test_transport_prioritizes_base_visible_tile_reference_and_enhancement() -> None:
    base = _candidate("base", "gaussian_base", deadline_ms=120)
    visible_tile = _candidate("tile-visible", "tile", deadline_ms=120, tile=TileSpec(row=0, column=0, rows=1, columns=1))
    reference = _candidate("reference", "reference_action", deadline_ms=120)
    enhancement = _candidate("enhancement", "gaussian_enhancement", deadline_ms=120, layer=1)

    plan = prioritize_transport_candidates(
        (enhancement, reference, visible_tile, base),
        now_ms=10,
        network=NetworkSample(timestamp_ms=0, throughput_bps=10_000_000, latency_ms=5),
    )

    assert plan.prioritized_candidate_ids == ("base", "tile-visible", "reference", "enhancement")
    assert [priority.priority_class for priority in plan.prioritized] == [
        TransportPriorityClass.BASE,
        TransportPriorityClass.VISIBLE_TILE,
        TransportPriorityClass.REFERENCE,
        TransportPriorityClass.ENHANCEMENT,
    ]
    assert plan.expired_candidate_ids == ()


def test_transport_marks_nonvisible_tiles_and_expired_candidates() -> None:
    visible = _candidate("tile-visible", "tile", deadline_ms=120, tile=TileSpec(row=0, column=0, rows=1, columns=2))
    hidden = _candidate("tile-hidden", "tile", deadline_ms=120, tile=TileSpec(row=0, column=1, rows=1, columns=2))
    expired = _candidate("expired", "gaussian_base", deadline_ms=5)

    plan = prioritize_transport_candidates(
        (hidden, expired, visible),
        now_ms=10,
        visible_tile_candidate_ids=("tile-visible",),
    )

    assert plan.prioritized_candidate_ids == ("tile-visible", "tile-hidden")
    assert plan.prioritized[0].priority_class == TransportPriorityClass.VISIBLE_TILE
    assert plan.prioritized[1].priority_class == TransportPriorityClass.TILE
    assert plan.expired_candidate_ids == ("expired",)


def test_retransmit_candidates_receive_explicit_class_and_bonus() -> None:
    base = _candidate("base", "gaussian_base", deadline_ms=120)
    retransmit = _candidate("reference-retry", "reference_action", deadline_ms=120, retransmit_priority=3)

    plan = prioritize_transport_candidates(
        (base, retransmit),
        now_ms=10,
        retransmit_candidate_ids=("reference-retry",),
    )

    assert plan.prioritized_candidate_ids[0] == "reference-retry"
    assert plan.prioritized[0].priority_class == TransportPriorityClass.RETRANSMIT
    assert plan.prioritized[0].retransmit is True
    assert plan.prioritized[0].score > plan.prioritized[1].score


def test_network_trace_sample_controls_transfer_estimate() -> None:
    candidate = _candidate("base", "gaussian_base", deadline_ms=20_000, size_bytes=1_000_000)
    trace = NetworkTrace(
        trace_id="trace-a",
        samples=(
            NetworkSample(timestamp_ms=0, throughput_bps=20_000_000, latency_ms=5),
            NetworkSample(timestamp_ms=100, throughput_bps=1_000_000, latency_ms=50, jitter_ms=10),
        ),
    )

    fast = prioritize_transport_candidates((candidate,), now_ms=50, network=trace)
    slow = prioritize_transport_candidates((candidate,), now_ms=150, network=trace)

    assert fast.prioritized[0].estimated_transfer_ms == 405.0
    assert slow.prioritized[0].estimated_transfer_ms == 8060.0
    assert slow.metadata["transport"]["network_sample"]["timestamp_ms"] == 100


def test_drop_expired_references_emits_clean_lifecycle_events() -> None:
    expired_requested, _ = ReferenceLifecycleStateMachine("ref-expired", deadline_ms=5).request(at_ms=1)
    active, _ = ReferenceLifecycleStateMachine("ref-active", deadline_ms=50).request(at_ms=1)
    candidate = ReferenceLifecycleStateMachine("ref-candidate", deadline_ms=5)

    result = drop_expired_references((expired_requested, active, candidate), now_ms=10)

    assert [state.reference_id for state in result.active_states] == ["ref-active"]
    assert [state.reference_id for state in result.dropped_states] == ["ref-expired", "ref-candidate"]
    assert result.dropped_states[0].status == LifecycleStatus.EXPIRED
    assert result.dropped_states[0].metadata["lifecycle"]["drop_reason"] == "deadline_missed"
    assert result.dropped_states[1].status == LifecycleStatus.DROPPED
    assert result.dropped_states[1].metadata["lifecycle"]["drop_reason"] == "expired"
    assert [event.drop_reason for event in result.events] == [DropReason.DEADLINE_MISSED, DropReason.EXPIRED]


def test_already_terminal_expired_references_are_reported_without_new_event() -> None:
    machine, _ = ReferenceLifecycleStateMachine("ref-terminal", deadline_ms=5).cancel(at_ms=3)

    result = drop_expired_references((machine,), now_ms=10)

    assert len(result.events) == 0
    assert result.dropped_states[0].status == LifecycleStatus.DROPPED


def test_transport_validation_rejects_bad_inputs() -> None:
    with pytest.raises(TransportError, match="now_ms"):
        prioritize_transport_candidates((), now_ms=-1)
    with pytest.raises(TransportError, match="weights"):
        prioritize_transport_candidates((), now_ms=0, weights=object())  # type: ignore[arg-type]
    with pytest.raises(TransportError, match="base"):
        TransportPriorityWeights(base=-1)
    with pytest.raises(TransportError, match="machines"):
        drop_expired_references((object(),), now_ms=1)  # type: ignore[arg-type]


def _candidate(
    candidate_id: str,
    candidate_kind: str,
    *,
    deadline_ms: int,
    size_bytes: int = 100_000,
    retransmit_priority: int = 0,
    tile: TileSpec | None = None,
    layer: int = 0,
) -> CandidateObject:
    return CandidateObject(
        candidate_id=candidate_id,
        object_id=f"object-{candidate_id}",
        candidate_kind=candidate_kind,
        decision_time_ms=0,
        layer=layer,
        resolution="720p",
        fov_deg=90,
        viewpoint=None,
        lookahead_ms=0,
        expiration_ms=deadline_ms,
        retransmit_priority=retransmit_priority,
        size_bytes=size_bytes,
        tile=tile,
        metadata={"test": {"phase": LifecyclePhase.CANDIDATE.value}},
    )
