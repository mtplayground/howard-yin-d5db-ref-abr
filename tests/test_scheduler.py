from __future__ import annotations

import pytest

from ref_abr.candidates import CandidateGenerationSpec
from ref_abr.domain import MediaType
from ref_abr.methods import ActionBudget, ObservationBudget, SchedulingObservation
from ref_abr.scheduler import (
    DiscreteEventScheduler,
    SchedulerClock,
    SchedulerConfig,
    SchedulerError,
    run_discrete_event_schedule,
)
from ref_abr.substrate import ParametricSubstrateValueProvider
from ref_abr.utility import ResourceBudget
from ref_abr.workloads import assemble_workload_manifest


def test_discrete_event_scheduler_drives_method_across_display_clock() -> None:
    method = RecordingSchedulerMethod()
    result = run_discrete_event_schedule(
        method,
        _workload(),
        config=_config(frame_count=3),
        substrate_provider=ParametricSubstrateValueProvider(),
    )

    assert len(result.epochs) == 3
    assert [epoch.deadline.display_time_ms for epoch in result.epochs] == [100, 120, 140]
    assert [epoch.deadline.target_deadline_ms for epoch in result.epochs] == [135, 155, 175]
    assert [epoch.controller_state.step_index for epoch in result.epochs] == [0, 1, 2]
    assert [seen["step_index"] for seen in method.seen] == [0, 1, 2]
    assert [seen["deadline_slack_ms"] for seen in method.seen] == [35, 35, 35]
    assert all(epoch.decision.controller_id == "ctrl-scheduler" for epoch in result.epochs)
    assert all(epoch.decision.metadata["adapter"]["method_id"] == "recording-scheduler" for epoch in result.epochs)
    assert result.metadata["scheduler"]["decision_count"] == 3


def test_scheduler_controller_state_carries_previous_decision_context() -> None:
    result = run_discrete_event_schedule(
        RecordingSchedulerMethod(),
        _workload(),
        config=_config(frame_count=2),
        substrate_provider=ParametricSubstrateValueProvider(),
    )

    first_state = result.epochs[0].controller_state.state
    second_state = result.epochs[1].controller_state.state
    assert first_state["previous_decision_id"] is None
    assert first_state["previous_selected_object_ids"] == ()
    assert second_state["previous_decision_id"] == result.epochs[0].decision.decision_id
    assert second_state["previous_selected_object_ids"] == result.epochs[0].decision.selected_object_ids
    assert second_state["deadline"]["frame_id"] == "sched-000001"


def test_scheduler_run_id_is_deterministic_for_same_inputs() -> None:
    first = run_discrete_event_schedule(
        RecordingSchedulerMethod(),
        _workload(),
        config=_config(frame_count=2),
        substrate_provider=ParametricSubstrateValueProvider(),
    )
    second = run_discrete_event_schedule(
        RecordingSchedulerMethod(),
        _workload(),
        config=_config(frame_count=2),
        substrate_provider=ParametricSubstrateValueProvider(),
    )

    assert first.run_id == second.run_id
    assert [decision.decision_id for decision in first.decisions] == [decision.decision_id for decision in second.decisions]


def test_scheduler_payload_includes_observation_and_deadline_records() -> None:
    result = DiscreteEventScheduler(
        method=RecordingSchedulerMethod(),
        workload=_workload(),
        config=_config(frame_count=1),
        substrate_provider=ParametricSubstrateValueProvider(),
    ).run()

    payload = result.as_payload()
    assert payload["run_id"] == result.run_id
    assert payload["epochs"][0]["deadline"]["slack_ms"] == 35
    assert payload["epochs"][0]["observation"]["metadata"]["scheduler"]["deadline"]["frame_id"] == "sched-000000"
    assert payload["epochs"][0]["controller_state"]["state"]["clock"]["display_time_ms"] == 100


def test_scheduler_validation_rejects_invalid_config_and_method() -> None:
    with pytest.raises(SchedulerError, match="display_interval_ms"):
        SchedulerClock(display_interval_ms=0, motion_to_photon_latency_ms=10)
    with pytest.raises(SchedulerError, match="frame_count"):
        _config(frame_count=0)
    with pytest.raises(SchedulerError, match="method"):
        DiscreteEventScheduler(
            method=object(),  # type: ignore[arg-type]
            workload=_workload(),
            config=_config(frame_count=1),
        )


def test_clock_from_fps_uses_integer_display_interval() -> None:
    clock = SchedulerClock.from_fps(60, motion_to_photon_latency_ms=20, start_time_ms=5)

    assert clock.display_interval_ms == 17
    assert clock.display_time_ms(2) == 39
    assert clock.target_deadline_ms(2) == 59


class RecordingSchedulerMethod:
    method_id = "recording-scheduler"
    method_name = "recording scheduler"

    def __init__(self) -> None:
        self.seen: list[dict[str, int | str]] = []

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, object]:
        deadline = observation.controller_state.state["deadline"]
        self.seen.append(
            {
                "step_index": observation.controller_state.step_index,
                "frame_id": observation.frame_id,
                "deadline_slack_ms": deadline["slack_ms"],
            }
        )
        selected = observation.candidates[:1]
        return {
            "selected_candidate_ids": [candidate.candidate_id for candidate in selected],
            "expected_utility": 0.5,
        }


def _config(*, frame_count: int) -> SchedulerConfig:
    return SchedulerConfig(
        frame_count=frame_count,
        clock=SchedulerClock(display_interval_ms=20, motion_to_photon_latency_ms=35, start_time_ms=100),
        observation_budget=ObservationBudget(max_candidates=20),
        action_budget=ActionBudget(max_selected_objects=1, max_selected_candidates=1, max_selected_bytes=1_000_000),
        candidate_generation_spec=CandidateGenerationSpec(
            resolutions=("720p",),
            fov_degrees=(90,),
            lookahead_ms=(0,),
            expiration_ms=(100,),
            retransmit_priorities=(0,),
            enhancement_layers=(1,),
            include_tiles=False,
            include_reference_actions=False,
        ),
        resource_budget=ResourceBudget(available_time_ms=100, available_bytes=1_000_000, available_memory_mb=1024),
        controller_id="ctrl-scheduler",
        frame_id_prefix="sched",
    )


def _workload():
    return assemble_workload_manifest(
        {
            "dataset": "scheduler-test",
            "sequences": [
                {
                    "scene": "scene",
                    "name": "seq",
                    "assets": [
                        {
                            "object_id": "splat-a",
                            "path": "splat-a.ply",
                            "size_bytes": 100_000,
                            "media_type": MediaType.GAUSSIAN_SPLAT.value,
                        },
                        {
                            "object_id": "splat-b",
                            "path": "splat-b.ply",
                            "size_bytes": 100_000,
                            "media_type": MediaType.GAUSSIAN_SPLAT.value,
                        },
                    ],
                }
            ],
        },
        split="calibration",
        config_id="scheduler-test-config",
        seed=20,
    )
