from __future__ import annotations

import math

import pytest

from ref_abr.viewport import (
    ViewportError,
    ViewportSweepConfig,
    generate_viewport_error_sweeps,
    load_viewport_trace,
    normalize_viewport_trace,
)


def test_normalize_6dof_pose_trace_with_degrees_and_defaults() -> None:
    trace = normalize_viewport_trace(
        {
            "poses": [
                {"time_ms": 20, "position": [1, 2, 3], "yaw": 370, "pitch": 190, "roll": -190, "fov": 100},
                {"time_ms": 0, "x": 0, "y": 0, "z": 0, "yaw_rad": math.pi},
            ]
        },
        trace_id="trace-a",
    )

    assert trace.trace_id == "trace-a"
    assert [pose.timestamp_ms for pose in trace.poses] == [0, 20]
    assert trace.poses[0].yaw_deg == 180
    assert trace.poses[0].fov_deg == 90
    assert trace.poses[1].yaw_deg == 10
    assert trace.poses[1].pitch_deg == -170
    assert trace.poses[1].roll_deg == 170


def test_load_viewport_trace_accepts_quaternion_samples(tmp_path) -> None:
    trace_path = tmp_path / "viewport.yml"
    trace_path.write_text(
        """
samples:
  - timestamp_ms: 0
    position_m: {x: 0, y: 0, z: 0}
    orientation:
      quaternion: [0, 0, 0, 1]
  - timestamp_ms: 16
    position_m: {x: 1, y: 0, z: 0}
    orientation:
      qx: 0
      qy: 0
      qz: 0.70710678
      qw: 0.70710678
    fov_deg: 95
""",
        encoding="utf-8",
    )

    trace = load_viewport_trace(trace_path)

    assert trace.source_uri == str(trace_path)
    assert trace.poses[0].yaw_deg == 0
    assert round(trace.poses[1].yaw_deg) == 90
    assert trace.metadata["provenance"]["input_format"] == "samples"


def test_generate_controlled_error_sweeps() -> None:
    trace = normalize_viewport_trace(
        {
            "frames": [
                {"timestamp_ms": 0, "yaw": 0, "pitch": 0, "roll": 0, "fov": 90},
                {"timestamp_ms": 16, "yaw": 10, "pitch": 0, "roll": 0, "fov": 90},
            ]
        },
        trace_id="base",
    )
    sweeps = generate_viewport_error_sweeps(
        trace,
        ViewportSweepConfig(
            angular_degrees=(5,),
            translation_meters=(0.25,),
            fov_degrees=(10,),
            horizon_degrees=(3,),
            sudden_turn_degrees=(30,),
            adversarial_degrees=(8,),
        ),
    )

    assert [sweep.metadata["error_sweep"]["kind"] for sweep in sweeps] == [
        "angular",
        "translational",
        "fov",
        "horizon",
        "sudden-turn",
        "adversarial",
    ]
    assert sweeps[0].poses[0].yaw_deg == 5
    assert sweeps[1].poses[0].x_m == 0.25
    assert sweeps[2].poses[0].fov_deg == 100
    assert sweeps[3].poses[0].roll_deg == 3
    assert sweeps[4].poses[0].yaw_deg == 0
    assert sweeps[4].poses[1].yaw_deg == 40
    assert sweeps[5].poses[0].yaw_deg == 8
    assert sweeps[5].poses[1].yaw_deg == 2


def test_invalid_viewport_trace_inputs_raise_clear_errors() -> None:
    with pytest.raises(ViewportError, match="non-empty poses"):
        normalize_viewport_trace({"poses": []})

    with pytest.raises(ViewportError, match="fov_deg"):
        normalize_viewport_trace({"poses": [{"timestamp_ms": 0, "fov": 200}]})

    with pytest.raises(ViewportError, match="quaternion"):
        normalize_viewport_trace({"poses": [{"timestamp_ms": 0, "orientation": [0, 0]}]})


def test_trace_id_is_stable_for_equivalent_pose_content() -> None:
    first = normalize_viewport_trace({"poses": [{"timestamp_ms": 0, "yaw": 360, "fov": 90}]})
    second = normalize_viewport_trace({"poses": [{"fov": 90, "yaw": 0, "timestamp_ms": 0}]})

    assert first.trace_id == second.trace_id
