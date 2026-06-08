from __future__ import annotations

import json

import pytest

from ref_abr.config import ConfigError, load_config_file, load_env_config, resolve_config
from ref_abr.devices import DeviceError, load_device_profiles
from ref_abr.network import NetworkError, load_network_trace
from ref_abr.viewport import ViewportError, load_viewport_trace
from ref_abr.workloads import WorkloadError, load_workload_manifest


def test_toy_workload_viewport_network_and_device_loaders_accept_fixture_files(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    workload_path = tmp_path / "workload.yaml"
    viewport_path = tmp_path / "viewport.json"
    network_path = tmp_path / "network.toml"
    devices_path = tmp_path / "devices.yaml"

    workload_path.write_text(
        """
dataset_name: toy-scenes
scenes:
  - scene_id: atrium
    sequences:
      - sequence_id: spin
        media_objects:
          - object_id: base-gaussian
            uri: assets/base.ply
            media_type: gaussian_splat
            size_bytes: 1024
          - object_id: reference-frame
            uri: refs/frame.png
            media_type: texture
            size_bytes: 512
            dependencies: [base-gaussian]
""",
        encoding="utf-8",
    )
    viewport_path.write_text(
        json.dumps(
            {
                "poses": [
                    {"timestamp_ms": 0, "position": {"x": 0, "y": 0, "z": 1}, "orientation": {"yaw": 370}, "fov": 100},
                    {"timestamp_ms": 16, "position": [0.1, 0.0, 1.0], "orientation": [10, 0, 0], "fov_deg": 95},
                ]
            }
        ),
        encoding="utf-8",
    )
    network_path.write_text(
        """
network_type = "wifi"
[[samples]]
timestamp_ms = 0
throughput_mbps = 12.5
latency_ms = 20
packet_loss_percent = 1.0
[[samples]]
timestamp_ms = 100
throughput_kbps = 8000
latency_ms = 35
jitter_ms = 4
""",
        encoding="utf-8",
    )
    devices_path.write_text(
        """
profiles:
  desktop-toy:
    device_class: desktop
    generation_ms: 5
    transfer_ms: 12
    restoration_ms: 4
    render_ms: 6
    memory_mb: 4096
    fps: 90
  mobile-toy:
    device_class: mobile
    generation_ms: 10
    transfer_ms: 20
    restoration_ms: 8
    render_ms: 12
    memory_mb: 1024
    fps: 60
""",
        encoding="utf-8",
    )

    workload = load_workload_manifest(workload_path, config_id="toy-config", split="train", seed=7, dataset_base_path=dataset_root)
    viewport = load_viewport_trace(viewport_path, trace_id="viewport-toy")
    network = load_network_trace(network_path, trace_id="network-toy")
    devices = load_device_profiles(devices_path)

    assert workload.config_id == "toy-config"
    assert workload.split == "train"
    assert [media.object_id for media in workload.media_objects] == ["base-gaussian", "reference-frame"]
    assert workload.media_objects[1].dependencies == ("base-gaussian",)
    assert viewport.trace_id == "viewport-toy"
    assert [pose.timestamp_ms for pose in viewport.poses] == [0, 16]
    assert viewport.poses[0].yaw_deg == 10.0
    assert network.samples[0].throughput_bps == 12_500_000
    assert network.samples[0].packet_loss == 0.01
    assert devices.by_id("mobile-toy").budgets.fps == 60.0


def test_config_split_resolution_env_defaults_and_overrides_use_toy_files(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        """
seed: 123
split: train
splits:
  train:
    identity: toy-train
  calibration:
    seed: 456
  final:
    identity: toy-final
dataset:
  name: toy
""",
        encoding="utf-8",
    )
    env_path.write_text(
        """
REF_ABR_ARTIFACT_ROOT=./toy-artifacts
REF_ABR_DEFAULT_SPLIT=calibration
REF_ABR_DEFAULT_SEED=99
REF_ABR_OVERWRITE_OUTPUTS=true
""",
        encoding="utf-8",
    )

    loaded = load_config_file(config_path)
    env_config = load_env_config(environ={}, env_file=env_path)
    resolved = resolve_config(
        config_path,
        overrides={"dataset.name": "toy-overridden", "splits.final.seed": "789"},
        split="final",
    )

    assert loaded["dataset"]["name"] == "toy"
    assert env_config.default_split == "calibration"
    assert env_config.default_seed == 99
    assert env_config.overwrite_outputs is True
    assert resolved.active_split == "final"
    assert resolved.splits["train"].identity == "toy-train"
    assert resolved.splits["calibration"].seed == 456
    assert resolved.splits["final"].identity == "toy-final"
    assert resolved.splits["final"].seed == 789
    assert resolved.values["dataset"]["name"] == "toy-overridden"


def test_malformed_loader_inputs_are_rejected_with_clear_errors(tmp_path) -> None:
    workload_path = tmp_path / "bad-workload.json"
    viewport_path = tmp_path / "bad-viewport.json"
    network_path = tmp_path / "bad-network.json"
    devices_path = tmp_path / "bad-devices.json"
    workload_path.write_text(json.dumps({"media_objects": [{"object_id": "missing-uri", "size_bytes": 1}]}), encoding="utf-8")
    viewport_path.write_text(json.dumps({"poses": []}), encoding="utf-8")
    network_path.write_text(json.dumps({"samples": [{"timestamp_ms": 0, "latency_ms": 1}]}), encoding="utf-8")
    devices_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "profile_id": "bad-mobile",
                        "device_class": "mobile",
                        "generation_ms": 1,
                        "transfer_ms": 1,
                        "restoration_ms": 1,
                        "render_ms": 1,
                        "memory_mb": 512,
                        "fps": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkloadError, match="missing uri"):
        load_workload_manifest(workload_path, config_id="bad", split="train", seed=0)
    with pytest.raises(ViewportError, match="non-empty"):
        load_viewport_trace(viewport_path)
    with pytest.raises(NetworkError, match="missing throughput"):
        load_network_trace(network_path)
    with pytest.raises(DeviceError, match="fps"):
        load_device_profiles(devices_path)


def test_malformed_config_and_split_inputs_are_rejected(tmp_path) -> None:
    unsupported = tmp_path / "config.txt"
    bad_yaml = tmp_path / "bad.yaml"
    override_conflict = tmp_path / "override.yaml"
    bad_split = tmp_path / "bad-split.yaml"
    unsupported.write_text("seed=1", encoding="utf-8")
    bad_yaml.write_text("seed: [", encoding="utf-8")
    override_conflict.write_text("seed: 1\ndataset: toy\n", encoding="utf-8")
    bad_split.write_text("seed: 1\nsplit: holdout\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Unsupported"):
        load_config_file(unsupported)
    with pytest.raises(ConfigError, match="Could not parse"):
        load_config_file(bad_yaml)
    with pytest.raises(ConfigError, match="conflicts"):
        resolve_config(override_conflict, overrides={"dataset.name": "bad"})
    with pytest.raises(ConfigError, match="split"):
        resolve_config(bad_split)
