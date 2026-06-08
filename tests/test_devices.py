from __future__ import annotations

import pytest

from ref_abr.devices import (
    DEVICE_CLASSES,
    DeviceBudgets,
    DeviceError,
    DeviceProfile,
    load_device_profiles,
    normalize_device_profiles,
)


def test_load_device_profiles_from_yaml_with_nested_budgets(tmp_path) -> None:
    profiles_path = tmp_path / "devices.yml"
    profiles_path.write_text(
        """
profiles:
  - profile_id: server-a100
    device_class: server
    budgets:
      generation_ms: 4.5
      transfer_ms: 8
      restoration_ms: 2
      render_ms: 6
      memory_gb: 40
      target_fps: 60
    metadata:
      gpu: A100
  - id: phone-reference
    kind: smartphone
    timing:
      generate_ms: 35
      network_transfer_ms: 120
      restore_ms: 15
      frame_render_ms: 18
    resources:
      memory_mb: 2048
    fps: 30
""",
        encoding="utf-8",
    )

    profile_set = load_device_profiles(profiles_path)

    assert profile_set.source_uri == str(profiles_path)
    assert [profile.profile_id for profile in profile_set.profiles] == ["server-a100", "phone-reference"]
    assert profile_set.profiles[0].budgets.memory_mb == 40 * 1024
    assert profile_set.profiles[1].device_class == "mobile"
    assert profile_set.profiles[1].budgets.transfer_ms == 120
    assert profile_set.by_id("server-a100").metadata["gpu"] == "A100"
    assert profile_set.metadata["provenance"]["device_classes"] == ["mobile", "server"]


def test_normalize_mapping_profiles_and_preserve_attributes() -> None:
    profile_set = normalize_device_profiles(
        {
            "devices": {
                "edge-small": {
                    "generation_ms": 10,
                    "transfer_ms": 20,
                    "restoration_ms": 5,
                    "render_ms": 11,
                    "memory_bytes": 512_000_000,
                    "fps": 45,
                    "accelerator": "integrated",
                },
                "laptop-pro": {
                    "class": "notebook",
                    "budgets": {
                        "generation": 22,
                        "transfer": 35,
                        "restoration": 9,
                        "render": 14,
                    },
                    "performance": {"frames_per_second": 90},
                    "resources": {"vram_gb": 8},
                },
            }
        }
    )

    assert [profile.profile_id for profile in profile_set.profiles] == ["edge-small", "laptop-pro"]
    assert profile_set.profiles[0].device_class == "edge"
    assert profile_set.profiles[0].budgets.memory_mb == 512
    assert profile_set.profiles[0].metadata["attributes"]["accelerator"] == "integrated"
    assert profile_set.profiles[1].device_class == "laptop"
    assert profile_set.profiles[1].budgets.memory_mb == 8192
    assert profile_set.profiles[1].budgets.fps == 90


def test_single_profile_without_id_gets_stable_generated_id() -> None:
    payload = {
        "device_class": "desktop",
        "generation_ms": 12,
        "transfer_ms": 16,
        "restoration_ms": 7,
        "render_ms": 8,
        "memory_mb": 4096,
        "fps": 60,
    }

    first = normalize_device_profiles(payload)
    second = normalize_device_profiles(dict(reversed(list(payload.items()))))

    assert first.profiles[0].profile_id == second.profiles[0].profile_id
    assert first.profiles[0].profile_id.startswith("desktop-")
    assert first.profiles[0].budgets == DeviceBudgets(12, 16, 7, 8, 4096, 60)


def test_device_profile_record_validation() -> None:
    assert DEVICE_CLASSES == ("server", "edge", "desktop", "laptop", "mobile")

    with pytest.raises(DeviceError, match="device_class"):
        DeviceProfile(
            profile_id="x",
            device_class="satellite",
            budgets=DeviceBudgets(1, 1, 1, 1, 1, 1),
        )

    with pytest.raises(DeviceError, match="memory_mb"):
        DeviceBudgets(1, 1, 1, 1, 0, 60)

    with pytest.raises(DeviceError, match="fps"):
        DeviceBudgets(1, 1, 1, 1, 1024, 0)


def test_invalid_device_profile_inputs_raise_clear_errors() -> None:
    with pytest.raises(DeviceError, match="at least one profile"):
        normalize_device_profiles({"profiles": []})

    with pytest.raises(DeviceError, match="generation_ms"):
        normalize_device_profiles(
            {
                "profiles": [
                    {
                        "profile_id": "mobile-a",
                        "device_class": "mobile",
                        "transfer_ms": 1,
                        "restoration_ms": 1,
                        "render_ms": 1,
                        "memory_mb": 512,
                        "fps": 30,
                    }
                ]
            }
        )

    with pytest.raises(DeviceError, match="duplicate profile_id"):
        normalize_device_profiles(
            {
                "profiles": [
                    {
                        "profile_id": "edge-a",
                        "device_class": "edge",
                        "generation_ms": 1,
                        "transfer_ms": 1,
                        "restoration_ms": 1,
                        "render_ms": 1,
                        "memory_mb": 512,
                        "fps": 30,
                    },
                    {
                        "profile_id": "edge-a",
                        "device_class": "edge",
                        "generation_ms": 2,
                        "transfer_ms": 2,
                        "restoration_ms": 2,
                        "render_ms": 2,
                        "memory_mb": 1024,
                        "fps": 60,
                    },
                ]
            }
        )
