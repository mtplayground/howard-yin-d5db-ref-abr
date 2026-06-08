from __future__ import annotations

import pytest

from ref_abr.devices import normalize_device_profiles
from ref_abr.network import normalize_network_trace
from ref_abr.replay import (
    ReplayCase,
    ReplayError,
    ReplaySourceRef,
    ReplaySubsetManifest,
    ReplaySubsetSpec,
    assemble_replay_subset,
    replay_subset_spec_from_mapping,
    validate_replay_subset_manifest,
)
from ref_abr.viewport import normalize_viewport_trace
from ref_abr.workloads import assemble_workload_manifest


def test_assemble_replay_subset_builds_deterministic_cross_product() -> None:
    workloads = (_workload("b"), _workload("a"))
    viewports = (_viewport("vp-b", yaw=10), _viewport("vp-a", yaw=0))
    networks = (_network("net-a", mbps=5),)
    devices = normalize_device_profiles(
        {
            "profiles": [
                _device("mobile-z", "mobile", 30),
                _device("edge-a", "edge", 60),
            ]
        },
        source_uri="memory://devices",
    )

    manifest = assemble_replay_subset(
        workloads=workloads,
        viewport_traces=viewports,
        network_traces=networks,
        device_profiles=devices,
        config_id="cfg-a",
        split="train",
        seed=9,
        spec=ReplaySubsetSpec(max_workloads=2, max_viewports=1, max_networks=1, max_devices=2),
    )

    assert manifest.config_id == "cfg-a"
    assert manifest.split == "train"
    assert len(manifest.cases) == 4
    assert [source.record_id for source in manifest.sources["workloads"]] == sorted(
        workload.manifest_id for workload in workloads
    )
    assert [source.record_id for source in manifest.sources["viewports"]] == ["vp-a"]
    assert [source.record_id for source in manifest.sources["devices"]] == ["edge-a", "mobile-z"]
    assert manifest.metadata["provenance"]["axis_counts"] == {
        "workloads": 2,
        "viewports": 1,
        "networks": 1,
        "devices": 2,
    }
    assert manifest.metadata["provenance"]["case_count"] == 4
    assert all(case.case_id.startswith("replay-case-") for case in manifest.cases)
    assert manifest.subset_id.startswith("replay-subset-")
    assert validate_replay_subset_manifest(manifest) is manifest


def test_replay_subset_id_is_stable_for_equivalent_inputs_in_different_order() -> None:
    workloads = (_workload("a"), _workload("b"))
    viewports = (_viewport("vp-a", yaw=0), _viewport("vp-b", yaw=10))
    networks = (_network("net-a", mbps=5), _network("net-b", mbps=7))
    devices = normalize_device_profiles(
        {"profiles": [_device("edge-a", "edge", 60), _device("mobile-z", "mobile", 30)]},
        source_uri="memory://devices",
    )
    spec = ReplaySubsetSpec(max_workloads=2, max_viewports=2, max_networks=1, max_devices=2, max_cases=5)

    first = assemble_replay_subset(
        workloads=workloads,
        viewport_traces=viewports,
        network_traces=networks,
        device_profiles=devices,
        config_id="cfg-a",
        split="calibration",
        seed=3,
        spec=spec,
    )
    second = assemble_replay_subset(
        workloads=tuple(reversed(workloads)),
        viewport_traces=tuple(reversed(viewports)),
        network_traces=tuple(reversed(networks)),
        device_profiles=tuple(reversed(devices.profiles)),
        config_id="cfg-a",
        split="calibration",
        seed=3,
        spec=spec,
    )

    assert first.subset_id == second.subset_id
    assert [case.as_payload() for case in first.cases] == [case.as_payload() for case in second.cases]
    assert len(first.cases) == 5


def test_replay_subset_spec_from_mapping() -> None:
    spec = replay_subset_spec_from_mapping(
        {
            "max_workloads": "1",
            "max_viewports": 2,
            "max_networks": None,
            "max_devices": 1,
            "max_cases": 3,
        }
    )

    assert spec.as_payload() == {
        "max_workloads": 1,
        "max_viewports": 2,
        "max_networks": None,
        "max_devices": 1,
        "max_cases": 3,
    }


def test_replay_subset_validation_rejects_incomplete_manifest() -> None:
    source = ReplaySourceRef(
        group="workloads",
        record_id="workload-a",
        record_type="workload_manifest",
        source_uri="memory://workload-a",
        provenance={"source_uri": "memory://workload-a"},
    )
    case = ReplayCase(
        case_id="case-a",
        workload_manifest_id="missing",
        viewport_trace_id="vp-a",
        network_trace_id="net-a",
        device_profile_id="device-a",
    )

    with pytest.raises(ReplayError, match="sources.viewports"):
        ReplaySubsetManifest(
            subset_id="subset-a",
            config_id="cfg-a",
            split="train",
            seed=0,
            cases=(case,),
            sources={"workloads": (source,), "viewports": (), "networks": (), "devices": ()},
            metadata={"provenance": {"assembly": "test"}},
        )

    valid_sources = {
        "workloads": (source,),
        "viewports": (
            ReplaySourceRef(
                group="viewports",
                record_id="vp-a",
                record_type="viewport_trace",
                provenance={"source_uri": "memory://vp-a"},
            ),
        ),
        "networks": (
            ReplaySourceRef(
                group="networks",
                record_id="net-a",
                record_type="network_trace",
                provenance={"source_uri": "memory://net-a"},
            ),
        ),
        "devices": (
            ReplaySourceRef(
                group="devices",
                record_id="device-a",
                record_type="device_profile",
                provenance={"source_uri": "memory://device-a"},
            ),
        ),
    }
    with pytest.raises(ReplayError, match="unknown source id"):
        ReplaySubsetManifest(
            subset_id="subset-a",
            config_id="cfg-a",
            split="train",
            seed=0,
            cases=(case,),
            sources=valid_sources,
            metadata={"provenance": {"assembly": "test"}},
        )


def test_replay_subset_validation_requires_source_provenance() -> None:
    case = ReplayCase(
        case_id="case-a",
        workload_manifest_id="workload-a",
        viewport_trace_id="vp-a",
        network_trace_id="net-a",
        device_profile_id="device-a",
    )
    sources = {
        "workloads": (ReplaySourceRef("workloads", "workload-a", "workload_manifest"),),
        "viewports": (ReplaySourceRef("viewports", "vp-a", "viewport_trace", provenance={"source_uri": "x"}),),
        "networks": (ReplaySourceRef("networks", "net-a", "network_trace", provenance={"source_uri": "x"}),),
        "devices": (ReplaySourceRef("devices", "device-a", "device_profile", provenance={"source_uri": "x"}),),
    }

    with pytest.raises(ReplayError, match="missing provenance"):
        ReplaySubsetManifest(
            subset_id="subset-a",
            config_id="cfg-a",
            split="train",
            seed=0,
            cases=(case,),
            sources=sources,
            metadata={"provenance": {"assembly": "test"}},
        )


def test_assemble_replay_subset_rejects_empty_axes_and_bad_spec() -> None:
    with pytest.raises(ReplayError, match="max_workloads"):
        ReplaySubsetSpec(max_workloads=0)

    with pytest.raises(ReplayError, match="workloads"):
        assemble_replay_subset(
            workloads=(),
            viewport_traces=(_viewport("vp-a", yaw=0),),
            network_traces=(_network("net-a", mbps=5),),
            device_profiles=normalize_device_profiles({"profiles": [_device("edge-a", "edge", 60)]}),
            config_id="cfg-a",
            split="train",
            seed=0,
        )


def _workload(name: str):
    return assemble_workload_manifest(
        {
            "dataset": "n3dv",
            "sequences": [
                {
                    "scene": name,
                    "name": "seq",
                    "assets": [{"object_id": f"{name}-splat", "path": f"{name}.ply", "size_bytes": 10}],
                }
            ],
        },
        config_id="cfg-a",
        split="train",
        seed=1,
        source_uri=f"memory://workload/{name}",
    )


def _viewport(trace_id: str, *, yaw: float):
    return normalize_viewport_trace(
        {"poses": [{"timestamp_ms": 0, "yaw": yaw, "fov": 90}]},
        trace_id=trace_id,
        source_uri=f"memory://viewport/{trace_id}",
    )


def _network(trace_id: str, *, mbps: float):
    return normalize_network_trace(
        {"network_type": "broadband", "samples": [{"timestamp_ms": 0, "throughput_mbps": mbps}]},
        trace_id=trace_id,
        source_uri=f"memory://network/{trace_id}",
    )


def _device(profile_id: str, device_class: str, fps: float) -> dict[str, object]:
    return {
        "profile_id": profile_id,
        "device_class": device_class,
        "generation_ms": 1,
        "transfer_ms": 1,
        "restoration_ms": 1,
        "render_ms": 1,
        "memory_mb": 1024,
        "fps": fps,
    }
