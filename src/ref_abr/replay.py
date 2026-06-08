"""Deterministic replay subset assembly and validation."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.config import stable_config_id
from ref_abr.devices import DeviceProfile, DeviceProfileSet
from ref_abr.domain import WorkloadManifest
from ref_abr.network import NetworkTrace
from ref_abr.viewport import ViewportTrace


class ReplayError(ValueError):
    """Raised when a replay subset cannot be assembled or validated."""


SOURCE_GROUPS: tuple[str, ...] = ("workloads", "viewports", "networks", "devices")


@dataclass(frozen=True)
class ReplaySubsetSpec:
    """Axis limits for the small deterministic replay cross product."""

    max_workloads: int | None = 2
    max_viewports: int | None = 2
    max_networks: int | None = 2
    max_devices: int | None = 2
    max_cases: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("max_workloads", "max_viewports", "max_networks", "max_devices", "max_cases"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _positive_int(value, field_name))

    def as_payload(self) -> dict[str, int | None]:
        return {
            "max_workloads": self.max_workloads,
            "max_viewports": self.max_viewports,
            "max_networks": self.max_networks,
            "max_devices": self.max_devices,
            "max_cases": self.max_cases,
        }


@dataclass(frozen=True)
class ReplaySourceRef:
    """Reference to a normalized source record used by replay cases."""

    group: str
    record_id: str
    record_type: str
    source_uri: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.group not in SOURCE_GROUPS:
            raise ReplayError(f"group must be one of: {', '.join(SOURCE_GROUPS)}.")
        _require_non_empty(self.record_id, "record_id")
        _require_non_empty(self.record_type, "record_type")
        if self.source_uri is not None:
            _require_non_empty(self.source_uri, "source_uri")
        object.__setattr__(self, "provenance", _plain_json_mapping(self.provenance, "provenance"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "record_id": self.record_id,
            "record_type": self.record_type,
            "source_uri": self.source_uri,
            "provenance": _to_payload(self.provenance),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class ReplayCase:
    """One deterministic replay combination."""

    case_id: str
    workload_manifest_id: str
    viewport_trace_id: str
    network_trace_id: str
    device_profile_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.case_id, "case_id")
        _require_non_empty(self.workload_manifest_id, "workload_manifest_id")
        _require_non_empty(self.viewport_trace_id, "viewport_trace_id")
        _require_non_empty(self.network_trace_id, "network_trace_id")
        _require_non_empty(self.device_profile_id, "device_profile_id")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "workload_manifest_id": self.workload_manifest_id,
            "viewport_trace_id": self.viewport_trace_id,
            "network_trace_id": self.network_trace_id,
            "device_profile_id": self.device_profile_id,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class ReplaySubsetManifest:
    """Reproducible subset manifest with cases and source references."""

    subset_id: str
    config_id: str
    split: str
    seed: int
    cases: tuple[ReplayCase, ...]
    sources: Mapping[str, tuple[ReplaySourceRef, ...]]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.subset_id, "subset_id")
        _require_non_empty(self.config_id, "config_id")
        _require_non_empty(self.split, "split")
        object.__setattr__(self, "seed", _non_negative_int(self.seed, "seed"))
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "sources", _freeze_sources(self.sources))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))
        validate_replay_subset_manifest(self)

    def as_payload(self) -> dict[str, Any]:
        return {
            "subset_id": self.subset_id,
            "config_id": self.config_id,
            "split": self.split,
            "seed": self.seed,
            "cases": [case.as_payload() for case in self.cases],
            "sources": {
                group: [source.as_payload() for source in self.sources[group]]
                for group in SOURCE_GROUPS
            },
            "metadata": _to_payload(self.metadata),
        }


def assemble_replay_subset(
    *,
    workloads: Iterable[WorkloadManifest],
    viewport_traces: Iterable[ViewportTrace],
    network_traces: Iterable[NetworkTrace],
    device_profiles: Iterable[DeviceProfile] | DeviceProfileSet,
    config_id: str,
    split: str,
    seed: int,
    spec: ReplaySubsetSpec | None = None,
) -> ReplaySubsetManifest:
    """Build a small deterministic cross-product replay subset."""

    subset_spec = spec or ReplaySubsetSpec()
    selected_workloads = _select_axis(tuple(workloads), subset_spec.max_workloads, _workload_id, "workloads")
    selected_viewports = _select_axis(tuple(viewport_traces), subset_spec.max_viewports, _viewport_id, "viewports")
    selected_networks = _select_axis(tuple(network_traces), subset_spec.max_networks, _network_id, "networks")
    selected_devices = _select_axis(_device_tuple(device_profiles), subset_spec.max_devices, _device_id, "devices")

    sources = {
        "workloads": tuple(_workload_source_ref(workload) for workload in selected_workloads),
        "viewports": tuple(_viewport_source_ref(trace) for trace in selected_viewports),
        "networks": tuple(_network_source_ref(trace) for trace in selected_networks),
        "devices": tuple(_device_source_ref(profile) for profile in selected_devices),
    }
    cases = tuple(
        _replay_case(
            config_id=config_id,
            split=split,
            seed=seed,
            workload=workload,
            viewport=viewport,
            network=network,
            device=device,
            index=index,
        )
        for index, (workload, viewport, network, device) in enumerate(
            itertools.product(selected_workloads, selected_viewports, selected_networks, selected_devices)
        )
    )
    if subset_spec.max_cases is not None:
        cases = cases[: subset_spec.max_cases]
    metadata = {
        "provenance": {
            "assembly": "deterministic_cross_product",
            "spec": subset_spec.as_payload(),
            "axis_counts": {
                "workloads": len(selected_workloads),
                "viewports": len(selected_viewports),
                "networks": len(selected_networks),
                "devices": len(selected_devices),
            },
            "case_count": len(cases),
        }
    }
    subset_id = _subset_id(
        config_id=config_id,
        split=split,
        seed=seed,
        cases=cases,
        sources=sources,
    )
    return ReplaySubsetManifest(
        subset_id=subset_id,
        config_id=config_id,
        split=split,
        seed=seed,
        cases=cases,
        sources=sources,
        metadata=metadata,
    )


def validate_replay_subset_manifest(manifest: ReplaySubsetManifest) -> ReplaySubsetManifest:
    """Validate replay subset completeness and source provenance."""

    if not isinstance(manifest, ReplaySubsetManifest):
        raise ReplayError("manifest must be a ReplaySubsetManifest record.")
    if not manifest.cases:
        raise ReplayError("cases must contain at least one replay case.")
    case_ids = [case.case_id for case in manifest.cases]
    _reject_duplicates(case_ids, "cases.case_id")

    source_ids_by_group: dict[str, set[str]] = {}
    for group in SOURCE_GROUPS:
        refs = tuple(manifest.sources.get(group, ()))
        if not refs:
            raise ReplayError(f"sources.{group} must contain at least one source reference.")
        source_ids = [source.record_id for source in refs]
        _reject_duplicates(source_ids, f"sources.{group}.record_id")
        source_ids_by_group[group] = set(source_ids)
        for source in refs:
            if source.group != group:
                raise ReplayError(f"sources.{group} contains source with mismatched group {source.group!r}.")
            if not source.provenance and not source.source_uri:
                raise ReplayError(f"sources.{group}.{source.record_id} is missing provenance or source_uri.")

    for case in manifest.cases:
        _require_reference(case.workload_manifest_id, source_ids_by_group["workloads"], f"{case.case_id}.workload_manifest_id")
        _require_reference(case.viewport_trace_id, source_ids_by_group["viewports"], f"{case.case_id}.viewport_trace_id")
        _require_reference(case.network_trace_id, source_ids_by_group["networks"], f"{case.case_id}.network_trace_id")
        _require_reference(case.device_profile_id, source_ids_by_group["devices"], f"{case.case_id}.device_profile_id")
    if not manifest.metadata.get("provenance"):
        raise ReplayError("metadata.provenance is required.")
    return manifest


def replay_subset_spec_from_mapping(values: Mapping[str, Any]) -> ReplaySubsetSpec:
    """Build a ReplaySubsetSpec from a config mapping."""

    mapping = _require_mapping(values, "replay_subset")
    defaults = ReplaySubsetSpec()
    return ReplaySubsetSpec(
        max_workloads=_optional_positive_int(mapping.get("max_workloads", defaults.max_workloads), "max_workloads"),
        max_viewports=_optional_positive_int(mapping.get("max_viewports", defaults.max_viewports), "max_viewports"),
        max_networks=_optional_positive_int(mapping.get("max_networks", defaults.max_networks), "max_networks"),
        max_devices=_optional_positive_int(mapping.get("max_devices", defaults.max_devices), "max_devices"),
        max_cases=_optional_positive_int(mapping.get("max_cases", defaults.max_cases), "max_cases"),
    )


def _replay_case(
    *,
    config_id: str,
    split: str,
    seed: int,
    workload: WorkloadManifest,
    viewport: ViewportTrace,
    network: NetworkTrace,
    device: DeviceProfile,
    index: int,
) -> ReplayCase:
    case_payload = {
        "config_id": config_id,
        "split": split,
        "seed": seed,
        "workload_manifest_id": workload.manifest_id,
        "viewport_trace_id": viewport.trace_id,
        "network_trace_id": network.trace_id,
        "device_profile_id": device.profile_id,
    }
    return ReplayCase(
        case_id=f"replay-case-{stable_config_id(case_payload)}",
        workload_manifest_id=workload.manifest_id,
        viewport_trace_id=viewport.trace_id,
        network_trace_id=network.trace_id,
        device_profile_id=device.profile_id,
        metadata={"provenance": {"source_index": index}},
    )


def _subset_id(
    *,
    config_id: str,
    split: str,
    seed: int,
    cases: tuple[ReplayCase, ...],
    sources: Mapping[str, tuple[ReplaySourceRef, ...]],
) -> str:
    payload = {
        "config_id": config_id,
        "split": split,
        "seed": seed,
        "cases": [case.as_payload() for case in cases],
        "sources": {
            group: [source.as_payload() for source in sources[group]]
            for group in SOURCE_GROUPS
        },
    }
    return f"replay-subset-{stable_config_id(payload)}"


def _workload_source_ref(workload: WorkloadManifest) -> ReplaySourceRef:
    payload = workload.as_payload()
    provenance = _source_provenance(payload)
    return ReplaySourceRef(
        group="workloads",
        record_id=workload.manifest_id,
        record_type="workload_manifest",
        source_uri=workload.source_uri,
        provenance=provenance,
        metadata={"config_id": workload.config_id, "split": workload.split, "seed": workload.seed},
    )


def _viewport_source_ref(trace: ViewportTrace) -> ReplaySourceRef:
    payload = trace.as_payload()
    return ReplaySourceRef(
        group="viewports",
        record_id=trace.trace_id,
        record_type="viewport_trace",
        source_uri=trace.source_uri,
        provenance=_source_provenance(payload),
        metadata={"sample_count": len(trace.poses)},
    )


def _network_source_ref(trace: NetworkTrace) -> ReplaySourceRef:
    payload = trace.as_payload()
    return ReplaySourceRef(
        group="networks",
        record_id=trace.trace_id,
        record_type="network_trace",
        source_uri=trace.source_uri,
        provenance=_source_provenance(payload),
        metadata={"sample_count": len(trace.samples)},
    )


def _device_source_ref(profile: DeviceProfile) -> ReplaySourceRef:
    payload = profile.as_payload()
    provenance = _source_provenance(payload)
    source_uri = _string_or_none(provenance.get("source_uri"))
    return ReplaySourceRef(
        group="devices",
        record_id=profile.profile_id,
        record_type="device_profile",
        source_uri=source_uri,
        provenance=provenance,
        metadata={"device_class": profile.device_class},
    )


def _source_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if isinstance(metadata, MappingABC):
        provenance = metadata.get("provenance")
        if isinstance(provenance, MappingABC):
            return _plain_json_mapping(provenance, "provenance")
        synthetic = metadata.get("synthetic")
        if isinstance(synthetic, MappingABC):
            return {"synthetic": _plain_json_mapping(synthetic, "synthetic")}
    return {}


def _select_axis(records: tuple[Any, ...], limit: int | None, id_fn: Any, field_name: str) -> tuple[Any, ...]:
    if not records:
        raise ReplayError(f"{field_name} must contain at least one record.")
    selected = tuple(sorted(records, key=lambda record: id_fn(record)))
    if limit is not None:
        selected = selected[:limit]
    return selected


def _device_tuple(device_profiles: Iterable[DeviceProfile] | DeviceProfileSet) -> tuple[DeviceProfile, ...]:
    if isinstance(device_profiles, DeviceProfileSet):
        return device_profiles.profiles
    return tuple(device_profiles)


def _workload_id(workload: WorkloadManifest) -> str:
    return workload.manifest_id


def _viewport_id(trace: ViewportTrace) -> str:
    return trace.trace_id


def _network_id(trace: NetworkTrace) -> str:
    return trace.trace_id


def _device_id(profile: DeviceProfile) -> str:
    return profile.profile_id


def _freeze_sources(sources: Mapping[str, Iterable[ReplaySourceRef]]) -> dict[str, tuple[ReplaySourceRef, ...]]:
    if not isinstance(sources, MappingABC):
        raise ReplayError("sources must be a mapping.")
    frozen: dict[str, tuple[ReplaySourceRef, ...]] = {}
    for group in SOURCE_GROUPS:
        refs = tuple(sources.get(group, ()))
        for ref in refs:
            if not isinstance(ref, ReplaySourceRef):
                raise ReplayError(f"sources.{group} must contain ReplaySourceRef records.")
        frozen[group] = refs
    return frozen


def _reject_duplicates(values: list[str], field_name: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ReplayError(f"{field_name} must not contain duplicate value(s): {', '.join(duplicates)}.")


def _require_reference(value: str, valid_values: set[str], field_name: str) -> None:
    if value not in valid_values:
        raise ReplayError(f"{field_name} references unknown source id {value!r}.")


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplayError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ReplayError(f"{field_name} must be a positive integer.")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ReplayError(f"{field_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise ReplayError(f"{field_name} must be a positive integer.")
    return parsed


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ReplayError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ReplayError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise ReplayError(f"{field_name} must be a non-negative integer.")
    return parsed


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _plain_json_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, MappingABC):
        raise ReplayError(f"{field_name} must be a mapping.")
    return {
        str(key): _plain_json_value(nested, f"{field_name}.{key}")
        for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
    }


def _plain_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, MappingABC):
        return _plain_json_mapping(value, field_name)
    if isinstance(value, list | tuple):
        return [_plain_json_value(item, field_name) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ReplayError(f"{field_name} must be finite.")
        return value
    raise ReplayError(f"{field_name} contains unsupported value type {type(value).__name__}.")


def _to_payload(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {key: _to_payload(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_to_payload(item) for item in value]
    if isinstance(value, list):
        return [_to_payload(item) for item in value]
    return value


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise ReplayError(f"{path} must be a mapping.")
    return value


__all__ = [
    "ReplayCase",
    "ReplayError",
    "ReplaySourceRef",
    "ReplaySubsetManifest",
    "ReplaySubsetSpec",
    "assemble_replay_subset",
    "replay_subset_spec_from_mapping",
    "validate_replay_subset_manifest",
]
