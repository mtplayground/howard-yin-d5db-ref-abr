"""Network trace normalization and synthetic boundary trace generation."""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ref_abr.config import ConfigError, load_config_file, stable_config_id


class NetworkError(ValueError):
    """Raised when network traces or synthetic settings are invalid."""


NETWORK_TYPES: tuple[str, ...] = ("mobile", "broadband", "lte", "5g", "wi-fi", "unknown")


@dataclass(frozen=True)
class NetworkSample:
    """Normalized network condition sample."""

    timestamp_ms: int
    throughput_bps: int
    latency_ms: float = 0.0
    packet_loss: float = 0.0
    jitter_ms: float = 0.0
    network_type: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_ms, bool) or not isinstance(self.timestamp_ms, int) or self.timestamp_ms < 0:
            raise NetworkError("timestamp_ms must be a non-negative integer.")
        if isinstance(self.throughput_bps, bool) or not isinstance(self.throughput_bps, int) or self.throughput_bps < 0:
            raise NetworkError("throughput_bps must be a non-negative integer.")
        object.__setattr__(self, "latency_ms", _non_negative_float(self.latency_ms, "latency_ms"))
        object.__setattr__(self, "packet_loss", _loss_ratio(self.packet_loss, "packet_loss"))
        object.__setattr__(self, "jitter_ms", _non_negative_float(self.jitter_ms, "jitter_ms"))
        object.__setattr__(self, "network_type", _network_type(self.network_type))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "throughput_bps": self.throughput_bps,
            "latency_ms": self.latency_ms,
            "packet_loss": self.packet_loss,
            "jitter_ms": self.jitter_ms,
            "network_type": self.network_type,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class NetworkTrace:
    """Normalized network trace."""

    trace_id: str
    samples: tuple[NetworkSample, ...]
    source_uri: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id:
            raise NetworkError("trace_id must be a non-empty string.")
        samples = tuple(self.samples)
        if not samples:
            raise NetworkError("samples must contain at least one network sample.")
        timestamps = [sample.timestamp_ms for sample in samples]
        if timestamps != sorted(timestamps):
            raise NetworkError("samples must be sorted by non-decreasing timestamp_ms.")
        if self.source_uri is not None and not self.source_uri:
            raise NetworkError("source_uri must be non-empty when provided.")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "samples": [sample.as_payload() for sample in self.samples],
            "source_uri": self.source_uri,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class NetworkBoundaryConfig:
    """Controls for deterministic synthetic network boundary traces."""

    duration_ms: int = 1000
    interval_ms: int = 100
    baseline_bps: int = 5_000_000
    low_bps: int = 1_000_000
    high_bps: int = 12_000_000
    threshold_bps: int = 3_000_000
    latency_ms: float = 40.0
    jitter_ms: float = 5.0
    jitter_fraction: float = 0.2

    def __post_init__(self) -> None:
        for field_name in ("duration_ms", "interval_ms", "baseline_bps", "low_bps", "high_bps", "threshold_bps"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise NetworkError(f"{field_name} must be a positive integer.")
        object.__setattr__(self, "latency_ms", _non_negative_float(self.latency_ms, "latency_ms"))
        object.__setattr__(self, "jitter_ms", _non_negative_float(self.jitter_ms, "jitter_ms"))
        jitter_fraction = _non_negative_float(self.jitter_fraction, "jitter_fraction")
        if jitter_fraction > 1:
            raise NetworkError("jitter_fraction must be between 0 and 1.")
        object.__setattr__(self, "jitter_fraction", jitter_fraction)


def load_network_trace(
    path: str | Path,
    *,
    trace_id: str | None = None,
    default_interval_ms: int = 100,
) -> NetworkTrace:
    """Load and normalize a network trace from JSON, TOML, YAML, or YML."""

    trace_path = Path(path)
    try:
        raw_trace = load_config_file(trace_path)
    except ConfigError as exc:
        raise NetworkError(str(exc)) from exc
    return normalize_network_trace(
        raw_trace,
        trace_id=trace_id,
        source_uri=str(trace_path),
        default_interval_ms=default_interval_ms,
    )


def normalize_network_trace(
    raw_trace: Mapping[str, Any],
    *,
    trace_id: str | None = None,
    source_uri: str | None = None,
    default_interval_ms: int = 100,
) -> NetworkTrace:
    """Normalize mobile/broadband/LTE/5G/Wi-Fi network traces."""

    root = _require_mapping(raw_trace, "trace")
    if isinstance(default_interval_ms, bool) or not isinstance(default_interval_ms, int) or default_interval_ms < 1:
        raise NetworkError("default_interval_ms must be a positive integer.")
    samples = _extract_samples(root)
    root_network_type = _network_type(str(_first_present(root, ("network_type", "access", "radio", "kind")) or "unknown"))
    normalized = tuple(
        sorted(
            (
                _sample_from_raw(
                    _require_mapping(sample, f"trace.samples[{index}]"),
                    index=index,
                    default_interval_ms=default_interval_ms,
                    root_network_type=root_network_type,
                )
                for index, sample in enumerate(samples)
            ),
            key=lambda sample: sample.timestamp_ms,
        )
    )
    resolved_trace_id = trace_id or _trace_id(normalized, source_uri)
    metadata = {
        "provenance": {
            "source_uri": source_uri,
            "sample_count": len(normalized),
            "input_format": _input_format(root),
            "network_type": root_network_type,
        }
    }
    return NetworkTrace(trace_id=resolved_trace_id, samples=normalized, source_uri=source_uri, metadata=metadata)


def generate_synthetic_boundary_traces(config: NetworkBoundaryConfig | None = None) -> tuple[NetworkTrace, ...]:
    """Generate step, outage, oscillation, burst, jitter, and threshold-near synthetic traces."""

    boundary = config or NetworkBoundaryConfig()
    timestamps = _timestamps(boundary)
    midpoint = len(timestamps) // 2
    traces = (
        _synthetic_trace(
            "step",
            boundary,
            timestamps,
            lambda index: boundary.low_bps if index < midpoint else boundary.high_bps,
        ),
        _synthetic_trace(
            "outage",
            boundary,
            timestamps,
            lambda index: 0 if index == midpoint else boundary.baseline_bps,
            packet_loss_fn=lambda index: 1.0 if index == midpoint else 0.0,
        ),
        _synthetic_trace(
            "oscillation",
            boundary,
            timestamps,
            lambda index: boundary.high_bps if index % 2 == 0 else boundary.low_bps,
        ),
        _synthetic_trace(
            "burst",
            boundary,
            timestamps,
            lambda index: boundary.high_bps if index == midpoint else boundary.baseline_bps,
        ),
        _synthetic_trace(
            "jitter",
            boundary,
            timestamps,
            lambda index: max(
                0,
                int(round(boundary.baseline_bps * (1 + (boundary.jitter_fraction if index % 2 == 0 else -boundary.jitter_fraction)))),
            ),
            jitter_fn=lambda index: boundary.jitter_ms * (2 if index % 2 == 0 else 1),
        ),
        _synthetic_trace(
            "threshold-near",
            boundary,
            timestamps,
            lambda index: int(round(boundary.threshold_bps * (1.05 if index % 2 == 0 else 0.95))),
        ),
    )
    return traces


def _sample_from_raw(
    sample: Mapping[str, Any],
    *,
    index: int,
    default_interval_ms: int,
    root_network_type: str,
) -> NetworkSample:
    timestamp_ms = _timestamp_ms(sample, index, default_interval_ms)
    throughput_bps = _throughput_bps(sample)
    latency_ms = _non_negative_float(_first_present(sample, ("latency_ms", "rtt_ms", "ping_ms")) or 0.0, "latency_ms")
    packet_loss = _packet_loss(sample)
    jitter_ms = _non_negative_float(_first_present(sample, ("jitter_ms", "jitter")) or 0.0, "jitter_ms")
    network_type = _network_type(str(_first_present(sample, ("network_type", "access", "radio", "kind")) or root_network_type))
    return NetworkSample(
        timestamp_ms=timestamp_ms,
        throughput_bps=throughput_bps,
        latency_ms=latency_ms,
        packet_loss=packet_loss,
        jitter_ms=jitter_ms,
        network_type=network_type,
        metadata={"source_index": index},
    )


def _extract_samples(root: Mapping[str, Any]) -> list[Any]:
    for key in ("samples", "trace", "network", "entries", "rows"):
        if key in root:
            samples = root[key]
            break
    else:
        samples = None
    if not isinstance(samples, list) or not samples:
        raise NetworkError("trace must contain a non-empty samples, trace, network, entries, or rows list.")
    return samples


def _timestamp_ms(sample: Mapping[str, Any], index: int, default_interval_ms: int) -> int:
    value = _first_present(sample, ("timestamp_ms", "time_ms", "t_ms", "elapsed_ms"))
    if value is None and "timestamp_s" in sample:
        value = _finite_float(sample["timestamp_s"], "timestamp_s") * 1000.0
    if value is None and "time_s" in sample:
        value = _finite_float(sample["time_s"], "time_s") * 1000.0
    if value is None and "t" in sample:
        value = _finite_float(sample["t"], "t") * 1000.0
    if value is None:
        return index * default_interval_ms
    parsed = _finite_float(value, "timestamp_ms")
    if parsed < 0:
        raise NetworkError("timestamp_ms must be non-negative.")
    return int(round(parsed))


def _throughput_bps(sample: Mapping[str, Any]) -> int:
    direct = _first_present(sample, ("throughput_bps", "bandwidth_bps", "capacity_bps", "bps"))
    if direct is not None:
        return _non_negative_int(round(_finite_float(direct, "throughput_bps")), "throughput_bps")
    mbps = _first_present(sample, ("throughput_mbps", "bandwidth_mbps", "capacity_mbps", "downlink_mbps", "dl_mbps", "mbps"))
    if mbps is not None:
        return _non_negative_int(round(_finite_float(mbps, "throughput_mbps") * 1_000_000), "throughput_mbps")
    kbps = _first_present(sample, ("throughput_kbps", "bandwidth_kbps", "capacity_kbps", "kbps"))
    if kbps is not None:
        return _non_negative_int(round(_finite_float(kbps, "throughput_kbps") * 1_000), "throughput_kbps")
    bytes_per_s = _first_present(sample, ("bytes_per_s", "bytes_per_sec"))
    if bytes_per_s is not None:
        return _non_negative_int(round(_finite_float(bytes_per_s, "bytes_per_s") * 8), "bytes_per_s")
    raise NetworkError("sample is missing throughput_bps, throughput_mbps, throughput_kbps, or bytes_per_s.")


def _packet_loss(sample: Mapping[str, Any]) -> float:
    ratio = _first_present(sample, ("packet_loss", "packet_loss_ratio", "loss", "loss_ratio"))
    if ratio is not None:
        return _loss_ratio(ratio, "packet_loss")
    percent = _first_present(sample, ("packet_loss_percent", "loss_percent"))
    if percent is not None:
        return _loss_ratio(_finite_float(percent, "packet_loss_percent") / 100.0, "packet_loss_percent")
    return 0.0


def _synthetic_trace(
    kind: str,
    config: NetworkBoundaryConfig,
    timestamps: tuple[int, ...],
    throughput_fn: Any,
    *,
    packet_loss_fn: Any | None = None,
    jitter_fn: Any | None = None,
) -> NetworkTrace:
    samples = tuple(
        NetworkSample(
            timestamp_ms=timestamp_ms,
            throughput_bps=throughput_fn(index),
            latency_ms=config.latency_ms,
            packet_loss=(packet_loss_fn(index) if packet_loss_fn else 0.0),
            jitter_ms=(jitter_fn(index) if jitter_fn else config.jitter_ms),
            network_type="unknown",
            metadata={"synthetic_index": index},
        )
        for index, timestamp_ms in enumerate(timestamps)
    )
    metadata = {
        "synthetic": {
            "kind": kind,
            "duration_ms": config.duration_ms,
            "interval_ms": config.interval_ms,
            "baseline_bps": config.baseline_bps,
            "low_bps": config.low_bps,
            "high_bps": config.high_bps,
            "threshold_bps": config.threshold_bps,
        }
    }
    trace_id = f"network-synthetic-{kind}-{stable_config_id({'kind': kind, 'samples': [sample.as_payload() for sample in samples]})}"
    return NetworkTrace(trace_id=trace_id, samples=samples, metadata=metadata)


def _timestamps(config: NetworkBoundaryConfig) -> tuple[int, ...]:
    timestamps = tuple(range(0, config.duration_ms + 1, config.interval_ms))
    if timestamps[-1] != config.duration_ms:
        timestamps = (*timestamps, config.duration_ms)
    return timestamps


def _trace_id(samples: tuple[NetworkSample, ...], source_uri: str | None) -> str:
    payload = {"source_uri": source_uri, "samples": [sample.as_payload() for sample in samples]}
    return f"network-{stable_config_id(payload)}"


def _input_format(root: Mapping[str, Any]) -> str:
    for key in ("samples", "trace", "network", "entries", "rows"):
        if key in root:
            return key
    return "unknown"


def _network_type(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "wifi": "wi-fi",
        "wi-fi": "wi-fi",
        "lte": "lte",
        "4g": "lte",
        "5g": "5g",
        "nr": "5g",
        "cellular": "mobile",
        "mobile": "mobile",
        "broadband": "broadband",
        "wired": "broadband",
        "unknown": "unknown",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in NETWORK_TYPES:
        raise NetworkError(f"network_type must be one of: {', '.join(NETWORK_TYPES)}.")
    return resolved


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise NetworkError(f"{field_name} must be numeric.")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise NetworkError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise NetworkError(f"{field_name} must be finite.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0:
        raise NetworkError(f"{field_name} must be non-negative.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NetworkError(f"{field_name} must be an integer.")
    if value < 0:
        raise NetworkError(f"{field_name} must be non-negative.")
    return value


def _loss_ratio(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if not 0.0 <= parsed <= 1.0:
        raise NetworkError(f"{field_name} must be between 0 and 1.")
    return parsed


def _plain_json_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, MappingABC):
        raise NetworkError(f"{field_name} must be a mapping.")
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
            raise NetworkError(f"{field_name} must be finite.")
        return value
    raise NetworkError(f"{field_name} contains unsupported value type {type(value).__name__}.")


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
        raise NetworkError(f"{path} must be a mapping.")
    return value


__all__ = [
    "NetworkBoundaryConfig",
    "NetworkError",
    "NetworkSample",
    "NetworkTrace",
    "generate_synthetic_boundary_traces",
    "load_network_trace",
    "normalize_network_trace",
]
