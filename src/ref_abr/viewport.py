"""Viewport pose trace normalization and controlled error sweeps."""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ref_abr.config import ConfigError, load_config_file, stable_config_id


class ViewportError(ValueError):
    """Raised when viewport traces or sweep settings are invalid."""


@dataclass(frozen=True)
class ViewportPose:
    """Normalized 6-DoF viewport pose sample."""

    timestamp_ms: int
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    fov_deg: float = 90.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_ms, bool) or not isinstance(self.timestamp_ms, int) or self.timestamp_ms < 0:
            raise ViewportError("timestamp_ms must be a non-negative integer.")
        object.__setattr__(self, "x_m", _finite_float(self.x_m, "x_m"))
        object.__setattr__(self, "y_m", _finite_float(self.y_m, "y_m"))
        object.__setattr__(self, "z_m", _finite_float(self.z_m, "z_m"))
        object.__setattr__(self, "yaw_deg", _normalize_yaw(_finite_float(self.yaw_deg, "yaw_deg")))
        object.__setattr__(self, "pitch_deg", _normalize_signed_angle(_finite_float(self.pitch_deg, "pitch_deg")))
        object.__setattr__(self, "roll_deg", _normalize_signed_angle(_finite_float(self.roll_deg, "roll_deg")))
        object.__setattr__(self, "fov_deg", _normalize_fov(_finite_float(self.fov_deg, "fov_deg")))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "position_m": {"x": self.x_m, "y": self.y_m, "z": self.z_m},
            "orientation_deg": {"yaw": self.yaw_deg, "pitch": self.pitch_deg, "roll": self.roll_deg},
            "fov_deg": self.fov_deg,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class ViewportTrace:
    """Normalized viewport pose trace."""

    trace_id: str
    poses: tuple[ViewportPose, ...]
    source_uri: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id:
            raise ViewportError("trace_id must be a non-empty string.")
        poses = tuple(self.poses)
        if not poses:
            raise ViewportError("poses must contain at least one sample.")
        timestamps = [pose.timestamp_ms for pose in poses]
        if timestamps != sorted(timestamps):
            raise ViewportError("poses must be sorted by non-decreasing timestamp_ms.")
        if self.source_uri is not None and not self.source_uri:
            raise ViewportError("source_uri must be non-empty when provided.")
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "poses": [pose.as_payload() for pose in self.poses],
            "source_uri": self.source_uri,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class ViewportSweepConfig:
    """Magnitude controls for deterministic viewport-error sweeps."""

    angular_degrees: tuple[float, ...] = (5.0, 10.0)
    translation_meters: tuple[float, ...] = (0.1, 0.25)
    fov_degrees: tuple[float, ...] = (5.0, 10.0)
    horizon_degrees: tuple[float, ...] = (5.0, 10.0)
    sudden_turn_degrees: tuple[float, ...] = (30.0,)
    adversarial_degrees: tuple[float, ...] = (10.0,)

    def __post_init__(self) -> None:
        for field_name in (
            "angular_degrees",
            "translation_meters",
            "fov_degrees",
            "horizon_degrees",
            "sudden_turn_degrees",
            "adversarial_degrees",
        ):
            parsed_values: list[float] = []
            for value in tuple(getattr(self, field_name)):
                parsed = _finite_float(value, field_name)
                if parsed < 0:
                    raise ViewportError(f"{field_name} magnitudes must be non-negative.")
                parsed_values.append(parsed)
            object.__setattr__(self, field_name, tuple(parsed_values))


def load_viewport_trace(
    path: str | Path,
    *,
    trace_id: str | None = None,
    default_fov_deg: float = 90.0,
    default_sample_interval_ms: int = 16,
) -> ViewportTrace:
    """Load and normalize a viewport trace from JSON, TOML, YAML, or YML."""

    trace_path = Path(path)
    try:
        raw_trace = load_config_file(trace_path)
    except ConfigError as exc:
        raise ViewportError(str(exc)) from exc
    return normalize_viewport_trace(
        raw_trace,
        trace_id=trace_id,
        source_uri=str(trace_path),
        default_fov_deg=default_fov_deg,
        default_sample_interval_ms=default_sample_interval_ms,
    )


def normalize_viewport_trace(
    raw_trace: Mapping[str, Any],
    *,
    trace_id: str | None = None,
    source_uri: str | None = None,
    default_fov_deg: float = 90.0,
    default_sample_interval_ms: int = 16,
) -> ViewportTrace:
    """Normalize flexible 6-DoF/360 pose trace metadata into a ViewportTrace."""

    root = _require_mapping(raw_trace, "trace")
    samples = _extract_samples(root)
    default_fov = _normalize_fov(_finite_float(default_fov_deg, "default_fov_deg"))
    if isinstance(default_sample_interval_ms, bool) or default_sample_interval_ms < 1:
        raise ViewportError("default_sample_interval_ms must be a positive integer.")

    poses = tuple(
        sorted(
            (
                _pose_from_sample(
                    sample=_require_mapping(sample, f"trace.samples[{index}]"),
                    index=index,
                    default_fov_deg=default_fov,
                    default_sample_interval_ms=default_sample_interval_ms,
                )
                for index, sample in enumerate(samples)
            ),
            key=lambda pose: pose.timestamp_ms,
        )
    )
    resolved_trace_id = trace_id or _trace_id(poses, source_uri)
    metadata = {
        "provenance": {
            "source_uri": source_uri,
            "sample_count": len(poses),
            "input_format": _input_format(root),
        }
    }
    return ViewportTrace(trace_id=resolved_trace_id, poses=poses, source_uri=source_uri, metadata=metadata)


def generate_viewport_error_sweeps(
    trace: ViewportTrace,
    config: ViewportSweepConfig | None = None,
) -> tuple[ViewportTrace, ...]:
    """Generate deterministic angular, translation, FoV, horizon, sudden-turn, and adversarial sweeps."""

    sweep_config = config or ViewportSweepConfig()
    sweeps: list[ViewportTrace] = []
    for magnitude in sweep_config.angular_degrees:
        sweeps.append(_sweep_trace(trace, "angular", magnitude, lambda pose, index: _with_pose_offsets(pose, yaw=magnitude, pitch=magnitude / 2)))
    for magnitude in sweep_config.translation_meters:
        sweeps.append(_sweep_trace(trace, "translational", magnitude, lambda pose, index: _with_pose_offsets(pose, x=magnitude)))
    for magnitude in sweep_config.fov_degrees:
        sweeps.append(_sweep_trace(trace, "fov", magnitude, lambda pose, index: _with_pose_offsets(pose, fov=magnitude)))
    for magnitude in sweep_config.horizon_degrees:
        sweeps.append(_sweep_trace(trace, "horizon", magnitude, lambda pose, index: _with_pose_offsets(pose, roll=magnitude)))
    for magnitude in sweep_config.sudden_turn_degrees:
        midpoint = len(trace.poses) // 2
        sweeps.append(
            _sweep_trace(
                trace,
                "sudden-turn",
                magnitude,
                lambda pose, index, midpoint=midpoint: _with_pose_offsets(pose, yaw=magnitude if index >= midpoint else 0.0),
            )
        )
    for magnitude in sweep_config.adversarial_degrees:
        sweeps.append(
            _sweep_trace(
                trace,
                "adversarial",
                magnitude,
                lambda pose, index: _with_pose_offsets(
                    pose,
                    yaw=magnitude if index % 2 == 0 else -magnitude,
                    pitch=-(magnitude / 2) if index % 2 == 0 else magnitude / 2,
                    x=magnitude / 100.0,
                ),
            )
        )
    return tuple(sweeps)


def _extract_samples(root: Mapping[str, Any]) -> list[Any]:
    if "poses" in root:
        samples = root["poses"]
    elif "samples" in root:
        samples = root["samples"]
    elif "frames" in root:
        samples = root["frames"]
    elif "trace" in root:
        nested = root["trace"]
        if isinstance(nested, list):
            samples = nested
        else:
            samples = _require_mapping(nested, "trace.trace").get("poses")
    else:
        samples = None
    if not isinstance(samples, list) or not samples:
        raise ViewportError("trace must contain a non-empty poses, samples, frames, or trace list.")
    return samples


def _pose_from_sample(
    *,
    sample: Mapping[str, Any],
    index: int,
    default_fov_deg: float,
    default_sample_interval_ms: int,
) -> ViewportPose:
    timestamp_ms = _timestamp_ms(sample, index, default_sample_interval_ms)
    x_m, y_m, z_m = _position_m(sample)
    yaw_deg, pitch_deg, roll_deg = _orientation_deg(sample)
    fov_deg = _fov_deg(sample, default_fov_deg)
    return ViewportPose(
        timestamp_ms=timestamp_ms,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        fov_deg=fov_deg,
        metadata={"source_index": index},
    )


def _timestamp_ms(sample: Mapping[str, Any], index: int, default_sample_interval_ms: int) -> int:
    value = _first_present(sample, ("timestamp_ms", "time_ms", "pts_ms", "t_ms"))
    if value is None and "timestamp_s" in sample:
        value = _finite_float(sample["timestamp_s"], "timestamp_s") * 1000.0
    if value is None and "time_s" in sample:
        value = _finite_float(sample["time_s"], "time_s") * 1000.0
    if value is None and "t" in sample:
        value = _finite_float(sample["t"], "t") * 1000.0
    if value is None:
        return index * default_sample_interval_ms
    parsed = _finite_float(value, "timestamp_ms")
    if parsed < 0:
        raise ViewportError("timestamp_ms must be non-negative.")
    return int(round(parsed))


def _position_m(sample: Mapping[str, Any]) -> tuple[float, float, float]:
    position = _first_present(sample, ("position_m", "position", "translation", "xyz"))
    if isinstance(position, MappingABC):
        return (
            _finite_float(_first_present(position, ("x", "tx", "px")) or 0.0, "position.x"),
            _finite_float(_first_present(position, ("y", "ty", "py")) or 0.0, "position.y"),
            _finite_float(_first_present(position, ("z", "tz", "pz")) or 0.0, "position.z"),
        )
    if isinstance(position, list | tuple):
        if len(position) != 3:
            raise ViewportError("position must contain exactly three values.")
        return (
            _finite_float(position[0], "position[0]"),
            _finite_float(position[1], "position[1]"),
            _finite_float(position[2], "position[2]"),
        )
    return (
        _finite_float(_first_present(sample, ("x_m", "x", "tx")) or 0.0, "x_m"),
        _finite_float(_first_present(sample, ("y_m", "y", "ty")) or 0.0, "y_m"),
        _finite_float(_first_present(sample, ("z_m", "z", "tz")) or 0.0, "z_m"),
    )


def _orientation_deg(sample: Mapping[str, Any]) -> tuple[float, float, float]:
    orientation = _first_present(sample, ("orientation_deg", "orientation", "rotation", "euler"))
    if isinstance(orientation, MappingABC):
        if "quaternion" in orientation:
            return _quaternion_to_euler_deg(orientation["quaternion"])
        if all(key in orientation for key in ("qx", "qy", "qz", "qw")):
            return _quaternion_to_euler_deg((orientation["qx"], orientation["qy"], orientation["qz"], orientation["qw"]))
        return (
            _angle_from_mapping(orientation, ("yaw_deg", "yaw", "heading"), ("yaw_rad",), "yaw"),
            _angle_from_mapping(orientation, ("pitch_deg", "pitch"), ("pitch_rad",), "pitch"),
            _angle_from_mapping(orientation, ("roll_deg", "roll"), ("roll_rad",), "roll"),
        )
    if isinstance(orientation, list | tuple):
        if len(orientation) == 3:
            return (
                _finite_float(orientation[0], "orientation[0]"),
                _finite_float(orientation[1], "orientation[1]"),
                _finite_float(orientation[2], "orientation[2]"),
            )
        if len(orientation) == 4:
            return _quaternion_to_euler_deg(orientation)
        raise ViewportError("orientation must contain three Euler values or four quaternion values.")
    if all(key in sample for key in ("qx", "qy", "qz", "qw")):
        return _quaternion_to_euler_deg((sample["qx"], sample["qy"], sample["qz"], sample["qw"]))
    return (
        _angle_from_mapping(sample, ("yaw_deg", "yaw", "heading"), ("yaw_rad",), "yaw"),
        _angle_from_mapping(sample, ("pitch_deg", "pitch"), ("pitch_rad",), "pitch"),
        _angle_from_mapping(sample, ("roll_deg", "roll", "horizon_deg"), ("roll_rad", "horizon_rad"), "roll"),
    )


def _fov_deg(sample: Mapping[str, Any], default_fov_deg: float) -> float:
    if "fov_rad" in sample:
        return math.degrees(_finite_float(sample["fov_rad"], "fov_rad"))
    value = _first_present(sample, ("fov_deg", "fov", "fov_y_deg", "fovy_deg"))
    return default_fov_deg if value is None else _finite_float(value, "fov_deg")


def _angle_from_mapping(
    mapping: Mapping[str, Any],
    degree_keys: tuple[str, ...],
    radian_keys: tuple[str, ...],
    field_name: str,
) -> float:
    value = _first_present(mapping, degree_keys)
    if value is not None:
        return _finite_float(value, field_name)
    value = _first_present(mapping, radian_keys)
    if value is not None:
        return math.degrees(_finite_float(value, field_name))
    return 0.0


def _quaternion_to_euler_deg(raw_quaternion: Any) -> tuple[float, float, float]:
    if isinstance(raw_quaternion, MappingABC):
        qx = _finite_float(raw_quaternion.get("x", raw_quaternion.get("qx", 0.0)), "quaternion.x")
        qy = _finite_float(raw_quaternion.get("y", raw_quaternion.get("qy", 0.0)), "quaternion.y")
        qz = _finite_float(raw_quaternion.get("z", raw_quaternion.get("qz", 0.0)), "quaternion.z")
        qw = _finite_float(raw_quaternion.get("w", raw_quaternion.get("qw", 1.0)), "quaternion.w")
    elif isinstance(raw_quaternion, list | tuple) and len(raw_quaternion) == 4:
        qx = _finite_float(raw_quaternion[0], "quaternion[0]")
        qy = _finite_float(raw_quaternion[1], "quaternion[1]")
        qz = _finite_float(raw_quaternion[2], "quaternion[2]")
        qw = _finite_float(raw_quaternion[3], "quaternion[3]")
    else:
        raise ViewportError("quaternion must be a mapping or four-value list.")
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        raise ViewportError("quaternion must not have zero length.")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def _sweep_trace(trace: ViewportTrace, kind: str, magnitude: float, transform: Any) -> ViewportTrace:
    poses = tuple(transform(pose, index) for index, pose in enumerate(trace.poses))
    metadata = {
        **trace.as_payload()["metadata"],
        "error_sweep": {"kind": kind, "magnitude": magnitude},
    }
    trace_id = f"{trace.trace_id}:{kind}:{_magnitude_label(magnitude)}"
    return ViewportTrace(trace_id=trace_id, poses=poses, source_uri=trace.source_uri, metadata=metadata)


def _with_pose_offsets(
    pose: ViewportPose,
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    fov: float = 0.0,
) -> ViewportPose:
    return ViewportPose(
        timestamp_ms=pose.timestamp_ms,
        x_m=pose.x_m + x,
        y_m=pose.y_m + y,
        z_m=pose.z_m + z,
        yaw_deg=pose.yaw_deg + yaw,
        pitch_deg=pose.pitch_deg + pitch,
        roll_deg=pose.roll_deg + roll,
        fov_deg=pose.fov_deg + fov,
        metadata=pose.as_payload()["metadata"],
    )


def _trace_id(poses: tuple[ViewportPose, ...], source_uri: str | None) -> str:
    payload = {"source_uri": source_uri, "poses": [pose.as_payload() for pose in poses]}
    return f"viewport-{stable_config_id(payload)}"


def _input_format(root: Mapping[str, Any]) -> str:
    if "poses" in root:
        return "poses"
    if "samples" in root:
        return "samples"
    if "frames" in root:
        return "frames"
    return "trace"


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ViewportError(f"{field_name} must be numeric.")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ViewportError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ViewportError(f"{field_name} must be finite.")
    return parsed


def _normalize_yaw(value: float) -> float:
    return value % 360.0


def _normalize_signed_angle(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


def _normalize_fov(value: float) -> float:
    if not 1.0 <= value <= 179.0:
        raise ViewportError("fov_deg must be between 1 and 179.")
    return value


def _magnitude_label(value: float) -> str:
    return str(value).replace("-", "neg").replace(".", "p")


def _plain_json_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, MappingABC):
        raise ViewportError(f"{field_name} must be a mapping.")
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
            raise ViewportError(f"{field_name} must be finite.")
        return value
    raise ViewportError(f"{field_name} contains unsupported value type {type(value).__name__}.")


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
        raise ViewportError(f"{path} must be a mapping.")
    return value


__all__ = [
    "ViewportError",
    "ViewportPose",
    "ViewportSweepConfig",
    "ViewportTrace",
    "generate_viewport_error_sweeps",
    "load_viewport_trace",
    "normalize_viewport_trace",
]
