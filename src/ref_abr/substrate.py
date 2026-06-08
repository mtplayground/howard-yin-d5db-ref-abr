"""Substrate-value provider interface and default parametric backend."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping as MappingABC
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from ref_abr.config import ConfigError, load_config_file, stable_config_id


class SubstrateError(ValueError):
    """Raised when substrate-value inputs or model coefficients are invalid."""


@runtime_checkable
class SubstrateValueProvider(Protocol):
    """Contract for providers that score one substrate query at a time."""

    provider_id: str

    def evaluate(self, query: "SubstrateQuery | Mapping[str, Any]") -> "SubstrateValue":
        """Return visible-quality, component-timing, and uncertainty values."""


@dataclass(frozen=True)
class ReferenceResolution:
    """Normalized reference resolution used by substrate-value models."""

    width_px: int
    height_px: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "width_px", _positive_int(self.width_px, "width_px"))
        object.__setattr__(self, "height_px", _positive_int(self.height_px, "height_px"))

    @property
    def pixel_count(self) -> int:
        return self.width_px * self.height_px

    @property
    def megapixels(self) -> float:
        return self.pixel_count / 1_000_000.0

    def as_payload(self) -> dict[str, int]:
        return {"width_px": self.width_px, "height_px": self.height_px}


@dataclass(frozen=True)
class SubstrateQuery:
    """Provider input for one layer/reference/viewport state."""

    layer: int
    ref_resolution: ReferenceResolution | Mapping[str, Any] | str | tuple[int, int]
    fov_deg: float
    view_mismatch_deg: float
    freshness_ms: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", _non_negative_int(self.layer, "layer"))
        object.__setattr__(self, "ref_resolution", coerce_ref_resolution(self.ref_resolution))
        object.__setattr__(self, "fov_deg", _fov_deg(self.fov_deg, "fov_deg"))
        object.__setattr__(self, "view_mismatch_deg", _non_negative_float(self.view_mismatch_deg, "view_mismatch_deg"))
        object.__setattr__(self, "freshness_ms", _non_negative_float(self.freshness_ms, "freshness_ms"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "ref_resolution": self.ref_resolution.as_payload(),
            "fov_deg": self.fov_deg,
            "view_mismatch_deg": self.view_mismatch_deg,
            "freshness_ms": self.freshness_ms,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class ComponentTiming:
    """Predicted component timing values in milliseconds."""

    generation_ms: float
    transfer_ms: float
    restoration_ms: float
    render_ms: float

    def __post_init__(self) -> None:
        for field_name in ("generation_ms", "transfer_ms", "restoration_ms", "render_ms"):
            object.__setattr__(self, field_name, _non_negative_float(getattr(self, field_name), field_name))

    @property
    def total_ms(self) -> float:
        return self.generation_ms + self.transfer_ms + self.restoration_ms + self.render_ms

    def as_payload(self) -> dict[str, float]:
        return {
            "generation_ms": self.generation_ms,
            "transfer_ms": self.transfer_ms,
            "restoration_ms": self.restoration_ms,
            "render_ms": self.render_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True)
class SubstrateUncertainty:
    """Recorded uncertainty for quality and timing predictions."""

    quality_stddev: float
    timing_stddev_ms: float
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "quality_stddev", _non_negative_float(self.quality_stddev, "quality_stddev"))
        object.__setattr__(self, "timing_stddev_ms", _non_negative_float(self.timing_stddev_ms, "timing_stddev_ms"))
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, "confidence"))

    def as_payload(self) -> dict[str, float]:
        return {
            "quality_stddev": self.quality_stddev,
            "timing_stddev_ms": self.timing_stddev_ms,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class SubstrateValue:
    """Provider output for one substrate query."""

    provider_id: str
    query: SubstrateQuery
    visible_quality: float
    component_timing: ComponentTiming
    uncertainty: SubstrateUncertainty
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or not self.provider_id.strip():
            raise SubstrateError("provider_id must be a non-empty string.")
        if not isinstance(self.query, SubstrateQuery):
            raise SubstrateError("query must be a SubstrateQuery record.")
        object.__setattr__(self, "visible_quality", _unit_interval(self.visible_quality, "visible_quality"))
        if not isinstance(self.component_timing, ComponentTiming):
            raise SubstrateError("component_timing must be a ComponentTiming record.")
        if not isinstance(self.uncertainty, SubstrateUncertainty):
            raise SubstrateError("uncertainty must be a SubstrateUncertainty record.")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "query": self.query.as_payload(),
            "visible_quality": self.visible_quality,
            "component_timing": self.component_timing.as_payload(),
            "uncertainty": self.uncertainty.as_payload(),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class ParametricSubstrateCoefficients:
    """Calibratable coefficients for the default analytic substrate model."""

    base_quality: float = 0.55
    layer_quality_gain: float = 0.09
    resolution_quality_gain: float = 0.18
    fov_penalty: float = 0.06
    mismatch_penalty: float = 0.16
    freshness_penalty: float = 0.035
    generation_base_ms: float = 1.5
    generation_layer_ms: float = 1.25
    generation_resolution_ms: float = 4.5
    generation_fov_ms: float = 0.012
    transfer_base_ms: float = 1.0
    transfer_layer_ms: float = 0.75
    transfer_resolution_ms: float = 7.0
    restoration_base_ms: float = 0.8
    restoration_layer_ms: float = 0.6
    restoration_resolution_ms: float = 2.0
    render_base_ms: float = 2.5
    render_resolution_ms: float = 4.0
    render_fov_ms: float = 0.018
    uncertainty_base: float = 0.025
    uncertainty_mismatch: float = 0.035
    uncertainty_freshness: float = 0.015
    timing_uncertainty_ms: float = 1.0

    def __post_init__(self) -> None:
        for field_name in _COEFFICIENT_NAMES:
            value = _non_negative_float(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None) -> "ParametricSubstrateCoefficients":
        """Build coefficients from a partial mapping, rejecting unknown names."""

        if values is None:
            return cls()
        mapping = _require_mapping(values, "coefficients")
        unknown = sorted(str(key) for key in mapping if str(key) not in _COEFFICIENT_NAMES)
        if unknown:
            raise SubstrateError(f"coefficients contains unknown field(s): {', '.join(unknown)}.")
        parsed = {str(key): _non_negative_float(value, f"coefficients.{key}") for key, value in mapping.items()}
        return cls(**parsed)

    def as_payload(self) -> dict[str, float]:
        return {field_name: getattr(self, field_name) for field_name in _COEFFICIENT_NAMES}


@dataclass(frozen=True)
class ParametricSubstrateValueProvider:
    """Default analytic backend for substrate visible-quality and timing values."""

    coefficients: ParametricSubstrateCoefficients = field(default_factory=ParametricSubstrateCoefficients)
    provider_id: str = "parametric-substrate-default"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or not self.provider_id.strip():
            raise SubstrateError("provider_id must be a non-empty string.")
        if not isinstance(self.coefficients, ParametricSubstrateCoefficients):
            raise SubstrateError("coefficients must be a ParametricSubstrateCoefficients record.")
        object.__setattr__(self, "provider_id", self.provider_id.strip())
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def evaluate(self, query: SubstrateQuery | Mapping[str, Any]) -> SubstrateValue:
        """Evaluate one layer/reference/view state with the parametric model."""

        substrate_query = coerce_substrate_query(query)
        coefficients = self.coefficients
        resolution_scale = _resolution_scale(substrate_query.ref_resolution)
        layer_scale = float(substrate_query.layer + 1)
        fov_scale = substrate_query.fov_deg / 90.0
        mismatch_scale = min(substrate_query.view_mismatch_deg, 180.0) / 45.0
        freshness_s = substrate_query.freshness_ms / 1000.0

        visible_quality = _clamp01(
            coefficients.base_quality
            + coefficients.layer_quality_gain * math.log1p(layer_scale)
            + coefficients.resolution_quality_gain * math.log1p(resolution_scale)
            - coefficients.fov_penalty * max(0.0, fov_scale - 1.0)
            - coefficients.mismatch_penalty * (mismatch_scale**2)
            - coefficients.freshness_penalty * freshness_s
        )
        timing = ComponentTiming(
            generation_ms=(
                coefficients.generation_base_ms
                + coefficients.generation_layer_ms * layer_scale
                + coefficients.generation_resolution_ms * resolution_scale
                + coefficients.generation_fov_ms * substrate_query.fov_deg
            ),
            transfer_ms=(
                coefficients.transfer_base_ms
                + coefficients.transfer_layer_ms * layer_scale
                + coefficients.transfer_resolution_ms * resolution_scale
            ),
            restoration_ms=(
                coefficients.restoration_base_ms
                + coefficients.restoration_layer_ms * layer_scale
                + coefficients.restoration_resolution_ms * resolution_scale
            ),
            render_ms=(
                coefficients.render_base_ms
                + coefficients.render_resolution_ms * resolution_scale
                + coefficients.render_fov_ms * substrate_query.fov_deg
            ),
        )
        quality_stddev = _clamp01(
            coefficients.uncertainty_base
            + coefficients.uncertainty_mismatch * (substrate_query.view_mismatch_deg / 180.0)
            + coefficients.uncertainty_freshness * freshness_s
        )
        uncertainty = SubstrateUncertainty(
            quality_stddev=quality_stddev,
            timing_stddev_ms=coefficients.timing_uncertainty_ms * (1.0 + 0.25 * resolution_scale),
            confidence=_clamp01(1.0 - 2.0 * quality_stddev),
        )
        metadata = {
            "model": {
                "kind": "parametric",
                "coefficient_id": stable_config_id(coefficients.as_payload()),
                "resolution_scale": resolution_scale,
                "layer_scale": layer_scale,
            },
            "provider": _to_payload(self.metadata),
        }
        return SubstrateValue(
            provider_id=self.provider_id,
            query=substrate_query,
            visible_quality=visible_quality,
            component_timing=timing,
            uncertainty=uncertainty,
            metadata=metadata,
        )

    def evaluate_many(self, queries: Iterable[SubstrateQuery | Mapping[str, Any]]) -> tuple[SubstrateValue, ...]:
        """Evaluate several substrate queries deterministically."""

        return tuple(self.evaluate(query) for query in queries)

    def as_payload(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "backend": "parametric",
            "coefficients": self.coefficients.as_payload(),
            "metadata": _to_payload(self.metadata),
        }


def load_parametric_substrate_provider(path: str | Path) -> ParametricSubstrateValueProvider:
    """Load a parametric substrate provider definition from JSON, TOML, YAML, or YML."""

    provider_path = Path(path)
    try:
        raw_provider = load_config_file(provider_path)
    except ConfigError as exc:
        raise SubstrateError(str(exc)) from exc
    return parametric_substrate_provider_from_mapping(raw_provider, source_uri=str(provider_path))


def parametric_substrate_provider_from_mapping(
    raw_provider: Mapping[str, Any],
    *,
    source_uri: str | None = None,
) -> ParametricSubstrateValueProvider:
    """Normalize a provider definition into the default parametric backend."""

    root = _require_mapping(raw_provider, "provider")
    backend = str(_first_present(root, ("backend", "type", "kind")) or "parametric").strip().lower()
    if backend not in {"parametric", "analytic", "default"}:
        raise SubstrateError("provider.backend must be parametric, analytic, or default.")
    provider_id = _string_or_none(_first_present(root, ("provider_id", "id", "name"))) or "parametric-substrate-default"
    coefficients = ParametricSubstrateCoefficients.from_mapping(root.get("coefficients"))
    metadata = _plain_json_mapping(root.get("metadata"), "metadata") if root.get("metadata") is not None else {}
    metadata["provenance"] = {"source_uri": source_uri}
    return ParametricSubstrateValueProvider(
        coefficients=coefficients,
        provider_id=provider_id,
        metadata=metadata,
    )


def coerce_substrate_query(query: SubstrateQuery | Mapping[str, Any]) -> SubstrateQuery:
    """Return a SubstrateQuery from an existing record or compatible mapping."""

    if isinstance(query, SubstrateQuery):
        return query
    raw_query = _require_mapping(query, "query")
    ref_resolution = _first_present(
        raw_query,
        ("ref_resolution", "reference_resolution", "resolution", "resolution_px", "ref_resolution_px"),
    )
    if ref_resolution is None:
        ref_resolution = {
            "width_px": _first_present(raw_query, ("ref_width_px", "width_px", "width")),
            "height_px": _first_present(raw_query, ("ref_height_px", "height_px", "height")),
        }
    return SubstrateQuery(
        layer=_required(raw_query, ("layer", "layer_index"), "query.layer"),
        ref_resolution=ref_resolution,
        fov_deg=_required(raw_query, ("fov_deg", "fov"), "query.fov_deg"),
        view_mismatch_deg=_required(
            raw_query,
            ("view_mismatch_deg", "mismatch_deg", "viewport_error_deg", "angular_error_deg"),
            "query.view_mismatch_deg",
        ),
        freshness_ms=_required(raw_query, ("freshness_ms", "age_ms", "staleness_ms"), "query.freshness_ms"),
        metadata=raw_query.get("metadata", {}),
    )


def coerce_ref_resolution(value: ReferenceResolution | Mapping[str, Any] | str | tuple[int, int]) -> ReferenceResolution:
    """Normalize common resolution representations into width/height pixels."""

    if isinstance(value, ReferenceResolution):
        return value
    if isinstance(value, MappingABC):
        width = _first_present(value, ("width_px", "width", "w"))
        height = _first_present(value, ("height_px", "height", "h"))
        if width is None or height is None:
            raise SubstrateError("ref_resolution must include width_px and height_px.")
        return ReferenceResolution(width_px=_positive_int(width, "ref_resolution.width_px"), height_px=_positive_int(height, "ref_resolution.height_px"))
    if isinstance(value, tuple):
        if len(value) != 2:
            raise SubstrateError("ref_resolution tuple must contain width and height.")
        return ReferenceResolution(width_px=_positive_int(value[0], "ref_resolution[0]"), height_px=_positive_int(value[1], "ref_resolution[1]"))
    if isinstance(value, str):
        return _resolution_from_string(value)
    raise SubstrateError("ref_resolution must be a ReferenceResolution, mapping, tuple, or string.")


def _resolution_from_string(value: str) -> ReferenceResolution:
    normalized = value.strip().lower().replace(" ", "")
    aliases = {
        "480p": (854, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "2160p": (3840, 2160),
        "4k": (3840, 2160),
    }
    if normalized in aliases:
        width, height = aliases[normalized]
        return ReferenceResolution(width, height)
    separator = "x" if "x" in normalized else "X" if "X" in normalized else None
    if separator:
        width_text, _, height_text = normalized.partition(separator)
        try:
            return ReferenceResolution(int(width_text), int(height_text))
        except ValueError as exc:
            raise SubstrateError("ref_resolution string must use WIDTHxHEIGHT pixels.") from exc
    raise SubstrateError("ref_resolution string must be a known label or WIDTHxHEIGHT pixels.")


def _resolution_scale(resolution: ReferenceResolution) -> float:
    return math.sqrt(resolution.pixel_count / (1920 * 1080))


def _required(mapping: Mapping[str, Any], keys: tuple[str, ...], field_name: str) -> Any:
    value = _first_present(mapping, keys)
    if value is None:
        raise SubstrateError(f"{field_name} is required.")
    return value


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise SubstrateError(f"{field_name} must be a positive integer.")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SubstrateError(f"{field_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise SubstrateError(f"{field_name} must be a positive integer.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise SubstrateError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SubstrateError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise SubstrateError(f"{field_name} must be a non-negative integer.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise SubstrateError(f"{field_name} must be numeric.")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise SubstrateError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise SubstrateError(f"{field_name} must be finite.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0:
        raise SubstrateError(f"{field_name} must be non-negative.")
    return parsed


def _fov_deg(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if not 1.0 <= parsed <= 179.0:
        raise SubstrateError(f"{field_name} must be between 1 and 179 degrees.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if not 0.0 <= parsed <= 1.0:
        raise SubstrateError(f"{field_name} must be between 0 and 1.")
    return parsed


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _plain_json_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, MappingABC):
        raise SubstrateError(f"{field_name} must be a mapping.")
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
            raise SubstrateError(f"{field_name} must be finite.")
        return value
    raise SubstrateError(f"{field_name} contains unsupported value type {type(value).__name__}.")


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
        raise SubstrateError(f"{path} must be a mapping.")
    return value


_COEFFICIENT_NAMES: tuple[str, ...] = tuple(field.name for field in fields(ParametricSubstrateCoefficients))


__all__ = [
    "ComponentTiming",
    "ParametricSubstrateCoefficients",
    "ParametricSubstrateValueProvider",
    "ReferenceResolution",
    "SubstrateError",
    "SubstrateQuery",
    "SubstrateUncertainty",
    "SubstrateValue",
    "SubstrateValueProvider",
    "coerce_ref_resolution",
    "coerce_substrate_query",
    "load_parametric_substrate_provider",
    "parametric_substrate_provider_from_mapping",
]
