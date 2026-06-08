"""Substrate-value provider interfaces and built-in backends."""

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


@dataclass(frozen=True)
class EmpiricalSubstrateTableRow:
    """One measured lookup-table row for a substrate query coordinate."""

    query: SubstrateQuery
    visible_quality: float
    component_timing: ComponentTiming
    uncertainty: SubstrateUncertainty = field(default_factory=lambda: SubstrateUncertainty(0.0, 0.0, 1.0))
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            "query": self.query.as_payload(),
            "visible_quality": self.visible_quality,
            "component_timing": self.component_timing.as_payload(),
            "uncertainty": self.uncertainty.as_payload(),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class EmpiricalSubstrateTable:
    """Validated empirical substrate lookup table."""

    rows: tuple[EmpiricalSubstrateTableRow, ...]
    table_id: str = ""
    source_uri: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if not rows:
            raise SubstrateError("rows must contain at least one empirical table row.")
        if self.source_uri is not None and not self.source_uri:
            raise SubstrateError("source_uri must be non-empty when provided.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))
        table_id = self.table_id.strip() if isinstance(self.table_id, str) else ""
        if not table_id:
            table_id = f"empirical-{stable_config_id({'rows': [row.as_payload() for row in rows]})}"
        object.__setattr__(self, "table_id", table_id)

    def as_payload(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "rows": [row.as_payload() for row in self.rows],
            "source_uri": self.source_uri,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class EmpiricalSubstrateValueProvider:
    """Lookup-table substrate backend with deterministic interpolation."""

    table: EmpiricalSubstrateTable
    provider_id: str = "empirical-substrate-lookup"
    interpolation: str = "inverse_distance"
    max_neighbors: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.table, EmpiricalSubstrateTable):
            raise SubstrateError("table must be an EmpiricalSubstrateTable record.")
        if not isinstance(self.provider_id, str) or not self.provider_id.strip():
            raise SubstrateError("provider_id must be a non-empty string.")
        interpolation = _interpolation_mode(self.interpolation)
        max_neighbors = self.max_neighbors
        if max_neighbors is not None:
            max_neighbors = _positive_int(max_neighbors, "max_neighbors")
        object.__setattr__(self, "provider_id", self.provider_id.strip())
        object.__setattr__(self, "interpolation", interpolation)
        object.__setattr__(self, "max_neighbors", max_neighbors)
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def evaluate(self, query: SubstrateQuery | Mapping[str, Any]) -> SubstrateValue:
        """Evaluate one substrate query from measured table rows."""

        substrate_query = coerce_substrate_query(query)
        neighbors = self._neighbors(substrate_query)
        if self.interpolation == "nearest" or len(neighbors) == 1:
            selected = neighbors[0][1]
            metadata = self._metadata("nearest", substrate_query, ((1.0, selected),))
            return SubstrateValue(
                provider_id=self.provider_id,
                query=substrate_query,
                visible_quality=selected.visible_quality,
                component_timing=selected.component_timing,
                uncertainty=selected.uncertainty,
                metadata=metadata,
            )

        weighted_rows = _idw_weights(neighbors)
        metadata = self._metadata("inverse_distance", substrate_query, weighted_rows)
        return SubstrateValue(
            provider_id=self.provider_id,
            query=substrate_query,
            visible_quality=_weighted_average(weighted_rows, lambda row: row.visible_quality),
            component_timing=ComponentTiming(
                generation_ms=_weighted_average(weighted_rows, lambda row: row.component_timing.generation_ms),
                transfer_ms=_weighted_average(weighted_rows, lambda row: row.component_timing.transfer_ms),
                restoration_ms=_weighted_average(weighted_rows, lambda row: row.component_timing.restoration_ms),
                render_ms=_weighted_average(weighted_rows, lambda row: row.component_timing.render_ms),
            ),
            uncertainty=SubstrateUncertainty(
                quality_stddev=_weighted_average(weighted_rows, lambda row: row.uncertainty.quality_stddev),
                timing_stddev_ms=_weighted_average(weighted_rows, lambda row: row.uncertainty.timing_stddev_ms),
                confidence=_weighted_average(weighted_rows, lambda row: row.uncertainty.confidence),
            ),
            metadata=metadata,
        )

    def evaluate_many(self, queries: Iterable[SubstrateQuery | Mapping[str, Any]]) -> tuple[SubstrateValue, ...]:
        """Evaluate several substrate queries deterministically."""

        return tuple(self.evaluate(query) for query in queries)

    def as_payload(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "backend": "empirical",
            "interpolation": self.interpolation,
            "max_neighbors": self.max_neighbors,
            "table": self.table.as_payload(),
            "metadata": _to_payload(self.metadata),
        }

    def _neighbors(self, query: SubstrateQuery) -> tuple[tuple[float, EmpiricalSubstrateTableRow], ...]:
        query_coordinates = _query_coordinates(query)
        row_coordinates = [_query_coordinates(row.query) for row in self.table.rows]
        ranges = _coordinate_ranges((*row_coordinates, query_coordinates))
        distances = tuple(
            sorted(
                (
                    (_normalized_distance(query_coordinates, coordinates, ranges), row)
                    for row, coordinates in zip(self.table.rows, row_coordinates, strict=True)
                ),
                key=lambda item: (item[0], stable_config_id(item[1].query.as_payload())),
            )
        )
        if self.max_neighbors is None:
            return distances
        return distances[: self.max_neighbors]

    def _metadata(
        self,
        interpolation_used: str,
        query: SubstrateQuery,
        weighted_rows: tuple[tuple[float, EmpiricalSubstrateTableRow], ...],
    ) -> dict[str, Any]:
        return {
            "model": {
                "kind": "empirical_lookup",
                "table_id": self.table.table_id,
                "interpolation": interpolation_used,
                "requested_interpolation": self.interpolation,
                "neighbor_count": len(weighted_rows),
                "neighbors": [
                    {
                        "weight": weight,
                        "query": row.query.as_payload(),
                        "metadata": _to_payload(row.metadata),
                    }
                    for weight, row in weighted_rows
                ],
            },
            "query_coordinates": _query_coordinates(query),
            "provider": _to_payload(self.metadata),
        }


def load_parametric_substrate_provider(path: str | Path) -> ParametricSubstrateValueProvider:
    """Load a parametric substrate provider definition from JSON, TOML, YAML, or YML."""

    provider_path = Path(path)
    try:
        raw_provider = load_config_file(provider_path)
    except ConfigError as exc:
        raise SubstrateError(str(exc)) from exc
    return parametric_substrate_provider_from_mapping(raw_provider, source_uri=str(provider_path))


def load_empirical_substrate_provider(path: str | Path) -> EmpiricalSubstrateValueProvider:
    """Load an empirical lookup-table provider from JSON, TOML, YAML, or YML."""

    provider_path = Path(path)
    try:
        raw_provider = load_config_file(provider_path)
    except ConfigError as exc:
        raise SubstrateError(str(exc)) from exc
    return empirical_substrate_provider_from_mapping(raw_provider, source_uri=str(provider_path))


def load_substrate_provider(path: str | Path) -> SubstrateValueProvider:
    """Load a substrate provider selected by its backend field."""

    provider_path = Path(path)
    try:
        raw_provider = load_config_file(provider_path)
    except ConfigError as exc:
        raise SubstrateError(str(exc)) from exc
    return substrate_provider_from_mapping(raw_provider, source_uri=str(provider_path))


def substrate_provider_from_mapping(
    raw_provider: Mapping[str, Any],
    *,
    source_uri: str | None = None,
) -> SubstrateValueProvider:
    """Normalize a provider definition into a swappable substrate backend."""

    root = _require_mapping(raw_provider, "provider")
    backend = _backend_name(_first_present(root, ("backend", "type", "kind")))
    if backend == "parametric":
        return parametric_substrate_provider_from_mapping(root, source_uri=source_uri)
    if backend == "empirical":
        return empirical_substrate_provider_from_mapping(root, source_uri=source_uri)
    raise SubstrateError("provider.backend must select parametric or empirical.")


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


def empirical_substrate_provider_from_mapping(
    raw_provider: Mapping[str, Any],
    *,
    source_uri: str | None = None,
) -> EmpiricalSubstrateValueProvider:
    """Normalize an empirical lookup-table provider definition."""

    root = _require_mapping(raw_provider, "provider")
    raw_backend = _first_present(root, ("backend", "type", "kind"))
    backend = "empirical" if raw_backend is None else _backend_name(raw_backend)
    if backend != "empirical":
        raise SubstrateError("provider.backend must be empirical, lookup, or lookup-table.")
    provider_id = _string_or_none(_first_present(root, ("provider_id", "id", "name"))) or "empirical-substrate-lookup"
    interpolation = str(_first_present(root, ("interpolation", "interpolation_mode")) or "inverse_distance")
    max_neighbors = _first_present(root, ("max_neighbors", "neighbors"))
    table = empirical_substrate_table_from_mapping(
        _table_mapping(root),
        table_id=_string_or_none(_first_present(root, ("table_id", "profile_id"))),
        source_uri=source_uri,
    )
    metadata = _plain_json_mapping(root.get("metadata"), "metadata") if root.get("metadata") is not None else {}
    metadata["provenance"] = {"source_uri": source_uri}
    return EmpiricalSubstrateValueProvider(
        table=table,
        provider_id=provider_id,
        interpolation=interpolation,
        max_neighbors=max_neighbors,
        metadata=metadata,
    )


def empirical_substrate_table_from_mapping(
    raw_table: Mapping[str, Any],
    *,
    table_id: str | None = None,
    source_uri: str | None = None,
) -> EmpiricalSubstrateTable:
    """Normalize empirical lookup-table rows into a validated table."""

    table = _require_mapping(raw_table, "table")
    rows = _first_present(table, ("rows", "samples", "measurements", "entries"))
    if not isinstance(rows, list) or not rows:
        raise SubstrateError("table.rows must contain at least one empirical row.")
    metadata = _plain_json_mapping(table.get("metadata"), "metadata") if table.get("metadata") is not None else {}
    return EmpiricalSubstrateTable(
        rows=tuple(_empirical_row_from_mapping(_require_mapping(row, f"table.rows[{index}]"), index=index) for index, row in enumerate(rows)),
        table_id=table_id or _string_or_none(_first_present(table, ("table_id", "id", "name"))) or "",
        source_uri=source_uri,
        metadata=metadata,
    )


def _empirical_row_from_mapping(row: Mapping[str, Any], *, index: int) -> EmpiricalSubstrateTableRow:
    row_query = row.get("query")
    if isinstance(row_query, MappingABC):
        query_payload = dict(row_query)
    else:
        query_payload = dict(row)
    query = coerce_substrate_query(query_payload)
    timing = _component_timing_from_mapping(row, f"table.rows[{index}]")
    uncertainty = _uncertainty_from_mapping(row, f"table.rows[{index}]")
    visible_quality = _required(
        row,
        ("visible_quality", "quality", "visible_quality_score", "quality_score"),
        f"table.rows[{index}].visible_quality",
    )
    metadata = _plain_json_mapping(row.get("metadata"), "metadata") if row.get("metadata") is not None else {}
    metadata["source_index"] = index
    return EmpiricalSubstrateTableRow(
        query=query,
        visible_quality=visible_quality,
        component_timing=timing,
        uncertainty=uncertainty,
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


def _component_timing_from_mapping(row: Mapping[str, Any], path: str) -> ComponentTiming:
    timing = row.get("component_timing") or row.get("timing")
    timing_mapping = _require_mapping(timing, f"{path}.component_timing") if timing is not None else row
    return ComponentTiming(
        generation_ms=_required(timing_mapping, ("generation_ms", "generation", "generate_ms"), f"{path}.generation_ms"),
        transfer_ms=_required(timing_mapping, ("transfer_ms", "transfer", "network_ms"), f"{path}.transfer_ms"),
        restoration_ms=_required(timing_mapping, ("restoration_ms", "restoration", "restore_ms"), f"{path}.restoration_ms"),
        render_ms=_required(timing_mapping, ("render_ms", "render", "rendering_ms"), f"{path}.render_ms"),
    )


def _uncertainty_from_mapping(row: Mapping[str, Any], path: str) -> SubstrateUncertainty:
    uncertainty = row.get("uncertainty")
    uncertainty_mapping = _require_mapping(uncertainty, f"{path}.uncertainty") if uncertainty is not None else row
    return SubstrateUncertainty(
        quality_stddev=_non_negative_float(
            _first_present(uncertainty_mapping, ("quality_stddev", "quality_sigma", "quality_uncertainty")) or 0.0,
            f"{path}.quality_stddev",
        ),
        timing_stddev_ms=_non_negative_float(
            _first_present(uncertainty_mapping, ("timing_stddev_ms", "timing_sigma_ms", "timing_uncertainty_ms")) or 0.0,
            f"{path}.timing_stddev_ms",
        ),
        confidence=_unit_interval(_first_present(uncertainty_mapping, ("confidence",)) or 1.0, f"{path}.confidence"),
    )


def _table_mapping(root: Mapping[str, Any]) -> Mapping[str, Any]:
    table = _first_present(root, ("table", "lookup_table", "profile"))
    if table is None:
        return root
    return _require_mapping(table, "provider.table")


def _backend_name(value: Any) -> str:
    normalized = str(value or "parametric").strip().lower().replace("_", "-")
    if normalized in {"parametric", "analytic", "default"}:
        return "parametric"
    if normalized in {"empirical", "lookup", "lookup-table", "table", "lut"}:
        return "empirical"
    return normalized


def _interpolation_mode(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "idw": "inverse_distance",
        "inverse_distance": "inverse_distance",
        "inverse_distance_weighted": "inverse_distance",
        "linear": "inverse_distance",
        "nearest": "nearest",
        "nearest_neighbor": "nearest",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in {"inverse_distance", "nearest"}:
        raise SubstrateError("interpolation must be inverse_distance, linear, idw, nearest, or nearest_neighbor.")
    return resolved


def _query_coordinates(query: SubstrateQuery) -> dict[str, float]:
    return {
        "layer": float(query.layer),
        "resolution_mp": query.ref_resolution.megapixels,
        "fov_deg": query.fov_deg,
        "view_mismatch_deg": query.view_mismatch_deg,
        "freshness_s": query.freshness_ms / 1000.0,
    }


def _coordinate_ranges(coordinates: tuple[Mapping[str, float], ...]) -> dict[str, float]:
    ranges: dict[str, float] = {}
    for key in coordinates[0]:
        values = [coordinate[key] for coordinate in coordinates]
        ranges[key] = max(values) - min(values)
    return ranges


def _normalized_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
    ranges: Mapping[str, float],
) -> float:
    squared_distance = 0.0
    for key, left_value in left.items():
        axis_range = ranges[key]
        if axis_range == 0:
            continue
        squared_distance += ((left_value - right[key]) / axis_range) ** 2
    return math.sqrt(squared_distance)


def _idw_weights(
    neighbors: tuple[tuple[float, EmpiricalSubstrateTableRow], ...],
) -> tuple[tuple[float, EmpiricalSubstrateTableRow], ...]:
    exact = tuple((distance, row) for distance, row in neighbors if distance == 0)
    if exact:
        equal_weight = 1.0 / len(exact)
        return tuple((equal_weight, row) for _, row in exact)
    raw_weights = tuple((1.0 / (distance**2), row) for distance, row in neighbors)
    total_weight = sum(weight for weight, _ in raw_weights)
    return tuple((weight / total_weight, row) for weight, row in raw_weights)


def _weighted_average(
    weighted_rows: tuple[tuple[float, EmpiricalSubstrateTableRow], ...],
    value_fn: Any,
) -> float:
    return sum(weight * value_fn(row) for weight, row in weighted_rows)


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
    "EmpiricalSubstrateTable",
    "EmpiricalSubstrateTableRow",
    "EmpiricalSubstrateValueProvider",
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
    "empirical_substrate_provider_from_mapping",
    "empirical_substrate_table_from_mapping",
    "load_empirical_substrate_provider",
    "load_parametric_substrate_provider",
    "load_substrate_provider",
    "parametric_substrate_provider_from_mapping",
    "substrate_provider_from_mapping",
]
