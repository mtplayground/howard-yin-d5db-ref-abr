"""Base external substrate provider backed by validated measurement traces."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from ref_abr.config import ConfigError, load_config_file, stable_config_id
from ref_abr.domain import ExternalMeasurementRecord
from ref_abr.external_measurements import (
    ExternalMeasurementError,
    load_external_measurement_records,
    materialize_external_measurement_record,
)
from ref_abr.substrate import (
    ComponentTiming,
    SubstrateError,
    SubstrateQuery,
    SubstrateUncertainty,
    SubstrateValue,
    coerce_substrate_query,
)


ExternalMatchPolicy = Literal["query", "first"]


@dataclass(frozen=True)
class ExternalSubstrateProviderConfig:
    """Configuration for mapping external trace records onto substrate values."""

    provider_id: str = "external-trace-substrate"
    match_policy: ExternalMatchPolicy = "query"
    quality_stddev: float = 0.0
    timing_stddev_ms: float = 0.0
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _non_empty_string(self.provider_id, "provider_id"))
        match_policy = str(self.match_policy).strip().lower().replace("-", "_")
        if match_policy in {"exact", "exact_query", "query"}:
            match_policy = "query"
        if match_policy not in {"query", "first"}:
            raise SubstrateError("match_policy must be query, exact_query, or first.")
        object.__setattr__(self, "match_policy", match_policy)
        object.__setattr__(self, "quality_stddev", _non_negative_float(self.quality_stddev, "quality_stddev"))
        object.__setattr__(self, "timing_stddev_ms", _non_negative_float(self.timing_stddev_ms, "timing_stddev_ms"))
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, "confidence"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "match_policy": self.match_policy,
            "quality_stddev": self.quality_stddev,
            "timing_stddev_ms": self.timing_stddev_ms,
            "confidence": self.confidence,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class ExternalTraceSubstrateProvider:
    """Substrate provider that serves values from external measurement records."""

    records: Sequence[ExternalMeasurementRecord | Mapping[str, Any]]
    config: ExternalSubstrateProviderConfig = field(default_factory=ExternalSubstrateProviderConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.config, ExternalSubstrateProviderConfig):
            raise SubstrateError("config must be an ExternalSubstrateProviderConfig record.")
        materialized = tuple(_coerce_external_record(record, index) for index, record in enumerate(self.records))
        if not materialized:
            raise SubstrateError("records must contain at least one external measurement record.")
        object.__setattr__(self, "records", materialized)
        object.__setattr__(self, "_query_index", _build_query_index(materialized))

    @property
    def provider_id(self) -> str:
        return self.config.provider_id

    def evaluate(self, query: SubstrateQuery | Mapping[str, Any]) -> SubstrateValue:
        """Map the best matching external measurement to a substrate value."""

        substrate_query = coerce_substrate_query(query)
        record = self._select_record(substrate_query)
        return external_measurement_to_substrate_value(
            record,
            substrate_query,
            provider_id=self.provider_id,
            uncertainty=SubstrateUncertainty(
                quality_stddev=self.config.quality_stddev,
                timing_stddev_ms=self.config.timing_stddev_ms,
                confidence=self.config.confidence,
            ),
            provider_metadata=self.config.metadata,
        )

    def evaluate_many(self, queries: Sequence[SubstrateQuery | Mapping[str, Any]]) -> tuple[SubstrateValue, ...]:
        """Evaluate several substrate queries deterministically."""

        return tuple(self.evaluate(query) for query in queries)

    def as_payload(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "backend": "external",
            "config": self.config.as_payload(),
            "record_count": len(self.records),
            "query_index_size": len(self._query_index),
            "metadata": _to_payload(self.config.metadata),
        }

    def _select_record(self, query: SubstrateQuery) -> ExternalMeasurementRecord:
        if self.config.match_policy == "first":
            return self.records[0]
        signature = query_signature(query)
        matches = self._query_index.get(signature)
        if matches:
            return matches[0]
        if len(self.records) == 1 and not self._query_index:
            return self.records[0]
        raise SubstrateError(
            "external measurement provider has no record matching query "
            f"{query.as_payload()}; add metadata.query to records or use match_policy='first'."
        )


def external_measurement_to_substrate_value(
    record: ExternalMeasurementRecord,
    query: SubstrateQuery,
    *,
    provider_id: str = "external-trace-substrate",
    uncertainty: SubstrateUncertainty | None = None,
    provider_metadata: Mapping[str, Any] | None = None,
) -> SubstrateValue:
    """Convert one external measurement into the canonical substrate output."""

    if not isinstance(record, ExternalMeasurementRecord):
        raise SubstrateError("record must be an ExternalMeasurementRecord.")
    if not isinstance(query, SubstrateQuery):
        raise SubstrateError("query must be a SubstrateQuery.")
    if uncertainty is None:
        uncertainty = SubstrateUncertainty(quality_stddev=0.0, timing_stddev_ms=0.0, confidence=1.0)
    if not isinstance(uncertainty, SubstrateUncertainty):
        raise SubstrateError("uncertainty must be a SubstrateUncertainty record.")

    return SubstrateValue(
        provider_id=provider_id,
        query=query,
        visible_quality=record.visible_quality,
        component_timing=ComponentTiming(
            generation_ms=record.generation_ms,
            transfer_ms=record.transfer_ms,
            restoration_ms=record.decode_ms + record.restore_ms,
            render_ms=record.render_ms,
        ),
        uncertainty=uncertainty,
        metadata={
            "model": {
                "kind": "external_trace",
                "record_id": record.record_id,
                "backend_id": record.backend_id,
                "object_id": record.object_id,
                "frame_id": record.frame_id,
                "candidate_kind": record.candidate_kind,
                "artifact_uri": record.artifact_uri,
                "size_bytes": record.size_bytes,
                "decode_ms": record.decode_ms,
                "restore_ms": record.restore_ms,
                "dropped_frame": record.dropped_frame,
                "deadline_hit": record.deadline_hit,
                "provenance": _to_payload(record.provenance),
            },
            "record_metadata": _to_payload(record.metadata),
            "provider": _to_payload(provider_metadata or {}),
        },
    )


def load_external_substrate_provider(path: str | Path) -> ExternalTraceSubstrateProvider:
    """Load an external trace substrate provider from JSON, TOML, YAML, or YML."""

    provider_path = Path(path)
    try:
        raw_provider = load_config_file(provider_path)
    except ConfigError as exc:
        raise SubstrateError(str(exc)) from exc
    return external_substrate_provider_from_mapping(raw_provider, source_uri=str(provider_path))


def external_substrate_provider_from_mapping(
    raw_provider: Mapping[str, Any],
    *,
    source_uri: str | None = None,
) -> ExternalTraceSubstrateProvider:
    """Normalize an external provider definition and its measurement records."""

    root = _require_mapping(raw_provider, "provider")
    backend = _backend_name(_first_present(root, ("backend", "type", "kind")))
    if backend != "external":
        raise SubstrateError("provider.backend must be external, trace, external-trace, or external-measurements.")

    provider_id = _string_or_none(_first_present(root, ("provider_id", "id", "name"))) or "external-trace-substrate"
    records = _external_records_from_mapping(root, source_uri=source_uri)
    uncertainty = _require_mapping(root.get("uncertainty"), "provider.uncertainty") if root.get("uncertainty") is not None else root
    metadata = _plain_json_mapping(root.get("metadata"), "metadata") if root.get("metadata") is not None else {}
    metadata["provenance"] = {"source_uri": source_uri}
    config = ExternalSubstrateProviderConfig(
        provider_id=provider_id,
        match_policy=str(_first_present(root, ("match_policy", "matching")) or "query"),
        quality_stddev=_non_negative_float(
            _first_present(uncertainty, ("quality_stddev", "quality_sigma", "quality_uncertainty")) or 0.0,
            "provider.quality_stddev",
        ),
        timing_stddev_ms=_non_negative_float(
            _first_present(uncertainty, ("timing_stddev_ms", "timing_sigma_ms", "timing_uncertainty_ms")) or 0.0,
            "provider.timing_stddev_ms",
        ),
        confidence=_unit_interval(_first_present(uncertainty, ("confidence",)) or 1.0, "provider.confidence"),
        metadata=metadata,
    )
    return ExternalTraceSubstrateProvider(records=records, config=config)


def query_signature(query: SubstrateQuery | Mapping[str, Any]) -> str:
    """Return the stable external-record lookup key for a substrate query."""

    return stable_config_id(coerce_substrate_query(query).as_payload())


def _external_records_from_mapping(
    root: Mapping[str, Any],
    *,
    source_uri: str | None,
) -> tuple[ExternalMeasurementRecord, ...]:
    records_path = _first_present(root, ("records_path", "trace_path", "measurements_path", "path"))
    inline_records = _first_present(root, ("records", "measurements", "external_measurements"))
    if records_path is not None and inline_records is not None:
        raise SubstrateError("provider must set either records_path or records, not both.")
    if records_path is not None:
        path = Path(str(records_path))
        if not path.is_absolute() and source_uri:
            path = Path(source_uri).parent / path
        try:
            return load_external_measurement_records(path)
        except ExternalMeasurementError as exc:
            raise SubstrateError(str(exc)) from exc
    if inline_records is None:
        raise SubstrateError("external provider requires records_path or records.")
    if not isinstance(inline_records, list):
        raise SubstrateError("provider.records must be a list of external measurement records.")
    return tuple(_coerce_external_record(record, index) for index, record in enumerate(inline_records))


def _coerce_external_record(record: ExternalMeasurementRecord | Mapping[str, Any], index: int) -> ExternalMeasurementRecord:
    if isinstance(record, ExternalMeasurementRecord):
        return record
    if isinstance(record, MappingABC):
        try:
            return materialize_external_measurement_record(record, path=f"records[{index}]")
        except ExternalMeasurementError as exc:
            raise SubstrateError(str(exc)) from exc
    raise SubstrateError(f"records[{index}] must be a mapping or ExternalMeasurementRecord.")


def _build_query_index(
    records: tuple[ExternalMeasurementRecord, ...],
) -> dict[str, tuple[ExternalMeasurementRecord, ...]]:
    indexed: dict[str, list[ExternalMeasurementRecord]] = {}
    for record in records:
        query_mapping = _record_query_mapping(record)
        if query_mapping is None:
            continue
        signature = query_signature(query_mapping)
        indexed.setdefault(signature, []).append(record)
    return {signature: tuple(matches) for signature, matches in indexed.items()}


def _record_query_mapping(record: ExternalMeasurementRecord) -> Mapping[str, Any] | None:
    for source in (record.metadata, record.provenance):
        query = source.get("query")
        if isinstance(query, MappingABC):
            return query
    return None


def _backend_name(value: Any) -> str:
    normalized = str(value or "external").strip().lower().replace("_", "-")
    if normalized in {"external", "trace", "external-trace", "external-measurement", "external-measurements"}:
        return "external"
    return normalized


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise SubstrateError(f"{path} must be a mapping.")
    return value


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SubstrateError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _non_negative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise SubstrateError(f"{field_name} must be numeric.")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise SubstrateError(f"{field_name} must be numeric.") from exc
    if not math_is_finite(parsed):
        raise SubstrateError(f"{field_name} must be finite.")
    if parsed < 0:
        raise SubstrateError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name)
    if parsed > 1:
        raise SubstrateError(f"{field_name} must be between 0 and 1.")
    return parsed


def _plain_json_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    mapping = _require_mapping(value, field_name)
    return {str(key): _plain_json_value(nested, f"{field_name}.{key}") for key, nested in mapping.items()}


def _plain_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _plain_json_value(nested, f"{field_name}.{key}") for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item, field_name) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math_is_finite(value):
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


def math_is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


__all__ = [
    "ExternalSubstrateProviderConfig",
    "ExternalTraceSubstrateProvider",
    "external_measurement_to_substrate_value",
    "external_substrate_provider_from_mapping",
    "load_external_substrate_provider",
    "query_signature",
]
