"""Freeze selected scheduling method configurations for downstream harnesses."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from ref_abr.config import stable_config_id
from ref_abr.domain import FrozenMethodManifest, MetricRecord
from ref_abr.entrypoints import EntrypointInvocation, EntrypointResult
from ref_abr.schema import DOMAIN_SCHEMA_VERSION, RECORD_TYPE_FIELD, SCHEMA_VERSION_FIELD, materialize_record, stamp_record


FreezeTieBreakerMode = Literal["maximize", "minimize"]
DEFAULT_PRIMARY_METRIC = "method_selection_quality"
DEFAULT_EXCLUDED_METHOD_IDS: tuple[str, ...] = (
    "perfect-information-oracle",
    "diagnostic-layered-3dgs",
    "diagnostic-viewport-tile",
    "learned-diagnostic-selector",
)
DEFAULT_EXCLUDED_METHOD_PREFIXES: tuple[str, ...] = ("diagnostic-", "oracle-")
DEFAULT_TIE_BREAKERS: tuple[Mapping[str, str], ...] = (
    {"metric_name": "method_selection_deadline_score", "mode": "maximize"},
    {"metric_name": "method_selection_resource_efficiency", "mode": "maximize"},
    {"metric_name": "method_selection_interpretability", "mode": "maximize"},
    {"metric_name": "method_selection_runtime_ms", "mode": "minimize"},
)


class FreezeMethodError(ValueError):
    """Raised when freeze_method inputs or exports are invalid."""


@dataclass(frozen=True)
class FreezeTieBreaker:
    """Metric used to deterministically break primary-metric ties."""

    metric_name: str
    mode: FreezeTieBreakerMode = "maximize"

    def __post_init__(self) -> None:
        _require_non_empty(self.metric_name, "metric_name")
        if self.mode not in {"maximize", "minimize"}:
            raise FreezeMethodError("tie breaker mode must be one of: maximize, minimize.")

    def score(self, value: float | None) -> float:
        parsed = 0.0 if value is None else _finite_float(value, self.metric_name)
        return parsed if self.mode == "maximize" else -parsed

    def as_payload(self) -> dict[str, str]:
        return {"metric_name": self.metric_name, "mode": self.mode}


@dataclass(frozen=True)
class FreezeMethodConfig:
    """Decision rule and manifest settings for freezing one method."""

    method_name: str = "RefABR frozen method"
    version: str = "1.0"
    primary_metric: str = DEFAULT_PRIMARY_METRIC
    primary_metric_mode: FreezeTieBreakerMode = "maximize"
    tie_breakers: Sequence[FreezeTieBreaker | Mapping[str, Any]] = DEFAULT_TIE_BREAKERS
    calibration_splits: Sequence[str] = ("calibration",)
    excluded_splits: Sequence[str] = ("final",)
    candidate_method_ids: Sequence[str] = ()
    excluded_method_ids: Sequence[str] = DEFAULT_EXCLUDED_METHOD_IDS
    excluded_method_prefixes: Sequence[str] = DEFAULT_EXCLUDED_METHOD_PREFIXES
    method_entrypoints: Mapping[str, str] = field(default_factory=dict)
    default_entrypoint: str = "ref_abr.methods:plan_schedule"
    method_parameters: Mapping[str, Any] = field(default_factory=dict)
    artifact_uri: str | None = None
    source_uri: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.method_name, "method_name")
        _require_non_empty(self.version, "version")
        _require_non_empty(self.primary_metric, "primary_metric")
        if self.primary_metric_mode not in {"maximize", "minimize"}:
            raise FreezeMethodError("primary_metric_mode must be one of: maximize, minimize.")
        object.__setattr__(self, "tie_breakers", _tie_breaker_tuple(self.tie_breakers))
        object.__setattr__(self, "calibration_splits", _string_tuple(self.calibration_splits, "calibration_splits"))
        object.__setattr__(self, "excluded_splits", _string_tuple(self.excluded_splits, "excluded_splits"))
        object.__setattr__(self, "candidate_method_ids", _string_tuple(self.candidate_method_ids, "candidate_method_ids", allow_empty=True))
        object.__setattr__(self, "excluded_method_ids", _string_tuple(self.excluded_method_ids, "excluded_method_ids", allow_empty=True))
        object.__setattr__(
            self,
            "excluded_method_prefixes",
            _string_tuple(self.excluded_method_prefixes, "excluded_method_prefixes", allow_empty=True),
        )
        object.__setattr__(self, "method_entrypoints", _string_mapping(self.method_entrypoints, "method_entrypoints"))
        _require_non_empty(self.default_entrypoint, "default_entrypoint")
        object.__setattr__(self, "method_parameters", _plain_json_mapping(self.method_parameters, "method_parameters"))
        if self.artifact_uri is not None:
            _require_non_empty(self.artifact_uri, "artifact_uri")
        if self.source_uri is not None:
            _require_non_empty(self.source_uri, "source_uri")
        object.__setattr__(self, "provenance", _plain_json_mapping(self.provenance, "provenance"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def config_id(self) -> str:
        return f"freeze-config-{stable_config_id(self.stable_payload())}"

    def stable_payload(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "version": self.version,
            "primary_metric": self.primary_metric,
            "primary_metric_mode": self.primary_metric_mode,
            "tie_breakers": [tie_breaker.as_payload() for tie_breaker in self.tie_breakers],
            "calibration_splits": list(self.calibration_splits),
            "excluded_splits": list(self.excluded_splits),
            "candidate_method_ids": list(self.candidate_method_ids),
            "excluded_method_ids": list(self.excluded_method_ids),
            "excluded_method_prefixes": list(self.excluded_method_prefixes),
            "method_entrypoints": dict(self.method_entrypoints),
            "default_entrypoint": self.default_entrypoint,
            "method_parameters": _to_payload(self.method_parameters),
            "artifact_uri": self.artifact_uri,
            "source_uri": self.source_uri,
            "provenance": _to_payload(self.provenance),
            "metadata": _to_payload(self.metadata),
        }

    def as_payload(self) -> dict[str, Any]:
        return {"config_id": self.config_id, **self.stable_payload()}


@dataclass(frozen=True)
class FreezeDecision:
    """Selected method and ranking evidence captured before manifest export."""

    selected_method_id: str
    primary_metric: str
    primary_metric_mode: FreezeTieBreakerMode
    ranked_methods: Sequence[Mapping[str, Any]]
    excluded_methods: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.selected_method_id, "selected_method_id")
        _require_non_empty(self.primary_metric, "primary_metric")
        if self.primary_metric_mode not in {"maximize", "minimize"}:
            raise FreezeMethodError("primary_metric_mode must be one of: maximize, minimize.")
        object.__setattr__(self, "ranked_methods", tuple(_plain_json_mapping(row, "ranked_methods") for row in self.ranked_methods))
        object.__setattr__(self, "excluded_methods", tuple(_plain_json_mapping(row, "excluded_methods") for row in self.excluded_methods))
        if not self.ranked_methods:
            raise FreezeMethodError("ranked_methods must not be empty.")

    @property
    def decision_id(self) -> str:
        return f"freeze-decision-{stable_config_id(self.as_payload(include_id=False))}"

    def as_payload(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "selected_method_id": self.selected_method_id,
            "primary_metric": self.primary_metric,
            "primary_metric_mode": self.primary_metric_mode,
            "ranked_methods": [_to_payload(row) for row in self.ranked_methods],
            "excluded_methods": [_to_payload(row) for row in self.excluded_methods],
        }
        if include_id:
            payload["decision_id"] = self.decision_id
        return payload


@dataclass(frozen=True)
class FreezeMethodResult:
    """Frozen method manifest plus the decision evidence used to create it."""

    manifest: FrozenMethodManifest
    decision: FreezeDecision
    output_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, FrozenMethodManifest):
            raise FreezeMethodError("manifest must be a FrozenMethodManifest record.")
        if not isinstance(self.decision, FreezeDecision):
            raise FreezeMethodError("decision must be a FreezeDecision record.")
        if self.output_path is not None:
            _require_non_empty(self.output_path, "output_path")

    def as_payload(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.as_payload(),
            "decision": self.decision.as_payload(),
            "output_path": self.output_path,
        }


def freeze_method(
    metrics: Sequence[MetricRecord],
    *,
    config: FreezeMethodConfig | None = None,
) -> FreezeMethodResult:
    """Apply the freeze decision rule and return a frozen method manifest."""

    freeze_config = config or FreezeMethodConfig()
    if not isinstance(freeze_config, FreezeMethodConfig):
        raise FreezeMethodError("config must be a FreezeMethodConfig record.")
    metric_records = _coerce_metrics(metrics)
    decision = apply_freeze_decision(metric_records, config=freeze_config)
    entrypoint = freeze_config.method_entrypoints.get(decision.selected_method_id, freeze_config.default_entrypoint)
    parameters = {
        **freeze_config.method_parameters,
        "frozen_method_id": decision.selected_method_id,
        "freeze_decision_id": decision.decision_id,
    }
    manifest_payload = {
        "selected_method_id": decision.selected_method_id,
        "version": freeze_config.version,
        "parameters": parameters,
        "calibration_splits": list(freeze_config.calibration_splits),
        "excluded_splits": list(freeze_config.excluded_splits),
        "primary_metric": freeze_config.primary_metric,
        "decision_id": decision.decision_id,
    }
    config_id = f"frozen-method-config-{stable_config_id(manifest_payload)}"
    artifact_uri = freeze_config.artifact_uri or f"frozen-method://{decision.selected_method_id}/{config_id}"
    manifest = FrozenMethodManifest(
        method_id=decision.selected_method_id,
        method_name=freeze_config.method_name,
        version=freeze_config.version,
        config_id=config_id,
        artifact_uri=artifact_uri,
        entrypoint=entrypoint,
        parameters=parameters,
        source_uri=freeze_config.source_uri,
        metadata={
            **freeze_config.metadata,
            "freeze_method": {
                "config": freeze_config.as_payload(),
                "decision": decision.as_payload(),
                "calibration_splits": list(freeze_config.calibration_splits),
                "excluded_splits": list(freeze_config.excluded_splits),
                "primary_metric": freeze_config.primary_metric,
                "provenance": {
                    **freeze_config.provenance,
                    "input_metric_count": len(metric_records),
                    "schema_version": DOMAIN_SCHEMA_VERSION,
                    "source": "freeze_method",
                },
            },
        },
    )
    return FreezeMethodResult(manifest=manifest, decision=decision)


def apply_freeze_decision(metrics: Sequence[MetricRecord], *, config: FreezeMethodConfig) -> FreezeDecision:
    """Rank eligible methods by primary metric, tie breakers, and method id."""

    metric_records = _coerce_metrics(metrics)
    scores = _aggregate_scores(metric_records)
    allowed_candidates = set(config.candidate_method_ids)
    ranked: list[Mapping[str, Any]] = []
    excluded: list[Mapping[str, Any]] = []
    for method_id in sorted(scores):
        reason = _exclusion_reason(method_id, config)
        if allowed_candidates and method_id not in allowed_candidates:
            reason = "not_configured_candidate"
        if reason is not None:
            excluded.append({"method_id": method_id, "reason": reason, "metrics": scores[method_id]})
            continue
        if config.primary_metric not in scores[method_id]:
            excluded.append({"method_id": method_id, "reason": "missing_primary_metric", "metrics": scores[method_id]})
            continue
        ranked.append(
            {
                "method_id": method_id,
                "metrics": scores[method_id],
                "score_vector": _score_vector(scores[method_id], config),
            }
        )
    if not ranked:
        raise FreezeMethodError("No eligible methods have the configured primary metric.")
    ranked = sorted(ranked, key=lambda row: (tuple(row["score_vector"]), _reverse_lexical_key(str(row["method_id"]))), reverse=True)
    return FreezeDecision(
        selected_method_id=str(ranked[0]["method_id"]),
        primary_metric=config.primary_metric,
        primary_metric_mode=config.primary_metric_mode,
        ranked_methods=tuple(ranked),
        excluded_methods=tuple(excluded),
    )


def export_frozen_method_manifest(output_root: str | Path, result: FreezeMethodResult | FrozenMethodManifest) -> Path:
    """Write a schema-stamped FrozenMethodManifest and return its path."""

    manifest = result.manifest if isinstance(result, FreezeMethodResult) else result
    if not isinstance(manifest, FrozenMethodManifest):
        raise FreezeMethodError("result must be a FreezeMethodResult or FrozenMethodManifest record.")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "frozen_method_manifest.json"
    content = json.dumps(stamp_record(manifest, record_type="frozen_method_manifest"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    _write_text_atomic(path, content)
    return path


def freeze_method_entrypoint(invocation: EntrypointInvocation) -> EntrypointResult:
    """Load method-selection evidence, freeze a method, and write its manifest."""

    if invocation.verb != "freeze_method":
        raise FreezeMethodError(f"freeze_method handler received verb {invocation.verb!r}.")
    raw_config = _entrypoint_config(invocation)
    freeze_config = _freeze_config_from_mapping(raw_config, invocation)
    output_root = _output_root(raw_config, invocation)
    metrics = _load_metric_evidence(raw_config, invocation)

    result = freeze_method(metrics, config=freeze_config)
    if invocation.dry_run:
        return EntrypointResult(
            status="dry_run",
            message="freeze_method decision resolved; dry run skipped manifest export.",
            payload={
                "config": freeze_config.as_payload(),
                "decision": result.decision.as_payload(),
                "manifest": result.manifest.as_payload(),
                "output_root": output_root.as_posix(),
            },
        )

    path = export_frozen_method_manifest(output_root, result)
    exported = FreezeMethodResult(manifest=result.manifest, decision=result.decision, output_path=path.as_posix())
    return EntrypointResult(
        status="ok",
        message=f"freeze_method froze {result.manifest.method_id}.",
        payload=exported.as_payload(),
    )


def _aggregate_scores(metrics: Sequence[MetricRecord]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for metric in metrics:
        method_id = metric.tags.get("method") or metric.tags.get("method_id")
        if method_id is None:
            raw_method_id = metric.metadata.get("method_id")
            method_id = raw_method_id if isinstance(raw_method_id, str) else None
        if method_id is None:
            continue
        grouped.setdefault(method_id, {}).setdefault(metric.metric_name, []).append(metric.value)
    return {
        method_id: {metric_name: sum(values) / len(values) for metric_name, values in sorted(metric_values.items())}
        for method_id, metric_values in sorted(grouped.items())
    }


def _score_vector(method_scores: Mapping[str, float], config: FreezeMethodConfig) -> tuple[float, ...]:
    primary = method_scores.get(config.primary_metric)
    if primary is None:
        raise FreezeMethodError("method_scores must include the primary metric.")
    primary_score = primary if config.primary_metric_mode == "maximize" else -primary
    return (primary_score, *(tie_breaker.score(method_scores.get(tie_breaker.metric_name)) for tie_breaker in config.tie_breakers))


def _exclusion_reason(method_id: str, config: FreezeMethodConfig) -> str | None:
    if method_id in config.excluded_method_ids:
        return "excluded_method_id"
    if any(method_id.startswith(prefix) for prefix in config.excluded_method_prefixes):
        return "excluded_method_prefix"
    return None


def _load_metric_evidence(raw_config: Mapping[str, Any], invocation: EntrypointInvocation) -> tuple[MetricRecord, ...]:
    if "metrics" in raw_config:
        return _metrics_from_rows(raw_config["metrics"], "metrics")
    if "metric_records" in raw_config:
        return _load_metric_records(_resolve_path(raw_config["metric_records"], invocation))
    if "candidate_selection_harness_result" in raw_config:
        return _load_harness_result_metrics(_resolve_path(raw_config["candidate_selection_harness_result"], invocation))
    if "candidate_selection_summary" in raw_config:
        return _load_candidate_selection_summary_metrics(_resolve_path(raw_config["candidate_selection_summary"], invocation))
    input_root = raw_config.get("input_root")
    if input_root is not None:
        root = _resolve_path(input_root, invocation)
        harness_result = root / "harness" / "harness_result.json"
        if harness_result.exists():
            return _load_harness_result_metrics(harness_result)
        summary = root / "candidate_method_selection_summary.json"
        if summary.exists():
            return _load_candidate_selection_summary_metrics(summary)
        metric_records = root / "metric_records.jsonl"
        if metric_records.exists():
            return _load_metric_records(metric_records)
    raise FreezeMethodError(
        "freeze_method requires metrics, metric_records, candidate_selection_harness_result, candidate_selection_summary, or input_root."
    )


def _load_metric_records(path: Path) -> tuple[MetricRecord, ...]:
    return tuple(_metric_from_payload(row, path) for row in _read_json_rows(path))


def _load_harness_result_metrics(path: Path) -> tuple[MetricRecord, ...]:
    rows = _read_json_rows(path)
    if len(rows) != 1:
        raise FreezeMethodError(f"Harness result path must contain one JSON object: {path}")
    root = rows[0]
    run_results = root.get("run_results")
    if not isinstance(run_results, Sequence) or isinstance(run_results, (str, bytes)):
        raise FreezeMethodError(f"{path} must contain run_results.")
    metrics: list[MetricRecord] = []
    for run_index, run_result in enumerate(run_results):
        if not isinstance(run_result, MappingABC):
            raise FreezeMethodError(f"{path}.run_results[{run_index}] must be a mapping.")
        for metric_payload in _sequence_of_mappings(run_result.get("metrics", ()), f"{path}.run_results[{run_index}].metrics"):
            metrics.append(_metric_from_payload(metric_payload, path))
    return tuple(metrics)


def _load_candidate_selection_summary_metrics(path: Path) -> tuple[MetricRecord, ...]:
    rows = _read_json_rows(path)
    if len(rows) != 1:
        raise FreezeMethodError(f"Candidate selection summary path must contain one JSON object: {path}")
    outcomes = rows[0].get("outcomes")
    if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)):
        raise FreezeMethodError(f"{path} must contain outcomes.")
    metrics: list[MetricRecord] = []
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, MappingABC):
            raise FreezeMethodError(f"{path}.outcomes[{index}] must be a mapping.")
        method_id = outcome.get("method_id")
        if not isinstance(method_id, str) or not method_id:
            raise FreezeMethodError(f"{path}.outcomes[{index}].method_id must be a non-empty string.")
        tags = {"method": method_id}
        if isinstance(outcome.get("selected_candidate_class"), str):
            tags["selected_candidate_class"] = outcome["selected_candidate_class"]
        for metric_name, field_name, unit in (
            ("method_selection_quality", "quality_score", "score"),
            ("method_selection_deadline_score", "deadline_score", "score"),
            ("method_selection_resource_efficiency", "resource_efficiency", "score"),
            ("method_selection_runtime_ms", "runtime_ms", "ms"),
            ("method_selection_interpretability", "interpretability_score", "score"),
        ):
            if field_name in outcome:
                metrics.append(
                    MetricRecord(
                        metric_name=metric_name,
                        value=_finite_float(outcome[field_name], field_name),
                        unit=unit,
                        tags=tags,
                        metadata={"candidate_selection_summary_path": path.as_posix(), "outcome_index": index},
                    )
                )
    return tuple(metrics)


def _metric_from_payload(row: Mapping[str, Any], path: Path) -> MetricRecord:
    payload = row.get("payload") if isinstance(row.get("payload"), MappingABC) else row
    if RECORD_TYPE_FIELD in payload:
        record = materialize_record(payload, expected_record_type="metric_record")
        if not isinstance(record, MetricRecord):
            raise FreezeMethodError(f"{path} contained {type(record).__name__}; expected MetricRecord.")
        return record
    try:
        return MetricRecord(**payload)
    except TypeError as exc:
        raise FreezeMethodError(f"Malformed metric record in {path}: {exc}") from exc


def _read_json_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.exists():
        raise FreezeMethodError(f"Input artifact does not exist: {path}")
    if not path.is_file():
        raise FreezeMethodError(f"Input artifact path is not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FreezeMethodError(f"Could not read input artifact {path}: {exc}") from exc
    if not text.strip():
        return ()
    try:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            parsed = json.loads(text)
            rows = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError as exc:
        raise FreezeMethodError(f"Could not parse input artifact {path}: {exc}") from exc
    parsed_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, MappingABC):
            raise FreezeMethodError(f"{path}:{index + 1} must be a mapping.")
        parsed_rows.append(row)
    return tuple(parsed_rows)


def _entrypoint_config(invocation: EntrypointInvocation) -> Mapping[str, Any]:
    values = {}
    if invocation.resolved_config is not None:
        raw_values = invocation.resolved_config.get("values")
        if isinstance(raw_values, MappingABC):
            values = raw_values
    raw_config = values.get("freeze_method", values)
    if not isinstance(raw_config, MappingABC):
        raise FreezeMethodError("freeze_method config must be a mapping.")
    return raw_config


def _freeze_config_from_mapping(raw_config: Mapping[str, Any], invocation: EntrypointInvocation) -> FreezeMethodConfig:
    resolved = invocation.resolved_config or {}
    source_uri = _string_or_none(raw_config.get("source_uri"))
    if source_uri is None and invocation.config is not None:
        source_uri = invocation.config.as_posix()
    return FreezeMethodConfig(
        method_name=raw_config.get("method_name", "RefABR frozen method"),
        version=raw_config.get("version", "1.0"),
        primary_metric=raw_config.get("primary_metric", DEFAULT_PRIMARY_METRIC),
        primary_metric_mode=raw_config.get("primary_metric_mode", "maximize"),
        tie_breakers=raw_config.get("tie_breakers", DEFAULT_TIE_BREAKERS),
        calibration_splits=raw_config.get("calibration_splits", ("calibration",)),
        excluded_splits=raw_config.get("excluded_splits", ("final",)),
        candidate_method_ids=raw_config.get("candidate_method_ids", ()),
        excluded_method_ids=raw_config.get("excluded_method_ids", DEFAULT_EXCLUDED_METHOD_IDS),
        excluded_method_prefixes=raw_config.get("excluded_method_prefixes", DEFAULT_EXCLUDED_METHOD_PREFIXES),
        method_entrypoints=raw_config.get("method_entrypoints", {}),
        default_entrypoint=raw_config.get("default_entrypoint", "ref_abr.methods:plan_schedule"),
        method_parameters=raw_config.get("method_parameters", {}),
        artifact_uri=_string_or_none(raw_config.get("artifact_uri")),
        source_uri=source_uri,
        provenance={
            "resolved_config_id": resolved.get("config_id"),
            "active_split": resolved.get("active_split"),
            **_plain_json_mapping(raw_config.get("provenance"), "provenance"),
        },
        metadata=raw_config.get("metadata", {}),
    )


def _output_root(raw_config: Mapping[str, Any], invocation: EntrypointInvocation) -> Path:
    configured = raw_config.get("output_root")
    if configured is not None:
        return _resolve_path(configured, invocation)
    if invocation.output_dir is not None:
        return invocation.output_dir
    raise FreezeMethodError("freeze_method requires output_root or --output-dir.")


def _resolve_path(value: Any, invocation: EntrypointInvocation) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FreezeMethodError("artifact paths must be non-empty strings.")
    path = Path(value)
    if path.is_absolute():
        return path
    if invocation.config is not None:
        return invocation.config.parent / path
    return path


def _metrics_from_rows(value: Any, field_name: str) -> tuple[MetricRecord, ...]:
    rows = _sequence_of_mappings(value, field_name)
    return tuple(_metric_from_payload(row, Path(f"<{field_name}>")) for row in rows)


def _coerce_metrics(metrics: Sequence[MetricRecord]) -> tuple[MetricRecord, ...]:
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence):
        raise FreezeMethodError("metrics must be a sequence of MetricRecord records.")
    coerced = tuple(metrics)
    for metric in coerced:
        if not isinstance(metric, MetricRecord):
            raise FreezeMethodError("metrics must contain MetricRecord records.")
    if not coerced:
        raise FreezeMethodError("metrics must not be empty.")
    return coerced


def _tie_breaker_tuple(values: Sequence[FreezeTieBreaker | Mapping[str, Any]]) -> tuple[FreezeTieBreaker, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FreezeMethodError("tie_breakers must be a sequence.")
    parsed: list[FreezeTieBreaker] = []
    for value in values:
        if isinstance(value, FreezeTieBreaker):
            parsed.append(value)
            continue
        if not isinstance(value, MappingABC):
            raise FreezeMethodError("tie_breakers must contain FreezeTieBreaker records or mappings.")
        parsed.append(FreezeTieBreaker(metric_name=value.get("metric_name"), mode=value.get("mode", "maximize")))
    return tuple(parsed)


def _sequence_of_mappings(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FreezeMethodError(f"{field_name} must be a sequence of mappings.")
    rows = tuple(value)
    for row in rows:
        if not isinstance(row, MappingABC):
            raise FreezeMethodError(f"{field_name} must contain mappings.")
    return rows


def _string_tuple(values: Sequence[str], field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FreezeMethodError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    if not parsed and not allow_empty:
        raise FreezeMethodError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _string_mapping(value: Mapping[str, str], field_name: str) -> dict[str, str]:
    if not isinstance(value, MappingABC):
        raise FreezeMethodError(f"{field_name} must be a mapping.")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        _require_non_empty(str(key), f"{field_name} key")
        _require_non_empty(item, f"{field_name}.{key}")
        parsed[str(key)] = item
    return {key: parsed[key] for key in sorted(parsed)}


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise FreezeMethodError(f"{field_name} must be a mapping.")
    return {str(key): _to_payload(item) for key, item in value.items()}


def _to_payload(value: Any) -> Any:
    if hasattr(value, "as_payload"):
        return value.as_payload()
    if isinstance(value, MappingABC):
        return {str(key): _to_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_payload(item) for item in value]
    if isinstance(value, list):
        return [_to_payload(item) for item in value]
    return value


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    _require_non_empty(value, "string value")
    return value


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise FreezeMethodError(f"{field_name} must be a non-empty string.")


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise FreezeMethodError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FreezeMethodError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise FreezeMethodError(f"{field_name} must be finite.")
    return parsed


def _reverse_lexical_key(value: str) -> tuple[int, ...]:
    return tuple(-ord(character) for character in value)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise FreezeMethodError(f"Failed to write frozen method manifest {path}: {exc}") from exc


__all__ = [
    "DEFAULT_EXCLUDED_METHOD_IDS",
    "DEFAULT_PRIMARY_METRIC",
    "FreezeDecision",
    "FreezeMethodConfig",
    "FreezeMethodError",
    "FreezeMethodResult",
    "FreezeTieBreaker",
    "apply_freeze_decision",
    "export_frozen_method_manifest",
    "freeze_method",
    "freeze_method_entrypoint",
]
