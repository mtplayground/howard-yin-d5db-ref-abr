"""Compute and export metric records from raw experiment artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ref_abr.accounting import CandidateResourceAccount, ComponentTimingAccount, ResourceAccountingSummary, ResourceUtilization
from ref_abr.artifacts import ArtifactProvenance, RawArtifactExportConfig, RawArtifactManifest, export_raw_artifacts
from ref_abr.config import stable_config_id
from ref_abr.deadline_lifecycle_metrics import DeadlineLifecycleMetricConfig, compute_deadline_lifecycle_metrics
from ref_abr.domain import FrameOutcome, MetricRecord, ScheduleDecision
from ref_abr.entrypoints import EntrypointInvocation, EntrypointResult
from ref_abr.lifecycle import ReferenceLifecycleEvent
from ref_abr.quality_metrics import QualityMetricConfig, compute_quality_metrics
from ref_abr.resource_stability_metrics import (
    ResourceStabilityMetricConfig,
    ViewportPredictionMetricSample,
    compute_resource_stability_viewport_metrics,
)
from ref_abr.schema import RECORD_TYPE_FIELD, SCHEMA_VERSION_FIELD, materialize_record
from ref_abr.statistical_confidence import StatisticalConfidenceConfig, paired_confidence_metric


DEFAULT_METRIC_SET: tuple[str, ...] = (
    "quality",
    "deadline_lifecycle",
    "resource_stability_viewport",
    "paired_baselines",
)
DEFAULT_GROUPING_KEYS: tuple[str, ...] = ("split",)
DEFAULT_PAIRED_METRIC_NAMES: tuple[str, ...] = (
    "deadline_hit_visible_quality",
    "full_frame_quality",
    "deadline_qoe",
    "viewport_coverage",
)


class ComputeMetricsError(ValueError):
    """Raised when metric computation inputs or configuration are invalid."""


@dataclass(frozen=True)
class ComputeMetricsConfig:
    """Controls for metric set wiring, grouping, and metric-record export."""

    run_id: str
    config_id: str | None = None
    split: str | None = None
    method_id: str | None = None
    baseline_method_id: str | None = None
    metric_set: Sequence[str] = DEFAULT_METRIC_SET
    grouping_keys: Sequence[str] = DEFAULT_GROUPING_KEYS
    paired_metric_names: Sequence[str] = DEFAULT_PAIRED_METRIC_NAMES
    useful_object_ids: Sequence[str] = ()
    confidence_level: float = 0.95
    bootstrap_iterations: int = 1000
    seed: int = 0
    output_format: str = "jsonl"
    tags: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, "run_id")
        for field_name in ("config_id", "split", "method_id", "baseline_method_id"):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_empty(value, field_name)
        object.__setattr__(self, "metric_set", _choice_tuple(self.metric_set, "metric_set", DEFAULT_METRIC_SET))
        object.__setattr__(self, "grouping_keys", _string_tuple(self.grouping_keys, "grouping_keys"))
        object.__setattr__(self, "paired_metric_names", _string_tuple(self.paired_metric_names, "paired_metric_names"))
        object.__setattr__(self, "useful_object_ids", _string_tuple(self.useful_object_ids, "useful_object_ids"))
        if self.output_format not in {"jsonl", "json"}:
            raise ComputeMetricsError("output_format must be one of: jsonl, json.")
        object.__setattr__(self, "tags", _string_mapping(self.tags, "tags"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))
        StatisticalConfidenceConfig(
            confidence_level=self.confidence_level,
            bootstrap_iterations=self.bootstrap_iterations,
            seed=self.seed,
        )

    @property
    def common_tags(self) -> dict[str, str]:
        tags = dict(self.tags)
        tags["run_id"] = self.run_id
        if self.config_id is not None:
            tags["config_id"] = self.config_id
        if self.method_id is not None:
            tags["method"] = self.method_id
        return dict(sorted(tags.items()))

    @property
    def baseline_tags(self) -> dict[str, str]:
        tags = dict(self.tags)
        tags["run_id"] = self.run_id
        if self.config_id is not None:
            tags["config_id"] = self.config_id
        if self.baseline_method_id is not None:
            tags["method"] = self.baseline_method_id
        else:
            tags["role"] = "baseline"
        return dict(sorted(tags.items()))

    @property
    def provenance_metadata(self) -> dict[str, Any]:
        return {
            **_to_payload(self.metadata),
            "compute_metrics": {
                "metric_set": list(self.metric_set),
                "grouping_keys": list(self.grouping_keys),
                "paired_metric_names": list(self.paired_metric_names),
            },
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_id": self.config_id,
            "split": self.split,
            "method_id": self.method_id,
            "baseline_method_id": self.baseline_method_id,
            "metric_set": list(self.metric_set),
            "grouping_keys": list(self.grouping_keys),
            "paired_metric_names": list(self.paired_metric_names),
            "useful_object_ids": list(self.useful_object_ids),
            "confidence_level": self.confidence_level,
            "bootstrap_iterations": self.bootstrap_iterations,
            "seed": self.seed,
            "output_format": self.output_format,
            "tags": dict(self.tags),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class ComputeMetricsInput:
    """Typed records consumed by the compute_metrics pipeline."""

    frame_outcomes: Sequence[FrameOutcome] = ()
    baseline_frame_outcomes: Sequence[FrameOutcome] = ()
    resource_records: Sequence[CandidateResourceAccount] = ()
    lifecycle_events: Sequence[ReferenceLifecycleEvent] = ()
    decisions: Sequence[ScheduleDecision] = ()
    viewport_samples: Sequence[ViewportPredictionMetricSample] = ()
    baseline_metric_records: Sequence[MetricRecord] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_outcomes", _coerce_sequence(self.frame_outcomes, FrameOutcome, "frame_outcomes"))
        object.__setattr__(self, "baseline_frame_outcomes", _coerce_sequence(self.baseline_frame_outcomes, FrameOutcome, "baseline_frame_outcomes"))
        object.__setattr__(self, "resource_records", _coerce_sequence(self.resource_records, CandidateResourceAccount, "resource_records"))
        object.__setattr__(self, "lifecycle_events", _coerce_sequence(self.lifecycle_events, ReferenceLifecycleEvent, "lifecycle_events"))
        object.__setattr__(self, "decisions", _coerce_sequence(self.decisions, ScheduleDecision, "decisions"))
        object.__setattr__(self, "viewport_samples", _coerce_sequence(self.viewport_samples, ViewportPredictionMetricSample, "viewport_samples"))
        object.__setattr__(self, "baseline_metric_records", _coerce_sequence(self.baseline_metric_records, MetricRecord, "baseline_metric_records"))


def compute_metric_records(
    records: ComputeMetricsInput,
    *,
    config: ComputeMetricsConfig,
) -> tuple[MetricRecord, ...]:
    """Compute configured metrics and paired-baseline confidence records."""

    if not isinstance(records, ComputeMetricsInput):
        raise ComputeMetricsError("records must be a ComputeMetricsInput record.")
    if not isinstance(config, ComputeMetricsConfig):
        raise ComputeMetricsError("config must be a ComputeMetricsConfig record.")

    metric_records: list[MetricRecord] = []
    common_metadata = config.provenance_metadata
    if "quality" in config.metric_set:
        metric_records.extend(
            compute_quality_metrics(
                records.frame_outcomes,
                baseline_outcomes=records.baseline_frame_outcomes,
                config=QualityMetricConfig(split=config.split, tags=config.common_tags, metadata=common_metadata),
            )
        )
    if "deadline_lifecycle" in config.metric_set:
        metric_records.extend(
            compute_deadline_lifecycle_metrics(
                frame_outcomes=records.frame_outcomes,
                resource_records=records.resource_records,
                useful_object_ids=config.useful_object_ids,
                lifecycle_events=records.lifecycle_events,
                config=DeadlineLifecycleMetricConfig(split=config.split, tags=config.common_tags, metadata=common_metadata),
            )
        )
    if "resource_stability_viewport" in config.metric_set:
        metric_records.extend(
            compute_resource_stability_viewport_metrics(
                resource_records=records.resource_records,
                frame_outcomes=records.frame_outcomes,
                decisions=records.decisions,
                viewport_samples=records.viewport_samples,
                config=ResourceStabilityMetricConfig(split=config.split, tags=config.common_tags, metadata=common_metadata),
            )
        )

    if "paired_baselines" in config.metric_set:
        baseline_records = tuple(records.baseline_metric_records)
        if not baseline_records and records.baseline_frame_outcomes:
            baseline_records = _compute_baseline_metric_records(records, config=config)
        metric_records.extend(_paired_baseline_metrics(tuple(metric_records), baseline_records, config=config))

    return tuple(_with_export_metadata(metric_records, config=config))


def export_metric_records(
    output_root: str | Path,
    metrics: Sequence[MetricRecord],
    *,
    config: ComputeMetricsConfig,
) -> RawArtifactManifest:
    """Export metric records with schema version and provenance envelopes."""

    metric_records = _coerce_sequence(metrics, MetricRecord, "metrics")
    provenance = ArtifactProvenance(
        run_id=config.run_id,
        config_id=config.config_id,
        split=config.split,
        method_id=config.method_id,
        source="compute_metrics",
        metadata=config.provenance_metadata,
    )
    return export_raw_artifacts(
        output_root,
        provenance=provenance,
        metric_records=metric_records,
        config=RawArtifactExportConfig(output_format=config.output_format),
    )


def compute_metrics_entrypoint(invocation: EntrypointInvocation) -> EntrypointResult:
    """Load configured artifacts, compute metrics, and write MetricRecord export."""

    if invocation.verb != "compute_metrics":
        raise ComputeMetricsError(f"compute_metrics handler received verb {invocation.verb!r}.")
    entrypoint_config = _entrypoint_config(invocation)
    compute_config = _compute_config_from_mapping(entrypoint_config, invocation)
    output_root = _output_root(entrypoint_config, invocation)
    input_records = _load_compute_input(entrypoint_config, invocation)

    if invocation.dry_run:
        return EntrypointResult(
            status="dry_run",
            message="compute_metrics configuration resolved; dry run skipped metric export.",
            payload={
                "config": compute_config.as_payload(),
                "input_counts": _input_counts(input_records),
                "output_root": output_root.as_posix(),
            },
        )

    metrics = compute_metric_records(input_records, config=compute_config)
    manifest = export_metric_records(output_root, metrics, config=compute_config)
    return EntrypointResult(
        status="ok",
        message=f"compute_metrics emitted {len(metrics)} MetricRecord(s).",
        payload={
            "metric_count": len(metrics),
            "metric_names": sorted({metric.metric_name for metric in metrics}),
            "manifest": manifest.as_payload(),
        },
    )


def _compute_baseline_metric_records(records: ComputeMetricsInput, *, config: ComputeMetricsConfig) -> tuple[MetricRecord, ...]:
    metadata = {
        **config.provenance_metadata,
        "baseline_source": "baseline_frame_outcomes",
    }
    baseline_metrics: list[MetricRecord] = []
    if "quality" in config.metric_set:
        baseline_metrics.extend(
            compute_quality_metrics(
                records.baseline_frame_outcomes,
                config=QualityMetricConfig(split=config.split, tags=config.baseline_tags, metadata=metadata),
            )
        )
    if "deadline_lifecycle" in config.metric_set:
        baseline_metrics.extend(
            compute_deadline_lifecycle_metrics(
                frame_outcomes=records.baseline_frame_outcomes,
                config=DeadlineLifecycleMetricConfig(split=config.split, tags=config.baseline_tags, metadata=metadata),
            )
        )
    if "resource_stability_viewport" in config.metric_set:
        baseline_metrics.extend(
            compute_resource_stability_viewport_metrics(
                frame_outcomes=records.baseline_frame_outcomes,
                config=ResourceStabilityMetricConfig(split=config.split, tags=config.baseline_tags, metadata=metadata),
            )
        )
    return tuple(baseline_metrics)


def _paired_baseline_metrics(
    treatment_metrics: tuple[MetricRecord, ...],
    baseline_metrics: tuple[MetricRecord, ...],
    *,
    config: ComputeMetricsConfig,
) -> tuple[MetricRecord, ...]:
    if not baseline_metrics:
        return ()
    paired: list[MetricRecord] = []
    for metric_name in config.paired_metric_names:
        treatment_candidates = tuple(metric for metric in treatment_metrics if metric.metric_name == metric_name)
        baseline_candidates = tuple(metric for metric in baseline_metrics if metric.metric_name == metric_name)
        if not treatment_candidates and not baseline_candidates:
            continue
        for group_key, grouped_treatment in _group_metrics(treatment_candidates, config.grouping_keys).items():
            grouped_baseline = _group_metrics(baseline_candidates, config.grouping_keys).get(group_key, ())
            confidence_config = StatisticalConfidenceConfig(
                confidence_level=config.confidence_level,
                bootstrap_iterations=config.bootstrap_iterations,
                seed=config.seed,
                split=config.split,
                tags={
                    **config.common_tags,
                    "source_metric": metric_name,
                    "paired_baseline": "true",
                    **_group_tags(group_key, config.grouping_keys),
                },
                metadata={
                    **config.provenance_metadata,
                    "paired_group": _group_payload(group_key, config.grouping_keys),
                },
            )
            paired.append(
                paired_confidence_metric(
                    grouped_treatment,
                    grouped_baseline,
                    metric_name=metric_name,
                    config=confidence_config,
                )
            )
    return tuple(paired)


def _group_metrics(metrics: Sequence[MetricRecord], grouping_keys: Sequence[str]) -> dict[tuple[str, ...], tuple[MetricRecord, ...]]:
    grouped: dict[tuple[str, ...], list[MetricRecord]] = {}
    for metric in metrics:
        key = tuple(_metric_group_value(metric, group_key) for group_key in grouping_keys)
        grouped.setdefault(key, []).append(metric)
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def _metric_group_value(metric: MetricRecord, group_key: str) -> str:
    if group_key == "metric_name":
        return metric.metric_name
    if group_key == "split":
        return metric.split or ""
    if group_key == "frame_id":
        return metric.frame_id or ""
    if group_key.startswith("metadata."):
        value = _nested_lookup(metric.metadata, group_key.removeprefix("metadata."))
        return "" if value is None else str(value)
    return metric.tags.get(group_key, "")


def _group_tags(group_key: tuple[str, ...], grouping_keys: Sequence[str]) -> dict[str, str]:
    return {
        f"group_{name.replace('.', '_')}": value
        for name, value in zip(grouping_keys, group_key, strict=True)
        if value
    }


def _group_payload(group_key: tuple[str, ...], grouping_keys: Sequence[str]) -> dict[str, str]:
    return {name: value for name, value in zip(grouping_keys, group_key, strict=True)}


def _with_export_metadata(metrics: Sequence[MetricRecord], *, config: ComputeMetricsConfig) -> tuple[MetricRecord, ...]:
    stamped: list[MetricRecord] = []
    for index, metric in enumerate(metrics):
        metadata = {
            **_to_payload(metric.metadata),
            "compute_metrics_export": {
                "sequence_index": index,
                "grouping_keys": list(config.grouping_keys),
                "metric_set": list(config.metric_set),
            },
        }
        stamped.append(
            MetricRecord(
                metric_name=metric.metric_name,
                value=metric.value,
                unit=metric.unit,
                tags=metric.tags,
                frame_id=metric.frame_id,
                split=metric.split,
                metadata=metadata,
            )
        )
    return tuple(stamped)


def _entrypoint_config(invocation: EntrypointInvocation) -> Mapping[str, Any]:
    values = {}
    if invocation.resolved_config is not None:
        raw_values = invocation.resolved_config.get("values")
        if isinstance(raw_values, MappingABC):
            values = raw_values
    raw_config = values.get("compute_metrics", values)
    if not isinstance(raw_config, MappingABC):
        raise ComputeMetricsError("compute_metrics config must be a mapping.")
    return raw_config


def _compute_config_from_mapping(raw_config: Mapping[str, Any], invocation: EntrypointInvocation) -> ComputeMetricsConfig:
    resolved = invocation.resolved_config or {}
    config_id = _string_or_none(raw_config.get("config_id")) or _string_or_none(resolved.get("config_id"))
    split = _string_or_none(raw_config.get("split")) or _string_or_none(resolved.get("active_split")) or invocation.split
    run_id = _string_or_none(raw_config.get("run_id")) or _default_run_id(raw_config, config_id=config_id, split=split)
    return ComputeMetricsConfig(
        run_id=run_id,
        config_id=config_id,
        split=split,
        method_id=_string_or_none(raw_config.get("method_id")),
        baseline_method_id=_string_or_none(raw_config.get("baseline_method_id")),
        metric_set=raw_config.get("metric_set", DEFAULT_METRIC_SET),
        grouping_keys=raw_config.get("grouping_keys", DEFAULT_GROUPING_KEYS),
        paired_metric_names=raw_config.get("paired_metric_names", DEFAULT_PAIRED_METRIC_NAMES),
        useful_object_ids=raw_config.get("useful_object_ids", ()),
        confidence_level=raw_config.get("confidence_level", 0.95),
        bootstrap_iterations=raw_config.get("bootstrap_iterations", 1000),
        seed=raw_config.get("seed", _seed_from_resolved(invocation.resolved_config)),
        output_format=raw_config.get("output_format", "jsonl"),
        tags=raw_config.get("tags", {}),
        metadata=raw_config.get("metadata", {}),
    )


def _output_root(raw_config: Mapping[str, Any], invocation: EntrypointInvocation) -> Path:
    configured = raw_config.get("output_root")
    if configured is not None:
        return _resolve_path(configured, invocation)
    if invocation.output_dir is not None:
        return invocation.output_dir
    raise ComputeMetricsError("compute_metrics requires output_root or --output-dir.")


def _load_compute_input(raw_config: Mapping[str, Any], invocation: EntrypointInvocation) -> ComputeMetricsInput:
    input_root = raw_config.get("input_root")
    return ComputeMetricsInput(
        frame_outcomes=_load_records(_configured_path(raw_config, "frame_outcomes", input_root, "frame_outcomes.jsonl", invocation), "frame_outcome", FrameOutcome),
        baseline_frame_outcomes=_load_records(
            _configured_path(raw_config, "baseline_frame_outcomes", input_root, "baseline_frame_outcomes.jsonl", invocation, required=False),
            "frame_outcome",
            FrameOutcome,
        ),
        resource_records=_load_resource_records(_configured_path(raw_config, "resource_records", input_root, "timing_records.jsonl", invocation, required=False)),
        lifecycle_events=_load_lifecycle_events(_configured_path(raw_config, "lifecycle_events", input_root, "lifecycle_events.jsonl", invocation, required=False)),
        decisions=_load_records(_configured_path(raw_config, "decisions", input_root, "decisions.jsonl", invocation, required=False), "schedule_decision", ScheduleDecision),
        viewport_samples=_load_viewport_samples(_configured_path(raw_config, "viewport_samples", input_root, "viewport_samples.jsonl", invocation, required=False)),
        baseline_metric_records=_load_records(
            _configured_path(raw_config, "baseline_metric_records", input_root, "baseline_metric_records.jsonl", invocation, required=False),
            "metric_record",
            MetricRecord,
        ),
    )


def _configured_path(
    raw_config: Mapping[str, Any],
    key: str,
    input_root: Any,
    default_name: str,
    invocation: EntrypointInvocation,
    *,
    required: bool = True,
) -> Path | None:
    if key in raw_config:
        return _resolve_path(raw_config[key], invocation)
    if input_root is not None:
        candidate = _resolve_path(input_root, invocation) / default_name
        if candidate.exists() or required:
            return candidate
    if required:
        raise ComputeMetricsError(f"compute_metrics requires {key} or input_root/{default_name}.")
    return None


def _load_records(path: Path | None, expected_record_type: str, expected_type: type) -> tuple[Any, ...]:
    if path is None:
        return ()
    loaded: list[Any] = []
    for payload in _read_artifact_payloads(path):
        record = materialize_record(payload, expected_record_type=expected_record_type)
        if not isinstance(record, expected_type):
            raise ComputeMetricsError(f"{path} contained {type(record).__name__}; expected {expected_type.__name__}.")
        loaded.append(record)
    return tuple(loaded)


def _load_lifecycle_events(path: Path | None) -> tuple[ReferenceLifecycleEvent, ...]:
    if path is None:
        return ()
    events: list[ReferenceLifecycleEvent] = []
    for payload in _read_artifact_payloads(path):
        raw = _unstamped_payload(payload, "reference_lifecycle_event")
        events.append(ReferenceLifecycleEvent(**raw))
    return tuple(events)


def _load_resource_records(path: Path | None) -> tuple[CandidateResourceAccount, ...]:
    if path is None:
        return ()
    accounts: list[CandidateResourceAccount] = []
    for payload in _read_artifact_payloads(path):
        record_type = _record_type(payload)
        raw = _unstamped_payload(payload, record_type)
        if record_type == "candidate_resource_account":
            accounts.append(_candidate_resource_account(raw))
        elif record_type == "resource_accounting_summary":
            accounts.extend(_resource_accounting_summary(raw).accounts)
        elif record_type == "component_timing_account":
            continue
        else:
            raise ComputeMetricsError(f"Unsupported resource record_type {record_type!r} in {path}.")
    return tuple(accounts)


def _load_viewport_samples(path: Path | None) -> tuple[ViewportPredictionMetricSample, ...]:
    if path is None:
        return ()
    samples: list[ViewportPredictionMetricSample] = []
    for payload in _read_artifact_payloads(path):
        raw = payload.get("payload") if isinstance(payload.get("payload"), MappingABC) else payload
        if RECORD_TYPE_FIELD in raw:
            raw = _unstamped_payload(raw, "viewport_prediction_metric_sample")
        samples.append(ViewportPredictionMetricSample(**raw))
    return tuple(samples)


def _read_artifact_payloads(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.exists():
        raise ComputeMetricsError(f"Artifact file does not exist: {path}")
    if not path.is_file():
        raise ComputeMetricsError(f"Artifact path is not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ComputeMetricsError(f"Could not read artifact file {path}: {exc}") from exc
    if not text.strip():
        return ()
    try:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            parsed = json.loads(text)
            rows = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError as exc:
        raise ComputeMetricsError(f"Could not parse artifact file {path}: {exc}") from exc
    payloads: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, MappingABC):
            raise ComputeMetricsError(f"{path}:{index + 1} must be a mapping.")
        payload = row.get("payload") if isinstance(row.get("payload"), MappingABC) else row
        payloads.append(payload)
    return tuple(payloads)


def _candidate_resource_account(raw: Mapping[str, Any]) -> CandidateResourceAccount:
    return CandidateResourceAccount(
        account_id=raw["account_id"],
        candidate_id=raw["candidate_id"],
        object_id=raw["object_id"],
        provider_id=raw["provider_id"],
        device_profile_id=raw["device_profile_id"],
        timing=_component_timing(raw["timing"]),
        utilization=_resource_utilization(raw["utilization"]),
        memory_mb=raw["memory_mb"],
        bandwidth_bps=raw.get("bandwidth_bps"),
        transfer_bytes=raw["transfer_bytes"],
        metadata=raw.get("metadata", {}),
    )


def _resource_accounting_summary(raw: Mapping[str, Any]) -> ResourceAccountingSummary:
    accounts = tuple(_candidate_resource_account(account) for account in _sequence_of_mappings(raw["accounts"], "accounts"))
    return ResourceAccountingSummary(
        summary_id=raw["summary_id"],
        accounts=accounts,
        total_timing=_component_timing(raw["total_timing"]),
        peak_memory_mb=raw["peak_memory_mb"],
        total_transfer_bytes=raw["total_transfer_bytes"],
        metadata=raw.get("metadata", {}),
    )


def _component_timing(raw: Mapping[str, Any]) -> ComponentTimingAccount:
    return ComponentTimingAccount(
        server_generation_ms=raw["server_generation_ms"],
        queue_ms=raw["queue_ms"],
        transfer_ms=raw["transfer_ms"],
        decode_ms=raw["decode_ms"],
        restore_ms=raw["restore_ms"],
        render_ms=raw["render_ms"],
    )


def _resource_utilization(raw: Mapping[str, Any]) -> ResourceUtilization:
    return ResourceUtilization(
        server_generation=raw["server_generation"],
        queue=raw["queue"],
        transfer_time=raw["transfer_time"],
        decode=raw["decode"],
        restore=raw["restore"],
        render=raw["render"],
        memory=raw["memory"],
        bandwidth=raw.get("bandwidth"),
    )


def _unstamped_payload(payload: Mapping[str, Any], expected_record_type: str) -> Mapping[str, Any]:
    record_type = _record_type(payload)
    if record_type != expected_record_type:
        raise ComputeMetricsError(f"Expected record_type {expected_record_type!r}; got {record_type!r}.")
    return {
        key: value
        for key, value in payload.items()
        if key not in {SCHEMA_VERSION_FIELD, RECORD_TYPE_FIELD}
    }


def _record_type(payload: Mapping[str, Any], *, default: str | None = None) -> str:
    record_type = payload.get(RECORD_TYPE_FIELD, default)
    if not isinstance(record_type, str) or not record_type:
        raise ComputeMetricsError("record_type must be present for artifact payloads.")
    return record_type


def _sequence_of_mappings(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComputeMetricsError(f"{field_name} must be a sequence of mappings.")
    rows = tuple(value)
    for row in rows:
        if not isinstance(row, MappingABC):
            raise ComputeMetricsError(f"{field_name} must contain mappings.")
    return rows


def _resolve_path(value: Any, invocation: EntrypointInvocation) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ComputeMetricsError("artifact paths must be non-empty strings.")
    path = Path(value)
    if path.is_absolute():
        return path
    if invocation.config is not None:
        return invocation.config.parent / path
    return path


def _default_run_id(raw_config: Mapping[str, Any], *, config_id: str | None, split: str | None) -> str:
    payload = {
        "config_id": config_id,
        "split": split,
        "metric_set": raw_config.get("metric_set", DEFAULT_METRIC_SET),
        "grouping_keys": raw_config.get("grouping_keys", DEFAULT_GROUPING_KEYS),
    }
    return f"metrics-run-{stable_config_id(payload)}"


def _seed_from_resolved(resolved_config: Mapping[str, Any] | None) -> int:
    if not isinstance(resolved_config, MappingABC):
        return 0
    seed = resolved_config.get("seed")
    if isinstance(seed, MappingABC):
        value = seed.get("value")
        if isinstance(value, int):
            return value
    return 0


def _input_counts(records: ComputeMetricsInput) -> dict[str, int]:
    return {
        "frame_outcomes": len(records.frame_outcomes),
        "baseline_frame_outcomes": len(records.baseline_frame_outcomes),
        "resource_records": len(records.resource_records),
        "lifecycle_events": len(records.lifecycle_events),
        "decisions": len(records.decisions),
        "viewport_samples": len(records.viewport_samples),
        "baseline_metric_records": len(records.baseline_metric_records),
    }


def _choice_tuple(values: Sequence[str], field_name: str, allowed: Sequence[str]) -> tuple[str, ...]:
    parsed = _string_tuple(values, field_name)
    allowed_values = set(allowed)
    unknown = sorted(value for value in parsed if value not in allowed_values)
    if unknown:
        raise ComputeMetricsError(f"{field_name} contains unknown value(s): {', '.join(unknown)}.")
    return parsed


def _coerce_sequence(values: Sequence[Any], expected_type: type, field_name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ComputeMetricsError(f"{field_name} must be a sequence.")
    records = tuple(values)
    for record in records:
        if not isinstance(record, expected_type):
            raise ComputeMetricsError(f"{field_name} must contain {expected_type.__name__} records.")
    return records


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    _require_non_empty(value, "string value")
    return value


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ComputeMetricsError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    return tuple(parsed)


def _string_mapping(value: Mapping[str, str], field_name: str) -> Mapping[str, str]:
    if not isinstance(value, MappingABC):
        raise ComputeMetricsError(f"{field_name} must be a mapping.")
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
        raise ComputeMetricsError(f"{field_name} must be a mapping.")
    return {str(key): _to_payload(item) for key, item in value.items()}


def _nested_lookup(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    cursor: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(cursor, MappingABC):
            return None
        cursor = cursor.get(part)
    return cursor


def _to_payload(value: Any) -> Any:
    if hasattr(value, "as_payload"):
        return value.as_payload()
    if isinstance(value, MappingABC):
        return {str(key): _to_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_payload(item) for item in value]
    if isinstance(value, list):
        return [_to_payload(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ComputeMetricsError(f"{field_name} must be a non-empty string.")


__all__ = [
    "ComputeMetricsConfig",
    "ComputeMetricsError",
    "ComputeMetricsInput",
    "DEFAULT_GROUPING_KEYS",
    "DEFAULT_METRIC_SET",
    "DEFAULT_PAIRED_METRIC_NAMES",
    "compute_metric_records",
    "compute_metrics_entrypoint",
    "export_metric_records",
]
