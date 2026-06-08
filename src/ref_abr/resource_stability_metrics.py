"""Resource-cost, control-stability, and viewport-prediction metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.accounting import CandidateResourceAccount, ResourceAccountingSummary
from ref_abr.config import stable_config_id
from ref_abr.domain import FrameOutcome, MetricRecord, ScheduleDecision


NO_RESOURCE_RECORDS = "no_resource_records_zero_cost"
NO_CONTROL_TRANSITIONS = "no_control_transitions_zero_rate"
NO_VIEWPORT_SAMPLES = "no_viewport_samples_zero_metric"
NO_RECOVERY_EVENTS = "no_recovery_events_zero_time"


class ResourceStabilityMetricError(ValueError):
    """Raised when resource/stability/viewport metric inputs are invalid."""


@dataclass(frozen=True)
class ResourceStabilityMetricConfig:
    """Shared labels for resource, control-stability, and viewport metric records."""

    split: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split is not None:
            _require_non_empty(self.split, "split")
        object.__setattr__(self, "tags", _string_mapping(self.tags, "tags"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "tags": dict(self.tags),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class ViewportPredictionMetricSample:
    """Per-frame viewport prediction measurements for metric export."""

    frame_id: str
    angular_error_deg: float
    coverage: float
    overfetch_ratio: float
    translational_error_m: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.frame_id, "frame_id")
        object.__setattr__(self, "angular_error_deg", _non_negative_float(self.angular_error_deg, "angular_error_deg"))
        object.__setattr__(self, "coverage", _unit_interval(self.coverage, "coverage"))
        object.__setattr__(self, "overfetch_ratio", _non_negative_float(self.overfetch_ratio, "overfetch_ratio"))
        if self.translational_error_m is not None:
            object.__setattr__(self, "translational_error_m", _non_negative_float(self.translational_error_m, "translational_error_m"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "angular_error_deg": self.angular_error_deg,
            "coverage": self.coverage,
            "overfetch_ratio": self.overfetch_ratio,
            "translational_error_m": self.translational_error_m,
            "metadata": _to_payload(self.metadata),
        }


def compute_resource_stability_viewport_metrics(
    *,
    resource_records: ResourceAccountingSummary | Sequence[CandidateResourceAccount] = (),
    frame_outcomes: Sequence[FrameOutcome] = (),
    decisions: Sequence[ScheduleDecision] = (),
    viewport_samples: Sequence[ViewportPredictionMetricSample] = (),
    config: ResourceStabilityMetricConfig | None = None,
) -> tuple[MetricRecord, ...]:
    """Compute issue-28 resource, stability, and viewport metric groups."""

    metric_config = config or ResourceStabilityMetricConfig()
    if not isinstance(metric_config, ResourceStabilityMetricConfig):
        raise ResourceStabilityMetricError("config must be a ResourceStabilityMetricConfig record.")
    return (
        *resource_cost_metrics(resource_records, config=metric_config),
        *control_stability_metrics(frame_outcomes, decisions=decisions, config=metric_config),
        *viewport_prediction_metrics(viewport_samples, config=metric_config),
    )


def resource_cost_metrics(
    resource_records: ResourceAccountingSummary | Sequence[CandidateResourceAccount],
    *,
    config: ResourceStabilityMetricConfig | None = None,
) -> tuple[MetricRecord, ...]:
    """Return byte, timing, and peak-memory resource cost metrics."""

    metric_config = config or ResourceStabilityMetricConfig()
    if not isinstance(metric_config, ResourceStabilityMetricConfig):
        raise ResourceStabilityMetricError("config must be a ResourceStabilityMetricConfig record.")
    accounts = _coerce_resource_accounts(resource_records)
    total_bytes = sum(account.transfer_bytes for account in accounts)
    total_timing_ms = sum(account.timing.total_ms for account in accounts)
    peak_memory_mb = max((account.memory_mb for account in accounts), default=0.0)
    null_rule = NO_RESOURCE_RECORDS if not accounts else "not_null"
    metadata = {"account_count": len(accounts), "null_rule": null_rule}
    return (
        _metric_record("resource_bytes_cost", total_bytes, "bytes", metric_config, metadata=metadata),
        _metric_record("resource_timing_cost_ms", total_timing_ms, "ms", metric_config, metadata=metadata),
        _metric_record("resource_memory_cost_mb", peak_memory_mb, "MB", metric_config, metadata=metadata),
    )


def control_stability_metrics(
    frame_outcomes: Sequence[FrameOutcome],
    *,
    decisions: Sequence[ScheduleDecision] = (),
    config: ResourceStabilityMetricConfig | None = None,
) -> tuple[MetricRecord, ...]:
    """Return quality variance, decision switch rate, and recovery time metrics."""

    metric_config = config or ResourceStabilityMetricConfig()
    if not isinstance(metric_config, ResourceStabilityMetricConfig):
        raise ResourceStabilityMetricError("config must be a ResourceStabilityMetricConfig record.")
    outcomes = _coerce_outcomes(frame_outcomes, "frame_outcomes")
    schedule_decisions = _coerce_decisions(decisions)
    quality_values = [_outcome_quality(outcome) for outcome in outcomes if _outcome_quality(outcome) is not None]
    variance = _population_variance(quality_values)
    switch_rate, switch_metadata = _switch_rate(schedule_decisions)
    recovery_time_ms, recovery_metadata = _recovery_time_ms(outcomes)
    return (
        _metric_record(
            "control_quality_variance",
            variance,
            "score^2",
            metric_config,
            metadata={"sample_count": len(quality_values), "null_rule": "not_null" if len(quality_values) >= 2 else "fewer_than_two_quality_samples_zero_variance"},
        ),
        _metric_record("control_switch_rate", switch_rate, "ratio", metric_config, metadata=switch_metadata),
        _metric_record("control_recovery_time_ms", recovery_time_ms, "ms", metric_config, metadata=recovery_metadata),
    )


def viewport_prediction_metrics(
    viewport_samples: Sequence[ViewportPredictionMetricSample],
    *,
    config: ResourceStabilityMetricConfig | None = None,
) -> tuple[MetricRecord, ...]:
    """Return mean viewport angular error, coverage, and overfetch metrics."""

    metric_config = config or ResourceStabilityMetricConfig()
    if not isinstance(metric_config, ResourceStabilityMetricConfig):
        raise ResourceStabilityMetricError("config must be a ResourceStabilityMetricConfig record.")
    samples = _coerce_viewport_samples(viewport_samples)
    null_rule = NO_VIEWPORT_SAMPLES if not samples else "not_null"
    metadata = {"sample_count": len(samples), "null_rule": null_rule}
    mean_error = _mean([sample.angular_error_deg for sample in samples])
    mean_coverage = _mean([sample.coverage for sample in samples])
    mean_overfetch = _mean([sample.overfetch_ratio for sample in samples])
    return (
        _metric_record("viewport_error_deg", mean_error, "degrees", metric_config, metadata=metadata),
        _metric_record("viewport_coverage", mean_coverage, "ratio", metric_config, metadata=metadata),
        _metric_record("viewport_overfetch_ratio", mean_overfetch, "ratio", metric_config, metadata=metadata),
    )


def _switch_rate(decisions: tuple[ScheduleDecision, ...]) -> tuple[float, dict[str, Any]]:
    if len(decisions) < 2:
        return 0.0, {"transition_count": 0, "switch_count": 0, "null_rule": NO_CONTROL_TRANSITIONS}
    switch_count = 0
    previous = set(decisions[0].selected_object_ids)
    for decision in decisions[1:]:
        current = set(decision.selected_object_ids)
        if current != previous:
            switch_count += 1
        previous = current
    transitions = len(decisions) - 1
    return switch_count / transitions, {"transition_count": transitions, "switch_count": switch_count, "null_rule": "not_null"}


def _recovery_time_ms(outcomes: tuple[FrameOutcome, ...]) -> tuple[float, dict[str, Any]]:
    recovery_times: list[float] = []
    pending_bad_time: int | None = None
    for outcome in outcomes:
        if _is_unstable_outcome(outcome):
            if pending_bad_time is None:
                pending_bad_time = outcome.scheduled_time_ms
            continue
        if pending_bad_time is not None:
            recovery_times.append(float(outcome.scheduled_time_ms - pending_bad_time))
            pending_bad_time = None
    if not recovery_times:
        return 0.0, {"recovery_count": 0, "null_rule": NO_RECOVERY_EVENTS}
    return _mean(recovery_times), {"recovery_count": len(recovery_times), "null_rule": "not_null"}


def _is_unstable_outcome(outcome: FrameOutcome) -> bool:
    payload = outcome.as_payload()
    metadata = payload.get("metadata")
    frame_evaluation = metadata.get("frame_evaluation") if isinstance(metadata, MappingABC) else None
    frame_evaluation = frame_evaluation if isinstance(frame_evaluation, MappingABC) else {}
    deadline_hit = bool(payload["deadline_hit"]) if payload["deadline_hit"] is not None else False
    return (
        payload["rendered_time_ms"] is None
        or not deadline_hit
        or bool(frame_evaluation.get("freeze", False))
        or bool(frame_evaluation.get("missing", False))
    )


def _outcome_quality(outcome: FrameOutcome) -> float | None:
    payload = outcome.as_payload()
    metadata = payload.get("metadata")
    frame_evaluation = metadata.get("frame_evaluation") if isinstance(metadata, MappingABC) else None
    if isinstance(frame_evaluation, MappingABC) and "quality_score" in frame_evaluation:
        return _unit_interval(frame_evaluation["quality_score"], "frame_evaluation.quality_score")
    if outcome.quality_score is not None:
        return _unit_interval(outcome.quality_score, "quality_score")
    return None


def _population_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _metric_record(
    metric_name: str,
    value: float | int,
    unit: str,
    config: ResourceStabilityMetricConfig,
    *,
    tags: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MetricRecord:
    metric_tags = {
        **dict(config.tags),
        **_string_mapping(tags or {}, "tags"),
    }
    metric_metadata = {
        **config.metadata,
        **_plain_json_mapping(metadata, "metadata"),
        "metric_id": _metric_id(metric_name, float(value), metric_tags),
    }
    return MetricRecord(
        metric_name=metric_name,
        value=_finite_float(value, "value"),
        unit=unit,
        tags=metric_tags,
        split=config.split,
        metadata=metric_metadata,
    )


def _metric_id(metric_name: str, value: float, tags: Mapping[str, str]) -> str:
    payload = {
        "metric_name": metric_name,
        "value": value,
        "tags": dict(tags),
    }
    return f"resource-stability-metric-{stable_config_id(payload)}"


def _coerce_resource_accounts(
    records: ResourceAccountingSummary | Sequence[CandidateResourceAccount],
) -> tuple[CandidateResourceAccount, ...]:
    if isinstance(records, ResourceAccountingSummary):
        return records.accounts
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ResourceStabilityMetricError("resource_records must be a ResourceAccountingSummary or sequence of CandidateResourceAccount records.")
    accounts = tuple(records)
    for account in accounts:
        if not isinstance(account, CandidateResourceAccount):
            raise ResourceStabilityMetricError("resource_records must contain CandidateResourceAccount records.")
    return accounts


def _coerce_outcomes(outcomes: Sequence[FrameOutcome], field_name: str) -> tuple[FrameOutcome, ...]:
    if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Sequence):
        raise ResourceStabilityMetricError(f"{field_name} must be a sequence of FrameOutcome records.")
    coerced = tuple(outcomes)
    for outcome in coerced:
        if not isinstance(outcome, FrameOutcome):
            raise ResourceStabilityMetricError(f"{field_name} must contain FrameOutcome records.")
    return coerced


def _coerce_decisions(decisions: Sequence[ScheduleDecision]) -> tuple[ScheduleDecision, ...]:
    if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
        raise ResourceStabilityMetricError("decisions must be a sequence of ScheduleDecision records.")
    coerced = tuple(decisions)
    for decision in coerced:
        if not isinstance(decision, ScheduleDecision):
            raise ResourceStabilityMetricError("decisions must contain ScheduleDecision records.")
    return coerced


def _coerce_viewport_samples(samples: Sequence[ViewportPredictionMetricSample]) -> tuple[ViewportPredictionMetricSample, ...]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise ResourceStabilityMetricError("viewport_samples must be a sequence of ViewportPredictionMetricSample records.")
    coerced = tuple(samples)
    for sample in coerced:
        if not isinstance(sample, ViewportPredictionMetricSample):
            raise ResourceStabilityMetricError("viewport_samples must contain ViewportPredictionMetricSample records.")
    return coerced


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise ResourceStabilityMetricError(f"{field_name} must be a mapping.")
    return {str(key): _to_payload(item) for key, item in value.items()}


def _string_mapping(value: Mapping[str, str], field_name: str) -> Mapping[str, str]:
    if not isinstance(value, MappingABC):
        raise ResourceStabilityMetricError(f"{field_name} must be a mapping.")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        _require_non_empty(str(key), f"{field_name} key")
        _require_non_empty(item, f"{field_name}.{key}")
        parsed[str(key)] = item
    return {key: parsed[key] for key in sorted(parsed)}


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


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ResourceStabilityMetricError(f"{field_name} must be a non-empty string.")


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ResourceStabilityMetricError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ResourceStabilityMetricError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise ResourceStabilityMetricError(f"{field_name} must be finite.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0:
        raise ResourceStabilityMetricError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if not 0.0 <= parsed <= 1.0:
        raise ResourceStabilityMetricError(f"{field_name} must be between 0 and 1.")
    return parsed


__all__ = [
    "NO_CONTROL_TRANSITIONS",
    "NO_RECOVERY_EVENTS",
    "NO_RESOURCE_RECORDS",
    "NO_VIEWPORT_SAMPLES",
    "ResourceStabilityMetricConfig",
    "ResourceStabilityMetricError",
    "ViewportPredictionMetricSample",
    "compute_resource_stability_viewport_metrics",
    "control_stability_metrics",
    "resource_cost_metrics",
    "viewport_prediction_metrics",
]
