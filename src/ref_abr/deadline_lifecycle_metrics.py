"""Deadline-QoE, useful-resource, and reference-lifecycle metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.accounting import CandidateResourceAccount, ResourceAccountingSummary
from ref_abr.config import stable_config_id
from ref_abr.domain import FrameOutcome, MetricRecord
from ref_abr.lifecycle import DropReason, LifecycleAction, LifecyclePhase, ReferenceLifecycleEvent
from ref_abr.quality_metrics import MISSING_RENDER_POLICY


NO_EVENTS_ZERO_RATE = "no_lifecycle_events_zero_rate"
NO_RESOURCE_ZERO_RATIO = "no_resource_records_zero_ratio"


class DeadlineLifecycleMetricError(ValueError):
    """Raised when deadline/lifecycle metric inputs are invalid."""


@dataclass(frozen=True)
class DeadlineQoeWeights:
    """Explicit weights and penalties for the deadline-QoE composite."""

    visible_quality: float = 0.45
    full_quality: float = 0.35
    deadline_hit: float = 0.20
    freeze_penalty: float = 0.15
    missing_penalty: float = 0.10

    def __post_init__(self) -> None:
        for field_name in ("visible_quality", "full_quality", "deadline_hit"):
            object.__setattr__(self, field_name, _non_negative_float(getattr(self, field_name), field_name))
        for field_name in ("freeze_penalty", "missing_penalty"):
            object.__setattr__(self, field_name, _unit_interval(getattr(self, field_name), field_name))

    @property
    def total_positive_weight(self) -> float:
        return self.visible_quality + self.full_quality + self.deadline_hit

    def as_payload(self) -> dict[str, float]:
        return {
            "visible_quality": self.visible_quality,
            "full_quality": self.full_quality,
            "deadline_hit": self.deadline_hit,
            "freeze_penalty": self.freeze_penalty,
            "missing_penalty": self.missing_penalty,
        }


@dataclass(frozen=True)
class DeadlineLifecycleMetricConfig:
    """Shared labels and weights for deadline/lifecycle metric records."""

    split: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    deadline_qoe_weights: DeadlineQoeWeights = field(default_factory=DeadlineQoeWeights)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split is not None:
            _require_non_empty(self.split, "split")
        object.__setattr__(self, "tags", _string_mapping(self.tags, "tags"))
        if not isinstance(self.deadline_qoe_weights, DeadlineQoeWeights):
            raise DeadlineLifecycleMetricError("deadline_qoe_weights must be a DeadlineQoeWeights record.")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "tags": dict(self.tags),
            "deadline_qoe_weights": self.deadline_qoe_weights.as_payload(),
            "metadata": _to_payload(self.metadata),
        }


def compute_deadline_lifecycle_metrics(
    *,
    frame_outcomes: Sequence[FrameOutcome] = (),
    resource_records: ResourceAccountingSummary | Sequence[CandidateResourceAccount] = (),
    useful_object_ids: Sequence[str] = (),
    lifecycle_events: Sequence[ReferenceLifecycleEvent] = (),
    config: DeadlineLifecycleMetricConfig | None = None,
) -> tuple[MetricRecord, ...]:
    """Compute issue-27 composite metrics with explicit zero-denominator rules."""

    metric_config = config or DeadlineLifecycleMetricConfig()
    if not isinstance(metric_config, DeadlineLifecycleMetricConfig):
        raise DeadlineLifecycleMetricError("config must be a DeadlineLifecycleMetricConfig record.")
    metrics: list[MetricRecord] = []
    for outcome in _coerce_outcomes(frame_outcomes, "frame_outcomes"):
        metrics.append(deadline_qoe(outcome, config=metric_config))
    metrics.append(useful_resource_ratio(resource_records, useful_object_ids=useful_object_ids, config=metric_config))
    metrics.extend(reference_lifecycle_rates(lifecycle_events, config=metric_config))
    return tuple(metrics)


def deadline_qoe(outcome: FrameOutcome, *, config: DeadlineLifecycleMetricConfig | None = None) -> MetricRecord:
    """Return a weighted QoE score that combines quality, deadline hit, and penalties."""

    metric_config = config or DeadlineLifecycleMetricConfig()
    if not isinstance(metric_config, DeadlineLifecycleMetricConfig):
        raise DeadlineLifecycleMetricError("config must be a DeadlineLifecycleMetricConfig record.")
    if not isinstance(outcome, FrameOutcome):
        raise DeadlineLifecycleMetricError("outcome must be a FrameOutcome record.")
    observed = _observed_frame(outcome)
    weights = metric_config.deadline_qoe_weights
    if not observed.rendered:
        value = 0.0
        null_rule = MISSING_RENDER_POLICY
    elif weights.total_positive_weight == 0.0:
        value = 0.0
        null_rule = "zero_when_all_deadline_qoe_weights_are_zero"
    else:
        weighted = (
            weights.visible_quality * observed.visible_quality
            + weights.full_quality * observed.full_quality
            + weights.deadline_hit * (1.0 if observed.deadline_hit else 0.0)
        ) / weights.total_positive_weight
        penalties = (
            weights.freeze_penalty * (1.0 if observed.freeze else 0.0)
            + weights.missing_penalty * observed.missing_ratio
        )
        value = _clamp01(weighted - penalties)
        null_rule = "not_null"
    return _metric_record(
        "deadline_qoe",
        value,
        "score",
        metric_config,
        frame_id=outcome.frame_id,
        metadata={
            **observed.as_metadata(),
            "weights": weights.as_payload(),
            "null_rule": null_rule,
            "definition": "weighted visible/full quality plus deadline-hit credit minus freeze and missing penalties",
        },
    )


def useful_resource_ratio(
    resource_records: ResourceAccountingSummary | Sequence[CandidateResourceAccount],
    *,
    useful_object_ids: Sequence[str],
    config: DeadlineLifecycleMetricConfig | None = None,
) -> MetricRecord:
    """Return useful transferred bytes divided by total transferred bytes."""

    metric_config = config or DeadlineLifecycleMetricConfig()
    if not isinstance(metric_config, DeadlineLifecycleMetricConfig):
        raise DeadlineLifecycleMetricError("config must be a DeadlineLifecycleMetricConfig record.")
    accounts = _coerce_resource_accounts(resource_records)
    useful_ids = set(_string_tuple(useful_object_ids, "useful_object_ids"))
    total_bytes = sum(account.transfer_bytes for account in accounts)
    useful_bytes = sum(account.transfer_bytes for account in accounts if account.object_id in useful_ids)
    if total_bytes == 0:
        value = 0.0
        null_rule = NO_RESOURCE_ZERO_RATIO
    else:
        value = useful_bytes / total_bytes
        null_rule = "not_null"
    return _metric_record(
        "useful_resource_ratio",
        _clamp01(value),
        "ratio",
        metric_config,
        metadata={
            "definition": "transfer bytes for useful objects divided by total transfer bytes",
            "useful_object_ids": sorted(useful_ids),
            "useful_bytes": useful_bytes,
            "total_bytes": total_bytes,
            "account_count": len(accounts),
            "null_rule": null_rule,
        },
    )


def reference_lifecycle_rates(
    lifecycle_events: Sequence[ReferenceLifecycleEvent],
    *,
    config: DeadlineLifecycleMetricConfig | None = None,
) -> tuple[MetricRecord, ...]:
    """Return late/stale/off-view/expired/useful rates over distinct references."""

    metric_config = config or DeadlineLifecycleMetricConfig()
    if not isinstance(metric_config, DeadlineLifecycleMetricConfig):
        raise DeadlineLifecycleMetricError("config must be a DeadlineLifecycleMetricConfig record.")
    events = _coerce_lifecycle_events(lifecycle_events)
    states = _classify_lifecycle_events(events)
    denominator = len(states)
    metrics: list[MetricRecord] = []
    for category in ("late", "stale", "off_view", "expired", "useful"):
        numerator = sum(1 for state in states.values() if getattr(state, category))
        value = 0.0 if denominator == 0 else numerator / denominator
        metrics.append(
            _metric_record(
                f"reference_lifecycle_{category}_rate",
                value,
                "ratio",
                metric_config,
                tags={"lifecycle_category": category},
                metadata={
                    "definition": f"share of distinct references classified as {category}",
                    "numerator": numerator,
                    "denominator": denominator,
                    "event_count": len(events),
                    "null_rule": NO_EVENTS_ZERO_RATE if denominator == 0 else "not_null",
                },
            )
        )
    return tuple(metrics)


@dataclass
class _LifecycleReferenceState:
    late: bool = False
    stale: bool = False
    off_view: bool = False
    expired: bool = False
    useful: bool = False


@dataclass(frozen=True)
class _ObservedFrame:
    visible_quality: float
    full_quality: float
    deadline_hit: bool
    rendered: bool
    freeze: bool
    missing_ratio: float
    source: str

    def as_metadata(self) -> dict[str, Any]:
        return {
            "visible_quality": self.visible_quality,
            "full_quality": self.full_quality,
            "deadline_hit": self.deadline_hit,
            "rendered": self.rendered,
            "freeze": self.freeze,
            "missing_ratio": self.missing_ratio,
            "quality_source": self.source,
        }


def _observed_frame(outcome: FrameOutcome) -> _ObservedFrame:
    payload = outcome.as_payload()
    metadata = payload.get("metadata")
    frame_evaluation = metadata.get("frame_evaluation") if isinstance(metadata, MappingABC) else None
    frame_evaluation = frame_evaluation if isinstance(frame_evaluation, MappingABC) else {}
    rendered = payload["rendered_time_ms"] is not None
    deadline_hit = bool(payload["deadline_hit"]) if payload["deadline_hit"] is not None else False
    visible_quality = _quality_value(frame_evaluation, "visible_quality", outcome.quality_score, "visible_quality")
    full_quality = _quality_value(frame_evaluation, "full_quality", outcome.quality_score, "full_quality")
    missing_ratio = _missing_ratio(frame_evaluation, outcome)
    freeze = bool(frame_evaluation.get("freeze", False))
    source = "frame_evaluation" if frame_evaluation else "quality_score"
    return _ObservedFrame(
        visible_quality=visible_quality,
        full_quality=full_quality,
        deadline_hit=deadline_hit,
        rendered=rendered,
        freeze=freeze,
        missing_ratio=missing_ratio,
        source=source,
    )


def _quality_value(
    frame_evaluation: Mapping[str, Any],
    field_name: str,
    fallback_quality_score: float | None,
    metric_field_name: str,
) -> float:
    if field_name in frame_evaluation:
        return _unit_interval(frame_evaluation[field_name], metric_field_name)
    if fallback_quality_score is not None:
        return _unit_interval(fallback_quality_score, "quality_score")
    return 0.0


def _missing_ratio(frame_evaluation: Mapping[str, Any], outcome: FrameOutcome) -> float:
    required = frame_evaluation.get("required_object_ids")
    if isinstance(required, list) and required:
        return _clamp01(len(outcome.missing_object_ids) / len(required))
    delivered_count = len(outcome.delivered_object_ids)
    missing_count = len(outcome.missing_object_ids)
    denominator = delivered_count + missing_count
    if denominator == 0:
        return 0.0
    return _clamp01(missing_count / denominator)


def _classify_lifecycle_events(events: tuple[ReferenceLifecycleEvent, ...]) -> dict[str, _LifecycleReferenceState]:
    states: dict[str, _LifecycleReferenceState] = {}
    for event in events:
        state = states.setdefault(event.reference_id, _LifecycleReferenceState())
        payload = event.as_payload()
        action = payload["action"]
        to_phase = payload["to_phase"]
        drop_reason = payload["drop_reason"]
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), MappingABC) else {}
        late_by_time = payload["deadline_ms"] is not None and payload["event_time_ms"] > payload["deadline_ms"]
        state.late = state.late or late_by_time or drop_reason == DropReason.DEADLINE_MISSED.value or _metadata_flag(metadata, "late")
        state.stale = state.stale or action == LifecycleAction.STALE.value or to_phase == LifecyclePhase.STALE.value or drop_reason == DropReason.STALE.value
        state.expired = state.expired or action == LifecycleAction.EXPIRE.value or to_phase == LifecyclePhase.EXPIRED.value or drop_reason in {
            DropReason.EXPIRED.value,
            DropReason.DEADLINE_MISSED.value,
        }
        state.useful = state.useful or action == LifecycleAction.USE.value or to_phase == LifecyclePhase.USED.value or _metadata_flag(metadata, "useful")
        state.off_view = state.off_view or _metadata_flag(metadata, "off_view")
    return states


def _metadata_flag(metadata: Mapping[str, Any], flag_name: str) -> bool:
    if bool(metadata.get(flag_name)):
        return True
    for key in ("lifecycle", "viewport", "classification"):
        nested = metadata.get(key)
        if isinstance(nested, MappingABC) and bool(nested.get(flag_name)):
            return True
    return False


def _metric_record(
    metric_name: str,
    value: float,
    unit: str,
    config: DeadlineLifecycleMetricConfig,
    *,
    frame_id: str | None = None,
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
        "metric_id": _metric_id(metric_name, value, frame_id, metric_tags),
    }
    return MetricRecord(
        metric_name=metric_name,
        value=_finite_float(value, "value"),
        unit=unit,
        tags=metric_tags,
        frame_id=frame_id,
        split=config.split,
        metadata=metric_metadata,
    )


def _metric_id(metric_name: str, value: float, frame_id: str | None, tags: Mapping[str, str]) -> str:
    payload = {
        "metric_name": metric_name,
        "value": value,
        "frame_id": frame_id,
        "tags": dict(tags),
    }
    return f"deadline-lifecycle-metric-{stable_config_id(payload)}"


def _coerce_outcomes(outcomes: Sequence[FrameOutcome], field_name: str) -> tuple[FrameOutcome, ...]:
    if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Sequence):
        raise DeadlineLifecycleMetricError(f"{field_name} must be a sequence of FrameOutcome records.")
    coerced = tuple(outcomes)
    for outcome in coerced:
        if not isinstance(outcome, FrameOutcome):
            raise DeadlineLifecycleMetricError(f"{field_name} must contain FrameOutcome records.")
    return coerced


def _coerce_resource_accounts(
    records: ResourceAccountingSummary | Sequence[CandidateResourceAccount],
) -> tuple[CandidateResourceAccount, ...]:
    if isinstance(records, ResourceAccountingSummary):
        return records.accounts
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise DeadlineLifecycleMetricError("resource_records must be a ResourceAccountingSummary or sequence of CandidateResourceAccount records.")
    accounts = tuple(records)
    for account in accounts:
        if not isinstance(account, CandidateResourceAccount):
            raise DeadlineLifecycleMetricError("resource_records must contain CandidateResourceAccount records.")
    return accounts


def _coerce_lifecycle_events(events: Sequence[ReferenceLifecycleEvent]) -> tuple[ReferenceLifecycleEvent, ...]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise DeadlineLifecycleMetricError("lifecycle_events must be a sequence of ReferenceLifecycleEvent records.")
    coerced = tuple(events)
    for event in coerced:
        if not isinstance(event, ReferenceLifecycleEvent):
            raise DeadlineLifecycleMetricError("lifecycle_events must contain ReferenceLifecycleEvent records.")
    return coerced


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DeadlineLifecycleMetricError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    return tuple(parsed)


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise DeadlineLifecycleMetricError(f"{field_name} must be a mapping.")
    return {str(key): _to_payload(item) for key, item in value.items()}


def _string_mapping(value: Mapping[str, str], field_name: str) -> Mapping[str, str]:
    if not isinstance(value, MappingABC):
        raise DeadlineLifecycleMetricError(f"{field_name} must be a mapping.")
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
        raise DeadlineLifecycleMetricError(f"{field_name} must be a non-empty string.")


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise DeadlineLifecycleMetricError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DeadlineLifecycleMetricError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise DeadlineLifecycleMetricError(f"{field_name} must be finite.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0:
        raise DeadlineLifecycleMetricError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if not 0.0 <= parsed <= 1.0:
        raise DeadlineLifecycleMetricError(f"{field_name} must be between 0 and 1.")
    return parsed


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = [
    "DeadlineLifecycleMetricConfig",
    "DeadlineLifecycleMetricError",
    "DeadlineQoeWeights",
    "NO_EVENTS_ZERO_RATE",
    "NO_RESOURCE_ZERO_RATIO",
    "compute_deadline_lifecycle_metrics",
    "deadline_qoe",
    "reference_lifecycle_rates",
    "useful_resource_ratio",
]
