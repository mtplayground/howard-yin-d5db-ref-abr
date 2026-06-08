"""Quality metrics derived from frame outcomes."""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.config import stable_config_id
from ref_abr.domain import FrameOutcome, MetricRecord


MISSING_RENDER_POLICY = "missing_render_zero_quality"


class QualityMetricError(ValueError):
    """Raised when quality metric inputs are invalid."""


@dataclass(frozen=True)
class QualityMetricConfig:
    """Tags and labels shared by emitted quality metric records."""

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


def compute_quality_metrics(
    outcomes: Sequence[FrameOutcome],
    *,
    baseline_outcomes: Sequence[FrameOutcome] = (),
    config: QualityMetricConfig | None = None,
) -> tuple[MetricRecord, ...]:
    """Compute frame quality metrics with explicit missing-render handling.

    Missing renders use zero quality for `deadline_hit_visible_quality` and
    `full_frame_quality`. `deadline_hit_visible_quality` is also zero when the
    frame misses its deadline. `restoration_gain` is emitted for matched
    frame-id pairs and is restored full-frame quality minus baseline full-frame
    quality, after applying the same missing-render zero-quality rule.
    """

    metric_config = config or QualityMetricConfig()
    if not isinstance(metric_config, QualityMetricConfig):
        raise QualityMetricError("config must be a QualityMetricConfig record.")
    frame_outcomes = _coerce_outcomes(outcomes, "outcomes")
    baseline_by_frame_id = _outcome_by_frame_id(baseline_outcomes, "baseline_outcomes")
    metrics: list[MetricRecord] = []
    for outcome in frame_outcomes:
        metrics.append(_deadline_hit_visible_quality(outcome, metric_config))
        metrics.append(_full_frame_quality(outcome, metric_config, role="restored"))
        baseline = baseline_by_frame_id.get(outcome.frame_id)
        if baseline is not None:
            metrics.append(_restoration_gain(outcome, baseline, metric_config))
    return tuple(metrics)


def deadline_hit_visible_quality(outcome: FrameOutcome, *, config: QualityMetricConfig | None = None) -> MetricRecord:
    """Return visible quality gated by both render presence and deadline hit."""

    metric_config = config or QualityMetricConfig()
    if not isinstance(metric_config, QualityMetricConfig):
        raise QualityMetricError("config must be a QualityMetricConfig record.")
    if not isinstance(outcome, FrameOutcome):
        raise QualityMetricError("outcome must be a FrameOutcome record.")
    return _deadline_hit_visible_quality(outcome, metric_config)


def full_frame_quality(outcome: FrameOutcome, *, config: QualityMetricConfig | None = None) -> MetricRecord:
    """Return full-frame quality, with missing renders scored as zero."""

    metric_config = config or QualityMetricConfig()
    if not isinstance(metric_config, QualityMetricConfig):
        raise QualityMetricError("config must be a QualityMetricConfig record.")
    if not isinstance(outcome, FrameOutcome):
        raise QualityMetricError("outcome must be a FrameOutcome record.")
    return _full_frame_quality(outcome, metric_config, role="restored")


def restoration_gain(
    restored_outcome: FrameOutcome,
    baseline_outcome: FrameOutcome,
    *,
    config: QualityMetricConfig | None = None,
) -> MetricRecord:
    """Return paired restored-minus-baseline full-frame quality gain."""

    metric_config = config or QualityMetricConfig()
    if not isinstance(metric_config, QualityMetricConfig):
        raise QualityMetricError("config must be a QualityMetricConfig record.")
    if not isinstance(restored_outcome, FrameOutcome):
        raise QualityMetricError("restored_outcome must be a FrameOutcome record.")
    if not isinstance(baseline_outcome, FrameOutcome):
        raise QualityMetricError("baseline_outcome must be a FrameOutcome record.")
    if restored_outcome.frame_id != baseline_outcome.frame_id:
        raise QualityMetricError("restored_outcome and baseline_outcome frame_id values must match.")
    return _restoration_gain(restored_outcome, baseline_outcome, metric_config)


def _deadline_hit_visible_quality(outcome: FrameOutcome, config: QualityMetricConfig) -> MetricRecord:
    observed = _observed_quality(outcome)
    value = observed.visible_quality if observed.rendered and observed.deadline_hit else 0.0
    return _metric_record(
        "deadline_hit_visible_quality",
        value,
        outcome,
        config,
        metadata={
            **observed.as_metadata(),
            "definition": "visible quality when rendered by deadline; zero for missing render or deadline miss",
        },
    )


def _full_frame_quality(outcome: FrameOutcome, config: QualityMetricConfig, *, role: str) -> MetricRecord:
    observed = _observed_quality(outcome)
    value = observed.full_quality if observed.rendered else 0.0
    return _metric_record(
        "full_frame_quality",
        value,
        outcome,
        config,
        tags={"role": role},
        metadata={
            **observed.as_metadata(),
            "definition": "full-frame quality with zero for missing render",
        },
    )


def _restoration_gain(restored_outcome: FrameOutcome, baseline_outcome: FrameOutcome, config: QualityMetricConfig) -> MetricRecord:
    restored = _observed_quality(restored_outcome)
    baseline = _observed_quality(baseline_outcome)
    restored_value = restored.full_quality if restored.rendered else 0.0
    baseline_value = baseline.full_quality if baseline.rendered else 0.0
    value = restored_value - baseline_value
    metadata = {
        "definition": "paired restored full-frame quality minus baseline full-frame quality",
        "missing_render_policy": MISSING_RENDER_POLICY,
        "restored": restored.as_metadata(),
        "baseline": baseline.as_metadata(),
        "baseline_frame_id": baseline_outcome.frame_id,
        "restored_value": restored_value,
        "baseline_value": baseline_value,
    }
    return _metric_record(
        "restoration_gain",
        value,
        restored_outcome,
        config,
        tags={"paired": "restored_minus_baseline"},
        metadata=metadata,
    )


@dataclass(frozen=True)
class _ObservedQuality:
    visible_quality: float
    full_quality: float
    rendered: bool
    deadline_hit: bool
    source: str

    def as_metadata(self) -> dict[str, Any]:
        return {
            "visible_quality": self.visible_quality,
            "full_quality": self.full_quality,
            "rendered": self.rendered,
            "deadline_hit": self.deadline_hit,
            "quality_source": self.source,
            "missing_render_policy": MISSING_RENDER_POLICY,
        }


def _observed_quality(outcome: FrameOutcome) -> _ObservedQuality:
    if not isinstance(outcome, FrameOutcome):
        raise QualityMetricError("outcome must be a FrameOutcome record.")
    payload = outcome.as_payload()
    rendered = payload["rendered_time_ms"] is not None
    deadline_hit = bool(payload["deadline_hit"]) if payload["deadline_hit"] is not None else False
    frame_evaluation = _frame_evaluation(payload)
    visible_quality = _quality_value(frame_evaluation, "visible_quality", outcome.quality_score, "visible_quality")
    full_quality_value = _quality_value(frame_evaluation, "full_quality", outcome.quality_score, "full_quality")
    source = "frame_evaluation" if frame_evaluation else "quality_score"
    return _ObservedQuality(
        visible_quality=visible_quality,
        full_quality=full_quality_value,
        rendered=rendered,
        deadline_hit=deadline_hit,
        source=source,
    )


def _frame_evaluation(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, MappingABC):
        return {}
    frame_evaluation = metadata.get("frame_evaluation")
    if not isinstance(frame_evaluation, MappingABC):
        return {}
    return frame_evaluation


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


def _metric_record(
    metric_name: str,
    value: float,
    outcome: FrameOutcome,
    config: QualityMetricConfig,
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
        "metric_id": _metric_id(metric_name, value, outcome, metric_tags),
    }
    return MetricRecord(
        metric_name=metric_name,
        value=_finite_float(value, "value"),
        unit="score",
        tags=metric_tags,
        frame_id=outcome.frame_id,
        split=config.split,
        metadata=metric_metadata,
    )


def _metric_id(metric_name: str, value: float, outcome: FrameOutcome, tags: Mapping[str, str]) -> str:
    payload = {
        "metric_name": metric_name,
        "value": value,
        "frame_id": outcome.frame_id,
        "tags": dict(tags),
    }
    return f"quality-metric-{stable_config_id(payload)}"


def _coerce_outcomes(outcomes: Sequence[FrameOutcome], field_name: str) -> tuple[FrameOutcome, ...]:
    if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Sequence):
        raise QualityMetricError(f"{field_name} must be a sequence of FrameOutcome records.")
    coerced = tuple(outcomes)
    for outcome in coerced:
        if not isinstance(outcome, FrameOutcome):
            raise QualityMetricError(f"{field_name} must contain FrameOutcome records.")
    return coerced


def _outcome_by_frame_id(outcomes: Sequence[FrameOutcome], field_name: str) -> dict[str, FrameOutcome]:
    coerced = _coerce_outcomes(outcomes, field_name)
    indexed: dict[str, FrameOutcome] = {}
    for outcome in coerced:
        if outcome.frame_id in indexed:
            raise QualityMetricError(f"{field_name} contains duplicate frame_id {outcome.frame_id!r}.")
        indexed[outcome.frame_id] = outcome
    return indexed


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise QualityMetricError(f"{field_name} must be a mapping.")
    return {str(key): _to_payload(item) for key, item in value.items()}


def _string_mapping(value: Mapping[str, str], field_name: str) -> Mapping[str, str]:
    if not isinstance(value, MappingABC):
        raise QualityMetricError(f"{field_name} must be a mapping.")
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
        raise QualityMetricError(f"{field_name} must be a non-empty string.")


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise QualityMetricError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise QualityMetricError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise QualityMetricError(f"{field_name} must be finite.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if not 0.0 <= parsed <= 1.0:
        raise QualityMetricError(f"{field_name} must be between 0 and 1.")
    return parsed


__all__ = [
    "MISSING_RENDER_POLICY",
    "QualityMetricConfig",
    "QualityMetricError",
    "compute_quality_metrics",
    "deadline_hit_visible_quality",
    "full_frame_quality",
    "restoration_gain",
]
