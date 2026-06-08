"""Paired confidence intervals and missing-data validation."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.config import stable_config_id
from ref_abr.domain import MetricRecord


class StatisticalConfidenceError(ValueError):
    """Raised when confidence or paired-validation inputs are invalid."""


@dataclass(frozen=True)
class StatisticalConfidenceConfig:
    """Controls for paired bootstrap confidence and promotion validation."""

    confidence_level: float = 0.95
    bootstrap_iterations: int = 1000
    seed: int = 0
    min_pairs: int = 2
    split: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence_level", _open_unit_interval(self.confidence_level, "confidence_level"))
        object.__setattr__(self, "bootstrap_iterations", _positive_int(self.bootstrap_iterations, "bootstrap_iterations"))
        object.__setattr__(self, "seed", _non_negative_int(self.seed, "seed"))
        object.__setattr__(self, "min_pairs", _positive_int(self.min_pairs, "min_pairs"))
        if self.split is not None:
            _require_non_empty(self.split, "split")
        object.__setattr__(self, "tags", _string_mapping(self.tags, "tags"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "confidence_level": self.confidence_level,
            "bootstrap_iterations": self.bootstrap_iterations,
            "seed": self.seed,
            "min_pairs": self.min_pairs,
            "split": self.split,
            "tags": dict(self.tags),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class PairedMetricSample:
    """One matched treatment/baseline metric tuple."""

    pair_id: str
    treatment_value: float
    baseline_value: float
    unit: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.pair_id, "pair_id")
        object.__setattr__(self, "treatment_value", _finite_float(self.treatment_value, "treatment_value"))
        object.__setattr__(self, "baseline_value", _finite_float(self.baseline_value, "baseline_value"))
        _require_non_empty(self.unit, "unit")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def delta(self) -> float:
        return self.treatment_value - self.baseline_value

    def as_payload(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "treatment_value": self.treatment_value,
            "baseline_value": self.baseline_value,
            "delta": self.delta,
            "unit": self.unit,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class PairedBaselineValidation:
    """Validation result for matched treatment/baseline metric tuples."""

    metric_name: str
    matched_pair_ids: tuple[str, ...]
    missing_baseline_pair_ids: tuple[str, ...] = ()
    missing_treatment_pair_ids: tuple[str, ...] = ()
    duplicate_treatment_pair_ids: tuple[str, ...] = ()
    duplicate_baseline_pair_ids: tuple[str, ...] = ()
    min_pairs: int = 2

    def __post_init__(self) -> None:
        _require_non_empty(self.metric_name, "metric_name")
        object.__setattr__(self, "matched_pair_ids", _string_tuple(self.matched_pair_ids, "matched_pair_ids"))
        object.__setattr__(self, "missing_baseline_pair_ids", _string_tuple(self.missing_baseline_pair_ids, "missing_baseline_pair_ids"))
        object.__setattr__(self, "missing_treatment_pair_ids", _string_tuple(self.missing_treatment_pair_ids, "missing_treatment_pair_ids"))
        object.__setattr__(self, "duplicate_treatment_pair_ids", _string_tuple(self.duplicate_treatment_pair_ids, "duplicate_treatment_pair_ids"))
        object.__setattr__(self, "duplicate_baseline_pair_ids", _string_tuple(self.duplicate_baseline_pair_ids, "duplicate_baseline_pair_ids"))
        object.__setattr__(self, "min_pairs", _positive_int(self.min_pairs, "min_pairs"))

    @property
    def promotion_blocked(self) -> bool:
        return bool(self.blocking_reasons)

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.missing_baseline_pair_ids:
            reasons.append("missing_paired_baselines")
        if self.missing_treatment_pair_ids:
            reasons.append("missing_paired_treatments")
        if self.duplicate_treatment_pair_ids:
            reasons.append("duplicate_treatment_pairs")
        if self.duplicate_baseline_pair_ids:
            reasons.append("duplicate_baseline_pairs")
        if len(self.matched_pair_ids) < self.min_pairs:
            reasons.append("insufficient_matched_pairs")
        return tuple(reasons)

    def as_payload(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "matched_pair_ids": list(self.matched_pair_ids),
            "missing_baseline_pair_ids": list(self.missing_baseline_pair_ids),
            "missing_treatment_pair_ids": list(self.missing_treatment_pair_ids),
            "duplicate_treatment_pair_ids": list(self.duplicate_treatment_pair_ids),
            "duplicate_baseline_pair_ids": list(self.duplicate_baseline_pair_ids),
            "min_pairs": self.min_pairs,
            "promotion_blocked": self.promotion_blocked,
            "blocking_reasons": list(self.blocking_reasons),
        }


@dataclass(frozen=True)
class PairedConfidenceInterval:
    """Paired bootstrap confidence interval over treatment-minus-baseline deltas."""

    metric_name: str
    unit: str
    sample_count: int
    mean_treatment: float
    mean_baseline: float
    mean_delta: float
    ci_lower: float
    ci_upper: float
    bootstrap_std_error: float
    validation: PairedBaselineValidation
    config: StatisticalConfidenceConfig

    def __post_init__(self) -> None:
        _require_non_empty(self.metric_name, "metric_name")
        _require_non_empty(self.unit, "unit")
        object.__setattr__(self, "sample_count", _non_negative_int(self.sample_count, "sample_count"))
        for field_name in ("mean_treatment", "mean_baseline", "mean_delta", "ci_lower", "ci_upper", "bootstrap_std_error"):
            object.__setattr__(self, field_name, _finite_float(getattr(self, field_name), field_name))
        if not isinstance(self.validation, PairedBaselineValidation):
            raise StatisticalConfidenceError("validation must be a PairedBaselineValidation record.")
        if not isinstance(self.config, StatisticalConfidenceConfig):
            raise StatisticalConfidenceError("config must be a StatisticalConfidenceConfig record.")

    def as_payload(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "unit": self.unit,
            "sample_count": self.sample_count,
            "mean_treatment": self.mean_treatment,
            "mean_baseline": self.mean_baseline,
            "mean_delta": self.mean_delta,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "bootstrap_std_error": self.bootstrap_std_error,
            "validation": self.validation.as_payload(),
            "config": self.config.as_payload(),
        }


def paired_confidence_metric(
    treatment_metrics: Sequence[MetricRecord],
    baseline_metrics: Sequence[MetricRecord],
    *,
    metric_name: str,
    config: StatisticalConfidenceConfig | None = None,
) -> MetricRecord:
    """Return a metric record containing paired bootstrap confidence metadata."""

    interval = paired_confidence_interval(
        treatment_metrics,
        baseline_metrics,
        metric_name=metric_name,
        config=config,
    )
    tags = {
        **dict(interval.config.tags),
        "source_metric": metric_name,
        "confidence_method": "paired_bootstrap",
    }
    metadata = {
        **interval.config.metadata,
        "confidence_interval": interval.as_payload(),
        "promotion_blocked": interval.validation.promotion_blocked,
        "blocking_reasons": list(interval.validation.blocking_reasons),
        "metric_id": _metric_id("paired_mean_delta", interval.mean_delta, tags),
    }
    return MetricRecord(
        metric_name="paired_mean_delta",
        value=interval.mean_delta,
        unit=interval.unit,
        tags=tags,
        split=interval.config.split,
        metadata=metadata,
    )


def paired_confidence_interval(
    treatment_metrics: Sequence[MetricRecord],
    baseline_metrics: Sequence[MetricRecord],
    *,
    metric_name: str,
    config: StatisticalConfidenceConfig | None = None,
) -> PairedConfidenceInterval:
    """Build matched pairs and compute a deterministic paired bootstrap CI."""

    _require_non_empty(metric_name, "metric_name")
    confidence_config = config or StatisticalConfidenceConfig()
    if not isinstance(confidence_config, StatisticalConfidenceConfig):
        raise StatisticalConfidenceError("config must be a StatisticalConfidenceConfig record.")
    samples, validation = paired_metric_samples(
        treatment_metrics,
        baseline_metrics,
        metric_name=metric_name,
        min_pairs=confidence_config.min_pairs,
    )
    return bootstrap_paired_confidence_interval(samples, metric_name=metric_name, validation=validation, config=confidence_config)


def paired_metric_samples(
    treatment_metrics: Sequence[MetricRecord],
    baseline_metrics: Sequence[MetricRecord],
    *,
    metric_name: str,
    min_pairs: int = 2,
) -> tuple[tuple[PairedMetricSample, ...], PairedBaselineValidation]:
    """Match treatment and baseline metric records by metric name and pair id."""

    _require_non_empty(metric_name, "metric_name")
    minimum_pairs = _positive_int(min_pairs, "min_pairs")
    treatment_records = _metric_index(_coerce_metrics(treatment_metrics, "treatment_metrics"), metric_name, "treatment")
    baseline_records = _metric_index(_coerce_metrics(baseline_metrics, "baseline_metrics"), metric_name, "baseline")
    treatment_ids = set(treatment_records.records)
    baseline_ids = set(baseline_records.records)
    matched_ids = tuple(sorted(treatment_ids & baseline_ids))
    samples: list[PairedMetricSample] = []
    for pair_id in matched_ids:
        treatment = treatment_records.records[pair_id]
        baseline = baseline_records.records[pair_id]
        if treatment.unit != baseline.unit:
            raise StatisticalConfidenceError(f"Paired metric unit mismatch for pair_id {pair_id!r}.")
        samples.append(
            PairedMetricSample(
                pair_id=pair_id,
                treatment_value=float(treatment.value),
                baseline_value=float(baseline.value),
                unit=treatment.unit,
                metadata={
                    "treatment": treatment.as_payload(),
                    "baseline": baseline.as_payload(),
                },
            )
        )
    validation = PairedBaselineValidation(
        metric_name=metric_name,
        matched_pair_ids=matched_ids,
        missing_baseline_pair_ids=tuple(sorted(treatment_ids - baseline_ids)),
        missing_treatment_pair_ids=tuple(sorted(baseline_ids - treatment_ids)),
        duplicate_treatment_pair_ids=treatment_records.duplicate_pair_ids,
        duplicate_baseline_pair_ids=baseline_records.duplicate_pair_ids,
        min_pairs=minimum_pairs,
    )
    return tuple(samples), validation


def bootstrap_paired_confidence_interval(
    samples: Sequence[PairedMetricSample],
    *,
    metric_name: str,
    validation: PairedBaselineValidation | None = None,
    config: StatisticalConfidenceConfig | None = None,
) -> PairedConfidenceInterval:
    """Compute a paired bootstrap confidence interval over matched deltas."""

    _require_non_empty(metric_name, "metric_name")
    confidence_config = config or StatisticalConfidenceConfig()
    if not isinstance(confidence_config, StatisticalConfidenceConfig):
        raise StatisticalConfidenceError("config must be a StatisticalConfidenceConfig record.")
    paired_samples = _coerce_samples(samples)
    resolved_validation = validation or PairedBaselineValidation(
        metric_name=metric_name,
        matched_pair_ids=tuple(sample.pair_id for sample in paired_samples),
        min_pairs=confidence_config.min_pairs,
    )
    if not isinstance(resolved_validation, PairedBaselineValidation):
        raise StatisticalConfidenceError("validation must be a PairedBaselineValidation record.")
    unit = paired_samples[0].unit if paired_samples else "delta"
    deltas = [sample.delta for sample in paired_samples]
    treatments = [sample.treatment_value for sample in paired_samples]
    baselines = [sample.baseline_value for sample in paired_samples]
    mean_delta = _mean(deltas)
    bootstrap_means = _bootstrap_means(deltas, confidence_config)
    lower, upper = _percentile_interval(bootstrap_means, confidence_config.confidence_level, fallback=mean_delta)
    return PairedConfidenceInterval(
        metric_name=metric_name,
        unit=unit,
        sample_count=len(paired_samples),
        mean_treatment=_mean(treatments),
        mean_baseline=_mean(baselines),
        mean_delta=mean_delta,
        ci_lower=lower,
        ci_upper=upper,
        bootstrap_std_error=_sample_stddev(bootstrap_means),
        validation=resolved_validation,
        config=confidence_config,
    )


@dataclass(frozen=True)
class _MetricIndex:
    records: Mapping[str, MetricRecord]
    duplicate_pair_ids: tuple[str, ...]


def _metric_index(metrics: tuple[MetricRecord, ...], metric_name: str, role: str) -> _MetricIndex:
    records: dict[str, MetricRecord] = {}
    duplicates: set[str] = set()
    for metric in metrics:
        if metric.metric_name != metric_name:
            continue
        pair_id = _pair_id(metric)
        if pair_id in records:
            duplicates.add(pair_id)
            continue
        records[pair_id] = metric
    return _MetricIndex(records=records, duplicate_pair_ids=tuple(sorted(duplicates)))


def _pair_id(metric: MetricRecord) -> str:
    if metric.frame_id is not None:
        return metric.frame_id
    metadata_pair_id = metric.metadata.get("pair_id") if isinstance(metric.metadata, MappingABC) else None
    if isinstance(metadata_pair_id, str) and metadata_pair_id:
        return metadata_pair_id
    tag_pair_id = metric.tags.get("pair_id") if isinstance(metric.tags, MappingABC) else None
    if isinstance(tag_pair_id, str) and tag_pair_id:
        return tag_pair_id
    raise StatisticalConfidenceError(f"Metric {metric.metric_name!r} must have frame_id, metadata.pair_id, or tags.pair_id for pairing.")


def _bootstrap_means(deltas: Sequence[float], config: StatisticalConfidenceConfig) -> tuple[float, ...]:
    if not deltas:
        return ()
    rng = random.Random(config.seed)
    sample_count = len(deltas)
    means: list[float] = []
    for _ in range(config.bootstrap_iterations):
        resampled = [deltas[rng.randrange(sample_count)] for _ in range(sample_count)]
        means.append(_mean(resampled))
    return tuple(means)


def _percentile_interval(values: Sequence[float], confidence_level: float, *, fallback: float) -> tuple[float, float]:
    if not values:
        return fallback, fallback
    ordered = sorted(values)
    alpha = 1.0 - confidence_level
    lower = _percentile(ordered, alpha / 2.0)
    upper = _percentile(ordered, 1.0 - alpha / 2.0)
    return lower, upper


def _percentile(ordered_values: Sequence[float], quantile: float) -> float:
    if not ordered_values:
        return 0.0
    if len(ordered_values) == 1:
        return ordered_values[0]
    position = quantile * (len(ordered_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered_values[lower_index]
    fraction = position - lower_index
    return ordered_values[lower_index] * (1.0 - fraction) + ordered_values[upper_index] * fraction


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _sample_stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _metric_id(metric_name: str, value: float, tags: Mapping[str, str]) -> str:
    payload = {
        "metric_name": metric_name,
        "value": value,
        "tags": dict(tags),
    }
    return f"stat-confidence-metric-{stable_config_id(payload)}"


def _coerce_metrics(metrics: Sequence[MetricRecord], field_name: str) -> tuple[MetricRecord, ...]:
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence):
        raise StatisticalConfidenceError(f"{field_name} must be a sequence of MetricRecord records.")
    coerced = tuple(metrics)
    for metric in coerced:
        if not isinstance(metric, MetricRecord):
            raise StatisticalConfidenceError(f"{field_name} must contain MetricRecord records.")
    return coerced


def _coerce_samples(samples: Sequence[PairedMetricSample]) -> tuple[PairedMetricSample, ...]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise StatisticalConfidenceError("samples must be a sequence of PairedMetricSample records.")
    coerced = tuple(samples)
    for sample in coerced:
        if not isinstance(sample, PairedMetricSample):
            raise StatisticalConfidenceError("samples must contain PairedMetricSample records.")
    return coerced


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise StatisticalConfidenceError(f"{field_name} must be a mapping.")
    return {str(key): _to_payload(item) for key, item in value.items()}


def _string_mapping(value: Mapping[str, str], field_name: str) -> Mapping[str, str]:
    if not isinstance(value, MappingABC):
        raise StatisticalConfidenceError(f"{field_name} must be a mapping.")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        _require_non_empty(str(key), f"{field_name} key")
        _require_non_empty(item, f"{field_name}.{key}")
        parsed[str(key)] = item
    return {key: parsed[key] for key in sorted(parsed)}


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise StatisticalConfidenceError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    return tuple(parsed)


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
        raise StatisticalConfidenceError(f"{field_name} must be a non-empty string.")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise StatisticalConfidenceError(f"{field_name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StatisticalConfidenceError(f"{field_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise StatisticalConfidenceError(f"{field_name} must be a positive integer.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise StatisticalConfidenceError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StatisticalConfidenceError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise StatisticalConfidenceError(f"{field_name} must be a non-negative integer.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise StatisticalConfidenceError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise StatisticalConfidenceError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise StatisticalConfidenceError(f"{field_name} must be finite.")
    return parsed


def _open_unit_interval(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if not 0.0 < parsed < 1.0:
        raise StatisticalConfidenceError(f"{field_name} must be between 0 and 1, exclusive.")
    return parsed


__all__ = [
    "PairedBaselineValidation",
    "PairedConfidenceInterval",
    "PairedMetricSample",
    "StatisticalConfidenceConfig",
    "StatisticalConfidenceError",
    "bootstrap_paired_confidence_interval",
    "paired_confidence_interval",
    "paired_confidence_metric",
    "paired_metric_samples",
]
