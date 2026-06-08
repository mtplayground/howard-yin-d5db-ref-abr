from __future__ import annotations

import pytest

from ref_abr.domain import MetricRecord
from ref_abr.statistical_confidence import (
    PairedMetricSample,
    StatisticalConfidenceConfig,
    StatisticalConfidenceError,
    bootstrap_paired_confidence_interval,
    paired_confidence_interval,
    paired_confidence_metric,
    paired_metric_samples,
)


def test_paired_confidence_interval_matches_tuples_and_bootstraps_deltas() -> None:
    treatment = (
        _metric("deadline_qoe", 0.8, "frame-1"),
        _metric("deadline_qoe", 0.7, "frame-2"),
        _metric("deadline_qoe", 0.9, "frame-3"),
    )
    baseline = (
        _metric("deadline_qoe", 0.6, "frame-1"),
        _metric("deadline_qoe", 0.65, "frame-2"),
        _metric("deadline_qoe", 0.7, "frame-3"),
    )

    interval = paired_confidence_interval(
        treatment,
        baseline,
        metric_name="deadline_qoe",
        config=StatisticalConfidenceConfig(seed=7, bootstrap_iterations=200),
    )

    assert interval.sample_count == 3
    assert interval.mean_treatment == pytest.approx(0.8)
    assert interval.mean_baseline == pytest.approx(0.65)
    assert interval.mean_delta == pytest.approx(0.15)
    assert interval.ci_lower <= interval.mean_delta <= interval.ci_upper
    assert interval.validation.promotion_blocked is False


def test_paired_confidence_metric_blocks_promotion_for_missing_baselines() -> None:
    treatment = (
        _metric("full_frame_quality", 0.8, "frame-1"),
        _metric("full_frame_quality", 0.7, "frame-2"),
    )
    baseline = (_metric("full_frame_quality", 0.6, "frame-1"),)

    metric = paired_confidence_metric(
        treatment,
        baseline,
        metric_name="full_frame_quality",
        config=StatisticalConfidenceConfig(seed=3, bootstrap_iterations=100),
    )

    confidence = metric.metadata["confidence_interval"]
    validation = confidence["validation"]
    assert metric.metric_name == "paired_mean_delta"
    assert metric.value == pytest.approx(0.2)
    assert metric.metadata["promotion_blocked"] is True
    assert "missing_paired_baselines" in metric.metadata["blocking_reasons"]
    assert validation["missing_baseline_pair_ids"] == ("frame-2",)


def test_paired_metric_samples_reports_duplicates_and_missing_treatment() -> None:
    treatment = (
        _metric("viewport_coverage", 0.8, "frame-1"),
        _metric("viewport_coverage", 0.9, "frame-1"),
    )
    baseline = (
        _metric("viewport_coverage", 0.7, "frame-1"),
        _metric("viewport_coverage", 0.6, "frame-2"),
    )

    samples, validation = paired_metric_samples(treatment, baseline, metric_name="viewport_coverage")

    assert len(samples) == 1
    assert validation.duplicate_treatment_pair_ids == ("frame-1",)
    assert validation.missing_treatment_pair_ids == ("frame-2",)
    assert validation.promotion_blocked is True
    assert "duplicate_treatment_pairs" in validation.blocking_reasons


def test_bootstrap_paired_confidence_interval_is_deterministic() -> None:
    samples = (
        PairedMetricSample(pair_id="a", treatment_value=0.8, baseline_value=0.6, unit="score"),
        PairedMetricSample(pair_id="b", treatment_value=0.7, baseline_value=0.65, unit="score"),
        PairedMetricSample(pair_id="c", treatment_value=0.9, baseline_value=0.7, unit="score"),
    )
    config = StatisticalConfidenceConfig(seed=11, bootstrap_iterations=250)

    first = bootstrap_paired_confidence_interval(samples, metric_name="deadline_qoe", config=config)
    second = bootstrap_paired_confidence_interval(samples, metric_name="deadline_qoe", config=config)

    assert first.as_payload() == second.as_payload()
    assert first.bootstrap_std_error > 0.0


def test_paired_confidence_rejects_unpairable_or_mismatched_metrics() -> None:
    with pytest.raises(StatisticalConfidenceError, match="must have frame_id"):
        paired_confidence_interval(
            (MetricRecord(metric_name="m", value=1.0, unit="score"),),
            (),
            metric_name="m",
        )

    with pytest.raises(StatisticalConfidenceError, match="unit mismatch"):
        paired_confidence_interval(
            (_metric("m", 1.0, "frame-1", unit="score"),),
            (_metric("m", 0.5, "frame-1", unit="ratio"),),
            metric_name="m",
        )


def _metric(metric_name: str, value: float, frame_id: str, *, unit: str = "score") -> MetricRecord:
    return MetricRecord(
        metric_name=metric_name,
        value=value,
        unit=unit,
        frame_id=frame_id,
        tags={"method": "candidate"},
    )
