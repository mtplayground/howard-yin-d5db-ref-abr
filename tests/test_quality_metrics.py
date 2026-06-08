from __future__ import annotations

import pytest

from ref_abr.domain import FrameOutcome
from ref_abr.quality_metrics import (
    MISSING_RENDER_POLICY,
    QualityMetricConfig,
    QualityMetricError,
    compute_quality_metrics,
    deadline_hit_visible_quality,
    full_frame_quality,
    restoration_gain,
)


def test_quality_metrics_use_frame_evaluation_values() -> None:
    outcome = _outcome("frame-1", rendered_time_ms=18, deadline_hit=True, visible_quality=0.75, full_quality=0.60)
    config = QualityMetricConfig(split="final", tags={"method": "ref"}, metadata={"run_id": "run-1"})

    metrics = compute_quality_metrics((outcome,), config=config)

    by_name = {metric.metric_name: metric for metric in metrics}
    assert by_name["deadline_hit_visible_quality"].value == pytest.approx(0.75)
    assert by_name["full_frame_quality"].value == pytest.approx(0.60)
    assert by_name["deadline_hit_visible_quality"].split == "final"
    assert by_name["deadline_hit_visible_quality"].tags["method"] == "ref"
    assert by_name["deadline_hit_visible_quality"].metadata["missing_render_policy"] == MISSING_RENDER_POLICY
    assert by_name["full_frame_quality"].metadata["definition"] == "full-frame quality with zero for missing render"


def test_deadline_hit_visible_quality_is_zero_for_deadline_miss() -> None:
    outcome = _outcome("frame-2", rendered_time_ms=30, deadline_hit=False, visible_quality=0.8, full_quality=0.7)

    visible_metric = deadline_hit_visible_quality(outcome)
    full_metric = full_frame_quality(outcome)

    assert visible_metric.value == pytest.approx(0.0)
    assert visible_metric.metadata["deadline_hit"] is False
    assert full_metric.value == pytest.approx(0.7)


def test_missing_render_quality_handling_is_zero_and_documented() -> None:
    outcome = _outcome("frame-3", rendered_time_ms=None, deadline_hit=False, visible_quality=0.9, full_quality=0.85)

    metrics = compute_quality_metrics((outcome,))

    assert [metric.value for metric in metrics] == [0.0, 0.0]
    for metric in metrics:
        assert metric.metadata["rendered"] is False
        assert metric.metadata["missing_render_policy"] == MISSING_RENDER_POLICY


def test_restoration_gain_is_paired_by_frame_id() -> None:
    restored = _outcome("frame-4", rendered_time_ms=20, deadline_hit=True, visible_quality=0.8, full_quality=0.75)
    baseline = _outcome("frame-4", rendered_time_ms=19, deadline_hit=True, visible_quality=0.5, full_quality=0.40)

    gain = restoration_gain(restored, baseline)

    assert gain.metric_name == "restoration_gain"
    assert gain.value == pytest.approx(0.35)
    assert gain.tags["paired"] == "restored_minus_baseline"
    assert gain.metadata["restored_value"] == pytest.approx(0.75)
    assert gain.metadata["baseline_value"] == pytest.approx(0.40)


def test_compute_quality_metrics_emits_restoration_gain_for_matched_pairs_only() -> None:
    restored = (
        _outcome("frame-5", rendered_time_ms=20, deadline_hit=True, visible_quality=0.8, full_quality=0.7),
        _outcome("frame-6", rendered_time_ms=20, deadline_hit=True, visible_quality=0.7, full_quality=0.6),
    )
    baseline = (_outcome("frame-5", rendered_time_ms=None, deadline_hit=False, visible_quality=0.5, full_quality=0.4),)

    metrics = compute_quality_metrics(restored, baseline_outcomes=baseline)

    gains = [metric for metric in metrics if metric.metric_name == "restoration_gain"]
    assert len(gains) == 1
    assert gains[0].frame_id == "frame-5"
    assert gains[0].value == pytest.approx(0.7)
    assert gains[0].metadata["baseline"]["rendered"] is False


def test_quality_metrics_reject_duplicate_baseline_frame_ids() -> None:
    outcome = _outcome("frame-7", rendered_time_ms=20, deadline_hit=True, visible_quality=0.8, full_quality=0.7)

    with pytest.raises(QualityMetricError, match="duplicate frame_id"):
        compute_quality_metrics((outcome,), baseline_outcomes=(outcome, outcome))

    with pytest.raises(QualityMetricError, match="frame_id values must match"):
        restoration_gain(outcome, _outcome("frame-8", rendered_time_ms=20, deadline_hit=True, visible_quality=0.2, full_quality=0.2))


def _outcome(
    frame_id: str,
    *,
    rendered_time_ms: int | None,
    deadline_hit: bool,
    visible_quality: float,
    full_quality: float,
) -> FrameOutcome:
    return FrameOutcome(
        frame_id=frame_id,
        scheduled_time_ms=10,
        rendered_time_ms=rendered_time_ms,
        deadline_ms=25,
        delivered_object_ids=("object-1",) if rendered_time_ms is not None else (),
        missing_object_ids=() if rendered_time_ms is not None else ("object-1",),
        quality_score=full_quality,
        deadline_hit=deadline_hit,
        metadata={
            "frame_evaluation": {
                "visible_quality": visible_quality,
                "full_quality": full_quality,
            }
        },
    )
