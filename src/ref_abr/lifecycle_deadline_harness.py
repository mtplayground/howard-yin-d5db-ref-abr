"""Reference lifecycle deadline harness."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from ref_abr.config import stable_config_id
from ref_abr.domain import MetricRecord
from ref_abr.harness import HarnessConfig, HarnessResult, HarnessRunResult, HarnessRunSpec, run_harness


LIFECYCLE_DEADLINE_METHOD_IDS: tuple[str, ...] = (
    "fixed-reference-cadence",
    "deadline-greedy",
    "quality-max-deadline-unaware",
    "no-lifecycle",
    "perfect-information-oracle",
)
LIFECYCLE_DEADLINE_METRIC_NAMES: tuple[str, ...] = (
    "lifecycle_deadline_success_rate",
    "lifecycle_deadline_risk",
    "lifecycle_late_rate",
    "lifecycle_expired_rate",
    "lifecycle_useful_rate",
    "viewport_error_penalty",
)
LifecycleDecision = Literal["use", "late", "expire", "skip"]


class LifecycleDeadlineHarnessError(ValueError):
    """Raised when reference lifecycle deadline harness inputs are invalid."""


@dataclass(frozen=True)
class LifecycleDeadlinePoint:
    """One latency/queue/restore/viewport-error injection point."""

    latency_ms: float
    queue_ms: float
    restore_ms: float
    viewport_error: float
    deadline_slack_ms: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "latency_ms", _non_negative_float(self.latency_ms, "latency_ms"))
        object.__setattr__(self, "queue_ms", _non_negative_float(self.queue_ms, "queue_ms"))
        object.__setattr__(self, "restore_ms", _non_negative_float(self.restore_ms, "restore_ms"))
        object.__setattr__(self, "viewport_error", _unit_interval(self.viewport_error, "viewport_error"))
        object.__setattr__(self, "deadline_slack_ms", _positive_float(self.deadline_slack_ms, "deadline_slack_ms"))

    @property
    def point_id(self) -> str:
        return f"lifecycle-deadline-point-{stable_config_id(self.stable_payload())}"

    def stable_payload(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "queue_ms": self.queue_ms,
            "restore_ms": self.restore_ms,
            "viewport_error": self.viewport_error,
            "deadline_slack_ms": self.deadline_slack_ms,
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            **self.stable_payload(),
        }


@dataclass(frozen=True)
class LifecycleDeadlineOutcome:
    """Method result at one reference lifecycle deadline injection point."""

    point: LifecycleDeadlinePoint
    method_id: str
    seed: int
    selected_lifecycle_decision: LifecycleDecision
    arrival_time_ms: float
    restore_complete_ms: float
    effective_deadline_ms: float
    late: bool
    expired: bool
    useful: bool
    lifecycle_deadline_risk: float
    lifecycle_deadline_success_rate: float
    viewport_error_penalty: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.point, LifecycleDeadlinePoint):
            raise LifecycleDeadlineHarnessError("point must be a LifecycleDeadlinePoint record.")
        _require_non_empty(self.method_id, "method_id")
        object.__setattr__(self, "seed", _non_negative_int(self.seed, "seed"))
        if self.selected_lifecycle_decision not in {"use", "late", "expire", "skip"}:
            raise LifecycleDeadlineHarnessError("selected_lifecycle_decision must be one of: use, late, expire, skip.")
        for field_name in ("arrival_time_ms", "restore_complete_ms", "effective_deadline_ms"):
            object.__setattr__(self, field_name, _non_negative_float(getattr(self, field_name), field_name))
        object.__setattr__(self, "lifecycle_deadline_risk", _unit_interval(self.lifecycle_deadline_risk, "lifecycle_deadline_risk"))
        object.__setattr__(
            self,
            "lifecycle_deadline_success_rate",
            _unit_interval(self.lifecycle_deadline_success_rate, "lifecycle_deadline_success_rate"),
        )
        object.__setattr__(self, "viewport_error_penalty", _unit_interval(self.viewport_error_penalty, "viewport_error_penalty"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def outcome_id(self) -> str:
        payload = {
            "point_id": self.point.point_id,
            "method_id": self.method_id,
            "seed": self.seed,
            "selected_lifecycle_decision": self.selected_lifecycle_decision,
            "risk": self.lifecycle_deadline_risk,
            "success": self.lifecycle_deadline_success_rate,
        }
        return f"lifecycle-deadline-outcome-{stable_config_id(payload)}"

    def metric_records(self, *, run_id: str, split: str | None = None) -> tuple[MetricRecord, ...]:
        tags = {
            "run_id": run_id,
            "method": self.method_id,
            "lifecycle_point_id": self.point.point_id,
            "lifecycle_decision": self.selected_lifecycle_decision,
        }
        base_metadata = {
            "lifecycle_deadline_outcome_id": self.outcome_id,
            "lifecycle_deadline_point": self.point.as_payload(),
            "lifecycle_deadline_outcome": self.as_payload(include_metrics=False),
        }
        return (
            _metric("lifecycle_deadline_success_rate", self.lifecycle_deadline_success_rate, "ratio", tags, base_metadata, split=split),
            _metric("lifecycle_deadline_risk", self.lifecycle_deadline_risk, "ratio", tags, base_metadata, split=split),
            _metric("lifecycle_late_rate", 1.0 if self.late else 0.0, "ratio", tags, base_metadata, split=split),
            _metric("lifecycle_expired_rate", 1.0 if self.expired else 0.0, "ratio", tags, base_metadata, split=split),
            _metric("lifecycle_useful_rate", 1.0 if self.useful else 0.0, "ratio", tags, base_metadata, split=split),
            _metric("viewport_error_penalty", self.viewport_error_penalty, "ratio", tags, base_metadata, split=split),
        )

    def as_payload(self, *, include_metrics: bool = True) -> dict[str, Any]:
        payload = {
            "outcome_id": self.outcome_id,
            "point": self.point.as_payload(),
            "method_id": self.method_id,
            "seed": self.seed,
            "selected_lifecycle_decision": self.selected_lifecycle_decision,
            "arrival_time_ms": self.arrival_time_ms,
            "restore_complete_ms": self.restore_complete_ms,
            "effective_deadline_ms": self.effective_deadline_ms,
            "late": self.late,
            "expired": self.expired,
            "useful": self.useful,
            "lifecycle_deadline_risk": self.lifecycle_deadline_risk,
            "lifecycle_deadline_success_rate": self.lifecycle_deadline_success_rate,
            "viewport_error_penalty": self.viewport_error_penalty,
            "metadata": _to_payload(self.metadata),
        }
        if include_metrics:
            payload["metric_names"] = list(LIFECYCLE_DEADLINE_METRIC_NAMES)
        return payload


@dataclass(frozen=True)
class LifecycleDeadlineConfig:
    """Sweep dimensions and comparison settings for the lifecycle deadline harness."""

    latency_ms: Sequence[float]
    queue_ms: Sequence[float]
    restore_ms: Sequence[float]
    viewport_errors: Sequence[float]
    deadline_slack_ms: Sequence[float]
    seeds: Sequence[int] = (0,)
    methods: Sequence[str] = LIFECYCLE_DEADLINE_METHOD_IDS
    baseline_method_id: str = "fixed-reference-cadence"
    run_mode: str = "full"
    split: str | None = "final"
    output_root: str | Path | None = None
    fixed_variables: Mapping[str, Any] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "latency_ms", _non_negative_float_tuple(self.latency_ms, "latency_ms"))
        object.__setattr__(self, "queue_ms", _non_negative_float_tuple(self.queue_ms, "queue_ms"))
        object.__setattr__(self, "restore_ms", _non_negative_float_tuple(self.restore_ms, "restore_ms"))
        object.__setattr__(self, "viewport_errors", _unit_interval_tuple(self.viewport_errors, "viewport_errors"))
        object.__setattr__(self, "deadline_slack_ms", _positive_float_tuple(self.deadline_slack_ms, "deadline_slack_ms"))
        object.__setattr__(self, "seeds", _int_tuple(self.seeds, "seeds"))
        object.__setattr__(self, "methods", _string_tuple(self.methods, "methods"))
        _require_non_empty(self.baseline_method_id, "baseline_method_id")
        if self.baseline_method_id not in self.methods:
            raise LifecycleDeadlineHarnessError("baseline_method_id must be included in methods.")
        if self.run_mode not in {"plan_only", "metrics_only", "full"}:
            raise LifecycleDeadlineHarnessError("run_mode must be one of: plan_only, metrics_only, full.")
        if self.split is not None:
            _require_non_empty(self.split, "split")
        if self.output_root is not None:
            object.__setattr__(self, "output_root", Path(self.output_root).as_posix())
        object.__setattr__(self, "fixed_variables", _plain_json_mapping(self.fixed_variables, "fixed_variables"))
        object.__setattr__(self, "tags", _string_mapping(self.tags, "tags"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def matrix_id(self) -> str:
        return f"reference-lifecycle-deadline-matrix-{stable_config_id(self.stable_payload())}"

    def surface_points(self) -> tuple[LifecycleDeadlinePoint, ...]:
        points: list[LifecycleDeadlinePoint] = []
        for latency in self.latency_ms:
            for queue in self.queue_ms:
                for restore in self.restore_ms:
                    for viewport_error in self.viewport_errors:
                        for slack in self.deadline_slack_ms:
                            points.append(
                                LifecycleDeadlinePoint(
                                    latency_ms=latency,
                                    queue_ms=queue,
                                    restore_ms=restore,
                                    viewport_error=viewport_error,
                                    deadline_slack_ms=slack,
                                )
                            )
        return tuple(points)

    def stable_payload(self) -> dict[str, Any]:
        return {
            "latency_ms": list(self.latency_ms),
            "queue_ms": list(self.queue_ms),
            "restore_ms": list(self.restore_ms),
            "viewport_errors": list(self.viewport_errors),
            "deadline_slack_ms": list(self.deadline_slack_ms),
            "seeds": list(self.seeds),
            "methods": list(self.methods),
            "baseline_method_id": self.baseline_method_id,
            "run_mode": self.run_mode,
            "split": self.split,
            "fixed_variables": _to_payload(self.fixed_variables),
            "tags": dict(self.tags),
            "metadata": _to_payload(self.metadata),
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "matrix_id": self.matrix_id,
            **self.stable_payload(),
            "output_root": self.output_root,
            "surface_point_count": len(self.surface_points()),
        }


@dataclass(frozen=True)
class LifecycleDeadlineResult:
    """Complete output of the reference lifecycle deadline harness."""

    matrix_id: str
    config: LifecycleDeadlineConfig
    surface_points: tuple[LifecycleDeadlinePoint, ...]
    outcomes: tuple[LifecycleDeadlineOutcome, ...]
    harness_result: HarnessResult

    def __post_init__(self) -> None:
        _require_non_empty(self.matrix_id, "matrix_id")
        if not isinstance(self.config, LifecycleDeadlineConfig):
            raise LifecycleDeadlineHarnessError("config must be a LifecycleDeadlineConfig record.")
        points = tuple(self.surface_points)
        outcomes = tuple(self.outcomes)
        for point in points:
            if not isinstance(point, LifecycleDeadlinePoint):
                raise LifecycleDeadlineHarnessError("surface_points must contain LifecycleDeadlinePoint records.")
        for outcome in outcomes:
            if not isinstance(outcome, LifecycleDeadlineOutcome):
                raise LifecycleDeadlineHarnessError("outcomes must contain LifecycleDeadlineOutcome records.")
        if not isinstance(self.harness_result, HarnessResult):
            raise LifecycleDeadlineHarnessError("harness_result must be a HarnessResult record.")
        object.__setattr__(self, "surface_points", points)
        object.__setattr__(self, "outcomes", outcomes)

    def as_payload(self) -> dict[str, Any]:
        return {
            "matrix_id": self.matrix_id,
            "config": self.config.as_payload(),
            "surface_points": [point.as_payload() for point in self.surface_points],
            "outcomes": [outcome.as_payload() for outcome in self.outcomes],
            "risk_curves": risk_curves_payload(self.outcomes),
            "harness_result": self.harness_result.as_payload(),
        }


def run_reference_lifecycle_deadline_harness(config: LifecycleDeadlineConfig) -> LifecycleDeadlineResult:
    """Run the issue-33 lifecycle deadline matrix through the generic harness."""

    if not isinstance(config, LifecycleDeadlineConfig):
        raise LifecycleDeadlineHarnessError("config must be a LifecycleDeadlineConfig record.")
    points = config.surface_points()
    point_by_id = {point.point_id: point for point in points}
    outcomes: list[LifecycleDeadlineOutcome] = []

    def executor(spec: HarnessRunSpec) -> HarnessRunResult:
        point = point_by_id[spec.workload_id]
        outcome = evaluate_lifecycle_deadline_point(point, method_id=spec.method_id, seed=spec.seed)
        outcomes.append(outcome)
        return HarnessRunResult(
            spec=spec,
            status="executed",
            metrics=outcome.metric_records(run_id=spec.run_id, split=config.split),
            metadata={"lifecycle_deadline_outcome_id": outcome.outcome_id},
        )

    harness_config = HarnessConfig(
        harness_name="reference-lifecycle-deadline",
        methods=config.methods,
        workloads=tuple(point.point_id for point in points),
        seeds=config.seeds,
        run_mode=config.run_mode,
        baseline_method_id=config.baseline_method_id,
        fixed_variables={
            **config.fixed_variables,
            "matrix_id": config.matrix_id,
            "sweep_dimensions": ("latency_ms", "queue_ms", "restore_ms", "viewport_error", "deadline_slack_ms"),
        },
        comparison_metric_names=("lifecycle_deadline_success_rate", "lifecycle_deadline_risk"),
        comparison_group_keys=("workload_id", "seed", "metric_name"),
        output_root=Path(config.output_root) / "harness" if config.output_root is not None else None,
        tags={**config.tags, "matrix_id": config.matrix_id},
        metadata={"lifecycle_deadline_config": config.as_payload(), **config.metadata},
    )
    harness_result = run_harness(harness_config, executor=executor)
    result = LifecycleDeadlineResult(
        matrix_id=config.matrix_id,
        config=config,
        surface_points=points,
        outcomes=tuple(outcomes),
        harness_result=harness_result,
    )
    if config.output_root is not None:
        export_reference_lifecycle_deadline_outputs(config.output_root, result)
    return result


def evaluate_lifecycle_deadline_point(
    point: LifecycleDeadlinePoint,
    *,
    method_id: str,
    seed: int = 0,
) -> LifecycleDeadlineOutcome:
    """Evaluate one method at one lifecycle deadline point with a deterministic analytic model."""

    if not isinstance(point, LifecycleDeadlinePoint):
        raise LifecycleDeadlineHarnessError("point must be a LifecycleDeadlinePoint record.")
    _require_non_empty(method_id, "method_id")
    parsed_seed = _non_negative_int(seed, "seed")
    behavior = _method_behavior(method_id)
    seed_jitter = ((parsed_seed % 11) - 5) * 0.002

    arrival_time_ms = point.latency_ms * behavior["latency_scale"] + point.queue_ms * behavior["queue_scale"]
    restore_complete_ms = arrival_time_ms + point.restore_ms * behavior["restore_scale"] + behavior["fixed_ms"]
    effective_deadline_ms = point.deadline_slack_ms + behavior["deadline_credit_ms"]
    viewport_penalty = _clamp01(point.viewport_error * behavior["viewport_sensitivity"])
    overrun_ratio = max(0.0, restore_complete_ms - effective_deadline_ms) / effective_deadline_ms
    late = restore_complete_ms > effective_deadline_ms
    expired = method_id != "no-lifecycle" and (restore_complete_ms > effective_deadline_ms * 1.2 or viewport_penalty >= 0.7)
    useful = method_id != "no-lifecycle" and not late and not expired and viewport_penalty <= 0.55

    if method_id == "no-lifecycle":
        selected = "skip"
        risk = _clamp01(0.72 + 0.18 * point.viewport_error + 0.04 * overrun_ratio + seed_jitter)
        success = 0.0
    else:
        selected = _selected_decision(late=late, expired=expired)
        risk = _clamp01(
            0.48 * min(overrun_ratio, 2.0)
            + 0.36 * viewport_penalty
            + behavior["risk_bias"]
            + (0.08 if late else 0.0)
            + (0.10 if expired else 0.0)
            + seed_jitter
        )
        if useful:
            success = _clamp01(1.0 - risk)
        elif late:
            success = _clamp01(0.35 * (1.0 - risk))
        else:
            success = _clamp01(0.55 * (1.0 - risk))

    return LifecycleDeadlineOutcome(
        point=point,
        method_id=method_id,
        seed=parsed_seed,
        selected_lifecycle_decision=selected,
        arrival_time_ms=arrival_time_ms,
        restore_complete_ms=restore_complete_ms,
        effective_deadline_ms=effective_deadline_ms,
        late=late,
        expired=expired,
        useful=useful,
        lifecycle_deadline_risk=risk,
        lifecycle_deadline_success_rate=success,
        viewport_error_penalty=viewport_penalty,
        metadata={
            "model": "deterministic_lifecycle_deadline_harness_v1",
            "latency_scale": behavior["latency_scale"],
            "queue_scale": behavior["queue_scale"],
            "restore_scale": behavior["restore_scale"],
            "fixed_ms": behavior["fixed_ms"],
            "deadline_credit_ms": behavior["deadline_credit_ms"],
            "risk_bias": behavior["risk_bias"],
            "seed_jitter": seed_jitter,
            "overrun_ratio": overrun_ratio,
        },
    )


def export_reference_lifecycle_deadline_outputs(
    output_root: str | Path,
    result: LifecycleDeadlineResult,
) -> tuple[Path, Path, Path]:
    """Write lifecycle matrix, risk curves, and a JSON summary."""

    if not isinstance(result, LifecycleDeadlineResult):
        raise LifecycleDeadlineHarnessError("result must be a LifecycleDeadlineResult record.")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    matrix_path = root / "lifecycle_matrix.jsonl"
    risk_curves_path = root / "lifecycle_risk_curves.json"
    summary_path = root / "lifecycle_deadline_summary.json"
    matrix_lines = "".join(
        json.dumps(outcome.as_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for outcome in result.outcomes
    )
    risk_content = json.dumps(risk_curves_payload(result.outcomes), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    summary_content = json.dumps(result.as_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    _write_text_atomic(matrix_path, matrix_lines)
    _write_text_atomic(risk_curves_path, risk_content)
    _write_text_atomic(summary_path, summary_content)
    return matrix_path, risk_curves_path, summary_path


def risk_curves_payload(outcomes: Sequence[LifecycleDeadlineOutcome]) -> dict[str, Any]:
    """Return per-method lifecycle risk curves sorted by injected conditions."""

    parsed = _outcome_tuple(outcomes)
    curves: list[dict[str, Any]] = []
    for method_id in sorted({outcome.method_id for outcome in parsed}):
        method_outcomes = sorted(
            (outcome for outcome in parsed if outcome.method_id == method_id),
            key=lambda outcome: (
                outcome.point.deadline_slack_ms,
                outcome.point.latency_ms,
                outcome.point.queue_ms,
                outcome.point.restore_ms,
                outcome.point.viewport_error,
                outcome.seed,
            ),
        )
        curves.append(
            {
                "curve_id": f"lifecycle-risk-curve-{stable_config_id({'method_id': method_id, 'outcomes': [item.outcome_id for item in method_outcomes]})}",
                "method_id": method_id,
                "points": [
                    {
                        "point_id": outcome.point.point_id,
                        "seed": outcome.seed,
                        "latency_ms": outcome.point.latency_ms,
                        "queue_ms": outcome.point.queue_ms,
                        "restore_ms": outcome.point.restore_ms,
                        "viewport_error": outcome.point.viewport_error,
                        "deadline_slack_ms": outcome.point.deadline_slack_ms,
                        "risk": outcome.lifecycle_deadline_risk,
                        "success_rate": outcome.lifecycle_deadline_success_rate,
                        "late": outcome.late,
                        "expired": outcome.expired,
                        "useful": outcome.useful,
                    }
                    for outcome in method_outcomes
                ],
            }
        )
    return {
        "record_type": "reference_lifecycle_deadline_risk_curves",
        "metric_names": list(LIFECYCLE_DEADLINE_METRIC_NAMES),
        "curves": curves,
    }


def _method_behavior(method_id: str) -> dict[str, float]:
    if method_id == "fixed-reference-cadence":
        return {
            "latency_scale": 1.0,
            "queue_scale": 1.0,
            "restore_scale": 1.0,
            "fixed_ms": 8.0,
            "deadline_credit_ms": 0.0,
            "viewport_sensitivity": 1.15,
            "risk_bias": 0.10,
        }
    if method_id == "deadline-greedy":
        return {
            "latency_scale": 0.94,
            "queue_scale": 0.62,
            "restore_scale": 0.92,
            "fixed_ms": 3.0,
            "deadline_credit_ms": 4.0,
            "viewport_sensitivity": 0.90,
            "risk_bias": 0.02,
        }
    if method_id == "quality-max-deadline-unaware":
        return {
            "latency_scale": 1.0,
            "queue_scale": 1.0,
            "restore_scale": 1.22,
            "fixed_ms": 10.0,
            "deadline_credit_ms": 0.0,
            "viewport_sensitivity": 1.05,
            "risk_bias": 0.14,
        }
    if method_id == "no-lifecycle":
        return {
            "latency_scale": 0.0,
            "queue_scale": 0.0,
            "restore_scale": 0.0,
            "fixed_ms": 0.0,
            "deadline_credit_ms": 0.0,
            "viewport_sensitivity": 1.0,
            "risk_bias": 0.70,
        }
    if method_id == "perfect-information-oracle":
        return {
            "latency_scale": 0.78,
            "queue_scale": 0.32,
            "restore_scale": 0.72,
            "fixed_ms": 0.0,
            "deadline_credit_ms": 8.0,
            "viewport_sensitivity": 0.50,
            "risk_bias": -0.08,
        }
    raise LifecycleDeadlineHarnessError(f"Unknown lifecycle deadline method_id {method_id!r}.")


def _selected_decision(*, late: bool, expired: bool) -> LifecycleDecision:
    if expired:
        return "expire"
    if late:
        return "late"
    return "use"


def _metric(
    metric_name: str,
    value: float,
    unit: str,
    tags: Mapping[str, str],
    metadata: Mapping[str, Any],
    *,
    split: str | None,
) -> MetricRecord:
    metric_metadata = {
        **_plain_json_mapping(metadata, "metadata"),
        "metric_id": f"lifecycle-deadline-metric-{stable_config_id({'metric_name': metric_name, 'value': value, 'tags': dict(tags)})}",
    }
    return MetricRecord(metric_name=metric_name, value=value, unit=unit, tags=tags, split=split, metadata=metric_metadata)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise LifecycleDeadlineHarnessError(f"Failed to write lifecycle deadline output {path}: {exc}") from exc


def _outcome_tuple(values: Sequence[LifecycleDeadlineOutcome]) -> tuple[LifecycleDeadlineOutcome, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise LifecycleDeadlineHarnessError("outcomes must be a sequence of LifecycleDeadlineOutcome records.")
    parsed = tuple(values)
    for value in parsed:
        if not isinstance(value, LifecycleDeadlineOutcome):
            raise LifecycleDeadlineHarnessError("outcomes must contain LifecycleDeadlineOutcome records.")
    return parsed


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise LifecycleDeadlineHarnessError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    if not parsed:
        raise LifecycleDeadlineHarnessError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _string_mapping(value: Mapping[str, str], field_name: str) -> Mapping[str, str]:
    if not isinstance(value, MappingABC):
        raise LifecycleDeadlineHarnessError(f"{field_name} must be a mapping.")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        _require_non_empty(str(key), f"{field_name} key")
        _require_non_empty(item, f"{field_name}.{key}")
        parsed[str(key)] = item
    return {key: parsed[key] for key in sorted(parsed)}


def _int_tuple(values: Sequence[int], field_name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise LifecycleDeadlineHarnessError(f"{field_name} must be a sequence of integers.")
    parsed: list[int] = []
    seen: set[int] = set()
    for value in values:
        parsed_value = _non_negative_int(value, field_name)
        if parsed_value not in seen:
            parsed.append(parsed_value)
            seen.add(parsed_value)
    if not parsed:
        raise LifecycleDeadlineHarnessError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _positive_float_tuple(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    parsed = tuple(_positive_float(value, field_name) for value in values)
    if not parsed:
        raise LifecycleDeadlineHarnessError(f"{field_name} must not be empty.")
    return tuple(dict.fromkeys(parsed))


def _non_negative_float_tuple(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    parsed = tuple(_non_negative_float(value, field_name) for value in values)
    if not parsed:
        raise LifecycleDeadlineHarnessError(f"{field_name} must not be empty.")
    return tuple(dict.fromkeys(parsed))


def _unit_interval_tuple(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    parsed = tuple(_unit_interval(value, field_name) for value in values)
    if not parsed:
        raise LifecycleDeadlineHarnessError(f"{field_name} must not be empty.")
    return tuple(dict.fromkeys(parsed))


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise LifecycleDeadlineHarnessError(f"{field_name} must be a mapping.")
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
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LifecycleDeadlineHarnessError(f"{field_name} must be a non-empty string.")


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise LifecycleDeadlineHarnessError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LifecycleDeadlineHarnessError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise LifecycleDeadlineHarnessError(f"{field_name} must be a non-negative integer.")
    return parsed


def _positive_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed <= 0.0:
        raise LifecycleDeadlineHarnessError(f"{field_name} must be positive.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0:
        raise LifecycleDeadlineHarnessError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if not 0.0 <= parsed <= 1.0:
        raise LifecycleDeadlineHarnessError(f"{field_name} must be between 0 and 1.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise LifecycleDeadlineHarnessError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LifecycleDeadlineHarnessError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise LifecycleDeadlineHarnessError(f"{field_name} must be finite.")
    return parsed


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = [
    "LIFECYCLE_DEADLINE_METHOD_IDS",
    "LIFECYCLE_DEADLINE_METRIC_NAMES",
    "LifecycleDeadlineConfig",
    "LifecycleDeadlineHarnessError",
    "LifecycleDeadlineOutcome",
    "LifecycleDeadlinePoint",
    "LifecycleDeadlineResult",
    "evaluate_lifecycle_deadline_point",
    "export_reference_lifecycle_deadline_outputs",
    "risk_curves_payload",
    "run_reference_lifecycle_deadline_harness",
]
