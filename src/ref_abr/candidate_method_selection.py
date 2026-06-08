"""Candidate method selection harness with runtime and interpretability traces."""

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


CANDIDATE_METHOD_SELECTION_METHOD_IDS: tuple[str, ...] = (
    "robust-deadline-aware-mpc",
    "deadline-aware-knapsack-allocator",
    "virtual-queue-deadline-controller",
    "robust-mpc-joint-space",
    "bola-slack-adapted",
    "bandwidth-greedy",
    "deadline-greedy",
    "quality-max-deadline-unaware",
    "diagnostic-layered-3dgs",
    "diagnostic-viewport-tile",
    "learned-diagnostic-selector",
)
CANDIDATE_METHOD_SELECTION_METRIC_NAMES: tuple[str, ...] = (
    "method_selection_quality",
    "method_selection_deadline_score",
    "method_selection_resource_efficiency",
    "method_selection_runtime_ms",
    "method_selection_interpretability",
)
CandidateSelectionClass = Literal["gaussian", "reference", "tile", "mixed", "diagnostic"]


class CandidateMethodSelectionError(ValueError):
    """Raised when candidate-method selection harness inputs are invalid."""


@dataclass(frozen=True)
class CandidateMethodSelectionPoint:
    """One reduced workload point for method-selection comparison."""

    scene_complexity: float
    budget_bytes: int
    viewport_risk: float
    queue_debt_ms: float
    deadline_slack_ms: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_complexity", _unit_interval(self.scene_complexity, "scene_complexity"))
        object.__setattr__(self, "budget_bytes", _positive_int(self.budget_bytes, "budget_bytes"))
        object.__setattr__(self, "viewport_risk", _unit_interval(self.viewport_risk, "viewport_risk"))
        object.__setattr__(self, "queue_debt_ms", _non_negative_float(self.queue_debt_ms, "queue_debt_ms"))
        object.__setattr__(self, "deadline_slack_ms", _positive_float(self.deadline_slack_ms, "deadline_slack_ms"))

    @property
    def point_id(self) -> str:
        return f"candidate-method-selection-point-{stable_config_id(self.stable_payload())}"

    def stable_payload(self) -> dict[str, Any]:
        return {
            "scene_complexity": self.scene_complexity,
            "budget_bytes": self.budget_bytes,
            "viewport_risk": self.viewport_risk,
            "queue_debt_ms": self.queue_debt_ms,
            "deadline_slack_ms": self.deadline_slack_ms,
        }

    def as_payload(self) -> dict[str, Any]:
        return {"point_id": self.point_id, **self.stable_payload()}


@dataclass(frozen=True)
class CandidateMethodSelectionOutcome:
    """Method result at one reduced workload point."""

    point: CandidateMethodSelectionPoint
    method_id: str
    seed: int
    selected_candidate_class: CandidateSelectionClass
    quality_score: float
    deadline_score: float
    resource_efficiency: float
    runtime_ms: float
    interpretability_score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.point, CandidateMethodSelectionPoint):
            raise CandidateMethodSelectionError("point must be a CandidateMethodSelectionPoint record.")
        _require_non_empty(self.method_id, "method_id")
        object.__setattr__(self, "seed", _non_negative_int(self.seed, "seed"))
        if self.selected_candidate_class not in {"gaussian", "reference", "tile", "mixed", "diagnostic"}:
            raise CandidateMethodSelectionError("selected_candidate_class must be one of: gaussian, reference, tile, mixed, diagnostic.")
        for field_name in ("quality_score", "deadline_score", "resource_efficiency", "interpretability_score"):
            object.__setattr__(self, field_name, _unit_interval(getattr(self, field_name), field_name))
        object.__setattr__(self, "runtime_ms", _non_negative_float(self.runtime_ms, "runtime_ms"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def outcome_id(self) -> str:
        payload = {
            "point_id": self.point.point_id,
            "method_id": self.method_id,
            "seed": self.seed,
            "selected_candidate_class": self.selected_candidate_class,
            "quality_score": self.quality_score,
            "runtime_ms": self.runtime_ms,
        }
        return f"candidate-method-selection-outcome-{stable_config_id(payload)}"

    def metric_records(self, *, run_id: str, split: str | None = None) -> tuple[MetricRecord, ...]:
        tags = {
            "run_id": run_id,
            "method": self.method_id,
            "selection_point_id": self.point.point_id,
            "selected_candidate_class": self.selected_candidate_class,
        }
        base_metadata = {
            "candidate_method_selection_outcome_id": self.outcome_id,
            "selection_point": self.point.as_payload(),
            "selection_outcome": self.as_payload(include_metrics=False),
        }
        return (
            _metric("method_selection_quality", self.quality_score, "score", tags, base_metadata, split=split),
            _metric("method_selection_deadline_score", self.deadline_score, "score", tags, base_metadata, split=split),
            _metric("method_selection_resource_efficiency", self.resource_efficiency, "score", tags, base_metadata, split=split),
            _metric("method_selection_runtime_ms", self.runtime_ms, "ms", tags, base_metadata, split=split),
            _metric("method_selection_interpretability", self.interpretability_score, "score", tags, base_metadata, split=split),
        )

    def as_payload(self, *, include_metrics: bool = True) -> dict[str, Any]:
        payload = {
            "outcome_id": self.outcome_id,
            "point": self.point.as_payload(),
            "method_id": self.method_id,
            "seed": self.seed,
            "selected_candidate_class": self.selected_candidate_class,
            "quality_score": self.quality_score,
            "deadline_score": self.deadline_score,
            "resource_efficiency": self.resource_efficiency,
            "runtime_ms": self.runtime_ms,
            "interpretability_score": self.interpretability_score,
            "metadata": _to_payload(self.metadata),
        }
        if include_metrics:
            payload["metric_names"] = list(CANDIDATE_METHOD_SELECTION_METRIC_NAMES)
        return payload


@dataclass(frozen=True)
class CandidateMethodSelectionConfig:
    """Reduced workload dimensions and comparison settings for method selection."""

    scene_complexities: Sequence[float]
    budget_bytes: Sequence[int]
    viewport_risks: Sequence[float]
    queue_debt_ms: Sequence[float]
    deadline_slack_ms: Sequence[float]
    seeds: Sequence[int] = (0,)
    methods: Sequence[str] = CANDIDATE_METHOD_SELECTION_METHOD_IDS
    baseline_method_id: str = "robust-deadline-aware-mpc"
    run_mode: str = "full"
    split: str | None = "final"
    output_root: str | Path | None = None
    fixed_variables: Mapping[str, Any] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_complexities", _unit_interval_tuple(self.scene_complexities, "scene_complexities"))
        object.__setattr__(self, "budget_bytes", _positive_int_tuple(self.budget_bytes, "budget_bytes"))
        object.__setattr__(self, "viewport_risks", _unit_interval_tuple(self.viewport_risks, "viewport_risks"))
        object.__setattr__(self, "queue_debt_ms", _non_negative_float_tuple(self.queue_debt_ms, "queue_debt_ms"))
        object.__setattr__(self, "deadline_slack_ms", _positive_float_tuple(self.deadline_slack_ms, "deadline_slack_ms"))
        object.__setattr__(self, "seeds", _int_tuple(self.seeds, "seeds"))
        object.__setattr__(self, "methods", _string_tuple(self.methods, "methods"))
        _require_non_empty(self.baseline_method_id, "baseline_method_id")
        if self.baseline_method_id not in self.methods:
            raise CandidateMethodSelectionError("baseline_method_id must be included in methods.")
        if self.run_mode not in {"plan_only", "metrics_only", "full"}:
            raise CandidateMethodSelectionError("run_mode must be one of: plan_only, metrics_only, full.")
        if self.split is not None:
            _require_non_empty(self.split, "split")
        if self.output_root is not None:
            object.__setattr__(self, "output_root", Path(self.output_root).as_posix())
        object.__setattr__(self, "fixed_variables", _plain_json_mapping(self.fixed_variables, "fixed_variables"))
        object.__setattr__(self, "tags", _string_mapping(self.tags, "tags"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def matrix_id(self) -> str:
        return f"candidate-method-selection-matrix-{stable_config_id(self.stable_payload())}"

    def selection_points(self) -> tuple[CandidateMethodSelectionPoint, ...]:
        points: list[CandidateMethodSelectionPoint] = []
        for complexity in self.scene_complexities:
            for budget in self.budget_bytes:
                for viewport_risk in self.viewport_risks:
                    for queue_debt in self.queue_debt_ms:
                        for slack in self.deadline_slack_ms:
                            points.append(
                                CandidateMethodSelectionPoint(
                                    scene_complexity=complexity,
                                    budget_bytes=budget,
                                    viewport_risk=viewport_risk,
                                    queue_debt_ms=queue_debt,
                                    deadline_slack_ms=slack,
                                )
                            )
        return tuple(points)

    def stable_payload(self) -> dict[str, Any]:
        return {
            "scene_complexities": list(self.scene_complexities),
            "budget_bytes": list(self.budget_bytes),
            "viewport_risks": list(self.viewport_risks),
            "queue_debt_ms": list(self.queue_debt_ms),
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
            "selection_point_count": len(self.selection_points()),
        }


@dataclass(frozen=True)
class CandidateMethodSelectionResult:
    """Complete output of the candidate method selection harness."""

    matrix_id: str
    config: CandidateMethodSelectionConfig
    selection_points: tuple[CandidateMethodSelectionPoint, ...]
    outcomes: tuple[CandidateMethodSelectionOutcome, ...]
    harness_result: HarnessResult

    def __post_init__(self) -> None:
        _require_non_empty(self.matrix_id, "matrix_id")
        if not isinstance(self.config, CandidateMethodSelectionConfig):
            raise CandidateMethodSelectionError("config must be a CandidateMethodSelectionConfig record.")
        points = tuple(self.selection_points)
        outcomes = tuple(self.outcomes)
        for point in points:
            if not isinstance(point, CandidateMethodSelectionPoint):
                raise CandidateMethodSelectionError("selection_points must contain CandidateMethodSelectionPoint records.")
        for outcome in outcomes:
            if not isinstance(outcome, CandidateMethodSelectionOutcome):
                raise CandidateMethodSelectionError("outcomes must contain CandidateMethodSelectionOutcome records.")
        if not isinstance(self.harness_result, HarnessResult):
            raise CandidateMethodSelectionError("harness_result must be a HarnessResult record.")
        object.__setattr__(self, "selection_points", points)
        object.__setattr__(self, "outcomes", outcomes)

    def as_payload(self) -> dict[str, Any]:
        return {
            "matrix_id": self.matrix_id,
            "config": self.config.as_payload(),
            "selection_points": [point.as_payload() for point in self.selection_points],
            "outcomes": [outcome.as_payload() for outcome in self.outcomes],
            "runtime_traces": runtime_trace_summary(self.outcomes),
            "interpretability_traces": interpretability_trace_summary(self.outcomes),
            "harness_result": self.harness_result.as_payload(),
        }


def run_candidate_method_selection_harness(config: CandidateMethodSelectionConfig) -> CandidateMethodSelectionResult:
    """Run the reduced candidate method selection harness."""

    if not isinstance(config, CandidateMethodSelectionConfig):
        raise CandidateMethodSelectionError("config must be a CandidateMethodSelectionConfig record.")
    points = config.selection_points()
    point_by_id = {point.point_id: point for point in points}
    outcomes: list[CandidateMethodSelectionOutcome] = []

    def executor(spec: HarnessRunSpec) -> HarnessRunResult:
        point = point_by_id[spec.workload_id]
        outcome = evaluate_candidate_method_selection_point(point, method_id=spec.method_id, seed=spec.seed)
        outcomes.append(outcome)
        return HarnessRunResult(
            spec=spec,
            status="executed",
            metrics=outcome.metric_records(run_id=spec.run_id, split=config.split),
            metadata={"candidate_method_selection_outcome_id": outcome.outcome_id},
        )

    harness_config = HarnessConfig(
        harness_name="candidate-method-selection",
        methods=config.methods,
        workloads=tuple(point.point_id for point in points),
        seeds=config.seeds,
        run_mode=config.run_mode,
        baseline_method_id=config.baseline_method_id,
        fixed_variables={
            **config.fixed_variables,
            "matrix_id": config.matrix_id,
            "reduced_workload": True,
            "sweep_dimensions": ("scene_complexity", "budget_bytes", "viewport_risk", "queue_debt_ms", "deadline_slack_ms"),
        },
        comparison_metric_names=(
            "method_selection_quality",
            "method_selection_deadline_score",
            "method_selection_resource_efficiency",
            "method_selection_runtime_ms",
            "method_selection_interpretability",
        ),
        comparison_group_keys=("workload_id", "seed", "metric_name"),
        output_root=Path(config.output_root) / "harness" if config.output_root is not None else None,
        tags={**config.tags, "matrix_id": config.matrix_id},
        metadata={"candidate_method_selection_config": config.as_payload(), **config.metadata},
    )
    harness_result = run_harness(harness_config, executor=executor)
    result = CandidateMethodSelectionResult(
        matrix_id=config.matrix_id,
        config=config,
        selection_points=points,
        outcomes=tuple(outcomes),
        harness_result=harness_result,
    )
    if config.output_root is not None:
        export_candidate_method_selection_outputs(config.output_root, result)
    return result


def evaluate_candidate_method_selection_point(
    point: CandidateMethodSelectionPoint,
    *,
    method_id: str,
    seed: int = 0,
) -> CandidateMethodSelectionOutcome:
    """Evaluate one method at one reduced workload point."""

    if not isinstance(point, CandidateMethodSelectionPoint):
        raise CandidateMethodSelectionError("point must be a CandidateMethodSelectionPoint record.")
    _require_non_empty(method_id, "method_id")
    if method_id not in CANDIDATE_METHOD_SELECTION_METHOD_IDS:
        raise CandidateMethodSelectionError(f"Unknown candidate selection method_id {method_id!r}.")
    seed = _non_negative_int(seed, "seed")
    context = _method_context(point, seed)
    selected_class = _selected_candidate_class(method_id, context)
    runtime_ms = _runtime_ms(method_id, point, selected_class)
    quality = _selected_quality(selected_class, context)
    deadline_score = _deadline_score(method_id, point, selected_class, runtime_ms)
    resource_efficiency = _resource_efficiency(point, selected_class)
    interpretability = _interpretability_score(method_id, selected_class, point)
    return CandidateMethodSelectionOutcome(
        point=point,
        method_id=method_id,
        seed=seed,
        selected_candidate_class=selected_class,
        quality_score=quality,
        deadline_score=deadline_score,
        resource_efficiency=resource_efficiency,
        runtime_ms=runtime_ms,
        interpretability_score=interpretability,
        metadata={
            "runtime_trace": _runtime_trace(method_id, runtime_ms, point),
            "interpretability_trace": _interpretability_trace(method_id, selected_class, point),
            "method_context": context,
        },
    )


def runtime_trace_summary(outcomes: Sequence[CandidateMethodSelectionOutcome]) -> dict[str, Any]:
    """Aggregate runtime traces by method."""

    grouped = _group_outcomes(outcomes)
    return {
        method_id: {
            "mean_runtime_ms": _mean(tuple(outcome.runtime_ms for outcome in method_outcomes)),
            "max_runtime_ms": max(outcome.runtime_ms for outcome in method_outcomes),
            "count": len(method_outcomes),
        }
        for method_id, method_outcomes in grouped.items()
    }


def interpretability_trace_summary(outcomes: Sequence[CandidateMethodSelectionOutcome]) -> dict[str, Any]:
    """Aggregate interpretability traces by method."""

    grouped = _group_outcomes(outcomes)
    return {
        method_id: {
            "mean_interpretability": _mean(tuple(outcome.interpretability_score for outcome in method_outcomes)),
            "selected_classes": sorted({outcome.selected_candidate_class for outcome in method_outcomes}),
            "trace_fields": sorted(
                {
                    key
                    for outcome in method_outcomes
                    for key in outcome.metadata.get("interpretability_trace", {})
                }
            ),
        }
        for method_id, method_outcomes in grouped.items()
    }


def export_candidate_method_selection_outputs(
    output_root: str | Path,
    result: CandidateMethodSelectionResult,
) -> tuple[Path, Path]:
    """Write JSONL candidate-selection outcomes and a JSON summary."""

    if not isinstance(result, CandidateMethodSelectionResult):
        raise CandidateMethodSelectionError("result must be a CandidateMethodSelectionResult record.")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    outcomes_path = root / "candidate_method_selection.jsonl"
    summary_path = root / "candidate_method_selection_summary.json"
    outcome_lines = "".join(
        json.dumps(outcome.as_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for outcome in result.outcomes
    )
    summary_content = json.dumps(result.as_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    _write_text_atomic(outcomes_path, outcome_lines)
    _write_text_atomic(summary_path, summary_content)
    return outcomes_path, summary_path


def _method_context(point: CandidateMethodSelectionPoint, seed: int) -> dict[str, float]:
    seed_adjustment = ((seed % 9) - 4) * 0.003
    gaussian_quality = _clamp01(0.56 + 0.10 * point.scene_complexity - 0.04 * point.viewport_risk + seed_adjustment)
    reference_quality = _clamp01(0.60 + 0.16 * point.scene_complexity - 0.24 * point.viewport_risk + 0.5 * seed_adjustment)
    tile_quality = _clamp01(0.50 + 0.13 * point.viewport_risk + 0.04 * point.scene_complexity)
    gaussian_cost = 260_000 * (0.75 + point.scene_complexity)
    reference_cost = 390_000 * (0.85 + point.scene_complexity + 0.25 * point.viewport_risk)
    tile_cost = 180_000 * (0.80 + point.viewport_risk)
    return {
        "gaussian_quality": gaussian_quality,
        "reference_quality": reference_quality,
        "tile_quality": tile_quality,
        "gaussian_cost": gaussian_cost,
        "reference_cost": reference_cost,
        "tile_cost": tile_cost,
        "budget_pressure": max(gaussian_cost, reference_cost, tile_cost) / point.budget_bytes,
        "queue_pressure": point.queue_debt_ms / max(1.0, point.deadline_slack_ms),
        "deadline_slack_ratio": point.deadline_slack_ms / (point.deadline_slack_ms + point.queue_debt_ms + 1.0),
    }


def _selected_candidate_class(method_id: str, context: Mapping[str, float]) -> CandidateSelectionClass:
    if method_id in {"diagnostic-layered-3dgs"}:
        return "diagnostic"
    if method_id in {"diagnostic-viewport-tile"}:
        return "tile"
    if method_id == "bandwidth-greedy":
        return "tile" if context["tile_cost"] <= context["gaussian_cost"] else "gaussian"
    if method_id == "deadline-greedy":
        return "gaussian" if context["queue_pressure"] >= 0.6 else "reference"
    if method_id == "quality-max-deadline-unaware":
        return "reference" if context["reference_quality"] >= context["gaussian_quality"] else "gaussian"
    if method_id == "bola-slack-adapted":
        if context["deadline_slack_ratio"] < 0.45:
            return "gaussian"
        return "reference" if context["reference_quality"] >= context["gaussian_quality"] else "gaussian"
    if method_id == "robust-mpc-joint-space":
        if context["budget_pressure"] > 1.35 or context["queue_pressure"] > 0.9:
            return "gaussian"
        return "reference" if context["reference_quality"] >= context["gaussian_quality"] + 0.02 else "mixed"
    if method_id == "deadline-aware-knapsack-allocator":
        if context["budget_pressure"] <= 1.0 and context["deadline_slack_ratio"] >= 0.55:
            return "mixed"
        return "gaussian"
    if method_id == "virtual-queue-deadline-controller":
        if context["queue_pressure"] > 0.7:
            return "gaussian"
        if context["budget_pressure"] < 1.2 and context["deadline_slack_ratio"] > 0.5:
            return "mixed"
        return "tile"
    if method_id == "learned-diagnostic-selector":
        if context["queue_pressure"] > 0.75:
            return "gaussian"
        if context["reference_quality"] - context["gaussian_quality"] > 0.08 and context["budget_pressure"] < 1.3:
            return "reference"
        if context["tile_quality"] > context["gaussian_quality"] and context["budget_pressure"] > 1.2:
            return "tile"
        return "mixed"
    if method_id == "robust-deadline-aware-mpc":
        if context["queue_pressure"] > 0.8 and context["deadline_slack_ratio"] < 0.5:
            return "gaussian"
        if context["budget_pressure"] <= 1.15 and context["reference_quality"] >= context["gaussian_quality"]:
            return "mixed"
        return "reference" if context["reference_quality"] >= context["gaussian_quality"] + 0.04 else "gaussian"
    raise CandidateMethodSelectionError(f"Unknown candidate selection method_id {method_id!r}.")


def _selected_quality(selection: CandidateSelectionClass, context: Mapping[str, float]) -> float:
    if selection == "gaussian":
        return context["gaussian_quality"]
    if selection == "reference":
        return context["reference_quality"]
    if selection == "tile":
        return context["tile_quality"]
    if selection == "mixed":
        return _clamp01(0.55 * context["reference_quality"] + 0.45 * context["gaussian_quality"] + 0.03)
    return _clamp01(0.60 * context["gaussian_quality"] + 0.40 * context["tile_quality"])


def _deadline_score(method_id: str, point: CandidateMethodSelectionPoint, selection: CandidateSelectionClass, runtime_ms: float) -> float:
    selection_latency = {
        "gaussian": 8.0,
        "reference": 18.0,
        "tile": 6.0,
        "mixed": 14.0,
        "diagnostic": 10.0,
    }[selection]
    controller_overhead = 0.25 * runtime_ms
    deadline_margin = point.deadline_slack_ms - point.queue_debt_ms - selection_latency - controller_overhead
    score = 0.5 + deadline_margin / max(1.0, 2.0 * point.deadline_slack_ms)
    if method_id == "virtual-queue-deadline-controller" and point.queue_debt_ms > 0:
        score += 0.06
    if method_id == "deadline-greedy":
        score += 0.04
    return _clamp01(score)


def _resource_efficiency(point: CandidateMethodSelectionPoint, selection: CandidateSelectionClass) -> float:
    cost_factor = {
        "gaussian": 0.62,
        "reference": 0.95,
        "tile": 0.45,
        "mixed": 0.82,
        "diagnostic": 0.55,
    }[selection]
    normalized_cost = cost_factor * (0.65 + point.scene_complexity + 0.25 * point.viewport_risk)
    pressure = normalized_cost * 400_000 / point.budget_bytes
    return _clamp01(1.0 / (1.0 + pressure))


def _runtime_ms(method_id: str, point: CandidateMethodSelectionPoint, selection: CandidateSelectionClass) -> float:
    base = {
        "robust-deadline-aware-mpc": 14.0,
        "deadline-aware-knapsack-allocator": 5.0,
        "virtual-queue-deadline-controller": 7.0,
        "robust-mpc-joint-space": 3.5,
        "bola-slack-adapted": 2.8,
        "bandwidth-greedy": 1.2,
        "deadline-greedy": 1.4,
        "quality-max-deadline-unaware": 1.6,
        "diagnostic-layered-3dgs": 2.0,
        "diagnostic-viewport-tile": 2.1,
        "learned-diagnostic-selector": 4.0,
    }[method_id]
    selection_multiplier = {"gaussian": 1.0, "reference": 1.2, "tile": 0.9, "mixed": 1.35, "diagnostic": 0.95}[selection]
    return round(base * selection_multiplier * (1.0 + 0.35 * point.scene_complexity + 0.15 * point.viewport_risk), 6)


def _interpretability_score(method_id: str, selection: CandidateSelectionClass, point: CandidateMethodSelectionPoint) -> float:
    base = {
        "robust-deadline-aware-mpc": 0.68,
        "deadline-aware-knapsack-allocator": 0.86,
        "virtual-queue-deadline-controller": 0.88,
        "robust-mpc-joint-space": 0.62,
        "bola-slack-adapted": 0.74,
        "bandwidth-greedy": 0.92,
        "deadline-greedy": 0.90,
        "quality-max-deadline-unaware": 0.78,
        "diagnostic-layered-3dgs": 0.96,
        "diagnostic-viewport-tile": 0.96,
        "learned-diagnostic-selector": 0.58,
    }[method_id]
    if selection == "mixed":
        base -= 0.04
    if point.viewport_risk > 0.6:
        base += 0.02
    return _clamp01(base)


def _runtime_trace(method_id: str, runtime_ms: float, point: CandidateMethodSelectionPoint) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "runtime_ms": runtime_ms,
        "runtime_bucket": "interactive" if runtime_ms <= 8.0 else "controller-heavy",
        "scene_complexity": point.scene_complexity,
        "queue_debt_ms": point.queue_debt_ms,
    }


def _interpretability_trace(method_id: str, selection: CandidateSelectionClass, point: CandidateMethodSelectionPoint) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "selected_candidate_class": selection,
        "explanation_type": _explanation_type(method_id),
        "uses_deadline_signal": method_id in {"robust-deadline-aware-mpc", "virtual-queue-deadline-controller", "deadline-greedy", "bola-slack-adapted"},
        "uses_viewport_signal": method_id in {"robust-deadline-aware-mpc", "virtual-queue-deadline-controller", "diagnostic-viewport-tile", "learned-diagnostic-selector"},
        "queue_debt_ms": point.queue_debt_ms,
        "viewport_risk": point.viewport_risk,
    }


def _explanation_type(method_id: str) -> str:
    if method_id in {"deadline-aware-knapsack-allocator", "virtual-queue-deadline-controller"}:
        return "priced_budget_terms"
    if method_id.startswith("diagnostic"):
        return "diagnostic_rule"
    if "greedy" in method_id:
        return "single_ordering_rule"
    if method_id == "learned-diagnostic-selector":
        return "learned_feature_attribution"
    return "controller_trace"


def _group_outcomes(outcomes: Sequence[CandidateMethodSelectionOutcome]) -> dict[str, tuple[CandidateMethodSelectionOutcome, ...]]:
    parsed = tuple(outcomes)
    for outcome in parsed:
        if not isinstance(outcome, CandidateMethodSelectionOutcome):
            raise CandidateMethodSelectionError("outcomes must contain CandidateMethodSelectionOutcome records.")
    grouped: dict[str, list[CandidateMethodSelectionOutcome]] = {}
    for outcome in parsed:
        grouped.setdefault(outcome.method_id, []).append(outcome)
    return {method_id: tuple(method_outcomes) for method_id, method_outcomes in sorted(grouped.items())}


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
        "metric_id": f"candidate-method-selection-metric-{stable_config_id({'metric_name': metric_name, 'value': value, 'tags': dict(tags)})}",
    }
    return MetricRecord(metric_name=metric_name, value=value, unit=unit, tags=tags, split=split, metadata=metric_metadata)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise CandidateMethodSelectionError(f"Failed to write candidate-method selection output {path}: {exc}") from exc


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CandidateMethodSelectionError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    if not parsed:
        raise CandidateMethodSelectionError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _string_mapping(value: Mapping[str, str], field_name: str) -> Mapping[str, str]:
    if not isinstance(value, MappingABC):
        raise CandidateMethodSelectionError(f"{field_name} must be a mapping.")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        _require_non_empty(str(key), f"{field_name} key")
        _require_non_empty(item, f"{field_name}.{key}")
        parsed[str(key)] = item
    return {key: parsed[key] for key in sorted(parsed)}


def _int_tuple(values: Sequence[int], field_name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CandidateMethodSelectionError(f"{field_name} must be a sequence of integers.")
    parsed: list[int] = []
    seen: set[int] = set()
    for value in values:
        parsed_value = _non_negative_int(value, field_name)
        if parsed_value not in seen:
            parsed.append(parsed_value)
            seen.add(parsed_value)
    if not parsed:
        raise CandidateMethodSelectionError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _positive_int_tuple(values: Sequence[int], field_name: str) -> tuple[int, ...]:
    parsed = tuple(_positive_int(value, field_name) for value in values)
    if not parsed:
        raise CandidateMethodSelectionError(f"{field_name} must not be empty.")
    return tuple(dict.fromkeys(parsed))


def _unit_interval_tuple(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    parsed = tuple(_unit_interval(value, field_name) for value in values)
    if not parsed:
        raise CandidateMethodSelectionError(f"{field_name} must not be empty.")
    return tuple(dict.fromkeys(parsed))


def _positive_float_tuple(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    parsed = tuple(_positive_float(value, field_name) for value in values)
    if not parsed:
        raise CandidateMethodSelectionError(f"{field_name} must not be empty.")
    return tuple(dict.fromkeys(parsed))


def _non_negative_float_tuple(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    parsed = tuple(_non_negative_float(value, field_name) for value in values)
    if not parsed:
        raise CandidateMethodSelectionError(f"{field_name} must not be empty.")
    return tuple(dict.fromkeys(parsed))


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise CandidateMethodSelectionError(f"{field_name} must be a mapping.")
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


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CandidateMethodSelectionError(f"{field_name} must be a non-empty string.")


def _positive_int(value: Any, field_name: str) -> int:
    parsed = _non_negative_int(value, field_name)
    if parsed <= 0:
        raise CandidateMethodSelectionError(f"{field_name} must be positive.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise CandidateMethodSelectionError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CandidateMethodSelectionError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise CandidateMethodSelectionError(f"{field_name} must be non-negative.")
    return parsed


def _positive_float(value: Any, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name)
    if parsed <= 0.0:
        raise CandidateMethodSelectionError(f"{field_name} must be positive.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0:
        raise CandidateMethodSelectionError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name)
    if parsed > 1.0:
        raise CandidateMethodSelectionError(f"{field_name} must be in [0, 1].")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise CandidateMethodSelectionError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CandidateMethodSelectionError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise CandidateMethodSelectionError(f"{field_name} must be finite.")
    return parsed


def _mean(values: Sequence[float]) -> float:
    parsed = tuple(_finite_float(value, "mean.value") for value in values)
    if not parsed:
        return 0.0
    return sum(parsed) / len(parsed)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


__all__ = [
    "CANDIDATE_METHOD_SELECTION_METHOD_IDS",
    "CANDIDATE_METHOD_SELECTION_METRIC_NAMES",
    "CandidateMethodSelectionConfig",
    "CandidateMethodSelectionError",
    "CandidateMethodSelectionOutcome",
    "CandidateMethodSelectionPoint",
    "CandidateMethodSelectionResult",
    "evaluate_candidate_method_selection_point",
    "export_candidate_method_selection_outputs",
    "interpretability_trace_summary",
    "run_candidate_method_selection_harness",
    "runtime_trace_summary",
]
