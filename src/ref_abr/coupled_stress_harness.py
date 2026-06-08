"""Coupled stress robustness harness."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ref_abr.config import stable_config_id
from ref_abr.domain import FrozenMethodManifest, MetricRecord
from ref_abr.harness import HarnessConfig, HarnessResult, HarnessRunResult, HarnessRunSpec, run_harness


COUPLED_STRESS_BASELINE_METHOD_IDS: tuple[str, ...] = (
    "deadline-aware-knapsack-allocator",
    "virtual-queue-deadline-controller",
    "robust-mpc-joint-space",
    "bola-slack-adapted",
    "deadline-greedy",
)
COUPLED_STRESS_AXES: tuple[str, ...] = ("bandwidth", "viewport", "server", "client", "deadline")
COUPLED_STRESS_METRIC_NAMES: tuple[str, ...] = (
    "stress_deadline_hit_qoe",
    "stress_recovery_ms",
    "stress_degradation_slope",
    "stress_deadline_hit_rate",
    "stress_visible_quality",
    "stress_resource_pressure",
)


class CoupledStressHarnessError(ValueError):
    """Raised when coupled stress harness inputs are invalid."""


@dataclass(frozen=True)
class CoupledStressPoint:
    """One scene/trace/viewport/device tuple under a coupled stress vector."""

    scene_id: str
    trace_id: str
    viewport_id: str
    device_profile_id: str
    stress_id: str
    stress_levels: Mapping[str, float]

    def __post_init__(self) -> None:
        _require_non_empty(self.scene_id, "scene_id")
        _require_non_empty(self.trace_id, "trace_id")
        _require_non_empty(self.viewport_id, "viewport_id")
        _require_non_empty(self.device_profile_id, "device_profile_id")
        _require_non_empty(self.stress_id, "stress_id")
        object.__setattr__(self, "stress_levels", _stress_levels_mapping(self.stress_levels, "stress_levels"))

    @property
    def point_id(self) -> str:
        return f"coupled-stress-point-{stable_config_id(self.stable_payload())}"

    def stable_payload(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "trace_id": self.trace_id,
            "viewport_id": self.viewport_id,
            "device_profile_id": self.device_profile_id,
            "stress_id": self.stress_id,
            "stress_levels": dict(self.stress_levels),
        }

    def as_payload(self) -> dict[str, Any]:
        return {"point_id": self.point_id, **self.stable_payload()}


@dataclass(frozen=True)
class CoupledStressOutcome:
    """Method result under one coupled stress point."""

    point: CoupledStressPoint
    method_id: str
    seed: int
    method_role: str
    stress_deadline_hit_qoe: float
    stress_recovery_ms: float
    stress_degradation_slope: float
    stress_deadline_hit_rate: float
    stress_visible_quality: float
    stress_resource_pressure: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.point, CoupledStressPoint):
            raise CoupledStressHarnessError("point must be a CoupledStressPoint record.")
        _require_non_empty(self.method_id, "method_id")
        object.__setattr__(self, "seed", _non_negative_int(self.seed, "seed"))
        if self.method_role not in {"frozen_refabr", "baseline"}:
            raise CoupledStressHarnessError("method_role must be one of: frozen_refabr, baseline.")
        for field_name in (
            "stress_deadline_hit_qoe",
            "stress_degradation_slope",
            "stress_deadline_hit_rate",
            "stress_visible_quality",
            "stress_resource_pressure",
        ):
            object.__setattr__(self, field_name, _unit_interval(getattr(self, field_name), field_name))
        object.__setattr__(self, "stress_recovery_ms", _non_negative_float(self.stress_recovery_ms, "stress_recovery_ms"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def outcome_id(self) -> str:
        payload = {
            "point_id": self.point.point_id,
            "method_id": self.method_id,
            "seed": self.seed,
            "stress_deadline_hit_qoe": self.stress_deadline_hit_qoe,
            "stress_recovery_ms": self.stress_recovery_ms,
        }
        return f"coupled-stress-outcome-{stable_config_id(payload)}"

    def metric_records(self, *, run_id: str, split: str | None = "final") -> tuple[MetricRecord, ...]:
        tags = {
            "run_id": run_id,
            "method": self.method_id,
            "method_role": self.method_role,
            "stress_id": self.point.stress_id,
            "scene_id": self.point.scene_id,
            "trace_id": self.point.trace_id,
            "viewport_id": self.point.viewport_id,
            "device_profile_id": self.point.device_profile_id,
        }
        base_metadata = {
            "coupled_stress_outcome_id": self.outcome_id,
            "coupled_stress_point": self.point.as_payload(),
            "coupled_stress_outcome": self.as_payload(include_metrics=False),
        }
        return (
            _metric("stress_deadline_hit_qoe", self.stress_deadline_hit_qoe, "score", tags, base_metadata, split=split),
            _metric("stress_recovery_ms", self.stress_recovery_ms, "ms", tags, base_metadata, split=split),
            _metric("stress_degradation_slope", self.stress_degradation_slope, "score_per_stress", tags, base_metadata, split=split),
            _metric("stress_deadline_hit_rate", self.stress_deadline_hit_rate, "ratio", tags, base_metadata, split=split),
            _metric("stress_visible_quality", self.stress_visible_quality, "score", tags, base_metadata, split=split),
            _metric("stress_resource_pressure", self.stress_resource_pressure, "ratio", tags, base_metadata, split=split),
        )

    def as_payload(self, *, include_metrics: bool = True) -> dict[str, Any]:
        payload = {
            "outcome_id": self.outcome_id,
            "point": self.point.as_payload(),
            "method_id": self.method_id,
            "seed": self.seed,
            "method_role": self.method_role,
            "stress_deadline_hit_qoe": self.stress_deadline_hit_qoe,
            "stress_recovery_ms": self.stress_recovery_ms,
            "stress_degradation_slope": self.stress_degradation_slope,
            "stress_deadline_hit_rate": self.stress_deadline_hit_rate,
            "stress_visible_quality": self.stress_visible_quality,
            "stress_resource_pressure": self.stress_resource_pressure,
            "metadata": _to_payload(self.metadata),
        }
        if include_metrics:
            payload["metric_names"] = list(COUPLED_STRESS_METRIC_NAMES)
        return payload


@dataclass(frozen=True)
class CoupledStressConfig:
    """Coupled stress robustness sweep and comparison settings."""

    scenes: Sequence[str]
    traces: Sequence[str]
    viewports: Sequence[str]
    devices: Sequence[str]
    stress_matrix: Sequence[Mapping[str, Any]]
    frozen_method_manifest: FrozenMethodManifest | Mapping[str, Any] | None = None
    frozen_method_id: str = "robust-deadline-aware-mpc"
    baseline_methods: Sequence[str] = COUPLED_STRESS_BASELINE_METHOD_IDS
    seeds: Sequence[int] = (0,)
    split: str = "final"
    run_mode: str = "full"
    output_root: str | Path | None = None
    fixed_variables: Mapping[str, Any] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenes", _string_tuple(self.scenes, "scenes"))
        object.__setattr__(self, "traces", _string_tuple(self.traces, "traces"))
        object.__setattr__(self, "viewports", _string_tuple(self.viewports, "viewports"))
        object.__setattr__(self, "devices", _string_tuple(self.devices, "devices"))
        object.__setattr__(self, "stress_matrix", _stress_matrix_tuple(self.stress_matrix))
        manifest = _coerce_frozen_manifest(self.frozen_method_manifest)
        object.__setattr__(self, "frozen_method_manifest", manifest)
        frozen_method_id = manifest.method_id if manifest is not None else self.frozen_method_id
        _require_non_empty(frozen_method_id, "frozen_method_id")
        object.__setattr__(self, "frozen_method_id", frozen_method_id)
        object.__setattr__(self, "baseline_methods", _string_tuple(self.baseline_methods, "baseline_methods"))
        if frozen_method_id in self.baseline_methods:
            raise CoupledStressHarnessError("frozen_method_id must not be duplicated in baseline_methods.")
        object.__setattr__(self, "seeds", _int_tuple(self.seeds, "seeds"))
        if self.split != "final":
            raise CoupledStressHarnessError("Coupled stress robustness harness must run on the final split.")
        if self.run_mode not in {"plan_only", "metrics_only", "full"}:
            raise CoupledStressHarnessError("run_mode must be one of: plan_only, metrics_only, full.")
        if self.output_root is not None:
            object.__setattr__(self, "output_root", Path(self.output_root).as_posix())
        object.__setattr__(self, "fixed_variables", _plain_json_mapping(self.fixed_variables, "fixed_variables"))
        object.__setattr__(self, "tags", _string_mapping(self.tags, "tags"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def methods(self) -> tuple[str, ...]:
        return (self.frozen_method_id, *self.baseline_methods)

    @property
    def harness_id(self) -> str:
        return f"coupled-stress-robustness-{stable_config_id(self.stable_payload())}"

    def stress_points(self) -> tuple[CoupledStressPoint, ...]:
        points: list[CoupledStressPoint] = []
        for scene in self.scenes:
            for trace in self.traces:
                for viewport in self.viewports:
                    for device in self.devices:
                        for stress in self.stress_matrix:
                            points.append(
                                CoupledStressPoint(
                                    scene_id=scene,
                                    trace_id=trace,
                                    viewport_id=viewport,
                                    device_profile_id=device,
                                    stress_id=str(stress["stress_id"]),
                                    stress_levels=stress["stress_levels"],
                                )
                            )
        return tuple(points)

    def stable_payload(self) -> dict[str, Any]:
        return {
            "scenes": list(self.scenes),
            "traces": list(self.traces),
            "viewports": list(self.viewports),
            "devices": list(self.devices),
            "stress_matrix": [_to_payload(row) for row in self.stress_matrix],
            "frozen_method_manifest": self.frozen_method_manifest.as_payload() if self.frozen_method_manifest is not None else None,
            "frozen_method_id": self.frozen_method_id,
            "baseline_methods": list(self.baseline_methods),
            "seeds": list(self.seeds),
            "split": self.split,
            "run_mode": self.run_mode,
            "fixed_variables": _to_payload(self.fixed_variables),
            "tags": dict(self.tags),
            "metadata": _to_payload(self.metadata),
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            **self.stable_payload(),
            "output_root": self.output_root,
            "stress_point_count": len(self.stress_points()),
        }


@dataclass(frozen=True)
class CoupledStressResult:
    """Complete coupled stress robustness harness output."""

    harness_id: str
    config: CoupledStressConfig
    stress_points: tuple[CoupledStressPoint, ...]
    outcomes: tuple[CoupledStressOutcome, ...]
    harness_result: HarnessResult

    def __post_init__(self) -> None:
        _require_non_empty(self.harness_id, "harness_id")
        if not isinstance(self.config, CoupledStressConfig):
            raise CoupledStressHarnessError("config must be a CoupledStressConfig record.")
        points = tuple(self.stress_points)
        outcomes = tuple(self.outcomes)
        for point in points:
            if not isinstance(point, CoupledStressPoint):
                raise CoupledStressHarnessError("stress_points must contain CoupledStressPoint records.")
        for outcome in outcomes:
            if not isinstance(outcome, CoupledStressOutcome):
                raise CoupledStressHarnessError("outcomes must contain CoupledStressOutcome records.")
        if not isinstance(self.harness_result, HarnessResult):
            raise CoupledStressHarnessError("harness_result must be a HarnessResult record.")
        object.__setattr__(self, "stress_points", points)
        object.__setattr__(self, "outcomes", outcomes)

    def as_payload(self) -> dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            "config": self.config.as_payload(),
            "stress_points": [point.as_payload() for point in self.stress_points],
            "outcomes": [outcome.as_payload() for outcome in self.outcomes],
            "coupled_stress_matrix": coupled_stress_matrix(self),
            "recovery_timelines": recovery_timelines(self),
            "degradation_slopes": degradation_slopes(self),
            "harness_result": self.harness_result.as_payload(),
        }


def run_coupled_stress_harness(config: CoupledStressConfig) -> CoupledStressResult:
    """Run the coupled bandwidth/viewport/server/client/deadline stress harness."""

    if not isinstance(config, CoupledStressConfig):
        raise CoupledStressHarnessError("config must be a CoupledStressConfig record.")
    points = config.stress_points()
    point_by_id = {point.point_id: point for point in points}
    outcomes: list[CoupledStressOutcome] = []

    def executor(spec: HarnessRunSpec) -> HarnessRunResult:
        point = point_by_id[spec.workload_id]
        outcome = evaluate_coupled_stress_point(
            point,
            method_id=spec.method_id,
            frozen_method_id=config.frozen_method_id,
            seed=spec.seed,
        )
        outcomes.append(outcome)
        return HarnessRunResult(
            spec=spec,
            status="executed",
            metrics=outcome.metric_records(run_id=spec.run_id, split=config.split),
            metadata={"coupled_stress_outcome_id": outcome.outcome_id},
        )

    harness_config = HarnessConfig(
        harness_name="coupled-stress-robustness",
        methods=config.methods,
        workloads=tuple(point.point_id for point in points),
        seeds=config.seeds,
        run_mode=config.run_mode,
        baseline_method_id=config.frozen_method_id,
        fixed_variables={
            **config.fixed_variables,
            "coupled_stress_harness_id": config.harness_id,
            "split": config.split,
            "frozen_method_id": config.frozen_method_id,
            "stress_axes": list(COUPLED_STRESS_AXES),
        },
        comparison_metric_names=COUPLED_STRESS_METRIC_NAMES,
        comparison_group_keys=("workload_id", "seed", "metric_name"),
        output_root=Path(config.output_root) / "harness" if config.output_root is not None else None,
        tags={**config.tags, "split": config.split, "frozen_method_id": config.frozen_method_id},
        metadata={
            "coupled_stress_config": config.as_payload(),
            "frozen_method_manifest": config.frozen_method_manifest.as_payload() if config.frozen_method_manifest is not None else None,
            **config.metadata,
        },
    )
    harness_result = run_harness(harness_config, executor=executor)
    result = CoupledStressResult(
        harness_id=config.harness_id,
        config=config,
        stress_points=points,
        outcomes=tuple(outcomes),
        harness_result=harness_result,
    )
    if config.output_root is not None:
        export_coupled_stress_outputs(config.output_root, result)
    return result


def evaluate_coupled_stress_point(
    point: CoupledStressPoint,
    *,
    method_id: str,
    frozen_method_id: str = "robust-deadline-aware-mpc",
    seed: int = 0,
) -> CoupledStressOutcome:
    """Evaluate one method under one deterministic coupled stress point."""

    if not isinstance(point, CoupledStressPoint):
        raise CoupledStressHarnessError("point must be a CoupledStressPoint record.")
    _require_non_empty(method_id, "method_id")
    _require_non_empty(frozen_method_id, "frozen_method_id")
    seed = _non_negative_int(seed, "seed")
    workload = _workload_profile(point, seed)
    method = _method_profile(method_id, frozen_method_id)
    pressure = _resource_pressure(point.stress_levels, workload, method)
    stress = workload["coupled_stress"]
    resilience = method["resilience"]
    degradation = _clamp01(stress * (1.0 - resilience) + pressure * 0.30)
    hit_rate = _clamp01(0.86 - degradation * 0.62 + method["deadline"] * 0.15 - point.stress_levels["deadline"] * 0.10)
    visible_quality = _clamp01(0.80 - degradation * 0.42 + method["quality"] * 0.14 - point.stress_levels["viewport"] * 0.08)
    qoe = _clamp01(0.56 * hit_rate + 0.34 * visible_quality + 0.10 * (1.0 - pressure))
    recovery = max(0.0, 55.0 + stress * 420.0 + pressure * 260.0 - resilience * 230.0)
    slope = _clamp01(degradation / max(0.10, stress))
    method_role = "frozen_refabr" if method_id == frozen_method_id else "baseline"
    timeline = _recovery_timeline(qoe=qoe, recovery_ms=recovery, stress=stress)
    return CoupledStressOutcome(
        point=point,
        method_id=method_id,
        seed=seed,
        method_role=method_role,
        stress_deadline_hit_qoe=round(qoe, 6),
        stress_recovery_ms=round(recovery, 6),
        stress_degradation_slope=round(slope, 6),
        stress_deadline_hit_rate=round(hit_rate, 6),
        stress_visible_quality=round(visible_quality, 6),
        stress_resource_pressure=round(pressure, 6),
        metadata={
            "stress_axes": dict(point.stress_levels),
            "workload_profile": workload,
            "method_profile": method,
            "recovery_timeline": timeline,
        },
    )


def coupled_stress_matrix(result: CoupledStressResult | Sequence[CoupledStressOutcome]) -> list[dict[str, Any]]:
    """Return aggregate stress-matrix rows by method and stress vector."""

    rows: list[dict[str, Any]] = []
    for (method_id, stress_id), outcomes in _group_by_method_stress(_result_outcomes(result)).items():
        point = outcomes[0].point
        rows.append(
            {
                "method_id": method_id,
                "method_role": outcomes[0].method_role,
                "stress_id": stress_id,
                "stress_levels": dict(point.stress_levels),
                "stress_intensity": _stress_intensity(point.stress_levels),
                "stress_deadline_hit_qoe": _mean(outcome.stress_deadline_hit_qoe for outcome in outcomes),
                "stress_deadline_hit_rate": _mean(outcome.stress_deadline_hit_rate for outcome in outcomes),
                "stress_visible_quality": _mean(outcome.stress_visible_quality for outcome in outcomes),
                "stress_recovery_ms": _mean(outcome.stress_recovery_ms for outcome in outcomes),
                "stress_degradation_slope": _mean(outcome.stress_degradation_slope for outcome in outcomes),
                "stress_resource_pressure": _mean(outcome.stress_resource_pressure for outcome in outcomes),
                "sample_count": len(outcomes),
            }
        )
    return sorted(rows, key=lambda row: (row["stress_intensity"], row["stress_id"], row["method_role"] != "frozen_refabr", row["method_id"]))


def recovery_timelines(result: CoupledStressResult | Sequence[CoupledStressOutcome]) -> list[dict[str, Any]]:
    """Return per-outcome recovery timeline rows."""

    rows: list[dict[str, Any]] = []
    for outcome in _result_outcomes(result):
        for sample in outcome.metadata.get("recovery_timeline", ()):
            rows.append(
                {
                    "method_id": outcome.method_id,
                    "method_role": outcome.method_role,
                    "stress_id": outcome.point.stress_id,
                    "scene_id": outcome.point.scene_id,
                    "trace_id": outcome.point.trace_id,
                    "viewport_id": outcome.point.viewport_id,
                    "device_profile_id": outcome.point.device_profile_id,
                    "seed": outcome.seed,
                    **_plain_json_mapping(sample, "recovery_timeline.sample"),
                }
            )
    return sorted(rows, key=lambda row: (row["method_id"], row["stress_id"], row["seed"], row["elapsed_ms"]))


def degradation_slopes(result: CoupledStressResult | Sequence[CoupledStressOutcome]) -> list[dict[str, Any]]:
    """Return degradation-slope rows for robustness plots."""

    rows: list[dict[str, Any]] = []
    for method_id, outcomes in _group_by_method(_result_outcomes(result)).items():
        ordered = sorted(outcomes, key=lambda outcome: (_stress_intensity(outcome.point.stress_levels), outcome.point.stress_id, outcome.seed))
        for outcome in ordered:
            rows.append(
                {
                    "method_id": method_id,
                    "method_role": outcome.method_role,
                    "stress_id": outcome.point.stress_id,
                    "stress_intensity": _stress_intensity(outcome.point.stress_levels),
                    "stress_degradation_slope": outcome.stress_degradation_slope,
                    "stress_deadline_hit_qoe": outcome.stress_deadline_hit_qoe,
                    "stress_recovery_ms": outcome.stress_recovery_ms,
                    "seed": outcome.seed,
                }
            )
    return rows


def export_coupled_stress_outputs(output_root: str | Path, result: CoupledStressResult) -> tuple[Path, Path, Path, Path, Path]:
    """Write coupled stress raw outcomes and plot-input JSON payloads."""

    if not isinstance(result, CoupledStressResult):
        raise CoupledStressHarnessError("result must be a CoupledStressResult record.")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    outcomes_path = root / "coupled_stress_outcomes.jsonl"
    matrix_path = root / "coupled_stress_matrix.json"
    timelines_path = root / "recovery_timelines.json"
    slopes_path = root / "degradation_slopes.json"
    summary_path = root / "coupled_stress_summary.json"
    _write_text_atomic(
        outcomes_path,
        "".join(json.dumps(outcome.as_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for outcome in result.outcomes),
    )
    _write_text_atomic(matrix_path, _json_payload(coupled_stress_matrix(result)))
    _write_text_atomic(timelines_path, _json_payload(recovery_timelines(result)))
    _write_text_atomic(slopes_path, _json_payload(degradation_slopes(result)))
    _write_text_atomic(summary_path, _json_payload(result.as_payload()))
    return outcomes_path, matrix_path, timelines_path, slopes_path, summary_path


def _workload_profile(point: CoupledStressPoint, seed: int) -> dict[str, float]:
    scene_complexity = _hash_fraction("scene", point.scene_id, seed)
    trace_variance = _hash_fraction("trace", point.trace_id, point.stress_id, seed)
    viewport_variance = _hash_fraction("viewport", point.viewport_id, point.stress_id, seed)
    device_headroom = _device_headroom(point.device_profile_id)
    stress_intensity = _stress_intensity(point.stress_levels)
    coupled_stress = _clamp01(
        stress_intensity * 0.62
        + point.stress_levels["bandwidth"] * trace_variance * 0.12
        + point.stress_levels["viewport"] * viewport_variance * 0.10
        + point.stress_levels["server"] * scene_complexity * 0.08
        + point.stress_levels["client"] * (1.0 - device_headroom) * 0.08
    )
    return {
        "scene_complexity": scene_complexity,
        "trace_variance": trace_variance,
        "viewport_variance": viewport_variance,
        "device_headroom": device_headroom,
        "stress_intensity": stress_intensity,
        "coupled_stress": coupled_stress,
    }


def _method_profile(method_id: str, frozen_method_id: str) -> dict[str, float]:
    if method_id == frozen_method_id:
        return {"quality": 0.24, "deadline": 0.27, "resilience": 0.76, "resource_flex": 0.72}
    profiles = {
        "deadline-aware-knapsack-allocator": {"quality": 0.20, "deadline": 0.20, "resilience": 0.61, "resource_flex": 0.62},
        "virtual-queue-deadline-controller": {"quality": 0.17, "deadline": 0.24, "resilience": 0.66, "resource_flex": 0.68},
        "robust-mpc-joint-space": {"quality": 0.21, "deadline": 0.18, "resilience": 0.58, "resource_flex": 0.58},
        "bola-slack-adapted": {"quality": 0.15, "deadline": 0.16, "resilience": 0.48, "resource_flex": 0.44},
        "deadline-greedy": {"quality": 0.08, "deadline": 0.22, "resilience": 0.42, "resource_flex": 0.50},
    }
    return profiles.get(method_id, {"quality": 0.12, "deadline": 0.12, "resilience": 0.40, "resource_flex": 0.40})


def _resource_pressure(stress_levels: Mapping[str, float], workload: Mapping[str, float], method: Mapping[str, float]) -> float:
    pressure = (
        stress_levels["bandwidth"] * 0.30
        + stress_levels["server"] * 0.23
        + stress_levels["client"] * 0.20
        + stress_levels["deadline"] * 0.17
        + stress_levels["viewport"] * 0.10
    )
    pressure += workload["scene_complexity"] * stress_levels["server"] * 0.12
    pressure += (1.0 - workload["device_headroom"]) * stress_levels["client"] * 0.12
    pressure -= method["resource_flex"] * 0.20
    return _clamp01(pressure)


def _recovery_timeline(*, qoe: float, recovery_ms: float, stress: float) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    for fraction in (0.0, 0.25, 0.50, 0.75, 1.0):
        elapsed = recovery_ms * fraction
        recovered = 1.0 if recovery_ms == 0.0 else min(1.0, elapsed / recovery_ms)
        quality = _clamp01(qoe - stress * 0.14 * (1.0 - recovered) + recovered * 0.06)
        samples.append(
            {
                "elapsed_ms": round(elapsed, 6),
                "recovered_fraction": round(recovered, 6),
                "deadline_hit_qoe": round(quality, 6),
            }
        )
    return samples


def _stress_intensity(stress_levels: Mapping[str, float]) -> float:
    return _clamp01(sum(stress_levels[axis] for axis in COUPLED_STRESS_AXES) / len(COUPLED_STRESS_AXES))


def _device_headroom(device_profile_id: str) -> float:
    lowered = device_profile_id.lower()
    if "server" in lowered or "edge" in lowered:
        return 0.95
    if "desktop" in lowered:
        return 0.78
    if "laptop" in lowered:
        return 0.62
    if "mobile" in lowered:
        return 0.38
    return 0.55


def _coerce_frozen_manifest(value: FrozenMethodManifest | Mapping[str, Any] | None) -> FrozenMethodManifest | None:
    if value is None:
        return None
    if isinstance(value, FrozenMethodManifest):
        return value
    if not isinstance(value, MappingABC):
        raise CoupledStressHarnessError("frozen_method_manifest must be a FrozenMethodManifest or mapping.")
    try:
        return FrozenMethodManifest(**value)
    except TypeError as exc:
        raise CoupledStressHarnessError(f"Malformed frozen_method_manifest: {exc}") from exc


def _stress_matrix_tuple(values: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CoupledStressHarnessError("stress_matrix must be a sequence of mappings.")
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, MappingABC):
            raise CoupledStressHarnessError("stress_matrix entries must be mappings.")
        stress_id = str(value.get("stress_id", "")).strip()
        _require_non_empty(stress_id, f"stress_matrix[{index}].stress_id")
        if stress_id in seen:
            raise CoupledStressHarnessError("stress_matrix must not contain duplicate stress_id values.")
        seen.add(stress_id)
        levels = _stress_levels_mapping(value.get("stress_levels"), f"stress_matrix[{index}].stress_levels")
        parsed.append({"stress_id": stress_id, "stress_levels": levels})
    if not parsed:
        raise CoupledStressHarnessError("stress_matrix must not be empty.")
    return tuple(parsed)


def _stress_levels_mapping(value: Any, field_name: str) -> Mapping[str, float]:
    if not isinstance(value, MappingABC):
        raise CoupledStressHarnessError(f"{field_name} must be a mapping.")
    missing = [axis for axis in COUPLED_STRESS_AXES if axis not in value]
    if missing:
        raise CoupledStressHarnessError(f"{field_name} must include stress axes: {', '.join(COUPLED_STRESS_AXES)}.")
    return {axis: _unit_interval(value[axis], f"{field_name}.{axis}") for axis in COUPLED_STRESS_AXES}


def _result_outcomes(result: CoupledStressResult | Sequence[CoupledStressOutcome]) -> tuple[CoupledStressOutcome, ...]:
    outcomes = result.outcomes if isinstance(result, CoupledStressResult) else tuple(result)
    for outcome in outcomes:
        if not isinstance(outcome, CoupledStressOutcome):
            raise CoupledStressHarnessError("outcomes must contain CoupledStressOutcome records.")
    return tuple(outcomes)


def _group_by_method(outcomes: Sequence[CoupledStressOutcome]) -> dict[str, tuple[CoupledStressOutcome, ...]]:
    grouped: dict[str, list[CoupledStressOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.method_id, []).append(outcome)
    return {method_id: tuple(method_outcomes) for method_id, method_outcomes in sorted(grouped.items())}


def _group_by_method_stress(outcomes: Sequence[CoupledStressOutcome]) -> dict[tuple[str, str], tuple[CoupledStressOutcome, ...]]:
    grouped: dict[tuple[str, str], list[CoupledStressOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault((outcome.method_id, outcome.point.stress_id), []).append(outcome)
    return {key: tuple(grouped[key]) for key in sorted(grouped)}


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
        "metric_id": f"coupled-stress-metric-{stable_config_id({'metric_name': metric_name, 'value': value, 'tags': dict(tags)})}",
    }
    return MetricRecord(metric_name=metric_name, value=value, unit=unit, tags=tags, split=split, metadata=metric_metadata)


def _json_payload(value: Any) -> str:
    return json.dumps(_to_payload(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise CoupledStressHarnessError(f"Failed to write coupled stress output {path}: {exc}") from exc


def _string_tuple(values: Sequence[str], field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CoupledStressHarnessError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    if not parsed and not allow_empty:
        raise CoupledStressHarnessError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _string_mapping(value: Mapping[str, str], field_name: str) -> Mapping[str, str]:
    if not isinstance(value, MappingABC):
        raise CoupledStressHarnessError(f"{field_name} must be a mapping.")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        _require_non_empty(str(key), f"{field_name} key")
        _require_non_empty(item, f"{field_name}.{key}")
        parsed[str(key)] = item
    return {key: parsed[key] for key in sorted(parsed)}


def _int_tuple(values: Sequence[int], field_name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CoupledStressHarnessError(f"{field_name} must be a sequence of integers.")
    parsed: list[int] = []
    seen: set[int] = set()
    for value in values:
        parsed_value = _non_negative_int(value, field_name)
        if parsed_value not in seen:
            parsed.append(parsed_value)
            seen.add(parsed_value)
    if not parsed:
        raise CoupledStressHarnessError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise CoupledStressHarnessError(f"{field_name} must be a mapping.")
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
        raise CoupledStressHarnessError(f"{field_name} must be a non-empty string.")


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise CoupledStressHarnessError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CoupledStressHarnessError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise CoupledStressHarnessError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0 or parsed > 1.0:
        raise CoupledStressHarnessError(f"{field_name} must be in [0, 1].")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0:
        raise CoupledStressHarnessError(f"{field_name} must be non-negative.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise CoupledStressHarnessError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CoupledStressHarnessError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise CoupledStressHarnessError(f"{field_name} must be finite.")
    return parsed


def _mean(values: Sequence[float]) -> float:
    parsed = tuple(_finite_float(value, "mean.value") for value in values)
    if not parsed:
        return 0.0
    return sum(parsed) / len(parsed)


def _hash_fraction(*parts: object) -> float:
    digest = stable_config_id({"parts": [str(part) for part in parts]})
    return int(digest[:8], 16) / 0xFFFFFFFF


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


__all__ = [
    "COUPLED_STRESS_AXES",
    "COUPLED_STRESS_BASELINE_METHOD_IDS",
    "COUPLED_STRESS_METRIC_NAMES",
    "CoupledStressConfig",
    "CoupledStressHarnessError",
    "CoupledStressOutcome",
    "CoupledStressPoint",
    "CoupledStressResult",
    "coupled_stress_matrix",
    "degradation_slopes",
    "evaluate_coupled_stress_point",
    "export_coupled_stress_outputs",
    "recovery_timelines",
    "run_coupled_stress_harness",
]
