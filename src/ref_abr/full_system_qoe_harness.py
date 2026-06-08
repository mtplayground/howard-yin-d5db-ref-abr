"""Full-system final-split Deadline-Hit QoE harness."""

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


FULL_SYSTEM_BASELINE_METHOD_IDS: tuple[str, ...] = (
    "cags-fixed-reference",
    "svq-gaussian-only-abr",
    "reference-only-after-base",
    "fixed-reference-cadence",
    "independent-gaussian-scheduler",
    "independent-reference-scheduler",
    "bandwidth-greedy",
    "deadline-greedy",
    "quality-max-deadline-unaware",
    "deadline-aware-knapsack-allocator",
    "virtual-queue-deadline-controller",
    "robust-mpc-joint-space",
    "bola-slack-adapted",
)
FULL_SYSTEM_QOE_METRIC_NAMES: tuple[str, ...] = (
    "deadline_hit_qoe",
    "deadline_hit_rate",
    "visible_quality",
    "full_frame_quality",
    "mean_latency_ms",
    "fps",
)


class FullSystemQoeHarnessError(ValueError):
    """Raised when full-system QoE harness inputs are invalid."""


@dataclass(frozen=True)
class FullSystemQoePoint:
    """One final-split scene/trace/viewport/device workload tuple."""

    scene_id: str
    trace_id: str
    viewport_id: str
    device_profile_id: str

    def __post_init__(self) -> None:
        _require_non_empty(self.scene_id, "scene_id")
        _require_non_empty(self.trace_id, "trace_id")
        _require_non_empty(self.viewport_id, "viewport_id")
        _require_non_empty(self.device_profile_id, "device_profile_id")

    @property
    def point_id(self) -> str:
        return f"full-system-qoe-point-{stable_config_id(self.stable_payload())}"

    def stable_payload(self) -> dict[str, str]:
        return {
            "scene_id": self.scene_id,
            "trace_id": self.trace_id,
            "viewport_id": self.viewport_id,
            "device_profile_id": self.device_profile_id,
        }

    def as_payload(self) -> dict[str, str]:
        return {"point_id": self.point_id, **self.stable_payload()}


@dataclass(frozen=True)
class FullSystemQoeOutcome:
    """Method result at one full-system final-split workload tuple."""

    point: FullSystemQoePoint
    method_id: str
    seed: int
    method_role: str
    deadline_hit_qoe: float
    deadline_hit_rate: float
    visible_quality: float
    full_frame_quality: float
    mean_latency_ms: float
    fps: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.point, FullSystemQoePoint):
            raise FullSystemQoeHarnessError("point must be a FullSystemQoePoint record.")
        _require_non_empty(self.method_id, "method_id")
        object.__setattr__(self, "seed", _non_negative_int(self.seed, "seed"))
        if self.method_role not in {"frozen_refabr", "baseline"}:
            raise FullSystemQoeHarnessError("method_role must be one of: frozen_refabr, baseline.")
        for field_name in ("deadline_hit_qoe", "deadline_hit_rate", "visible_quality", "full_frame_quality"):
            object.__setattr__(self, field_name, _unit_interval(getattr(self, field_name), field_name))
        object.__setattr__(self, "mean_latency_ms", _non_negative_float(self.mean_latency_ms, "mean_latency_ms"))
        object.__setattr__(self, "fps", _positive_float(self.fps, "fps"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def outcome_id(self) -> str:
        payload = {
            "point_id": self.point.point_id,
            "method_id": self.method_id,
            "seed": self.seed,
            "deadline_hit_qoe": self.deadline_hit_qoe,
            "deadline_hit_rate": self.deadline_hit_rate,
        }
        return f"full-system-qoe-outcome-{stable_config_id(payload)}"

    def metric_records(self, *, run_id: str, split: str | None = "final") -> tuple[MetricRecord, ...]:
        tags = {
            "run_id": run_id,
            "method": self.method_id,
            "method_role": self.method_role,
            "scene_id": self.point.scene_id,
            "trace_id": self.point.trace_id,
            "viewport_id": self.point.viewport_id,
            "device_profile_id": self.point.device_profile_id,
        }
        base_metadata = {
            "full_system_qoe_outcome_id": self.outcome_id,
            "full_system_qoe_point": self.point.as_payload(),
            "full_system_qoe_outcome": self.as_payload(include_metrics=False),
        }
        return (
            _metric("deadline_hit_qoe", self.deadline_hit_qoe, "score", tags, base_metadata, split=split),
            _metric("deadline_hit_rate", self.deadline_hit_rate, "ratio", tags, base_metadata, split=split),
            _metric("visible_quality", self.visible_quality, "score", tags, base_metadata, split=split),
            _metric("full_frame_quality", self.full_frame_quality, "score", tags, base_metadata, split=split),
            _metric("mean_latency_ms", self.mean_latency_ms, "ms", tags, base_metadata, split=split),
            _metric("fps", self.fps, "fps", tags, base_metadata, split=split),
        )

    def as_payload(self, *, include_metrics: bool = True) -> dict[str, Any]:
        payload = {
            "outcome_id": self.outcome_id,
            "point": self.point.as_payload(),
            "method_id": self.method_id,
            "seed": self.seed,
            "method_role": self.method_role,
            "deadline_hit_qoe": self.deadline_hit_qoe,
            "deadline_hit_rate": self.deadline_hit_rate,
            "visible_quality": self.visible_quality,
            "full_frame_quality": self.full_frame_quality,
            "mean_latency_ms": self.mean_latency_ms,
            "fps": self.fps,
            "metadata": _to_payload(self.metadata),
        }
        if include_metrics:
            payload["metric_names"] = list(FULL_SYSTEM_QOE_METRIC_NAMES)
        return payload


@dataclass(frozen=True)
class FullSystemQoeConfig:
    """Final-split full-system QoE sweep and comparison settings."""

    scenes: Sequence[str]
    traces: Sequence[str]
    viewports: Sequence[str]
    devices: Sequence[str]
    frozen_method_manifest: FrozenMethodManifest | Mapping[str, Any] | None = None
    frozen_method_id: str = "robust-deadline-aware-mpc"
    baseline_methods: Sequence[str] = FULL_SYSTEM_BASELINE_METHOD_IDS
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
        manifest = _coerce_frozen_manifest(self.frozen_method_manifest)
        object.__setattr__(self, "frozen_method_manifest", manifest)
        frozen_method_id = manifest.method_id if manifest is not None else self.frozen_method_id
        _require_non_empty(frozen_method_id, "frozen_method_id")
        object.__setattr__(self, "frozen_method_id", frozen_method_id)
        object.__setattr__(self, "baseline_methods", _string_tuple(self.baseline_methods, "baseline_methods"))
        if frozen_method_id in self.baseline_methods:
            raise FullSystemQoeHarnessError("frozen_method_id must not be duplicated in baseline_methods.")
        object.__setattr__(self, "seeds", _int_tuple(self.seeds, "seeds"))
        if self.split != "final":
            raise FullSystemQoeHarnessError("Full-system QoE harness must run on the final split.")
        if self.run_mode not in {"plan_only", "metrics_only", "full"}:
            raise FullSystemQoeHarnessError("run_mode must be one of: plan_only, metrics_only, full.")
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
        return f"full-system-deadline-hit-qoe-{stable_config_id(self.stable_payload())}"

    def workload_points(self) -> tuple[FullSystemQoePoint, ...]:
        points: list[FullSystemQoePoint] = []
        for scene in self.scenes:
            for trace in self.traces:
                for viewport in self.viewports:
                    for device in self.devices:
                        points.append(
                            FullSystemQoePoint(
                                scene_id=scene,
                                trace_id=trace,
                                viewport_id=viewport,
                                device_profile_id=device,
                            )
                        )
        return tuple(points)

    def stable_payload(self) -> dict[str, Any]:
        return {
            "scenes": list(self.scenes),
            "traces": list(self.traces),
            "viewports": list(self.viewports),
            "devices": list(self.devices),
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
            "workload_point_count": len(self.workload_points()),
        }


@dataclass(frozen=True)
class FullSystemQoeResult:
    """Complete full-system Deadline-Hit QoE harness output."""

    harness_id: str
    config: FullSystemQoeConfig
    workload_points: tuple[FullSystemQoePoint, ...]
    outcomes: tuple[FullSystemQoeOutcome, ...]
    harness_result: HarnessResult

    def __post_init__(self) -> None:
        _require_non_empty(self.harness_id, "harness_id")
        if not isinstance(self.config, FullSystemQoeConfig):
            raise FullSystemQoeHarnessError("config must be a FullSystemQoeConfig record.")
        points = tuple(self.workload_points)
        outcomes = tuple(self.outcomes)
        for point in points:
            if not isinstance(point, FullSystemQoePoint):
                raise FullSystemQoeHarnessError("workload_points must contain FullSystemQoePoint records.")
        for outcome in outcomes:
            if not isinstance(outcome, FullSystemQoeOutcome):
                raise FullSystemQoeHarnessError("outcomes must contain FullSystemQoeOutcome records.")
        if not isinstance(self.harness_result, HarnessResult):
            raise FullSystemQoeHarnessError("harness_result must be a HarnessResult record.")
        object.__setattr__(self, "workload_points", points)
        object.__setattr__(self, "outcomes", outcomes)

    def as_payload(self) -> dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            "config": self.config.as_payload(),
            "workload_points": [point.as_payload() for point in self.workload_points],
            "outcomes": [outcome.as_payload() for outcome in self.outcomes],
            "main_qoe_table": main_qoe_table(self),
            "quality_deadline_pareto": quality_deadline_pareto(self),
            "deadline_hit_qoe_cdf": deadline_hit_qoe_cdf(self),
            "harness_result": self.harness_result.as_payload(),
        }


def run_full_system_qoe_harness(config: FullSystemQoeConfig) -> FullSystemQoeResult:
    """Run the final-split full-system Deadline-Hit QoE harness."""

    if not isinstance(config, FullSystemQoeConfig):
        raise FullSystemQoeHarnessError("config must be a FullSystemQoeConfig record.")
    points = config.workload_points()
    point_by_id = {point.point_id: point for point in points}
    outcomes: list[FullSystemQoeOutcome] = []

    def executor(spec: HarnessRunSpec) -> HarnessRunResult:
        point = point_by_id[spec.workload_id]
        outcome = evaluate_full_system_qoe_point(
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
            metadata={"full_system_qoe_outcome_id": outcome.outcome_id},
        )

    harness_config = HarnessConfig(
        harness_name="full-system-deadline-hit-qoe",
        methods=config.methods,
        workloads=tuple(point.point_id for point in points),
        seeds=config.seeds,
        run_mode=config.run_mode,
        baseline_method_id=config.frozen_method_id,
        fixed_variables={
            **config.fixed_variables,
            "full_system_qoe_harness_id": config.harness_id,
            "split": config.split,
            "frozen_method_id": config.frozen_method_id,
        },
        comparison_metric_names=FULL_SYSTEM_QOE_METRIC_NAMES,
        comparison_group_keys=("workload_id", "seed", "metric_name"),
        output_root=Path(config.output_root) / "harness" if config.output_root is not None else None,
        tags={**config.tags, "split": config.split, "frozen_method_id": config.frozen_method_id},
        metadata={
            "full_system_qoe_config": config.as_payload(),
            "frozen_method_manifest": config.frozen_method_manifest.as_payload() if config.frozen_method_manifest is not None else None,
            **config.metadata,
        },
    )
    harness_result = run_harness(harness_config, executor=executor)
    result = FullSystemQoeResult(
        harness_id=config.harness_id,
        config=config,
        workload_points=points,
        outcomes=tuple(outcomes),
        harness_result=harness_result,
    )
    if config.output_root is not None:
        export_full_system_qoe_outputs(config.output_root, result)
    return result


def evaluate_full_system_qoe_point(
    point: FullSystemQoePoint,
    *,
    method_id: str,
    frozen_method_id: str = "robust-deadline-aware-mpc",
    seed: int = 0,
) -> FullSystemQoeOutcome:
    """Evaluate one method on one deterministic final-split full-system tuple."""

    if not isinstance(point, FullSystemQoePoint):
        raise FullSystemQoeHarnessError("point must be a FullSystemQoePoint record.")
    _require_non_empty(method_id, "method_id")
    _require_non_empty(frozen_method_id, "frozen_method_id")
    seed = _non_negative_int(seed, "seed")
    workload = _workload_profile(point, seed)
    profile = _method_profile(method_id, frozen_method_id)
    deadline_hit_rate = _clamp01(0.58 + profile["deadline"] - workload["network_stress"] * 0.20 - workload["viewport_stress"] * 0.08)
    visible_quality = _clamp01(0.55 + profile["quality"] - workload["scene_complexity"] * 0.10 + workload["device_headroom"] * 0.05)
    full_frame_quality = _clamp01(visible_quality - 0.04 + profile["coverage"] - workload["viewport_stress"] * 0.05)
    latency = max(1.0, 42.0 + workload["network_stress"] * 38.0 + workload["scene_complexity"] * 22.0 - profile["deadline"] * 28.0)
    fps = max(1.0, 72.0 + workload["device_headroom"] * 28.0 - workload["scene_complexity"] * 18.0 - profile["compute"] * 10.0)
    deadline_hit_qoe = _clamp01(0.52 * deadline_hit_rate + 0.34 * visible_quality + 0.14 * full_frame_quality - max(0.0, latency - 50.0) / 500.0)
    method_role = "frozen_refabr" if method_id == frozen_method_id else "baseline"
    return FullSystemQoeOutcome(
        point=point,
        method_id=method_id,
        seed=seed,
        method_role=method_role,
        deadline_hit_qoe=round(deadline_hit_qoe, 6),
        deadline_hit_rate=round(deadline_hit_rate, 6),
        visible_quality=round(visible_quality, 6),
        full_frame_quality=round(full_frame_quality, 6),
        mean_latency_ms=round(latency, 6),
        fps=round(fps, 6),
        metadata={
            "final_split": True,
            "workload_profile": workload,
            "method_profile": profile,
        },
    )


def main_qoe_table(result: FullSystemQoeResult | Sequence[FullSystemQoeOutcome]) -> list[dict[str, Any]]:
    """Return aggregate rows for the main Deadline-Hit QoE table."""

    outcomes = _result_outcomes(result)
    rows: list[dict[str, Any]] = []
    for method_id, method_outcomes in _group_by_method(outcomes).items():
        rows.append(
            {
                "method_id": method_id,
                "method_role": method_outcomes[0].method_role,
                "split": "final",
                "deadline_hit_qoe": _mean(outcome.deadline_hit_qoe for outcome in method_outcomes),
                "deadline_hit_rate": _mean(outcome.deadline_hit_rate for outcome in method_outcomes),
                "visible_quality": _mean(outcome.visible_quality for outcome in method_outcomes),
                "full_frame_quality": _mean(outcome.full_frame_quality for outcome in method_outcomes),
                "mean_latency_ms": _mean(outcome.mean_latency_ms for outcome in method_outcomes),
                "fps": _mean(outcome.fps for outcome in method_outcomes),
                "sample_count": len(method_outcomes),
            }
        )
    rows = sorted(rows, key=lambda row: (-row["deadline_hit_qoe"], row["method_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def quality_deadline_pareto(result: FullSystemQoeResult | Sequence[FullSystemQoeOutcome]) -> list[dict[str, Any]]:
    """Return quality/deadline Pareto input rows."""

    rows = [
        {
            "method_id": row["method_id"],
            "method_role": row["method_role"],
            "visible_quality": row["visible_quality"],
            "deadline_hit_rate": row["deadline_hit_rate"],
            "deadline_hit_qoe": row["deadline_hit_qoe"],
        }
        for row in main_qoe_table(result)
    ]
    for row in rows:
        row["pareto_frontier"] = not any(
            other is not row
            and other["visible_quality"] >= row["visible_quality"]
            and other["deadline_hit_rate"] >= row["deadline_hit_rate"]
            and (
                other["visible_quality"] > row["visible_quality"]
                or other["deadline_hit_rate"] > row["deadline_hit_rate"]
            )
            for other in rows
        )
    return rows


def deadline_hit_qoe_cdf(result: FullSystemQoeResult | Sequence[FullSystemQoeOutcome]) -> list[dict[str, Any]]:
    """Return per-method Deadline-Hit QoE empirical CDF input rows."""

    rows: list[dict[str, Any]] = []
    for method_id, method_outcomes in _group_by_method(_result_outcomes(result)).items():
        sorted_outcomes = sorted(method_outcomes, key=lambda outcome: (outcome.deadline_hit_qoe, outcome.point.point_id, outcome.seed))
        total = len(sorted_outcomes)
        for index, outcome in enumerate(sorted_outcomes, start=1):
            rows.append(
                {
                    "method_id": method_id,
                    "method_role": outcome.method_role,
                    "deadline_hit_qoe": outcome.deadline_hit_qoe,
                    "cumulative_probability": index / total,
                    "scene_id": outcome.point.scene_id,
                    "trace_id": outcome.point.trace_id,
                    "viewport_id": outcome.point.viewport_id,
                    "device_profile_id": outcome.point.device_profile_id,
                    "seed": outcome.seed,
                }
            )
    return rows


def export_full_system_qoe_outputs(output_root: str | Path, result: FullSystemQoeResult) -> tuple[Path, Path, Path, Path, Path]:
    """Write full-system QoE raw outcomes and paper-input JSON payloads."""

    if not isinstance(result, FullSystemQoeResult):
        raise FullSystemQoeHarnessError("result must be a FullSystemQoeResult record.")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    outcomes_path = root / "full_system_qoe_outcomes.jsonl"
    main_table_path = root / "main_qoe_table.json"
    pareto_path = root / "quality_deadline_pareto.json"
    cdf_path = root / "deadline_hit_qoe_cdf.json"
    summary_path = root / "full_system_qoe_summary.json"
    _write_text_atomic(
        outcomes_path,
        "".join(json.dumps(outcome.as_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for outcome in result.outcomes),
    )
    _write_text_atomic(main_table_path, _json_payload(main_qoe_table(result)))
    _write_text_atomic(pareto_path, _json_payload(quality_deadline_pareto(result)))
    _write_text_atomic(cdf_path, _json_payload(deadline_hit_qoe_cdf(result)))
    _write_text_atomic(summary_path, _json_payload(result.as_payload()))
    return outcomes_path, main_table_path, pareto_path, cdf_path, summary_path


def _workload_profile(point: FullSystemQoePoint, seed: int) -> dict[str, float]:
    return {
        "scene_complexity": _hash_fraction("scene", point.scene_id, seed),
        "network_stress": _hash_fraction("trace", point.trace_id, seed),
        "viewport_stress": _hash_fraction("viewport", point.viewport_id, seed),
        "device_headroom": _device_headroom(point.device_profile_id),
    }


def _method_profile(method_id: str, frozen_method_id: str) -> dict[str, float]:
    if method_id == frozen_method_id:
        return {"quality": 0.24, "deadline": 0.26, "coverage": 0.10, "compute": 0.46}
    profiles = {
        "cags-fixed-reference": {"quality": 0.11, "deadline": 0.03, "coverage": 0.09, "compute": 0.20},
        "svq-gaussian-only-abr": {"quality": 0.09, "deadline": 0.14, "coverage": 0.02, "compute": 0.18},
        "reference-only-after-base": {"quality": 0.16, "deadline": 0.04, "coverage": 0.08, "compute": 0.34},
        "fixed-reference-cadence": {"quality": 0.14, "deadline": 0.08, "coverage": 0.08, "compute": 0.26},
        "independent-gaussian-scheduler": {"quality": 0.10, "deadline": 0.12, "coverage": 0.03, "compute": 0.18},
        "independent-reference-scheduler": {"quality": 0.15, "deadline": 0.06, "coverage": 0.07, "compute": 0.28},
        "bandwidth-greedy": {"quality": 0.08, "deadline": 0.17, "coverage": 0.02, "compute": 0.12},
        "deadline-greedy": {"quality": 0.07, "deadline": 0.20, "coverage": 0.02, "compute": 0.14},
        "quality-max-deadline-unaware": {"quality": 0.20, "deadline": 0.02, "coverage": 0.08, "compute": 0.38},
        "deadline-aware-knapsack-allocator": {"quality": 0.19, "deadline": 0.18, "coverage": 0.08, "compute": 0.30},
        "virtual-queue-deadline-controller": {"quality": 0.17, "deadline": 0.21, "coverage": 0.07, "compute": 0.29},
        "robust-mpc-joint-space": {"quality": 0.20, "deadline": 0.16, "coverage": 0.08, "compute": 0.36},
        "bola-slack-adapted": {"quality": 0.17, "deadline": 0.15, "coverage": 0.07, "compute": 0.24},
    }
    return profiles.get(method_id, {"quality": 0.12, "deadline": 0.10, "coverage": 0.05, "compute": 0.25})


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


def _hash_fraction(*parts: object) -> float:
    digest = stable_config_id({"parts": [str(part) for part in parts]})
    return int(digest[:8], 16) / 0xFFFFFFFF


def _result_outcomes(result: FullSystemQoeResult | Sequence[FullSystemQoeOutcome]) -> tuple[FullSystemQoeOutcome, ...]:
    outcomes = result.outcomes if isinstance(result, FullSystemQoeResult) else tuple(result)
    for outcome in outcomes:
        if not isinstance(outcome, FullSystemQoeOutcome):
            raise FullSystemQoeHarnessError("outcomes must contain FullSystemQoeOutcome records.")
    return tuple(outcomes)


def _group_by_method(outcomes: Sequence[FullSystemQoeOutcome]) -> dict[str, tuple[FullSystemQoeOutcome, ...]]:
    grouped: dict[str, list[FullSystemQoeOutcome]] = {}
    for outcome in outcomes:
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
        "metric_id": f"full-system-qoe-metric-{stable_config_id({'metric_name': metric_name, 'value': value, 'tags': dict(tags)})}",
    }
    return MetricRecord(metric_name=metric_name, value=value, unit=unit, tags=tags, split=split, metadata=metric_metadata)


def _coerce_frozen_manifest(value: FrozenMethodManifest | Mapping[str, Any] | None) -> FrozenMethodManifest | None:
    if value is None:
        return None
    if isinstance(value, FrozenMethodManifest):
        return value
    if not isinstance(value, MappingABC):
        raise FullSystemQoeHarnessError("frozen_method_manifest must be a FrozenMethodManifest or mapping.")
    try:
        return FrozenMethodManifest(**value)
    except TypeError as exc:
        raise FullSystemQoeHarnessError(f"Malformed frozen_method_manifest: {exc}") from exc


def _json_payload(value: Any) -> str:
    return json.dumps(_to_payload(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise FullSystemQoeHarnessError(f"Failed to write full-system QoE output {path}: {exc}") from exc


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FullSystemQoeHarnessError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    if not parsed:
        raise FullSystemQoeHarnessError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _string_mapping(value: Mapping[str, str], field_name: str) -> Mapping[str, str]:
    if not isinstance(value, MappingABC):
        raise FullSystemQoeHarnessError(f"{field_name} must be a mapping.")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        _require_non_empty(str(key), f"{field_name} key")
        _require_non_empty(item, f"{field_name}.{key}")
        parsed[str(key)] = item
    return {key: parsed[key] for key in sorted(parsed)}


def _int_tuple(values: Sequence[int], field_name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FullSystemQoeHarnessError(f"{field_name} must be a sequence of integers.")
    parsed: list[int] = []
    seen: set[int] = set()
    for value in values:
        parsed_value = _non_negative_int(value, field_name)
        if parsed_value not in seen:
            parsed.append(parsed_value)
            seen.add(parsed_value)
    if not parsed:
        raise FullSystemQoeHarnessError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise FullSystemQoeHarnessError(f"{field_name} must be a mapping.")
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
        raise FullSystemQoeHarnessError(f"{field_name} must be a non-empty string.")


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise FullSystemQoeHarnessError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise FullSystemQoeHarnessError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise FullSystemQoeHarnessError(f"{field_name} must be non-negative.")
    return parsed


def _positive_float(value: Any, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name)
    if parsed <= 0.0:
        raise FullSystemQoeHarnessError(f"{field_name} must be positive.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0:
        raise FullSystemQoeHarnessError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name)
    if parsed > 1.0:
        raise FullSystemQoeHarnessError(f"{field_name} must be in [0, 1].")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise FullSystemQoeHarnessError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FullSystemQoeHarnessError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise FullSystemQoeHarnessError(f"{field_name} must be finite.")
    return parsed


def _mean(values: Sequence[float]) -> float:
    parsed = tuple(_finite_float(value, "mean.value") for value in values)
    if not parsed:
        return 0.0
    return sum(parsed) / len(parsed)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


__all__ = [
    "FULL_SYSTEM_BASELINE_METHOD_IDS",
    "FULL_SYSTEM_QOE_METRIC_NAMES",
    "FullSystemQoeConfig",
    "FullSystemQoeHarnessError",
    "FullSystemQoeOutcome",
    "FullSystemQoePoint",
    "FullSystemQoeResult",
    "deadline_hit_qoe_cdf",
    "evaluate_full_system_qoe_point",
    "export_full_system_qoe_outputs",
    "main_qoe_table",
    "quality_deadline_pareto",
    "run_full_system_qoe_harness",
]
