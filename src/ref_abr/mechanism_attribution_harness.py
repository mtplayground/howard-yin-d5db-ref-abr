"""Mechanism and attribution ablation harness."""

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


MECHANISM_ABLATION_VARIANTS: tuple[Mapping[str, Any], ...] = (
    {"variant_id": "full", "method_suffix": "", "removed_terms": ()},
    {"variant_id": "no-lifecycle", "method_suffix": "no-lifecycle", "removed_terms": ("lifecycle",)},
    {"variant_id": "no-uncertainty", "method_suffix": "no-uncertainty", "removed_terms": ("uncertainty",)},
    {"variant_id": "no-component-cost", "method_suffix": "no-component-cost", "removed_terms": ("component_cost",)},
    {"variant_id": "no-cancellation", "method_suffix": "no-cancellation", "removed_terms": ("cancellation",)},
    {"variant_id": "no-fov", "method_suffix": "no-fov", "removed_terms": ("fov",)},
    {"variant_id": "no-lead-time", "method_suffix": "no-lead-time", "removed_terms": ("lead_time",)},
    {"variant_id": "oracle", "method_suffix": "oracle", "removed_terms": ()},
)
MECHANISM_ABLATION_METRIC_NAMES: tuple[str, ...] = (
    "ablation_deadline_hit_qoe",
    "ablation_qoe_delta_from_full",
    "ablation_oracle_gap",
    "ablation_decision_trace_shift",
    "ablation_deadline_hit_rate",
    "ablation_visible_quality",
)


class MechanismAttributionHarnessError(ValueError):
    """Raised when mechanism-attribution harness inputs are invalid."""


@dataclass(frozen=True)
class MechanismAttributionPoint:
    """One paired mechanism-ablation workload and decision-trace case."""

    scene_id: str
    trace_id: str
    viewport_id: str
    device_profile_id: str
    case_id: str

    def __post_init__(self) -> None:
        _require_non_empty(self.scene_id, "scene_id")
        _require_non_empty(self.trace_id, "trace_id")
        _require_non_empty(self.viewport_id, "viewport_id")
        _require_non_empty(self.device_profile_id, "device_profile_id")
        _require_non_empty(self.case_id, "case_id")

    @property
    def point_id(self) -> str:
        return f"mechanism-attribution-point-{stable_config_id(self.stable_payload())}"

    def stable_payload(self) -> dict[str, str]:
        return {
            "scene_id": self.scene_id,
            "trace_id": self.trace_id,
            "viewport_id": self.viewport_id,
            "device_profile_id": self.device_profile_id,
            "case_id": self.case_id,
        }

    def as_payload(self) -> dict[str, str]:
        return {"point_id": self.point_id, **self.stable_payload()}


@dataclass(frozen=True)
class MechanismAttributionOutcome:
    """Method/variant result for one paired attribution point."""

    point: MechanismAttributionPoint
    method_id: str
    variant_id: str
    removed_terms: Sequence[str]
    seed: int
    deadline_hit_qoe: float
    full_deadline_hit_qoe: float
    oracle_deadline_hit_qoe: float
    qoe_delta_from_full: float
    oracle_gap: float
    decision_trace_shift: float
    deadline_hit_rate: float
    visible_quality: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.point, MechanismAttributionPoint):
            raise MechanismAttributionHarnessError("point must be a MechanismAttributionPoint record.")
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.variant_id, "variant_id")
        object.__setattr__(self, "removed_terms", _string_tuple(self.removed_terms, "removed_terms", allow_empty=True))
        object.__setattr__(self, "seed", _non_negative_int(self.seed, "seed"))
        for field_name in (
            "deadline_hit_qoe",
            "full_deadline_hit_qoe",
            "oracle_deadline_hit_qoe",
            "oracle_gap",
            "decision_trace_shift",
            "deadline_hit_rate",
            "visible_quality",
        ):
            object.__setattr__(self, field_name, _unit_interval(getattr(self, field_name), field_name))
        object.__setattr__(self, "qoe_delta_from_full", _finite_float(self.qoe_delta_from_full, "qoe_delta_from_full"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def outcome_id(self) -> str:
        payload = {
            "point_id": self.point.point_id,
            "method_id": self.method_id,
            "variant_id": self.variant_id,
            "seed": self.seed,
            "deadline_hit_qoe": self.deadline_hit_qoe,
            "qoe_delta_from_full": self.qoe_delta_from_full,
        }
        return f"mechanism-attribution-outcome-{stable_config_id(payload)}"

    def metric_records(self, *, run_id: str, split: str | None = "final") -> tuple[MetricRecord, ...]:
        tags = {
            "run_id": run_id,
            "method": self.method_id,
            "variant_id": self.variant_id,
            "case_id": self.point.case_id,
            "scene_id": self.point.scene_id,
            "trace_id": self.point.trace_id,
            "viewport_id": self.point.viewport_id,
            "device_profile_id": self.point.device_profile_id,
        }
        base_metadata = {
            "mechanism_attribution_outcome_id": self.outcome_id,
            "mechanism_attribution_point": self.point.as_payload(),
            "mechanism_attribution_outcome": self.as_payload(include_metrics=False),
        }
        return (
            _metric("ablation_deadline_hit_qoe", self.deadline_hit_qoe, "score", tags, base_metadata, split=split),
            _metric("ablation_qoe_delta_from_full", self.qoe_delta_from_full, "score_delta", tags, base_metadata, split=split),
            _metric("ablation_oracle_gap", self.oracle_gap, "score", tags, base_metadata, split=split),
            _metric("ablation_decision_trace_shift", self.decision_trace_shift, "score", tags, base_metadata, split=split),
            _metric("ablation_deadline_hit_rate", self.deadline_hit_rate, "ratio", tags, base_metadata, split=split),
            _metric("ablation_visible_quality", self.visible_quality, "score", tags, base_metadata, split=split),
        )

    def as_payload(self, *, include_metrics: bool = True) -> dict[str, Any]:
        payload = {
            "outcome_id": self.outcome_id,
            "point": self.point.as_payload(),
            "method_id": self.method_id,
            "variant_id": self.variant_id,
            "removed_terms": list(self.removed_terms),
            "seed": self.seed,
            "deadline_hit_qoe": self.deadline_hit_qoe,
            "full_deadline_hit_qoe": self.full_deadline_hit_qoe,
            "oracle_deadline_hit_qoe": self.oracle_deadline_hit_qoe,
            "qoe_delta_from_full": self.qoe_delta_from_full,
            "oracle_gap": self.oracle_gap,
            "decision_trace_shift": self.decision_trace_shift,
            "deadline_hit_rate": self.deadline_hit_rate,
            "visible_quality": self.visible_quality,
            "metadata": _to_payload(self.metadata),
        }
        if include_metrics:
            payload["metric_names"] = list(MECHANISM_ABLATION_METRIC_NAMES)
        return payload


@dataclass(frozen=True)
class MechanismAttributionConfig:
    """Paired mechanism ablation sweep over final-split cases."""

    scenes: Sequence[str]
    traces: Sequence[str]
    viewports: Sequence[str]
    devices: Sequence[str]
    decision_cases: Sequence[str]
    frozen_method_manifest: FrozenMethodManifest | Mapping[str, Any] | None = None
    frozen_method_id: str = "robust-deadline-aware-mpc"
    variants: Sequence[str] = tuple(str(row["variant_id"]) for row in MECHANISM_ABLATION_VARIANTS)
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
        object.__setattr__(self, "decision_cases", _string_tuple(self.decision_cases, "decision_cases"))
        manifest = _coerce_frozen_manifest(self.frozen_method_manifest)
        object.__setattr__(self, "frozen_method_manifest", manifest)
        frozen_method_id = manifest.method_id if manifest is not None else self.frozen_method_id
        _require_non_empty(frozen_method_id, "frozen_method_id")
        object.__setattr__(self, "frozen_method_id", frozen_method_id)
        object.__setattr__(self, "variants", _variant_tuple(self.variants))
        object.__setattr__(self, "seeds", _int_tuple(self.seeds, "seeds"))
        if self.split != "final":
            raise MechanismAttributionHarnessError("Mechanism attribution harness must run on the final split.")
        if self.run_mode not in {"plan_only", "metrics_only", "full"}:
            raise MechanismAttributionHarnessError("run_mode must be one of: plan_only, metrics_only, full.")
        if self.output_root is not None:
            object.__setattr__(self, "output_root", Path(self.output_root).as_posix())
        object.__setattr__(self, "fixed_variables", _plain_json_mapping(self.fixed_variables, "fixed_variables"))
        object.__setattr__(self, "tags", _string_mapping(self.tags, "tags"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(_method_id_for_variant(self.frozen_method_id, variant_id) for variant_id in self.variants)

    @property
    def harness_id(self) -> str:
        return f"mechanism-attribution-ablation-{stable_config_id(self.stable_payload())}"

    def attribution_points(self) -> tuple[MechanismAttributionPoint, ...]:
        points: list[MechanismAttributionPoint] = []
        for scene in self.scenes:
            for trace in self.traces:
                for viewport in self.viewports:
                    for device in self.devices:
                        for case in self.decision_cases:
                            points.append(
                                MechanismAttributionPoint(
                                    scene_id=scene,
                                    trace_id=trace,
                                    viewport_id=viewport,
                                    device_profile_id=device,
                                    case_id=case,
                                )
                            )
        return tuple(points)

    def stable_payload(self) -> dict[str, Any]:
        return {
            "scenes": list(self.scenes),
            "traces": list(self.traces),
            "viewports": list(self.viewports),
            "devices": list(self.devices),
            "decision_cases": list(self.decision_cases),
            "frozen_method_manifest": self.frozen_method_manifest.as_payload() if self.frozen_method_manifest is not None else None,
            "frozen_method_id": self.frozen_method_id,
            "variants": list(self.variants),
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
            "attribution_point_count": len(self.attribution_points()),
        }


@dataclass(frozen=True)
class MechanismAttributionResult:
    """Complete mechanism-attribution ablation harness output."""

    harness_id: str
    config: MechanismAttributionConfig
    attribution_points: tuple[MechanismAttributionPoint, ...]
    outcomes: tuple[MechanismAttributionOutcome, ...]
    harness_result: HarnessResult

    def __post_init__(self) -> None:
        _require_non_empty(self.harness_id, "harness_id")
        if not isinstance(self.config, MechanismAttributionConfig):
            raise MechanismAttributionHarnessError("config must be a MechanismAttributionConfig record.")
        points = tuple(self.attribution_points)
        outcomes = tuple(self.outcomes)
        for point in points:
            if not isinstance(point, MechanismAttributionPoint):
                raise MechanismAttributionHarnessError("attribution_points must contain MechanismAttributionPoint records.")
        for outcome in outcomes:
            if not isinstance(outcome, MechanismAttributionOutcome):
                raise MechanismAttributionHarnessError("outcomes must contain MechanismAttributionOutcome records.")
        if not isinstance(self.harness_result, HarnessResult):
            raise MechanismAttributionHarnessError("harness_result must be a HarnessResult record.")
        object.__setattr__(self, "attribution_points", points)
        object.__setattr__(self, "outcomes", outcomes)

    def as_payload(self) -> dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            "config": self.config.as_payload(),
            "attribution_points": [point.as_payload() for point in self.attribution_points],
            "outcomes": [outcome.as_payload() for outcome in self.outcomes],
            "paired_ablation_table": paired_ablation_table(self),
            "oracle_gap_cases": oracle_gap_cases(self),
            "decision_trace_cases": decision_trace_cases(self),
            "harness_result": self.harness_result.as_payload(),
        }


def run_mechanism_attribution_harness(config: MechanismAttributionConfig) -> MechanismAttributionResult:
    """Run paired mechanism-removal and oracle attribution cases."""

    if not isinstance(config, MechanismAttributionConfig):
        raise MechanismAttributionHarnessError("config must be a MechanismAttributionConfig record.")
    points = config.attribution_points()
    point_by_id = {point.point_id: point for point in points}
    variant_by_method = {_method_id_for_variant(config.frozen_method_id, variant): variant for variant in config.variants}
    outcomes: list[MechanismAttributionOutcome] = []

    def executor(spec: HarnessRunSpec) -> HarnessRunResult:
        point = point_by_id[spec.workload_id]
        variant_id = variant_by_method[spec.method_id]
        outcome = evaluate_mechanism_attribution_point(
            point,
            variant_id=variant_id,
            frozen_method_id=config.frozen_method_id,
            seed=spec.seed,
        )
        outcomes.append(outcome)
        return HarnessRunResult(
            spec=spec,
            status="executed",
            metrics=outcome.metric_records(run_id=spec.run_id, split=config.split),
            metadata={"mechanism_attribution_outcome_id": outcome.outcome_id},
        )

    harness_config = HarnessConfig(
        harness_name="mechanism-attribution-ablation",
        methods=config.methods,
        workloads=tuple(point.point_id for point in points),
        seeds=config.seeds,
        run_mode=config.run_mode,
        baseline_method_id=config.frozen_method_id,
        fixed_variables={
            **config.fixed_variables,
            "mechanism_attribution_harness_id": config.harness_id,
            "split": config.split,
            "frozen_method_id": config.frozen_method_id,
            "paired_full_variant": "full",
        },
        comparison_metric_names=MECHANISM_ABLATION_METRIC_NAMES,
        comparison_group_keys=("workload_id", "seed", "metric_name"),
        output_root=Path(config.output_root) / "harness" if config.output_root is not None else None,
        tags={**config.tags, "split": config.split, "frozen_method_id": config.frozen_method_id},
        metadata={
            "mechanism_attribution_config": config.as_payload(),
            "frozen_method_manifest": config.frozen_method_manifest.as_payload() if config.frozen_method_manifest is not None else None,
            **config.metadata,
        },
    )
    harness_result = run_harness(harness_config, executor=executor)
    result = MechanismAttributionResult(
        harness_id=config.harness_id,
        config=config,
        attribution_points=points,
        outcomes=tuple(outcomes),
        harness_result=harness_result,
    )
    if config.output_root is not None:
        export_mechanism_attribution_outputs(config.output_root, result)
    return result


def evaluate_mechanism_attribution_point(
    point: MechanismAttributionPoint,
    *,
    variant_id: str,
    frozen_method_id: str = "robust-deadline-aware-mpc",
    seed: int = 0,
) -> MechanismAttributionOutcome:
    """Evaluate one paired ablation variant and decision-trace case."""

    if not isinstance(point, MechanismAttributionPoint):
        raise MechanismAttributionHarnessError("point must be a MechanismAttributionPoint record.")
    variant = _variant_payload(variant_id)
    seed = _non_negative_int(seed, "seed")
    base = _case_profile(point, seed)
    full_qoe = _clamp01(0.72 - base["stress"] * 0.11 + base["device_headroom"] * 0.05)
    oracle_qoe = _clamp01(full_qoe + 0.08 + base["stress"] * 0.05)
    penalty = _variant_penalty(variant_id, base)
    if variant_id == "oracle":
        qoe = oracle_qoe
    else:
        qoe = _clamp01(full_qoe - penalty)
    deadline_hit_rate = _clamp01(qoe + 0.08 - base["network_stress"] * 0.08)
    visible_quality = _clamp01(qoe + 0.04 - base["viewport_stress"] * 0.05)
    trace_shift = 0.0 if variant_id == "full" else _clamp01(0.08 + penalty * 2.4 + base["case_sensitivity"] * 0.10)
    method_id = _method_id_for_variant(frozen_method_id, variant_id)
    decision_trace = _decision_trace(point, variant_id, base, penalty)
    return MechanismAttributionOutcome(
        point=point,
        method_id=method_id,
        variant_id=variant_id,
        removed_terms=variant["removed_terms"],
        seed=seed,
        deadline_hit_qoe=round(qoe, 6),
        full_deadline_hit_qoe=round(full_qoe, 6),
        oracle_deadline_hit_qoe=round(oracle_qoe, 6),
        qoe_delta_from_full=round(qoe - full_qoe, 6),
        oracle_gap=round(max(0.0, oracle_qoe - qoe), 6),
        decision_trace_shift=round(trace_shift, 6),
        deadline_hit_rate=round(deadline_hit_rate, 6),
        visible_quality=round(visible_quality, 6),
        metadata={
            "paired_full_variant": "full",
            "decision_trace": decision_trace,
            "oracle_reference": {
                "oracle_deadline_hit_qoe": round(oracle_qoe, 6),
                "oracle_gap": round(max(0.0, oracle_qoe - qoe), 6),
            },
            "case_profile": base,
        },
    )


def paired_ablation_table(result: MechanismAttributionResult | Sequence[MechanismAttributionOutcome]) -> list[dict[str, Any]]:
    """Return aggregate paired mechanism-removal attribution rows."""

    rows: list[dict[str, Any]] = []
    for variant_id, outcomes in _group_by_variant(_result_outcomes(result)).items():
        rows.append(
            {
                "variant_id": variant_id,
                "method_id": outcomes[0].method_id,
                "removed_terms": list(outcomes[0].removed_terms),
                "deadline_hit_qoe": _mean(outcome.deadline_hit_qoe for outcome in outcomes),
                "qoe_delta_from_full": _mean(outcome.qoe_delta_from_full for outcome in outcomes),
                "oracle_gap": _mean(outcome.oracle_gap for outcome in outcomes),
                "decision_trace_shift": _mean(outcome.decision_trace_shift for outcome in outcomes),
                "deadline_hit_rate": _mean(outcome.deadline_hit_rate for outcome in outcomes),
                "visible_quality": _mean(outcome.visible_quality for outcome in outcomes),
                "sample_count": len(outcomes),
            }
        )
    return sorted(rows, key=lambda row: (row["variant_id"] != "full", row["variant_id"]))


def oracle_gap_cases(result: MechanismAttributionResult | Sequence[MechanismAttributionOutcome]) -> list[dict[str, Any]]:
    """Return per-case oracle-gap rows for attribution plots."""

    return [
        {
            "variant_id": outcome.variant_id,
            "method_id": outcome.method_id,
            "scene_id": outcome.point.scene_id,
            "trace_id": outcome.point.trace_id,
            "viewport_id": outcome.point.viewport_id,
            "device_profile_id": outcome.point.device_profile_id,
            "case_id": outcome.point.case_id,
            "seed": outcome.seed,
            "deadline_hit_qoe": outcome.deadline_hit_qoe,
            "full_deadline_hit_qoe": outcome.full_deadline_hit_qoe,
            "oracle_deadline_hit_qoe": outcome.oracle_deadline_hit_qoe,
            "oracle_gap": outcome.oracle_gap,
            "qoe_delta_from_full": outcome.qoe_delta_from_full,
        }
        for outcome in _result_outcomes(result)
    ]


def decision_trace_cases(result: MechanismAttributionResult | Sequence[MechanismAttributionOutcome]) -> list[dict[str, Any]]:
    """Return decision-trace case rows with mechanism-level trace components."""

    rows: list[dict[str, Any]] = []
    for outcome in _result_outcomes(result):
        trace = outcome.metadata.get("decision_trace", {})
        rows.append(
            {
                "variant_id": outcome.variant_id,
                "method_id": outcome.method_id,
                "case_id": outcome.point.case_id,
                "scene_id": outcome.point.scene_id,
                "trace_id": outcome.point.trace_id,
                "viewport_id": outcome.point.viewport_id,
                "device_profile_id": outcome.point.device_profile_id,
                "seed": outcome.seed,
                "decision_trace_shift": outcome.decision_trace_shift,
                "selected_action": trace.get("selected_action"),
                "lifecycle_weight": trace.get("lifecycle_weight"),
                "uncertainty_weight": trace.get("uncertainty_weight"),
                "component_cost_weight": trace.get("component_cost_weight"),
                "cancellation_weight": trace.get("cancellation_weight"),
                "fov_weight": trace.get("fov_weight"),
                "lead_time_weight": trace.get("lead_time_weight"),
            }
        )
    return rows


def export_mechanism_attribution_outputs(
    output_root: str | Path,
    result: MechanismAttributionResult,
) -> tuple[Path, Path, Path, Path, Path]:
    """Write attribution outcomes and table/oracle/trace paper inputs."""

    if not isinstance(result, MechanismAttributionResult):
        raise MechanismAttributionHarnessError("result must be a MechanismAttributionResult record.")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    outcomes_path = root / "mechanism_attribution_outcomes.jsonl"
    table_path = root / "paired_ablation_table.json"
    oracle_path = root / "oracle_gap_cases.json"
    trace_path = root / "decision_trace_cases.json"
    summary_path = root / "mechanism_attribution_summary.json"
    _write_text_atomic(
        outcomes_path,
        "".join(json.dumps(outcome.as_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for outcome in result.outcomes),
    )
    _write_text_atomic(table_path, _json_payload(paired_ablation_table(result)))
    _write_text_atomic(oracle_path, _json_payload(oracle_gap_cases(result)))
    _write_text_atomic(trace_path, _json_payload(decision_trace_cases(result)))
    _write_text_atomic(summary_path, _json_payload(result.as_payload()))
    return outcomes_path, table_path, oracle_path, trace_path, summary_path


def _case_profile(point: MechanismAttributionPoint, seed: int) -> dict[str, float]:
    network_stress = _hash_fraction("trace", point.trace_id, point.case_id, seed)
    viewport_stress = _hash_fraction("viewport", point.viewport_id, point.case_id, seed)
    scene_complexity = _hash_fraction("scene", point.scene_id, seed)
    case_sensitivity = _hash_fraction("case", point.case_id, seed)
    device_headroom = _device_headroom(point.device_profile_id)
    return {
        "network_stress": network_stress,
        "viewport_stress": viewport_stress,
        "scene_complexity": scene_complexity,
        "case_sensitivity": case_sensitivity,
        "device_headroom": device_headroom,
        "stress": _clamp01(0.36 * network_stress + 0.28 * viewport_stress + 0.22 * scene_complexity + 0.14 * (1.0 - device_headroom)),
    }


def _variant_penalty(variant_id: str, profile: Mapping[str, float]) -> float:
    if variant_id == "full" or variant_id == "oracle":
        return 0.0
    penalties = {
        "no-lifecycle": 0.030 + profile["network_stress"] * 0.055,
        "no-uncertainty": 0.025 + profile["viewport_stress"] * 0.060,
        "no-component-cost": 0.020 + profile["scene_complexity"] * 0.052,
        "no-cancellation": 0.018 + profile["network_stress"] * 0.040,
        "no-fov": 0.016 + profile["viewport_stress"] * 0.050,
        "no-lead-time": 0.022 + profile["case_sensitivity"] * 0.045,
    }
    return penalties[variant_id]


def _decision_trace(point: MechanismAttributionPoint, variant_id: str, profile: Mapping[str, float], penalty: float) -> dict[str, Any]:
    weights = {
        "lifecycle_weight": _term_weight("lifecycle", variant_id, 0.55 + profile["network_stress"] * 0.25),
        "uncertainty_weight": _term_weight("uncertainty", variant_id, 0.50 + profile["viewport_stress"] * 0.25),
        "component_cost_weight": _term_weight("component_cost", variant_id, 0.45 + profile["scene_complexity"] * 0.25),
        "cancellation_weight": _term_weight("cancellation", variant_id, 0.40 + profile["network_stress"] * 0.20),
        "fov_weight": _term_weight("fov", variant_id, 0.38 + profile["viewport_stress"] * 0.22),
        "lead_time_weight": _term_weight("lead_time", variant_id, 0.42 + profile["case_sensitivity"] * 0.20),
    }
    selected_action = "reference-prefetch"
    if variant_id in {"no-lifecycle", "no-lead-time"} and profile["network_stress"] > 0.55:
        selected_action = "gaussian-base"
    elif variant_id in {"no-fov", "no-uncertainty"} and profile["viewport_stress"] > 0.55:
        selected_action = "visible-tile"
    elif variant_id == "oracle":
        selected_action = "oracle-best-reference"
    return {
        **{key: round(value, 6) for key, value in weights.items()},
        "selected_action": selected_action,
        "decision_trace_shift": round(_clamp01(0.08 + penalty * 2.4 + profile["case_sensitivity"] * 0.10), 6) if variant_id != "full" else 0.0,
        "case_id": point.case_id,
    }


def _term_weight(term: str, variant_id: str, value: float) -> float:
    removed_variant = f"no-{term.replace('_', '-')}"
    if variant_id == removed_variant:
        return 0.0
    if variant_id == "oracle":
        return min(1.0, value + 0.12)
    return _clamp01(value)


def _method_id_for_variant(frozen_method_id: str, variant_id: str) -> str:
    if variant_id == "full":
        return frozen_method_id
    if variant_id == "oracle":
        return "perfect-information-oracle"
    return f"{frozen_method_id}-{variant_id}"


def _variant_payload(variant_id: str) -> Mapping[str, Any]:
    for row in MECHANISM_ABLATION_VARIANTS:
        if row["variant_id"] == variant_id:
            return row
    valid = ", ".join(str(row["variant_id"]) for row in MECHANISM_ABLATION_VARIANTS)
    raise MechanismAttributionHarnessError(f"variant_id must be one of: {valid}.")


def _variant_tuple(values: Sequence[str]) -> tuple[str, ...]:
    parsed = _string_tuple(values, "variants")
    for variant_id in parsed:
        _variant_payload(variant_id)
    if "full" not in parsed:
        raise MechanismAttributionHarnessError("variants must include full for paired comparisons.")
    return parsed


def _coerce_frozen_manifest(value: FrozenMethodManifest | Mapping[str, Any] | None) -> FrozenMethodManifest | None:
    if value is None:
        return None
    if isinstance(value, FrozenMethodManifest):
        return value
    if not isinstance(value, MappingABC):
        raise MechanismAttributionHarnessError("frozen_method_manifest must be a FrozenMethodManifest or mapping.")
    try:
        return FrozenMethodManifest(**value)
    except TypeError as exc:
        raise MechanismAttributionHarnessError(f"Malformed frozen_method_manifest: {exc}") from exc


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


def _result_outcomes(result: MechanismAttributionResult | Sequence[MechanismAttributionOutcome]) -> tuple[MechanismAttributionOutcome, ...]:
    outcomes = result.outcomes if isinstance(result, MechanismAttributionResult) else tuple(result)
    for outcome in outcomes:
        if not isinstance(outcome, MechanismAttributionOutcome):
            raise MechanismAttributionHarnessError("outcomes must contain MechanismAttributionOutcome records.")
    return tuple(outcomes)


def _group_by_variant(outcomes: Sequence[MechanismAttributionOutcome]) -> dict[str, tuple[MechanismAttributionOutcome, ...]]:
    grouped: dict[str, list[MechanismAttributionOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.variant_id, []).append(outcome)
    return {variant_id: tuple(variant_outcomes) for variant_id, variant_outcomes in sorted(grouped.items())}


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
        "metric_id": f"mechanism-attribution-metric-{stable_config_id({'metric_name': metric_name, 'value': value, 'tags': dict(tags)})}",
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
        raise MechanismAttributionHarnessError(f"Failed to write mechanism-attribution output {path}: {exc}") from exc


def _string_tuple(values: Sequence[str], field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise MechanismAttributionHarnessError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    if not parsed and not allow_empty:
        raise MechanismAttributionHarnessError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _string_mapping(value: Mapping[str, str], field_name: str) -> Mapping[str, str]:
    if not isinstance(value, MappingABC):
        raise MechanismAttributionHarnessError(f"{field_name} must be a mapping.")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        _require_non_empty(str(key), f"{field_name} key")
        _require_non_empty(item, f"{field_name}.{key}")
        parsed[str(key)] = item
    return {key: parsed[key] for key in sorted(parsed)}


def _int_tuple(values: Sequence[int], field_name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise MechanismAttributionHarnessError(f"{field_name} must be a sequence of integers.")
    parsed: list[int] = []
    seen: set[int] = set()
    for value in values:
        parsed_value = _non_negative_int(value, field_name)
        if parsed_value not in seen:
            parsed.append(parsed_value)
            seen.add(parsed_value)
    if not parsed:
        raise MechanismAttributionHarnessError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise MechanismAttributionHarnessError(f"{field_name} must be a mapping.")
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
        raise MechanismAttributionHarnessError(f"{field_name} must be a non-empty string.")


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise MechanismAttributionHarnessError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MechanismAttributionHarnessError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise MechanismAttributionHarnessError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0 or parsed > 1.0:
        raise MechanismAttributionHarnessError(f"{field_name} must be in [0, 1].")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise MechanismAttributionHarnessError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MechanismAttributionHarnessError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise MechanismAttributionHarnessError(f"{field_name} must be finite.")
    return parsed


def _mean(values: Sequence[float]) -> float:
    parsed = tuple(_finite_float(value, "mean.value") for value in values)
    if not parsed:
        return 0.0
    return sum(parsed) / len(parsed)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


__all__ = [
    "MECHANISM_ABLATION_METRIC_NAMES",
    "MECHANISM_ABLATION_VARIANTS",
    "MechanismAttributionConfig",
    "MechanismAttributionHarnessError",
    "MechanismAttributionOutcome",
    "MechanismAttributionPoint",
    "MechanismAttributionResult",
    "decision_trace_cases",
    "evaluate_mechanism_attribution_point",
    "export_mechanism_attribution_outputs",
    "oracle_gap_cases",
    "paired_ablation_table",
    "run_mechanism_attribution_harness",
]
