"""Gaussian-reference substitution surface harness."""

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


SUBSTITUTION_METHOD_IDS: tuple[str, ...] = (
    "svq-gaussian-only-abr",
    "reference-only-after-base",
    "quality-max-deadline-unaware",
    "deadline-aware-knapsack-allocator",
    "perfect-information-oracle",
)
SUBSTITUTION_METRIC_NAMES: tuple[str, ...] = (
    "substitution_gain",
    "substitution_utility_score",
    "substitution_selected_reference",
    "substitution_budget_pressure",
)
SubstitutionAction = Literal["gaussian", "reference", "mixed"]


class SubstitutionSurfaceError(ValueError):
    """Raised when substitution-surface harness inputs are invalid."""


@dataclass(frozen=True)
class SubstitutionSurfacePoint:
    """One point in the layer/resolution/FoV/mismatch/budget/slack surface."""

    layer: int
    ref_resolution: str
    fov_deg: float
    view_mismatch: float
    budget_bytes: int
    slack_ms: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", _non_negative_int(self.layer, "layer"))
        _require_non_empty(self.ref_resolution, "ref_resolution")
        object.__setattr__(self, "fov_deg", _positive_float(self.fov_deg, "fov_deg"))
        object.__setattr__(self, "view_mismatch", _unit_interval(self.view_mismatch, "view_mismatch"))
        object.__setattr__(self, "budget_bytes", _positive_int(self.budget_bytes, "budget_bytes"))
        object.__setattr__(self, "slack_ms", _non_negative_float(self.slack_ms, "slack_ms"))

    @property
    def point_id(self) -> str:
        return f"substitution-point-{stable_config_id(self.stable_payload())}"

    def stable_payload(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "ref_resolution": self.ref_resolution,
            "fov_deg": self.fov_deg,
            "view_mismatch": self.view_mismatch,
            "budget_bytes": self.budget_bytes,
            "slack_ms": self.slack_ms,
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            **self.stable_payload(),
        }


@dataclass(frozen=True)
class SubstitutionOutcome:
    """Method result at one substitution surface point."""

    point: SubstitutionSurfacePoint
    method_id: str
    seed: int
    selected_action: SubstitutionAction
    gaussian_quality: float
    reference_quality: float
    substitution_gain: float
    utility_score: float
    budget_pressure: float
    deadline_slack_ratio: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.point, SubstitutionSurfacePoint):
            raise SubstitutionSurfaceError("point must be a SubstitutionSurfacePoint record.")
        _require_non_empty(self.method_id, "method_id")
        object.__setattr__(self, "seed", _non_negative_int(self.seed, "seed"))
        if self.selected_action not in {"gaussian", "reference", "mixed"}:
            raise SubstitutionSurfaceError("selected_action must be one of: gaussian, reference, mixed.")
        for field_name in ("gaussian_quality", "reference_quality"):
            object.__setattr__(self, field_name, _unit_interval(getattr(self, field_name), field_name))
        object.__setattr__(self, "substitution_gain", _finite_float(self.substitution_gain, "substitution_gain"))
        object.__setattr__(self, "utility_score", _finite_float(self.utility_score, "utility_score"))
        object.__setattr__(self, "budget_pressure", _non_negative_float(self.budget_pressure, "budget_pressure"))
        object.__setattr__(self, "deadline_slack_ratio", _non_negative_float(self.deadline_slack_ratio, "deadline_slack_ratio"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def outcome_id(self) -> str:
        payload = {
            "point_id": self.point.point_id,
            "method_id": self.method_id,
            "seed": self.seed,
            "selected_action": self.selected_action,
            "substitution_gain": self.substitution_gain,
            "utility_score": self.utility_score,
        }
        return f"substitution-outcome-{stable_config_id(payload)}"

    def metric_records(self, *, run_id: str, split: str | None = None) -> tuple[MetricRecord, ...]:
        tags = {
            "run_id": run_id,
            "method": self.method_id,
            "surface_point_id": self.point.point_id,
            "selected_action": self.selected_action,
        }
        base_metadata = {
            "substitution_outcome_id": self.outcome_id,
            "surface_point": self.point.as_payload(),
            "substitution_outcome": self.as_payload(include_metrics=False),
        }
        return (
            _metric("substitution_gain", self.substitution_gain, "score", tags, base_metadata, split=split),
            _metric("substitution_utility_score", self.utility_score, "score", tags, base_metadata, split=split),
            _metric("substitution_selected_reference", 1.0 if self.selected_action in {"reference", "mixed"} else 0.0, "ratio", tags, base_metadata, split=split),
            _metric("substitution_budget_pressure", self.budget_pressure, "ratio", tags, base_metadata, split=split),
        )

    def as_payload(self, *, include_metrics: bool = True) -> dict[str, Any]:
        payload = {
            "outcome_id": self.outcome_id,
            "point": self.point.as_payload(),
            "method_id": self.method_id,
            "seed": self.seed,
            "selected_action": self.selected_action,
            "gaussian_quality": self.gaussian_quality,
            "reference_quality": self.reference_quality,
            "substitution_gain": self.substitution_gain,
            "utility_score": self.utility_score,
            "budget_pressure": self.budget_pressure,
            "deadline_slack_ratio": self.deadline_slack_ratio,
            "metadata": _to_payload(self.metadata),
        }
        if include_metrics:
            payload["metric_names"] = list(SUBSTITUTION_METRIC_NAMES)
        return payload


@dataclass(frozen=True)
class SubstitutionSurfaceConfig:
    """Sweep dimensions and comparison settings for the substitution harness."""

    layers: Sequence[int]
    ref_resolutions: Sequence[str]
    fov_degrees: Sequence[float]
    view_mismatches: Sequence[float]
    budget_bytes: Sequence[int]
    slack_ms: Sequence[float]
    seeds: Sequence[int] = (0,)
    methods: Sequence[str] = SUBSTITUTION_METHOD_IDS
    baseline_method_id: str = "svq-gaussian-only-abr"
    run_mode: str = "full"
    split: str | None = "final"
    output_root: str | Path | None = None
    fixed_variables: Mapping[str, Any] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", _int_tuple(self.layers, "layers"))
        object.__setattr__(self, "ref_resolutions", _string_tuple(self.ref_resolutions, "ref_resolutions"))
        object.__setattr__(self, "fov_degrees", _positive_float_tuple(self.fov_degrees, "fov_degrees"))
        object.__setattr__(self, "view_mismatches", _unit_interval_tuple(self.view_mismatches, "view_mismatches"))
        object.__setattr__(self, "budget_bytes", _positive_int_tuple(self.budget_bytes, "budget_bytes"))
        object.__setattr__(self, "slack_ms", _non_negative_float_tuple(self.slack_ms, "slack_ms"))
        object.__setattr__(self, "seeds", _int_tuple(self.seeds, "seeds"))
        object.__setattr__(self, "methods", _string_tuple(self.methods, "methods"))
        _require_non_empty(self.baseline_method_id, "baseline_method_id")
        if self.baseline_method_id not in self.methods:
            raise SubstitutionSurfaceError("baseline_method_id must be included in methods.")
        if self.run_mode not in {"plan_only", "metrics_only", "full"}:
            raise SubstitutionSurfaceError("run_mode must be one of: plan_only, metrics_only, full.")
        if self.split is not None:
            _require_non_empty(self.split, "split")
        if self.output_root is not None:
            object.__setattr__(self, "output_root", Path(self.output_root).as_posix())
        object.__setattr__(self, "fixed_variables", _plain_json_mapping(self.fixed_variables, "fixed_variables"))
        object.__setattr__(self, "tags", _string_mapping(self.tags, "tags"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def surface_id(self) -> str:
        return f"gaussian-reference-substitution-surface-{stable_config_id(self.stable_payload())}"

    def surface_points(self) -> tuple[SubstitutionSurfacePoint, ...]:
        points: list[SubstitutionSurfacePoint] = []
        for layer in self.layers:
            for ref_resolution in self.ref_resolutions:
                for fov_deg in self.fov_degrees:
                    for view_mismatch in self.view_mismatches:
                        for budget in self.budget_bytes:
                            for slack in self.slack_ms:
                                points.append(
                                    SubstitutionSurfacePoint(
                                        layer=layer,
                                        ref_resolution=ref_resolution,
                                        fov_deg=fov_deg,
                                        view_mismatch=view_mismatch,
                                        budget_bytes=budget,
                                        slack_ms=slack,
                                    )
                                )
        return tuple(points)

    def stable_payload(self) -> dict[str, Any]:
        return {
            "layers": list(self.layers),
            "ref_resolutions": list(self.ref_resolutions),
            "fov_degrees": list(self.fov_degrees),
            "view_mismatches": list(self.view_mismatches),
            "budget_bytes": list(self.budget_bytes),
            "slack_ms": list(self.slack_ms),
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
            "surface_id": self.surface_id,
            **self.stable_payload(),
            "output_root": self.output_root,
            "surface_point_count": len(self.surface_points()),
        }


@dataclass(frozen=True)
class SubstitutionSurfaceResult:
    """Complete output of the Gaussian-reference substitution surface harness."""

    surface_id: str
    config: SubstitutionSurfaceConfig
    surface_points: tuple[SubstitutionSurfacePoint, ...]
    outcomes: tuple[SubstitutionOutcome, ...]
    harness_result: HarnessResult

    def __post_init__(self) -> None:
        _require_non_empty(self.surface_id, "surface_id")
        if not isinstance(self.config, SubstitutionSurfaceConfig):
            raise SubstitutionSurfaceError("config must be a SubstitutionSurfaceConfig record.")
        points = tuple(self.surface_points)
        outcomes = tuple(self.outcomes)
        for point in points:
            if not isinstance(point, SubstitutionSurfacePoint):
                raise SubstitutionSurfaceError("surface_points must contain SubstitutionSurfacePoint records.")
        for outcome in outcomes:
            if not isinstance(outcome, SubstitutionOutcome):
                raise SubstitutionSurfaceError("outcomes must contain SubstitutionOutcome records.")
        if not isinstance(self.harness_result, HarnessResult):
            raise SubstitutionSurfaceError("harness_result must be a HarnessResult record.")
        object.__setattr__(self, "surface_points", points)
        object.__setattr__(self, "outcomes", outcomes)

    def as_payload(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "config": self.config.as_payload(),
            "surface_points": [point.as_payload() for point in self.surface_points],
            "outcomes": [outcome.as_payload() for outcome in self.outcomes],
            "harness_result": self.harness_result.as_payload(),
        }


def run_gaussian_reference_substitution_surface(config: SubstitutionSurfaceConfig) -> SubstitutionSurfaceResult:
    """Run the issue-32 substitution surface sweep through the generic harness."""

    if not isinstance(config, SubstitutionSurfaceConfig):
        raise SubstitutionSurfaceError("config must be a SubstitutionSurfaceConfig record.")
    points = config.surface_points()
    point_by_id = {point.point_id: point for point in points}
    outcomes: list[SubstitutionOutcome] = []

    def executor(spec: HarnessRunSpec) -> HarnessRunResult:
        point = point_by_id[spec.workload_id]
        outcome = evaluate_substitution_point(point, method_id=spec.method_id, seed=spec.seed)
        outcomes.append(outcome)
        return HarnessRunResult(
            spec=spec,
            status="executed",
            metrics=outcome.metric_records(run_id=spec.run_id, split=config.split),
            metadata={"substitution_outcome_id": outcome.outcome_id},
        )

    harness_config = HarnessConfig(
        harness_name="gaussian-reference-substitution-surface",
        methods=config.methods,
        workloads=tuple(point.point_id for point in points),
        seeds=config.seeds,
        run_mode=config.run_mode,
        baseline_method_id=config.baseline_method_id,
        fixed_variables={
            **config.fixed_variables,
            "surface_id": config.surface_id,
            "sweep_dimensions": ("layer", "ref_resolution", "fov_deg", "view_mismatch", "budget_bytes", "slack_ms"),
        },
        comparison_metric_names=("substitution_gain", "substitution_utility_score"),
        comparison_group_keys=("workload_id", "seed", "metric_name"),
        output_root=Path(config.output_root) / "harness" if config.output_root is not None else None,
        tags={**config.tags, "surface_id": config.surface_id},
        metadata={"surface_config": config.as_payload(), **config.metadata},
    )
    harness_result = run_harness(harness_config, executor=executor)
    result = SubstitutionSurfaceResult(
        surface_id=config.surface_id,
        config=config,
        surface_points=points,
        outcomes=tuple(outcomes),
        harness_result=harness_result,
    )
    if config.output_root is not None:
        export_substitution_surface_outputs(config.output_root, result)
    return result


def evaluate_substitution_point(point: SubstitutionSurfacePoint, *, method_id: str, seed: int = 0) -> SubstitutionOutcome:
    """Evaluate one method at one surface point with a deterministic analytic model."""

    if not isinstance(point, SubstitutionSurfacePoint):
        raise SubstitutionSurfaceError("point must be a SubstitutionSurfacePoint record.")
    _require_non_empty(method_id, "method_id")
    parsed_seed = _non_negative_int(seed, "seed")
    gaussian_quality = _gaussian_quality(point, parsed_seed)
    reference_quality = _reference_quality(point, parsed_seed)
    gaussian_cost = _gaussian_cost_bytes(point)
    reference_cost = _reference_cost_bytes(point)
    reference_time = _reference_time_ms(point)
    budget_pressure = reference_cost / point.budget_bytes
    gaussian_pressure = gaussian_cost / point.budget_bytes
    deadline_slack_ratio = point.slack_ms / reference_time
    gaussian_utility = _utility_score(
        gaussian_quality,
        budget_pressure=gaussian_pressure,
        deadline_slack_ratio=max(1.0, deadline_slack_ratio * 1.2),
        mismatch_penalty=0.0,
    )
    reference_utility = _utility_score(
        reference_quality,
        budget_pressure=budget_pressure,
        deadline_slack_ratio=deadline_slack_ratio,
        mismatch_penalty=point.view_mismatch,
    )
    selected_action = _selected_action(
        method_id,
        gaussian_quality=gaussian_quality,
        reference_quality=reference_quality,
        gaussian_utility=gaussian_utility,
        reference_utility=reference_utility,
        budget_pressure=budget_pressure,
        deadline_slack_ratio=deadline_slack_ratio,
    )
    selected_quality = _selected_quality(selected_action, gaussian_quality=gaussian_quality, reference_quality=reference_quality)
    utility_score = _selected_utility(selected_action, gaussian_utility=gaussian_utility, reference_utility=reference_utility)
    if method_id == "perfect-information-oracle":
        utility_score = max(gaussian_utility, reference_utility) + 0.03
    return SubstitutionOutcome(
        point=point,
        method_id=method_id,
        seed=parsed_seed,
        selected_action=selected_action,
        gaussian_quality=gaussian_quality,
        reference_quality=reference_quality,
        substitution_gain=selected_quality - gaussian_quality,
        utility_score=utility_score,
        budget_pressure=budget_pressure,
        deadline_slack_ratio=deadline_slack_ratio,
        metadata={
            "gaussian_cost_bytes": gaussian_cost,
            "reference_cost_bytes": reference_cost,
            "reference_time_ms": reference_time,
            "gaussian_utility": gaussian_utility,
            "reference_utility": reference_utility,
            "model": "deterministic_substitution_surface_v1",
        },
    )


def export_substitution_surface_outputs(output_root: str | Path, result: SubstitutionSurfaceResult) -> tuple[Path, Path]:
    """Write JSONL substitution outcomes and a JSON summary."""

    if not isinstance(result, SubstitutionSurfaceResult):
        raise SubstitutionSurfaceError("result must be a SubstitutionSurfaceResult record.")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    outcomes_path = root / "substitution_surface.jsonl"
    summary_path = root / "substitution_surface_summary.json"
    outcome_lines = "".join(json.dumps(outcome.as_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for outcome in result.outcomes)
    summary_content = json.dumps(result.as_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    _write_text_atomic(outcomes_path, outcome_lines)
    _write_text_atomic(summary_path, summary_content)
    return outcomes_path, summary_path


def _selected_action(
    method_id: str,
    *,
    gaussian_quality: float,
    reference_quality: float,
    gaussian_utility: float,
    reference_utility: float,
    budget_pressure: float,
    deadline_slack_ratio: float,
) -> SubstitutionAction:
    if method_id == "svq-gaussian-only-abr":
        return "gaussian"
    if method_id == "reference-only-after-base":
        return "reference"
    if method_id == "quality-max-deadline-unaware":
        return "reference" if reference_quality >= gaussian_quality else "gaussian"
    if method_id == "deadline-aware-knapsack-allocator":
        if reference_utility > gaussian_utility and budget_pressure <= 1.2 and deadline_slack_ratio >= 0.75:
            return "reference"
        if reference_utility > gaussian_utility * 0.95 and budget_pressure <= 1.5:
            return "mixed"
        return "gaussian"
    if method_id == "perfect-information-oracle":
        return "reference" if reference_utility >= gaussian_utility else "gaussian"
    raise SubstitutionSurfaceError(f"Unknown substitution method_id {method_id!r}.")


def _selected_quality(action: SubstitutionAction, *, gaussian_quality: float, reference_quality: float) -> float:
    if action == "gaussian":
        return gaussian_quality
    if action == "reference":
        return reference_quality
    return 0.5 * (gaussian_quality + reference_quality)


def _selected_utility(action: SubstitutionAction, *, gaussian_utility: float, reference_utility: float) -> float:
    if action == "gaussian":
        return gaussian_utility
    if action == "reference":
        return reference_utility
    return 0.5 * (gaussian_utility + reference_utility)


def _gaussian_quality(point: SubstitutionSurfacePoint, seed: int) -> float:
    seed_adjustment = ((seed % 7) - 3) * 0.003
    quality = 0.52 + 0.045 * min(point.layer, 5) + 0.05 * _fov_factor(point.fov_deg) + seed_adjustment
    return _clamp01(quality)


def _reference_quality(point: SubstitutionSurfacePoint, seed: int) -> float:
    seed_adjustment = ((seed % 5) - 2) * 0.004
    quality = (
        0.48
        + 0.25 * _resolution_factor(point.ref_resolution)
        + 0.08 * _fov_factor(point.fov_deg)
        - 0.32 * point.view_mismatch
        + 0.025 * min(point.layer, 4)
        + seed_adjustment
    )
    return _clamp01(quality)


def _gaussian_cost_bytes(point: SubstitutionSurfacePoint) -> int:
    return int(round(260_000 * (1.0 + 0.18 * point.layer) * (0.75 + 0.5 * _fov_factor(point.fov_deg))))


def _reference_cost_bytes(point: SubstitutionSurfacePoint) -> int:
    base = {
        "360p": 380_000,
        "540p": 620_000,
        "720p": 900_000,
        "1080p": 1_550_000,
        "1440p": 2_300_000,
    }.get(point.ref_resolution, 900_000)
    return int(round(base * (1.0 + 0.22 * point.layer) * (0.65 + 0.7 * _fov_factor(point.fov_deg))))


def _reference_time_ms(point: SubstitutionSurfacePoint) -> float:
    return 12.0 + 4.0 * point.layer + 22.0 * _resolution_factor(point.ref_resolution) + 8.0 * _fov_factor(point.fov_deg)


def _utility_score(quality: float, *, budget_pressure: float, deadline_slack_ratio: float, mismatch_penalty: float) -> float:
    budget_penalty = max(0.0, budget_pressure - 1.0) * 0.18
    deadline_penalty = max(0.0, 1.0 - deadline_slack_ratio) * 0.28
    return quality - budget_penalty - deadline_penalty - 0.12 * mismatch_penalty


def _resolution_factor(ref_resolution: str) -> float:
    table = {
        "360p": 0.20,
        "540p": 0.35,
        "720p": 0.55,
        "1080p": 0.78,
        "1440p": 0.92,
    }
    if ref_resolution in table:
        return table[ref_resolution]
    digits = "".join(ch for ch in ref_resolution if ch.isdigit())
    if digits:
        return _clamp01(int(digits) / 1600.0)
    return 0.55


def _fov_factor(fov_deg: float) -> float:
    return _clamp01((fov_deg - 45.0) / 90.0)


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
        "metric_id": f"substitution-metric-{stable_config_id({'metric_name': metric_name, 'value': value, 'tags': dict(tags)})}",
    }
    return MetricRecord(metric_name=metric_name, value=value, unit=unit, tags=tags, split=split, metadata=metric_metadata)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise SubstitutionSurfaceError(f"Failed to write substitution output {path}: {exc}") from exc


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SubstitutionSurfaceError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    if not parsed:
        raise SubstitutionSurfaceError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _string_mapping(value: Mapping[str, str], field_name: str) -> Mapping[str, str]:
    if not isinstance(value, MappingABC):
        raise SubstitutionSurfaceError(f"{field_name} must be a mapping.")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        _require_non_empty(str(key), f"{field_name} key")
        _require_non_empty(item, f"{field_name}.{key}")
        parsed[str(key)] = item
    return {key: parsed[key] for key in sorted(parsed)}


def _int_tuple(values: Sequence[int], field_name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SubstitutionSurfaceError(f"{field_name} must be a sequence of integers.")
    parsed: list[int] = []
    seen: set[int] = set()
    for value in values:
        parsed_value = _non_negative_int(value, field_name)
        if parsed_value not in seen:
            parsed.append(parsed_value)
            seen.add(parsed_value)
    if not parsed:
        raise SubstitutionSurfaceError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _positive_int_tuple(values: Sequence[int], field_name: str) -> tuple[int, ...]:
    parsed = tuple(_positive_int(value, field_name) for value in values)
    if not parsed:
        raise SubstitutionSurfaceError(f"{field_name} must not be empty.")
    return tuple(dict.fromkeys(parsed))


def _positive_float_tuple(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    parsed = tuple(_positive_float(value, field_name) for value in values)
    if not parsed:
        raise SubstitutionSurfaceError(f"{field_name} must not be empty.")
    return tuple(dict.fromkeys(parsed))


def _non_negative_float_tuple(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    parsed = tuple(_non_negative_float(value, field_name) for value in values)
    if not parsed:
        raise SubstitutionSurfaceError(f"{field_name} must not be empty.")
    return tuple(dict.fromkeys(parsed))


def _unit_interval_tuple(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    parsed = tuple(_unit_interval(value, field_name) for value in values)
    if not parsed:
        raise SubstitutionSurfaceError(f"{field_name} must not be empty.")
    return tuple(dict.fromkeys(parsed))


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise SubstitutionSurfaceError(f"{field_name} must be a mapping.")
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
        raise SubstitutionSurfaceError(f"{field_name} must be a non-empty string.")


def _positive_int(value: Any, field_name: str) -> int:
    parsed = _non_negative_int(value, field_name)
    if parsed <= 0:
        raise SubstitutionSurfaceError(f"{field_name} must be a positive integer.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise SubstitutionSurfaceError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SubstitutionSurfaceError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise SubstitutionSurfaceError(f"{field_name} must be a non-negative integer.")
    return parsed


def _positive_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed <= 0.0:
        raise SubstitutionSurfaceError(f"{field_name} must be positive.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0:
        raise SubstitutionSurfaceError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if not 0.0 <= parsed <= 1.0:
        raise SubstitutionSurfaceError(f"{field_name} must be between 0 and 1.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise SubstitutionSurfaceError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SubstitutionSurfaceError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise SubstitutionSurfaceError(f"{field_name} must be finite.")
    return parsed


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = [
    "SUBSTITUTION_METHOD_IDS",
    "SUBSTITUTION_METRIC_NAMES",
    "SubstitutionOutcome",
    "SubstitutionSurfaceConfig",
    "SubstitutionSurfaceError",
    "SubstitutionSurfacePoint",
    "SubstitutionSurfaceResult",
    "evaluate_substitution_point",
    "export_substitution_surface_outputs",
    "run_gaussian_reference_substitution_surface",
]
