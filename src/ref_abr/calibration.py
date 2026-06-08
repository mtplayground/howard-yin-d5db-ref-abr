"""Utility and lifecycle calibration output records."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ref_abr.config import stable_config_id
from ref_abr.lifecycle_deadline_harness import LifecycleDeadlineOutcome, LifecycleDeadlineResult
from ref_abr.schema import DOMAIN_SCHEMA_VERSION, RECORD_TYPE_FIELD, SCHEMA_VERSION_FIELD
from ref_abr.substrate import ParametricSubstrateCoefficients, ParametricSubstrateValueProvider
from ref_abr.substitution_surface import SubstitutionOutcome, SubstitutionSurfaceResult
from ref_abr.utility import UtilityModelWeights


CALIBRATION_RECORD_TYPE = "utility_lifecycle_calibration"
CALIBRATION_OUTPUT_FILENAME = "utility_lifecycle_calibration.json"


class CalibrationError(ValueError):
    """Raised when calibration inputs or persisted calibration output are invalid."""


@dataclass(frozen=True)
class CalibrationUncertainty:
    """Aggregate uncertainty from substitution and lifecycle calibration sources."""

    substitution_stddev: float
    lifecycle_risk_stddev: float
    utility_weight_stddev: float
    substrate_coefficient_stddev: float
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "substitution_stddev", _non_negative_float(self.substitution_stddev, "substitution_stddev"))
        object.__setattr__(self, "lifecycle_risk_stddev", _non_negative_float(self.lifecycle_risk_stddev, "lifecycle_risk_stddev"))
        object.__setattr__(self, "utility_weight_stddev", _non_negative_float(self.utility_weight_stddev, "utility_weight_stddev"))
        object.__setattr__(
            self,
            "substrate_coefficient_stddev",
            _non_negative_float(self.substrate_coefficient_stddev, "substrate_coefficient_stddev"),
        )
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, "confidence"))

    def as_payload(self) -> dict[str, float]:
        return {
            "substitution_stddev": self.substitution_stddev,
            "lifecycle_risk_stddev": self.lifecycle_risk_stddev,
            "utility_weight_stddev": self.utility_weight_stddev,
            "substrate_coefficient_stddev": self.substrate_coefficient_stddev,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CalibrationParameter:
    """One calibrated scalar with source and uncertainty metadata."""

    name: str
    value: float
    uncertainty: float
    unit: str
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "name")
        object.__setattr__(self, "value", _finite_float(self.value, "value"))
        object.__setattr__(self, "uncertainty", _non_negative_float(self.uncertainty, "uncertainty"))
        _require_non_empty(self.unit, "unit")
        _require_non_empty(self.source, "source")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "unit": self.unit,
            "source": self.source,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class UtilityLifecycleCalibration:
    """Persisted calibration manifest consumed by utility and candidate methods."""

    calibration_id: str
    split: str
    utility_model_weights: UtilityModelWeights
    substrate_coefficients: ParametricSubstrateCoefficients
    lifecycle_coefficients: Mapping[str, float]
    uncertainty: CalibrationUncertainty
    parameters: tuple[CalibrationParameter, ...]
    source_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.calibration_id, "calibration_id")
        _require_non_empty(self.split, "split")
        if not isinstance(self.utility_model_weights, UtilityModelWeights):
            raise CalibrationError("utility_model_weights must be a UtilityModelWeights record.")
        if not isinstance(self.substrate_coefficients, ParametricSubstrateCoefficients):
            raise CalibrationError("substrate_coefficients must be a ParametricSubstrateCoefficients record.")
        if not isinstance(self.uncertainty, CalibrationUncertainty):
            raise CalibrationError("uncertainty must be a CalibrationUncertainty record.")
        lifecycle_coefficients = _float_mapping(self.lifecycle_coefficients, "lifecycle_coefficients")
        parameters = tuple(self.parameters)
        parameter_names = [parameter.name for parameter in parameters]
        duplicates = sorted({name for name in parameter_names if parameter_names.count(name) > 1})
        if duplicates:
            raise CalibrationError(f"parameters must not contain duplicate name values: {', '.join(duplicates)}.")
        for parameter in parameters:
            if not isinstance(parameter, CalibrationParameter):
                raise CalibrationError("parameters must contain CalibrationParameter records.")
        object.__setattr__(self, "lifecycle_coefficients", lifecycle_coefficients)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "source_ids", _string_tuple(self.source_ids, "source_ids"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @classmethod
    def from_parts(
        cls,
        *,
        split: str,
        utility_model_weights: UtilityModelWeights,
        substrate_coefficients: ParametricSubstrateCoefficients,
        lifecycle_coefficients: Mapping[str, float],
        uncertainty: CalibrationUncertainty,
        parameters: Sequence[CalibrationParameter],
        source_ids: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> "UtilityLifecycleCalibration":
        """Build a calibration manifest with a content-derived identifier."""

        _require_non_empty(split, "split")
        if not isinstance(utility_model_weights, UtilityModelWeights):
            raise CalibrationError("utility_model_weights must be a UtilityModelWeights record.")
        if not isinstance(substrate_coefficients, ParametricSubstrateCoefficients):
            raise CalibrationError("substrate_coefficients must be a ParametricSubstrateCoefficients record.")
        if not isinstance(uncertainty, CalibrationUncertainty):
            raise CalibrationError("uncertainty must be a CalibrationUncertainty record.")
        parsed_parameters = tuple(parameters)
        for parameter in parsed_parameters:
            if not isinstance(parameter, CalibrationParameter):
                raise CalibrationError("parameters must contain CalibrationParameter records.")
        payload = {
            "split": split,
            "utility_model_weights": utility_model_weights.as_payload(),
            "substrate_coefficients": substrate_coefficients.as_payload(),
            "lifecycle_coefficients": _float_mapping(lifecycle_coefficients, "lifecycle_coefficients"),
            "uncertainty": uncertainty.as_payload(),
            "parameters": [parameter.as_payload() for parameter in parsed_parameters],
            "source_ids": list(_string_tuple(source_ids, "source_ids")),
            "metadata": _plain_json_mapping(metadata, "metadata"),
        }
        return cls(
            calibration_id=f"utility-lifecycle-calibration-{stable_config_id(payload)}",
            split=split,
            utility_model_weights=utility_model_weights,
            substrate_coefficients=substrate_coefficients,
            lifecycle_coefficients=lifecycle_coefficients,
            uncertainty=uncertainty,
            parameters=parsed_parameters,
            source_ids=tuple(source_ids),
            metadata=metadata or {},
        )

    def to_utility_model_weights(self) -> UtilityModelWeights:
        """Return the calibrated weights record for utility estimation."""

        return self.utility_model_weights

    def to_substrate_provider(self, *, provider_id: str | None = None) -> ParametricSubstrateValueProvider:
        """Return a parametric substrate provider carrying calibrated coefficients."""

        resolved_provider_id = provider_id or f"calibrated-substrate-{self.calibration_id.removeprefix('utility-lifecycle-calibration-')}"
        _require_non_empty(resolved_provider_id, "provider_id")
        return ParametricSubstrateValueProvider(
            coefficients=self.substrate_coefficients,
            provider_id=resolved_provider_id,
            metadata={
                "calibration_id": self.calibration_id,
                "calibration_split": self.split,
                "calibration_uncertainty": self.uncertainty.as_payload(),
                "lifecycle_coefficients": dict(self.lifecycle_coefficients),
            },
        )

    def parameter_map(self) -> dict[str, CalibrationParameter]:
        """Return calibrated parameters keyed by name."""

        return {parameter.name: parameter for parameter in self.parameters}

    def stable_payload(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "utility_model_weights": self.utility_model_weights.as_payload(),
            "substrate_coefficients": self.substrate_coefficients.as_payload(),
            "lifecycle_coefficients": dict(self.lifecycle_coefficients),
            "uncertainty": self.uncertainty.as_payload(),
            "parameters": [parameter.as_payload() for parameter in self.parameters],
            "source_ids": list(self.source_ids),
            "metadata": _to_payload(self.metadata),
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            SCHEMA_VERSION_FIELD: DOMAIN_SCHEMA_VERSION,
            RECORD_TYPE_FIELD: CALIBRATION_RECORD_TYPE,
            "calibration_id": self.calibration_id,
            **self.stable_payload(),
        }


def build_utility_lifecycle_calibration(
    *,
    substitution_result: SubstitutionSurfaceResult,
    lifecycle_result: LifecycleDeadlineResult,
    split: str = "calibration",
    metadata: Mapping[str, Any] | None = None,
) -> UtilityLifecycleCalibration:
    """Derive utility/lifecycle coefficients from issue-32 and issue-33 harness outputs."""

    if not isinstance(substitution_result, SubstitutionSurfaceResult):
        raise CalibrationError("substitution_result must be a SubstitutionSurfaceResult record.")
    if not isinstance(lifecycle_result, LifecycleDeadlineResult):
        raise CalibrationError("lifecycle_result must be a LifecycleDeadlineResult record.")
    _require_non_empty(split, "split")
    substitution_stats = _substitution_stats(substitution_result.outcomes)
    lifecycle_stats = _lifecycle_stats(lifecycle_result.outcomes)

    weights = UtilityModelWeights(
        visible_qoe_weight=1.0 + 0.50 * max(0.0, substitution_stats["mean_gain"]),
        lifecycle_risk_weight=0.25 + 0.75 * lifecycle_stats["mean_risk"] + 0.25 * lifecycle_stats["expired_rate"],
        deadline_miss_weight=0.70 + 0.60 * lifecycle_stats["late_rate"],
        time_price_weight=0.16 + 0.10 * lifecycle_stats["late_rate"],
        transfer_price_weight=0.10 + 0.10 * min(2.0, substitution_stats["mean_budget_pressure"]) / 2.0,
        memory_price_weight=0.04,
        debt_weight=0.18 + 0.22 * lifecycle_stats["expired_rate"],
        uncertainty_weight=0.25 + 0.45 * _clamp01(substitution_stats["gain_stddev"] + lifecycle_stats["risk_stddev"]),
    )
    substrate_coefficients = ParametricSubstrateCoefficients(
        base_quality=_clamp_non_negative(0.55 + 0.20 * substitution_stats["mean_gain"]),
        layer_quality_gain=_clamp_non_negative(0.09 + 0.03 * substitution_stats["selected_reference_rate"]),
        resolution_quality_gain=_clamp_non_negative(0.18 + 0.05 * substitution_stats["mean_gain"]),
        fov_penalty=_clamp_non_negative(0.06 + 0.03 * lifecycle_stats["viewport_penalty_mean"]),
        mismatch_penalty=_clamp_non_negative(0.16 + 0.08 * lifecycle_stats["viewport_penalty_mean"]),
        freshness_penalty=_clamp_non_negative(0.035 + 0.02 * lifecycle_stats["late_rate"]),
        timing_uncertainty_ms=_clamp_non_negative(1.0 + 2.0 * lifecycle_stats["risk_stddev"]),
        uncertainty_base=_clamp_non_negative(0.025 + 0.05 * substitution_stats["gain_stddev"]),
        uncertainty_mismatch=_clamp_non_negative(0.035 + 0.04 * lifecycle_stats["viewport_penalty_mean"]),
        uncertainty_freshness=_clamp_non_negative(0.015 + 0.03 * lifecycle_stats["late_rate"]),
    )
    lifecycle_coefficients = {
        "risk_intercept": _clamp01(lifecycle_stats["mean_risk"]),
        "late_penalty": _clamp01(lifecycle_stats["late_rate"]),
        "expired_penalty": _clamp01(lifecycle_stats["expired_rate"]),
        "useful_credit": _clamp01(lifecycle_stats["useful_rate"]),
        "viewport_error_weight": _clamp01(lifecycle_stats["viewport_penalty_mean"]),
        "deadline_slack_sensitivity": _clamp_non_negative(1.0 + lifecycle_stats["late_rate"] - lifecycle_stats["useful_rate"]),
    }
    uncertainty = CalibrationUncertainty(
        substitution_stddev=substitution_stats["gain_stddev"],
        lifecycle_risk_stddev=lifecycle_stats["risk_stddev"],
        utility_weight_stddev=_stdev(tuple(weights.as_payload().values())),
        substrate_coefficient_stddev=_stdev(tuple(substrate_coefficients.as_payload().values())),
        confidence=_clamp01(1.0 - 0.5 * substitution_stats["gain_stddev"] - 0.5 * lifecycle_stats["risk_stddev"]),
    )
    parameters = _calibration_parameters(
        weights=weights,
        substrate_coefficients=substrate_coefficients,
        lifecycle_coefficients=lifecycle_coefficients,
        uncertainty=uncertainty,
    )
    return UtilityLifecycleCalibration.from_parts(
        split=split,
        utility_model_weights=weights,
        substrate_coefficients=substrate_coefficients,
        lifecycle_coefficients=lifecycle_coefficients,
        uncertainty=uncertainty,
        parameters=parameters,
        source_ids=(substitution_result.surface_id, lifecycle_result.matrix_id),
        metadata={
            "source_metrics": {
                "substitution": substitution_stats,
                "lifecycle": lifecycle_stats,
            },
            **_plain_json_mapping(metadata, "metadata"),
        },
    )


def calibration_from_mapping(payload: Mapping[str, Any]) -> UtilityLifecycleCalibration:
    """Materialize a persisted utility/lifecycle calibration payload."""

    root = _require_mapping(payload, "calibration")
    version = root.get(SCHEMA_VERSION_FIELD)
    if version != DOMAIN_SCHEMA_VERSION:
        raise CalibrationError(f"calibration.{SCHEMA_VERSION_FIELD} must be {DOMAIN_SCHEMA_VERSION}; got {version!r}.")
    record_type = root.get(RECORD_TYPE_FIELD)
    if record_type != CALIBRATION_RECORD_TYPE:
        raise CalibrationError(f"calibration.{RECORD_TYPE_FIELD} must be {CALIBRATION_RECORD_TYPE!r}; got {record_type!r}.")
    weights = UtilityModelWeights(**_require_mapping(root.get("utility_model_weights"), "calibration.utility_model_weights"))
    coefficients = ParametricSubstrateCoefficients.from_mapping(
        _require_mapping(root.get("substrate_coefficients"), "calibration.substrate_coefficients")
    )
    uncertainty = CalibrationUncertainty(
        **_require_mapping(root.get("uncertainty"), "calibration.uncertainty")
    )
    parameters = tuple(
        CalibrationParameter(**_require_mapping(item, f"calibration.parameters[{index}]"))
        for index, item in enumerate(_sequence(root.get("parameters"), "calibration.parameters"))
    )
    return UtilityLifecycleCalibration(
        calibration_id=_string_field(root.get("calibration_id"), "calibration.calibration_id"),
        split=_string_field(root.get("split"), "calibration.split"),
        utility_model_weights=weights,
        substrate_coefficients=coefficients,
        lifecycle_coefficients=_float_mapping(
            _require_mapping(root.get("lifecycle_coefficients"), "calibration.lifecycle_coefficients"),
            "calibration.lifecycle_coefficients",
        ),
        uncertainty=uncertainty,
        parameters=parameters,
        source_ids=_string_tuple(_sequence(root.get("source_ids"), "calibration.source_ids"), "calibration.source_ids"),
        metadata=_plain_json_mapping(root.get("metadata"), "calibration.metadata") if root.get("metadata") is not None else {},
    )


def export_utility_lifecycle_calibration(
    output_path: str | Path,
    calibration: UtilityLifecycleCalibration,
) -> Path:
    """Persist a schema-stamped calibration manifest as deterministic JSON."""

    if not isinstance(calibration, UtilityLifecycleCalibration):
        raise CalibrationError("calibration must be a UtilityLifecycleCalibration record.")
    path = _output_file_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(calibration.as_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise CalibrationError(f"Failed to write calibration output {path}: {exc}") from exc
    return path


def load_utility_lifecycle_calibration(path: str | Path) -> UtilityLifecycleCalibration:
    """Load a persisted utility/lifecycle calibration manifest."""

    calibration_path = Path(path)
    if calibration_path.is_dir():
        calibration_path = calibration_path / CALIBRATION_OUTPUT_FILENAME
    if not calibration_path.exists():
        raise CalibrationError(f"Calibration file does not exist: {calibration_path}")
    if not calibration_path.is_file():
        raise CalibrationError(f"Calibration path is not a file: {calibration_path}")
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"Could not read calibration file {calibration_path}: {exc}") from exc
    return calibration_from_mapping(_require_mapping(payload, "calibration"))


def load_calibrated_utility_model_weights(path: str | Path) -> UtilityModelWeights:
    """Load calibrated utility weights for candidate utility estimation."""

    return load_utility_lifecycle_calibration(path).to_utility_model_weights()


def load_calibrated_substrate_provider(
    path: str | Path,
    *,
    provider_id: str | None = None,
) -> ParametricSubstrateValueProvider:
    """Load a calibrated parametric substrate provider for candidate generation."""

    return load_utility_lifecycle_calibration(path).to_substrate_provider(provider_id=provider_id)


def _substitution_stats(outcomes: Sequence[SubstitutionOutcome]) -> dict[str, float]:
    parsed = _substitution_outcome_tuple(outcomes)
    gains = tuple(outcome.substitution_gain for outcome in parsed)
    pressures = tuple(outcome.budget_pressure for outcome in parsed)
    selected_reference = tuple(1.0 if outcome.selected_action in {"reference", "mixed"} else 0.0 for outcome in parsed)
    return {
        "outcome_count": float(len(parsed)),
        "mean_gain": _mean(gains),
        "gain_stddev": _stdev(gains),
        "mean_budget_pressure": _mean(pressures),
        "selected_reference_rate": _mean(selected_reference),
    }


def _lifecycle_stats(outcomes: Sequence[LifecycleDeadlineOutcome]) -> dict[str, float]:
    parsed = _lifecycle_outcome_tuple(outcomes)
    risks = tuple(outcome.lifecycle_deadline_risk for outcome in parsed)
    late = tuple(1.0 if outcome.late else 0.0 for outcome in parsed)
    expired = tuple(1.0 if outcome.expired else 0.0 for outcome in parsed)
    useful = tuple(1.0 if outcome.useful else 0.0 for outcome in parsed)
    viewport_penalties = tuple(outcome.viewport_error_penalty for outcome in parsed)
    return {
        "outcome_count": float(len(parsed)),
        "mean_risk": _mean(risks),
        "risk_stddev": _stdev(risks),
        "late_rate": _mean(late),
        "expired_rate": _mean(expired),
        "useful_rate": _mean(useful),
        "viewport_penalty_mean": _mean(viewport_penalties),
    }


def _calibration_parameters(
    *,
    weights: UtilityModelWeights,
    substrate_coefficients: ParametricSubstrateCoefficients,
    lifecycle_coefficients: Mapping[str, float],
    uncertainty: CalibrationUncertainty,
) -> tuple[CalibrationParameter, ...]:
    parameters: list[CalibrationParameter] = []
    weight_uncertainty = uncertainty.utility_weight_stddev
    substrate_uncertainty = uncertainty.substrate_coefficient_stddev
    lifecycle_uncertainty = uncertainty.lifecycle_risk_stddev
    for name, value in weights.as_payload().items():
        parameters.append(CalibrationParameter(f"utility.{name}", value, weight_uncertainty, "weight", "utility_model"))
    for name, value in substrate_coefficients.as_payload().items():
        parameters.append(CalibrationParameter(f"substrate.{name}", value, substrate_uncertainty, "coefficient", "substrate_model"))
    for name, value in sorted(lifecycle_coefficients.items()):
        parameters.append(CalibrationParameter(f"lifecycle.{name}", value, lifecycle_uncertainty, "coefficient", "lifecycle_model"))
    return tuple(parameters)


def _output_file_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix:
        return path
    return path / CALIBRATION_OUTPUT_FILENAME


def _substitution_outcome_tuple(values: Sequence[SubstitutionOutcome]) -> tuple[SubstitutionOutcome, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CalibrationError("substitution outcomes must be a sequence of SubstitutionOutcome records.")
    parsed = tuple(values)
    if not parsed:
        raise CalibrationError("substitution outcomes must not be empty.")
    for value in parsed:
        if not isinstance(value, SubstitutionOutcome):
            raise CalibrationError("substitution outcomes must contain SubstitutionOutcome records.")
    return parsed


def _lifecycle_outcome_tuple(values: Sequence[LifecycleDeadlineOutcome]) -> tuple[LifecycleDeadlineOutcome, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CalibrationError("lifecycle outcomes must be a sequence of LifecycleDeadlineOutcome records.")
    parsed = tuple(values)
    if not parsed:
        raise CalibrationError("lifecycle outcomes must not be empty.")
    for value in parsed:
        if not isinstance(value, LifecycleDeadlineOutcome):
            raise CalibrationError("lifecycle outcomes must contain LifecycleDeadlineOutcome records.")
    return parsed


def _sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CalibrationError(f"{field_name} must be a sequence.")
    return tuple(value)


def _string_tuple(values: Sequence[Any], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CalibrationError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        parsed_value = _string_field(value, field_name)
        if parsed_value not in seen:
            parsed.append(parsed_value)
            seen.add(parsed_value)
    if not parsed:
        raise CalibrationError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _float_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, float]:
    mapping = _require_mapping(value, field_name)
    parsed = {str(key): _finite_float(item, f"{field_name}.{key}") for key, item in mapping.items()}
    if not parsed:
        raise CalibrationError(f"{field_name} must not be empty.")
    return {key: parsed[key] for key in sorted(parsed)}


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    mapping = _require_mapping(value, field_name)
    return {str(key): _to_payload(item) for key, item in mapping.items()}


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


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise CalibrationError(f"{field_name} must be a mapping.")
    return value


def _string_field(value: Any, field_name: str) -> str:
    _require_non_empty(value, field_name)
    return value.strip()


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationError(f"{field_name} must be a non-empty string.")


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0:
        raise CalibrationError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if not 0.0 <= parsed <= 1.0:
        raise CalibrationError(f"{field_name} must be between 0 and 1.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise CalibrationError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise CalibrationError(f"{field_name} must be finite.")
    return parsed


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise CalibrationError("Cannot compute mean over an empty sequence.")
    return sum(values) / len(values)


def _stdev(values: Sequence[float]) -> float:
    if not values:
        raise CalibrationError("Cannot compute stddev over an empty sequence.")
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_non_negative(value: float) -> float:
    return max(0.0, value)


__all__ = [
    "CALIBRATION_OUTPUT_FILENAME",
    "CALIBRATION_RECORD_TYPE",
    "CalibrationError",
    "CalibrationParameter",
    "CalibrationUncertainty",
    "UtilityLifecycleCalibration",
    "build_utility_lifecycle_calibration",
    "calibration_from_mapping",
    "export_utility_lifecycle_calibration",
    "load_calibrated_substrate_provider",
    "load_calibrated_utility_model_weights",
    "load_utility_lifecycle_calibration",
]
