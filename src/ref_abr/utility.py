"""Utility and deadline estimation for candidate actions."""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.candidates import CandidateObject, CandidateSet
from ref_abr.config import stable_config_id
from ref_abr.devices import DeviceBudgets, DeviceProfile
from ref_abr.substrate import (
    ComponentTiming,
    SubstrateQuery,
    SubstrateUncertainty,
    SubstrateValue,
    SubstrateValueProvider,
    coerce_ref_resolution,
)


class UtilityError(ValueError):
    """Raised when utility model inputs are malformed or incomplete."""


@dataclass(frozen=True)
class ViewportRisk:
    """Normalized viewport-prediction risk components for one decision epoch."""

    mismatch_probability: float = 0.0
    occlusion_probability: float = 0.0
    motion_instability: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mismatch_probability", _unit_interval(self.mismatch_probability, "mismatch_probability"))
        object.__setattr__(self, "occlusion_probability", _unit_interval(self.occlusion_probability, "occlusion_probability"))
        object.__setattr__(self, "motion_instability", _unit_interval(self.motion_instability, "motion_instability"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def aggregate(self) -> float:
        """Combined viewport risk in [0, 1]."""

        return _clamp01(
            0.55 * self.mismatch_probability
            + 0.25 * self.occlusion_probability
            + 0.20 * self.motion_instability
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "mismatch_probability": self.mismatch_probability,
            "occlusion_probability": self.occlusion_probability,
            "motion_instability": self.motion_instability,
            "aggregate": self.aggregate,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class ResourceBudget:
    """Per-decision resource budgets and carried debts."""

    available_time_ms: float
    available_bytes: int
    available_memory_mb: float
    queue_debt_ms: float = 0.0
    transfer_debt_bytes: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "available_time_ms", _positive_float(self.available_time_ms, "available_time_ms"))
        object.__setattr__(self, "available_bytes", _positive_int(self.available_bytes, "available_bytes"))
        object.__setattr__(self, "available_memory_mb", _positive_float(self.available_memory_mb, "available_memory_mb"))
        object.__setattr__(self, "queue_debt_ms", _non_negative_float(self.queue_debt_ms, "queue_debt_ms"))
        object.__setattr__(self, "transfer_debt_bytes", _non_negative_int(self.transfer_debt_bytes, "transfer_debt_bytes"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @classmethod
    def from_device_profile(
        cls,
        profile: DeviceProfile,
        *,
        available_bytes: int,
        queue_debt_ms: float = 0.0,
        transfer_debt_bytes: int = 0,
    ) -> "ResourceBudget":
        """Build resource budgets from a device profile and network byte budget."""

        if not isinstance(profile, DeviceProfile):
            raise UtilityError("profile must be a DeviceProfile record.")
        budgets = profile.budgets
        return cls(
            available_time_ms=budgets.generation_ms + budgets.transfer_ms + budgets.restoration_ms + budgets.render_ms,
            available_bytes=available_bytes,
            available_memory_mb=budgets.memory_mb,
            queue_debt_ms=queue_debt_ms,
            transfer_debt_bytes=transfer_debt_bytes,
            metadata={"device_profile_id": profile.profile_id, "device_class": profile.device_class},
        )

    @classmethod
    def from_device_budgets(
        cls,
        budgets: DeviceBudgets,
        *,
        available_bytes: int,
        queue_debt_ms: float = 0.0,
        transfer_debt_bytes: int = 0,
    ) -> "ResourceBudget":
        """Build resource budgets from raw device timing and memory budgets."""

        if not isinstance(budgets, DeviceBudgets):
            raise UtilityError("budgets must be a DeviceBudgets record.")
        return cls(
            available_time_ms=budgets.generation_ms + budgets.transfer_ms + budgets.restoration_ms + budgets.render_ms,
            available_bytes=available_bytes,
            available_memory_mb=budgets.memory_mb,
            queue_debt_ms=queue_debt_ms,
            transfer_debt_bytes=transfer_debt_bytes,
            metadata={"fps": budgets.fps},
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "available_time_ms": self.available_time_ms,
            "available_bytes": self.available_bytes,
            "available_memory_mb": self.available_memory_mb,
            "queue_debt_ms": self.queue_debt_ms,
            "transfer_debt_bytes": self.transfer_debt_bytes,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class UtilityModelWeights:
    """Calibratable weights for the utility and deadline model."""

    visible_qoe_weight: float = 1.0
    lifecycle_risk_weight: float = 0.35
    deadline_miss_weight: float = 0.9
    time_price_weight: float = 0.18
    transfer_price_weight: float = 0.12
    memory_price_weight: float = 0.04
    debt_weight: float = 0.20
    uncertainty_weight: float = 0.30

    def __post_init__(self) -> None:
        for field_name in (
            "visible_qoe_weight",
            "lifecycle_risk_weight",
            "deadline_miss_weight",
            "time_price_weight",
            "transfer_price_weight",
            "memory_price_weight",
            "debt_weight",
            "uncertainty_weight",
        ):
            object.__setattr__(self, field_name, _non_negative_float(getattr(self, field_name), field_name))

    def as_payload(self) -> dict[str, float]:
        return {
            "visible_qoe_weight": self.visible_qoe_weight,
            "lifecycle_risk_weight": self.lifecycle_risk_weight,
            "deadline_miss_weight": self.deadline_miss_weight,
            "time_price_weight": self.time_price_weight,
            "transfer_price_weight": self.transfer_price_weight,
            "memory_price_weight": self.memory_price_weight,
            "debt_weight": self.debt_weight,
            "uncertainty_weight": self.uncertainty_weight,
        }


@dataclass(frozen=True)
class ResourcePrice:
    """Normalized resource price components for one candidate."""

    time_price: float
    transfer_price: float
    memory_price: float

    def __post_init__(self) -> None:
        for field_name in ("time_price", "transfer_price", "memory_price"):
            object.__setattr__(self, field_name, _non_negative_float(getattr(self, field_name), field_name))

    @property
    def total(self) -> float:
        return self.time_price + self.transfer_price + self.memory_price

    def as_payload(self) -> dict[str, float]:
        return {
            "time_price": self.time_price,
            "transfer_price": self.transfer_price,
            "memory_price": self.memory_price,
            "total": self.total,
        }


@dataclass(frozen=True)
class ResourceDebt:
    """Budget-overrun and queued-debt components for one candidate."""

    time_debt_ms: float
    transfer_debt_bytes: int
    memory_debt_mb: float
    carried_queue_debt_ms: float
    carried_transfer_debt_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_debt_ms", _non_negative_float(self.time_debt_ms, "time_debt_ms"))
        object.__setattr__(self, "transfer_debt_bytes", _non_negative_int(self.transfer_debt_bytes, "transfer_debt_bytes"))
        object.__setattr__(self, "memory_debt_mb", _non_negative_float(self.memory_debt_mb, "memory_debt_mb"))
        object.__setattr__(self, "carried_queue_debt_ms", _non_negative_float(self.carried_queue_debt_ms, "carried_queue_debt_ms"))
        object.__setattr__(self, "carried_transfer_debt_bytes", _non_negative_int(self.carried_transfer_debt_bytes, "carried_transfer_debt_bytes"))

    @property
    def normalized_total(self) -> float:
        return (
            self.time_debt_ms
            + self.memory_debt_mb
            + (self.transfer_debt_bytes / 1_000_000.0)
            + self.carried_queue_debt_ms
            + (self.carried_transfer_debt_bytes / 1_000_000.0)
        )

    def as_payload(self) -> dict[str, float | int]:
        return {
            "time_debt_ms": self.time_debt_ms,
            "transfer_debt_bytes": self.transfer_debt_bytes,
            "memory_debt_mb": self.memory_debt_mb,
            "carried_queue_debt_ms": self.carried_queue_debt_ms,
            "carried_transfer_debt_bytes": self.carried_transfer_debt_bytes,
            "normalized_total": self.normalized_total,
        }


@dataclass(frozen=True)
class UtilityUncertainty:
    """Uncertainty recorded for utility and deadline estimates."""

    quality_stddev: float
    timing_stddev_ms: float
    deadline_probability_stddev: float
    utility_stddev: float
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "quality_stddev", _non_negative_float(self.quality_stddev, "quality_stddev"))
        object.__setattr__(self, "timing_stddev_ms", _non_negative_float(self.timing_stddev_ms, "timing_stddev_ms"))
        object.__setattr__(
            self,
            "deadline_probability_stddev",
            _non_negative_float(self.deadline_probability_stddev, "deadline_probability_stddev"),
        )
        object.__setattr__(self, "utility_stddev", _non_negative_float(self.utility_stddev, "utility_stddev"))
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, "confidence"))

    def as_payload(self) -> dict[str, float]:
        return {
            "quality_stddev": self.quality_stddev,
            "timing_stddev_ms": self.timing_stddev_ms,
            "deadline_probability_stddev": self.deadline_probability_stddev,
            "utility_stddev": self.utility_stddev,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CandidateUtilityEstimate:
    """Utility estimate for a single feasible candidate."""

    estimate_id: str
    candidate_id: str
    visible_qoe_gain: float
    lifecycle_risk: float
    deadline_miss_probability: float
    resource_price: ResourcePrice
    resource_debt: ResourceDebt
    expected_utility: float
    uncertainty: UtilityUncertainty
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.estimate_id, "estimate_id")
        _require_non_empty(self.candidate_id, "candidate_id")
        object.__setattr__(self, "visible_qoe_gain", _non_negative_float(self.visible_qoe_gain, "visible_qoe_gain"))
        object.__setattr__(self, "lifecycle_risk", _unit_interval(self.lifecycle_risk, "lifecycle_risk"))
        object.__setattr__(
            self,
            "deadline_miss_probability",
            _unit_interval(self.deadline_miss_probability, "deadline_miss_probability"),
        )
        if not isinstance(self.resource_price, ResourcePrice):
            raise UtilityError("resource_price must be a ResourcePrice record.")
        if not isinstance(self.resource_debt, ResourceDebt):
            raise UtilityError("resource_debt must be a ResourceDebt record.")
        if not isinstance(self.uncertainty, UtilityUncertainty):
            raise UtilityError("uncertainty must be a UtilityUncertainty record.")
        object.__setattr__(self, "expected_utility", _finite_float(self.expected_utility, "expected_utility"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "estimate_id": self.estimate_id,
            "candidate_id": self.candidate_id,
            "visible_qoe_gain": self.visible_qoe_gain,
            "lifecycle_risk": self.lifecycle_risk,
            "deadline_miss_probability": self.deadline_miss_probability,
            "resource_price": self.resource_price.as_payload(),
            "resource_debt": self.resource_debt.as_payload(),
            "expected_utility": self.expected_utility,
            "uncertainty": self.uncertainty.as_payload(),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class UtilityEstimateSet:
    """Utility estimates for a candidate set."""

    estimate_set_id: str
    candidate_set_id: str
    estimates: tuple[CandidateUtilityEstimate, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.estimate_set_id, "estimate_set_id")
        _require_non_empty(self.candidate_set_id, "candidate_set_id")
        estimates = tuple(self.estimates)
        if not estimates:
            raise UtilityError("estimates must contain at least one utility estimate.")
        estimate_ids = [estimate.estimate_id for estimate in estimates]
        duplicates = sorted({estimate_id for estimate_id in estimate_ids if estimate_ids.count(estimate_id) > 1})
        if duplicates:
            raise UtilityError(f"estimates must not contain duplicate estimate_id values: {', '.join(duplicates)}.")
        object.__setattr__(self, "estimates", estimates)
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "estimate_set_id": self.estimate_set_id,
            "candidate_set_id": self.candidate_set_id,
            "estimates": [estimate.as_payload() for estimate in self.estimates],
            "metadata": _to_payload(self.metadata),
        }


def estimate_candidate_utility(
    candidate: CandidateObject,
    *,
    substrate_value: SubstrateValue | Mapping[str, Any] | None = None,
    substrate_provider: SubstrateValueProvider | None = None,
    viewport_risk: ViewportRisk | Mapping[str, Any] | None = None,
    budgets: ResourceBudget | DeviceProfile | DeviceBudgets | Mapping[str, Any] | None = None,
    model_weights: UtilityModelWeights | Mapping[str, Any] | None = None,
) -> CandidateUtilityEstimate:
    """Estimate visible utility, deadline risk, and resource cost for one candidate."""

    if not isinstance(candidate, CandidateObject):
        raise UtilityError("candidate must be a CandidateObject record.")
    risk = coerce_viewport_risk(viewport_risk)
    resource_budget = coerce_resource_budget(budgets)
    weights = coerce_utility_model_weights(model_weights)
    substrate = coerce_substrate_value(substrate_value) if substrate_value is not None else _substrate_for_candidate(candidate, substrate_provider)

    timing = substrate.component_timing
    uncertainty = substrate.uncertainty
    visible_qoe_gain = _clamp01(substrate.visible_quality * (1.0 - 0.70 * risk.aggregate))
    lifecycle_risk = _lifecycle_risk(candidate, risk, uncertainty)
    deadline_miss_probability = _deadline_miss_probability(candidate, timing, uncertainty, resource_budget)
    memory_mb = _candidate_memory_mb(candidate)
    resource_price = ResourcePrice(
        time_price=weights.time_price_weight * min(4.0, timing.total_ms / resource_budget.available_time_ms),
        transfer_price=weights.transfer_price_weight * min(4.0, candidate.size_bytes / resource_budget.available_bytes),
        memory_price=weights.memory_price_weight * min(4.0, memory_mb / resource_budget.available_memory_mb),
    )
    resource_debt = ResourceDebt(
        time_debt_ms=max(0.0, timing.total_ms - min(float(candidate.expiration_ms), resource_budget.available_time_ms)),
        transfer_debt_bytes=max(0, candidate.size_bytes - resource_budget.available_bytes),
        memory_debt_mb=max(0.0, memory_mb - resource_budget.available_memory_mb),
        carried_queue_debt_ms=resource_budget.queue_debt_ms,
        carried_transfer_debt_bytes=resource_budget.transfer_debt_bytes,
    )
    utility_uncertainty = _utility_uncertainty(uncertainty, deadline_miss_probability, risk, weights)
    debt_penalty = weights.debt_weight * _normalized_debt(resource_debt, resource_budget)
    expected_utility = (
        weights.visible_qoe_weight * visible_qoe_gain
        - weights.lifecycle_risk_weight * lifecycle_risk
        - weights.deadline_miss_weight * deadline_miss_probability
        - resource_price.total
        - debt_penalty
        - weights.uncertainty_weight * utility_uncertainty.utility_stddev
    )
    payload = {
        "candidate_id": candidate.candidate_id,
        "visible_qoe_gain": visible_qoe_gain,
        "lifecycle_risk": lifecycle_risk,
        "deadline_miss_probability": deadline_miss_probability,
        "resource_price": resource_price.as_payload(),
        "resource_debt": resource_debt.as_payload(),
        "expected_utility": expected_utility,
        "uncertainty": utility_uncertainty.as_payload(),
        "weights": weights.as_payload(),
        "budgets": resource_budget.as_payload(),
        "viewport_risk": risk.as_payload(),
        "substrate": substrate.as_payload(),
    }
    return CandidateUtilityEstimate(
        estimate_id=f"utility-{stable_config_id(payload)}",
        candidate_id=candidate.candidate_id,
        visible_qoe_gain=visible_qoe_gain,
        lifecycle_risk=lifecycle_risk,
        deadline_miss_probability=deadline_miss_probability,
        resource_price=resource_price,
        resource_debt=resource_debt,
        expected_utility=expected_utility,
        uncertainty=utility_uncertainty,
        metadata={
            "model": {
                "kind": "utility_deadline",
                "weights": weights.as_payload(),
                "debt_penalty": debt_penalty,
                "candidate_kind": candidate.candidate_kind,
            },
            "inputs": {
                "substrate_provider_id": substrate.provider_id,
                "viewport_risk": risk.as_payload(),
                "budgets": resource_budget.as_payload(),
            },
        },
    )


def estimate_candidate_set_utility(
    candidate_set: CandidateSet,
    *,
    substrate_provider: SubstrateValueProvider | None = None,
    viewport_risk: ViewportRisk | Mapping[str, Any] | None = None,
    budgets: ResourceBudget | DeviceProfile | DeviceBudgets | Mapping[str, Any] | None = None,
    model_weights: UtilityModelWeights | Mapping[str, Any] | None = None,
) -> UtilityEstimateSet:
    """Estimate utility for every candidate in a candidate set."""

    if not isinstance(candidate_set, CandidateSet):
        raise UtilityError("candidate_set must be a CandidateSet record.")
    estimates = tuple(
        estimate_candidate_utility(
            candidate,
            substrate_provider=substrate_provider,
            viewport_risk=viewport_risk,
            budgets=budgets,
            model_weights=model_weights,
        )
        for candidate in candidate_set.candidates
    )
    payload = {
        "candidate_set_id": candidate_set.candidate_set_id,
        "estimates": [estimate.as_payload() for estimate in estimates],
    }
    return UtilityEstimateSet(
        estimate_set_id=f"utility-set-{stable_config_id(payload)}",
        candidate_set_id=candidate_set.candidate_set_id,
        estimates=estimates,
        metadata={
            "provenance": {
                "candidate_set_id": candidate_set.candidate_set_id,
                "candidate_count": len(candidate_set.candidates),
                "estimate_count": len(estimates),
            }
        },
    )


def coerce_viewport_risk(value: ViewportRisk | Mapping[str, Any] | None) -> ViewportRisk:
    """Normalize viewport risk from a record or mapping."""

    if value is None:
        return ViewportRisk()
    if isinstance(value, ViewportRisk):
        return value
    mapping = _require_mapping(value, "viewport_risk")
    return ViewportRisk(
        mismatch_probability=_first_present(mapping, ("mismatch_probability", "mismatch_risk", "view_mismatch")) or 0.0,
        occlusion_probability=_first_present(mapping, ("occlusion_probability", "occlusion_risk")) or 0.0,
        motion_instability=_first_present(mapping, ("motion_instability", "motion_risk", "sudden_turn_probability")) or 0.0,
        metadata=_plain_json_mapping(mapping.get("metadata"), "viewport_risk.metadata") if mapping.get("metadata") is not None else {},
    )


def coerce_resource_budget(
    value: ResourceBudget | DeviceProfile | DeviceBudgets | Mapping[str, Any] | None,
) -> ResourceBudget:
    """Normalize resource budgets from explicit budgets or device records."""

    if value is None:
        return ResourceBudget(available_time_ms=33.333, available_bytes=1_000_000, available_memory_mb=1024.0)
    if isinstance(value, ResourceBudget):
        return value
    if isinstance(value, DeviceProfile):
        available_bytes = _metadata_available_bytes(value.metadata)
        return ResourceBudget.from_device_profile(value, available_bytes=available_bytes)
    if isinstance(value, DeviceBudgets):
        return ResourceBudget.from_device_budgets(value, available_bytes=1_000_000)
    mapping = _require_mapping(value, "budgets")
    if "device_profile" in mapping and isinstance(mapping["device_profile"], DeviceProfile):
        return ResourceBudget.from_device_profile(
            mapping["device_profile"],
            available_bytes=_positive_int(
                _first_present(mapping, ("available_bytes", "byte_budget", "transfer_bytes")) or 1_000_000,
                "available_bytes",
            ),
            queue_debt_ms=_first_present(mapping, ("queue_debt_ms", "time_debt_ms")) or 0.0,
            transfer_debt_bytes=_first_present(mapping, ("transfer_debt_bytes", "byte_debt")) or 0,
        )
    if "device_budgets" in mapping and isinstance(mapping["device_budgets"], DeviceBudgets):
        return ResourceBudget.from_device_budgets(
            mapping["device_budgets"],
            available_bytes=_positive_int(
                _first_present(mapping, ("available_bytes", "byte_budget", "transfer_bytes")) or 1_000_000,
                "available_bytes",
            ),
            queue_debt_ms=_first_present(mapping, ("queue_debt_ms", "time_debt_ms")) or 0.0,
            transfer_debt_bytes=_first_present(mapping, ("transfer_debt_bytes", "byte_debt")) or 0,
        )
    return ResourceBudget(
        available_time_ms=_first_present(mapping, ("available_time_ms", "time_budget_ms", "deadline_budget_ms")) or 33.333,
        available_bytes=_first_present(mapping, ("available_bytes", "byte_budget", "transfer_bytes")) or 1_000_000,
        available_memory_mb=_first_present(mapping, ("available_memory_mb", "memory_mb", "memory_budget_mb")) or 1024.0,
        queue_debt_ms=_first_present(mapping, ("queue_debt_ms", "time_debt_ms")) or 0.0,
        transfer_debt_bytes=_first_present(mapping, ("transfer_debt_bytes", "byte_debt")) or 0,
        metadata=_plain_json_mapping(mapping.get("metadata"), "budgets.metadata") if mapping.get("metadata") is not None else {},
    )


def coerce_utility_model_weights(value: UtilityModelWeights | Mapping[str, Any] | None) -> UtilityModelWeights:
    """Normalize model weights from a record or mapping."""

    if value is None:
        return UtilityModelWeights()
    if isinstance(value, UtilityModelWeights):
        return value
    mapping = _require_mapping(value, "model_weights")
    allowed = set(UtilityModelWeights().as_payload())
    unknown = sorted(str(key) for key in mapping if str(key) not in allowed)
    if unknown:
        raise UtilityError(f"model_weights contains unknown field(s): {', '.join(unknown)}.")
    return UtilityModelWeights(**{str(key): value for key, value in mapping.items()})


def coerce_substrate_value(value: SubstrateValue | Mapping[str, Any]) -> SubstrateValue:
    """Normalize a substrate-value payload into a typed record."""

    if isinstance(value, SubstrateValue):
        return value
    mapping = _require_mapping(value, "substrate")
    provider_id = _string_or_none(mapping.get("provider_id")) or "embedded-substrate"
    query_mapping = _require_mapping(mapping.get("query"), "substrate.query")
    timing_mapping = _require_mapping(mapping.get("component_timing"), "substrate.component_timing")
    uncertainty_mapping = _require_mapping(mapping.get("uncertainty"), "substrate.uncertainty")
    return SubstrateValue(
        provider_id=provider_id,
        query=SubstrateQuery(
            layer=_first_present(query_mapping, ("layer", "layer_index")),
            ref_resolution=_first_present(query_mapping, ("ref_resolution", "resolution"))
            or {
                "width_px": _first_present(query_mapping, ("ref_width_px", "width_px", "width")),
                "height_px": _first_present(query_mapping, ("ref_height_px", "height_px", "height")),
            },
            fov_deg=_first_present(query_mapping, ("fov_deg", "fov")),
            view_mismatch_deg=_first_present(query_mapping, ("view_mismatch_deg", "mismatch_deg")) or 0.0,
            freshness_ms=_first_present(query_mapping, ("freshness_ms", "age_ms")) or 0.0,
            metadata=_plain_json_mapping(query_mapping.get("metadata"), "substrate.query.metadata")
            if query_mapping.get("metadata") is not None
            else {},
        ),
        visible_quality=_first_present(mapping, ("visible_quality", "quality", "qoe")) or 0.0,
        component_timing=ComponentTiming(
            generation_ms=_first_present(timing_mapping, ("generation_ms", "generate_ms")) or 0.0,
            transfer_ms=_first_present(timing_mapping, ("transfer_ms", "network_ms")) or 0.0,
            restoration_ms=_first_present(timing_mapping, ("restoration_ms", "restore_ms")) or 0.0,
            render_ms=_first_present(timing_mapping, ("render_ms", "rendering_ms")) or 0.0,
        ),
        uncertainty=SubstrateUncertainty(
            quality_stddev=_first_present(uncertainty_mapping, ("quality_stddev", "quality_sigma")) or 0.0,
            timing_stddev_ms=_first_present(uncertainty_mapping, ("timing_stddev_ms", "timing_sigma_ms")) or 0.0,
            confidence=_first_present(uncertainty_mapping, ("confidence",)) if "confidence" in uncertainty_mapping else 1.0,
        ),
        metadata=_plain_json_mapping(mapping.get("metadata"), "substrate.metadata") if mapping.get("metadata") is not None else {},
    )


def _substrate_for_candidate(
    candidate: CandidateObject,
    substrate_provider: SubstrateValueProvider | None,
) -> SubstrateValue:
    embedded = candidate.metadata.get("substrate")
    if embedded is not None:
        return coerce_substrate_value(_require_mapping(embedded, "candidate.metadata.substrate"))
    if substrate_provider is None:
        raise UtilityError("candidate metadata must include substrate values or substrate_provider must be supplied.")
    return substrate_provider.evaluate(
        {
            "layer": candidate.layer,
            "ref_resolution": candidate.resolution,
            "fov_deg": candidate.fov_deg,
            "view_mismatch_deg": 0.0,
            "freshness_ms": candidate.lookahead_ms,
        }
    )


def _lifecycle_risk(candidate: CandidateObject, risk: ViewportRisk, uncertainty: SubstrateUncertainty) -> float:
    freshness_ratio = candidate.lookahead_ms / max(1.0, float(candidate.expiration_ms))
    priority_relief = min(float(candidate.retransmit_priority), 10.0) / 20.0
    uncertainty_risk = (uncertainty.quality_stddev + (1.0 - uncertainty.confidence)) / 2.0
    dependency_risk = min(len(candidate.dependencies), 5) * 0.025
    return _clamp01(0.50 * risk.aggregate + 0.25 * freshness_ratio + 0.20 * uncertainty_risk + dependency_risk - priority_relief)


def _deadline_miss_probability(
    candidate: CandidateObject,
    timing: ComponentTiming,
    uncertainty: SubstrateUncertainty,
    budgets: ResourceBudget,
) -> float:
    effective_deadline_ms = min(float(candidate.expiration_ms), budgets.available_time_ms)
    slack_ms = effective_deadline_ms - timing.total_ms - budgets.queue_debt_ms
    sigma_ms = max(1.0, uncertainty.timing_stddev_ms)
    return _logistic_probability(-slack_ms / sigma_ms)


def _utility_uncertainty(
    substrate_uncertainty: SubstrateUncertainty,
    deadline_miss_probability: float,
    risk: ViewportRisk,
    weights: UtilityModelWeights,
) -> UtilityUncertainty:
    deadline_stddev = math.sqrt(deadline_miss_probability * (1.0 - deadline_miss_probability))
    quality_component = weights.visible_qoe_weight * substrate_uncertainty.quality_stddev
    deadline_component = weights.deadline_miss_weight * deadline_stddev
    risk_component = weights.lifecycle_risk_weight * risk.aggregate * 0.10
    utility_stddev = math.sqrt(quality_component**2 + deadline_component**2 + risk_component**2)
    confidence = _clamp01(substrate_uncertainty.confidence * (1.0 - 0.25 * risk.aggregate))
    return UtilityUncertainty(
        quality_stddev=substrate_uncertainty.quality_stddev,
        timing_stddev_ms=substrate_uncertainty.timing_stddev_ms,
        deadline_probability_stddev=deadline_stddev,
        utility_stddev=utility_stddev,
        confidence=confidence,
    )


def _candidate_memory_mb(candidate: CandidateObject) -> float:
    resolution = coerce_ref_resolution(candidate.resolution)
    layer_scale = max(1, candidate.layer + 1)
    tile_scale = 1.0
    if candidate.tile is not None:
        tile_scale = 1.0 / float(candidate.tile.rows * candidate.tile.columns)
    return max(candidate.size_bytes / 1_000_000.0, resolution.megapixels * 4.0 * layer_scale * tile_scale)


def _normalized_debt(debt: ResourceDebt, budgets: ResourceBudget) -> float:
    return (
        debt.time_debt_ms / budgets.available_time_ms
        + debt.transfer_debt_bytes / budgets.available_bytes
        + debt.memory_debt_mb / budgets.available_memory_mb
        + debt.carried_queue_debt_ms / budgets.available_time_ms
        + debt.carried_transfer_debt_bytes / budgets.available_bytes
    )


def _metadata_available_bytes(metadata: Mapping[str, Any]) -> int:
    raw_value = _first_present(metadata, ("available_bytes", "byte_budget", "transfer_bytes"))
    if raw_value is None:
        return 1_000_000
    return _positive_int(raw_value, "available_bytes")


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise UtilityError(f"{field_name} must be a mapping.")
    return value


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
    return value


def _first_present(mapping: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise UtilityError("value must be a non-empty string when provided.")
    return value.strip()


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise UtilityError(f"{field_name} must be a non-empty string.")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise UtilityError(f"{field_name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise UtilityError(f"{field_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise UtilityError(f"{field_name} must be positive.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise UtilityError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise UtilityError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise UtilityError(f"{field_name} must be non-negative.")
    return parsed


def _positive_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed <= 0:
        raise UtilityError(f"{field_name} must be positive.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0:
        raise UtilityError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0 or parsed > 1.0:
        raise UtilityError(f"{field_name} must be between 0 and 1.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise UtilityError(f"{field_name} must be a finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise UtilityError(f"{field_name} must be a finite number.") from exc
    if not math.isfinite(parsed):
        raise UtilityError(f"{field_name} must be finite.")
    return parsed


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _logistic_probability(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return _clamp01(1.0 / (1.0 + math.exp(-value)))
