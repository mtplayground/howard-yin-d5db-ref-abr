"""Per-frame outcome evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.accounting import CandidateResourceAccount, ComponentTimingAccount, ResourceAccountingSummary
from ref_abr.candidates import CandidateObject
from ref_abr.config import stable_config_id
from ref_abr.domain import FrameOutcome
from ref_abr.utility import CandidateUtilityEstimate, UtilityEstimateSet


class OutcomeEvaluationError(ValueError):
    """Raised when frame outcome evaluation inputs are invalid."""


@dataclass(frozen=True)
class FrameEvaluationConfig:
    """Weights and penalties for converting frame facts into a quality score."""

    visible_quality_weight: float = 0.65
    full_quality_weight: float = 0.35
    deadline_miss_penalty: float = 0.25
    missing_object_penalty: float = 0.20
    dropped_object_penalty: float = 0.15
    freeze_penalty: float = 0.20
    frame_interval_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "visible_quality_weight", _non_negative_float(self.visible_quality_weight, "visible_quality_weight"))
        object.__setattr__(self, "full_quality_weight", _non_negative_float(self.full_quality_weight, "full_quality_weight"))
        for field_name in (
            "deadline_miss_penalty",
            "missing_object_penalty",
            "dropped_object_penalty",
            "freeze_penalty",
        ):
            object.__setattr__(self, field_name, _unit_interval(getattr(self, field_name), field_name))
        if self.frame_interval_ms is not None:
            object.__setattr__(self, "frame_interval_ms", _positive_float(self.frame_interval_ms, "frame_interval_ms"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def normalized_quality_weights(self) -> tuple[float, float]:
        total = self.visible_quality_weight + self.full_quality_weight
        if total <= 0.0:
            return 0.5, 0.5
        return self.visible_quality_weight / total, self.full_quality_weight / total

    def as_payload(self) -> dict[str, Any]:
        return {
            "visible_quality_weight": self.visible_quality_weight,
            "full_quality_weight": self.full_quality_weight,
            "deadline_miss_penalty": self.deadline_miss_penalty,
            "missing_object_penalty": self.missing_object_penalty,
            "dropped_object_penalty": self.dropped_object_penalty,
            "freeze_penalty": self.freeze_penalty,
            "frame_interval_ms": self.frame_interval_ms,
            "metadata": _to_payload(self.metadata),
        }


def evaluate_frame_outcome(
    *,
    frame_id: str,
    scheduled_time_ms: int,
    deadline_ms: int,
    required_object_ids: Sequence[str],
    delivered_object_ids: Sequence[str] = (),
    visible_object_ids: Sequence[str] | None = None,
    dropped_object_ids: Sequence[str] = (),
    frozen: bool = False,
    previous_frame_id: str | None = None,
    accounting: ResourceAccountingSummary | Sequence[CandidateResourceAccount] | None = None,
    utility_estimates: UtilityEstimateSet | Sequence[CandidateUtilityEstimate] = (),
    candidate_by_id: Mapping[str, CandidateObject] | None = None,
    selected_candidate_ids: Sequence[str] = (),
    config: FrameEvaluationConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FrameOutcome:
    """Evaluate coverage, quality, timing, deadline, and penalties for one frame."""

    _require_non_empty(frame_id, "frame_id")
    scheduled_time = _non_negative_int(scheduled_time_ms, "scheduled_time_ms")
    deadline = _non_negative_int(deadline_ms, "deadline_ms")
    required_ids = _ordered_unique_strings(required_object_ids, "required_object_ids")
    delivered_ids = _ordered_unique_strings(delivered_object_ids, "delivered_object_ids")
    visible_ids = (
        _ordered_unique_strings(visible_object_ids, "visible_object_ids")
        if visible_object_ids is not None
        else required_ids
    )
    dropped_ids = _ordered_unique_strings(dropped_object_ids, "dropped_object_ids")
    selected_ids = _ordered_unique_strings(selected_candidate_ids, "selected_candidate_ids")
    if not isinstance(frozen, bool):
        raise OutcomeEvaluationError("frozen must be boolean.")
    if previous_frame_id is not None:
        _require_non_empty(previous_frame_id, "previous_frame_id")
    evaluation_config = config or FrameEvaluationConfig()
    if not isinstance(evaluation_config, FrameEvaluationConfig):
        raise OutcomeEvaluationError("config must be a FrameEvaluationConfig record.")

    accounts = _coerce_accounts(accounting)
    estimates = _coerce_estimates(utility_estimates)
    candidates = _coerce_candidates(candidate_by_id)
    selected_account_ids = _account_ids_for_selection(accounts, selected_ids)
    selected_accounts = [account for account in accounts if not selected_ids or account.candidate_id in selected_ids]
    timing = _sum_timing(selected_accounts)
    latency_ms = timing.total_ms if timing is not None else (0.0 if delivered_ids or frozen or not required_ids else None)
    rendered_time_ms = scheduled_time + int(round(latency_ms)) if latency_ms is not None else None
    deadline_hit = rendered_time_ms is not None and rendered_time_ms <= deadline

    delivered_set = set(delivered_ids)
    required_set = set(required_ids)
    visible_set = set(visible_ids)
    missing_ids = tuple(object_id for object_id in required_ids if object_id not in delivered_set)
    coverage = _ratio(len(required_set & delivered_set), len(required_ids), empty_value=1.0)
    visible_coverage = _ratio(len(visible_set & delivered_set), len(visible_ids), empty_value=1.0)
    quality_by_object = _quality_by_object(estimates, candidates)
    visible_quality = _average_quality(visible_ids, delivered_set, quality_by_object, default_quality=visible_coverage)
    full_quality = _average_quality(required_ids, delivered_set, quality_by_object, default_quality=coverage)
    dropped_ratio = _ratio(len(set(dropped_ids) & (required_set | visible_set)), max(len(required_set | visible_set), 1), empty_value=0.0)
    missing_ratio = _ratio(len(missing_ids), len(required_ids), empty_value=0.0)
    freeze_flag = frozen or (bool(required_ids) and not delivered_ids)
    penalties = {
        "deadline_miss": 0.0 if deadline_hit else evaluation_config.deadline_miss_penalty,
        "missing": evaluation_config.missing_object_penalty * missing_ratio,
        "dropped": evaluation_config.dropped_object_penalty * dropped_ratio,
        "freeze": evaluation_config.freeze_penalty if freeze_flag else 0.0,
    }
    visible_weight, full_weight = evaluation_config.normalized_quality_weights()
    weighted_quality = visible_weight * visible_quality + full_weight * full_quality
    quality_score = _clamp01(weighted_quality - sum(penalties.values()))
    fps = _fps(latency_ms, evaluation_config.frame_interval_ms)
    outcome_metadata = _plain_json_mapping(metadata, "metadata") if metadata is not None else {}
    frame_evaluation = {
        "evaluation_id": _evaluation_id(
            frame_id=frame_id,
            scheduled_time_ms=scheduled_time,
            deadline_ms=deadline,
            required_object_ids=required_ids,
            delivered_object_ids=delivered_ids,
            missing_object_ids=missing_ids,
            dropped_object_ids=dropped_ids,
            selected_candidate_ids=selected_ids,
            quality_score=quality_score,
            latency_ms=latency_ms,
            deadline_hit=deadline_hit,
        ),
        "coverage": coverage,
        "visible_coverage": visible_coverage,
        "visible_quality": visible_quality,
        "full_quality": full_quality,
        "weighted_quality": weighted_quality,
        "quality_score": quality_score,
        "latency_ms": latency_ms,
        "fps": fps,
        "deadline_hit": deadline_hit,
        "dropped": bool(dropped_ids),
        "missing": bool(missing_ids),
        "freeze": freeze_flag,
        "penalties": penalties,
        "required_object_ids": list(required_ids),
        "visible_object_ids": list(visible_ids),
        "dropped_object_ids": list(dropped_ids),
        "selected_candidate_ids": list(selected_ids),
        "selected_account_ids": list(selected_account_ids),
        "previous_frame_id": previous_frame_id,
        "timing": timing.as_payload() if timing is not None else None,
        "config": evaluation_config.as_payload(),
    }
    outcome_metadata["frame_evaluation"] = frame_evaluation
    return FrameOutcome(
        frame_id=frame_id,
        scheduled_time_ms=scheduled_time,
        rendered_time_ms=rendered_time_ms,
        deadline_ms=deadline,
        delivered_object_ids=delivered_ids,
        missing_object_ids=missing_ids,
        quality_score=quality_score,
        deadline_hit=deadline_hit,
        metadata=outcome_metadata,
    )


def _evaluation_id(**payload: Any) -> str:
    return f"frame-outcome-{stable_config_id(payload)}"


def _coerce_accounts(
    accounting: ResourceAccountingSummary | Sequence[CandidateResourceAccount] | None,
) -> tuple[CandidateResourceAccount, ...]:
    if accounting is None:
        return ()
    if isinstance(accounting, ResourceAccountingSummary):
        return accounting.accounts
    if isinstance(accounting, (str, bytes)) or not isinstance(accounting, Sequence):
        raise OutcomeEvaluationError("accounting must be a ResourceAccountingSummary or sequence of CandidateResourceAccount records.")
    accounts = tuple(accounting)
    for account in accounts:
        if not isinstance(account, CandidateResourceAccount):
            raise OutcomeEvaluationError("accounting must contain CandidateResourceAccount records.")
    return accounts


def _coerce_estimates(
    utility_estimates: UtilityEstimateSet | Sequence[CandidateUtilityEstimate],
) -> tuple[CandidateUtilityEstimate, ...]:
    if isinstance(utility_estimates, UtilityEstimateSet):
        return utility_estimates.estimates
    if isinstance(utility_estimates, (str, bytes)) or not isinstance(utility_estimates, Sequence):
        raise OutcomeEvaluationError("utility_estimates must be a UtilityEstimateSet or sequence of CandidateUtilityEstimate records.")
    estimates = tuple(utility_estimates)
    for estimate in estimates:
        if not isinstance(estimate, CandidateUtilityEstimate):
            raise OutcomeEvaluationError("utility_estimates must contain CandidateUtilityEstimate records.")
    return estimates


def _coerce_candidates(candidate_by_id: Mapping[str, CandidateObject] | None) -> Mapping[str, CandidateObject]:
    if candidate_by_id is None:
        return {}
    if not isinstance(candidate_by_id, MappingABC):
        raise OutcomeEvaluationError("candidate_by_id must be a mapping.")
    candidates: dict[str, CandidateObject] = {}
    for candidate_id, candidate in candidate_by_id.items():
        _require_non_empty(str(candidate_id), "candidate_by_id key")
        if not isinstance(candidate, CandidateObject):
            raise OutcomeEvaluationError("candidate_by_id values must be CandidateObject records.")
        candidates[str(candidate_id)] = candidate
    return candidates


def _account_ids_for_selection(accounts: tuple[CandidateResourceAccount, ...], selected_candidate_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not selected_candidate_ids:
        return tuple(account.account_id for account in accounts)
    selected = set(selected_candidate_ids)
    return tuple(account.account_id for account in accounts if account.candidate_id in selected)


def _sum_timing(accounts: Sequence[CandidateResourceAccount]) -> ComponentTimingAccount | None:
    if not accounts:
        return None
    return ComponentTimingAccount(
        server_generation_ms=sum(account.timing.server_generation_ms for account in accounts),
        queue_ms=sum(account.timing.queue_ms for account in accounts),
        transfer_ms=sum(account.timing.transfer_ms for account in accounts),
        decode_ms=sum(account.timing.decode_ms for account in accounts),
        restore_ms=sum(account.timing.restore_ms for account in accounts),
        render_ms=sum(account.timing.render_ms for account in accounts),
    )


def _quality_by_object(
    estimates: tuple[CandidateUtilityEstimate, ...],
    candidates: Mapping[str, CandidateObject],
) -> dict[str, float]:
    qualities: dict[str, list[float]] = {}
    for estimate in estimates:
        candidate = candidates.get(estimate.candidate_id)
        object_id = candidate.object_id if candidate is not None else _string_or_none(estimate.metadata.get("object_id"))
        if object_id is None:
            continue
        qualities.setdefault(object_id, []).append(_clamp01(estimate.visible_qoe_gain))
    return {object_id: sum(values) / len(values) for object_id, values in qualities.items()}


def _average_quality(
    object_ids: tuple[str, ...],
    delivered_set: set[str],
    quality_by_object: Mapping[str, float],
    *,
    default_quality: float,
) -> float:
    if not object_ids:
        return 1.0
    if not quality_by_object:
        return _clamp01(default_quality)
    values = []
    for object_id in object_ids:
        if object_id not in delivered_set:
            values.append(0.0)
            continue
        values.append(_clamp01(quality_by_object.get(object_id, 1.0)))
    return _clamp01(sum(values) / len(values))


def _fps(latency_ms: float | None, frame_interval_ms: float | None) -> float | None:
    interval = frame_interval_ms if frame_interval_ms is not None else latency_ms
    if interval is None:
        return None
    return 1000.0 / max(1.0, interval)


def _ratio(numerator: int | float, denominator: int | float, *, empty_value: float) -> float:
    if denominator <= 0:
        return empty_value
    return _clamp01(float(numerator) / float(denominator))


def _ordered_unique_strings(values: Sequence[str] | None, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise OutcomeEvaluationError(f"{field_name} must be a sequence of strings.")
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return tuple(ordered)


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise OutcomeEvaluationError(f"{field_name} must be a mapping.")
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


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise OutcomeEvaluationError(f"{field_name} must be a non-empty string.")


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise OutcomeEvaluationError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise OutcomeEvaluationError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise OutcomeEvaluationError(f"{field_name} must be a non-negative integer.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise OutcomeEvaluationError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise OutcomeEvaluationError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise OutcomeEvaluationError(f"{field_name} must be finite.")
    return parsed


def _positive_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed <= 0.0:
        raise OutcomeEvaluationError(f"{field_name} must be positive.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0:
        raise OutcomeEvaluationError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if not 0.0 <= parsed <= 1.0:
        raise OutcomeEvaluationError(f"{field_name} must be between 0 and 1.")
    return parsed


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = [
    "FrameEvaluationConfig",
    "OutcomeEvaluationError",
    "evaluate_frame_outcome",
]
