"""Minimum canonical baseline scheduling methods."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.candidates import CandidateObject
from ref_abr.domain import LifecycleStatus
from ref_abr.methods import ActionBudget, SchedulingObservation
from ref_abr.utility import CandidateUtilityEstimate


class BaselineError(ValueError):
    """Raised when a baseline is configured with invalid parameters."""


GAUSSIAN_CANDIDATE_KINDS: tuple[str, ...] = ("gaussian_base", "gaussian_enhancement", "tile")
ALL_CANDIDATE_KINDS: tuple[str, ...] = (*GAUSSIAN_CANDIDATE_KINDS, "reference_action")
ACTIVE_BASE_STATUSES: tuple[LifecycleStatus, ...] = (
    LifecycleStatus.REQUESTED,
    LifecycleStatus.IN_FLIGHT,
    LifecycleStatus.AVAILABLE,
)


@dataclass(frozen=True)
class CAGSFixedReferenceBaseline:
    """CAGS-style fixed-reference baseline.

    The method does not adapt quality. It deterministically selects reference-action
    candidates at one fixed reference resolution, falling back to base Gaussian
    candidates only when no reference action is visible.
    """

    fixed_resolution: str | None = None
    method_id: str = "cags-fixed-reference"
    method_name: str = "CAGS fixed-reference"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.method_name, "method_name")
        if self.fixed_resolution is not None:
            _require_non_empty(self.fixed_resolution, "fixed_resolution")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, Any]:
        candidates = _filter_resolution(observation.candidates, self.fixed_resolution)
        selected = _select_with_budget(
            _sorted_candidates(candidates, kind_preference=("reference_action", "gaussian_base")),
            action_budget,
            allowed_kinds=("reference_action",),
        )
        if not selected:
            selected = _select_with_budget(
                _sorted_candidates(candidates, kind_preference=("gaussian_base",)),
                action_budget,
                allowed_kinds=("gaussian_base",),
            )
        return _decision_payload(self, observation, selected, "fixed_reference")


@dataclass(frozen=True)
class SVQGaussianOnlyABRBaseline:
    """SVQ-style Gaussian-only ABR baseline.

    The method excludes reference-action candidates and selects Gaussian base,
    enhancement, or tile candidates by expected utility when available.
    """

    method_id: str = "svq-gaussian-only-abr"
    method_name: str = "SVQ Gaussian-only ABR"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.method_name, "method_name")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, Any]:
        utility_by_candidate = _utility_by_candidate(observation.utility_estimates)
        ranked = tuple(
            sorted(
                (candidate for candidate in observation.candidates if candidate.candidate_kind in GAUSSIAN_CANDIDATE_KINDS),
                key=lambda candidate: _utility_sort_key(candidate, utility_by_candidate),
            )
        )
        selected = _select_with_budget(ranked, action_budget, allowed_kinds=GAUSSIAN_CANDIDATE_KINDS)
        return _decision_payload(self, observation, selected, "gaussian_only_abr", utility_by_candidate)


@dataclass(frozen=True)
class ReferenceOnlyAfterBaseBaseline:
    """Reference-only-after-base baseline.

    The method first schedules base Gaussian candidates for objects whose base is
    not yet requested, in flight, or available. Once the base is active, it only
    schedules reference-action candidates for those objects.
    """

    method_id: str = "reference-only-after-base"
    method_name: str = "Reference-only-after-base"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.method_name, "method_name")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, Any]:
        active_base_object_ids = _active_base_object_ids(observation)
        base_candidates = tuple(
            candidate
            for candidate in observation.candidates
            if candidate.candidate_kind == "gaussian_base" and candidate.object_id not in active_base_object_ids
        )
        if base_candidates:
            selected = _select_with_budget(
                _sorted_candidates(base_candidates, kind_preference=("gaussian_base",)),
                action_budget,
                allowed_kinds=("gaussian_base",),
            )
            return _decision_payload(self, observation, selected, "base_before_reference")

        reference_candidates = tuple(
            candidate
            for candidate in observation.candidates
            if candidate.candidate_kind == "reference_action" and candidate.object_id in active_base_object_ids
        )
        selected = _select_with_budget(
            _sorted_candidates(reference_candidates, kind_preference=("reference_action",)),
            action_budget,
            allowed_kinds=("reference_action",),
        )
        return _decision_payload(self, observation, selected, "reference_after_base")


@dataclass(frozen=True)
class FixedReferenceCadenceBaseline:
    """Fixed-reference baseline that only schedules on cadence-aligned epochs."""

    cadence_ms: int
    phase_ms: int = 0
    method_id: str = "fixed-reference-cadence"
    method_name: str = "Fixed-reference cadence"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cadence_ms", _positive_int(self.cadence_ms, "cadence_ms"))
        object.__setattr__(self, "phase_ms", _non_negative_int(self.phase_ms, "phase_ms"))
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.method_name, "method_name")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, Any]:
        if (observation.decision_time_ms - self.phase_ms) % self.cadence_ms != 0:
            return _decision_payload(self, observation, (), "fixed_reference_cadence_skip")
        selected = _select_with_budget(
            _sorted_candidates(observation.candidates, kind_preference=("reference_action",)),
            action_budget,
            allowed_kinds=("reference_action",),
        )
        return _decision_payload(self, observation, selected, "fixed_reference_cadence")


@dataclass(frozen=True)
class IndependentGaussianSchedulerBaseline:
    """Independent Gaussian scheduler that ignores reference-action candidates."""

    method_id: str = "independent-gaussian"
    method_name: str = "Independent Gaussian scheduler"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.method_name, "method_name")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, Any]:
        utility_by_candidate = _utility_by_candidate(observation.utility_estimates)
        ranked = tuple(
            sorted(
                (candidate for candidate in observation.candidates if candidate.candidate_kind in GAUSSIAN_CANDIDATE_KINDS),
                key=lambda candidate: _utility_sort_key(candidate, utility_by_candidate),
            )
        )
        selected = _select_with_budget(ranked, action_budget, allowed_kinds=GAUSSIAN_CANDIDATE_KINDS)
        return _decision_payload(self, observation, selected, "independent_gaussian", utility_by_candidate)


@dataclass(frozen=True)
class IndependentReferenceSchedulerBaseline:
    """Independent reference scheduler that ignores Gaussian candidates."""

    method_id: str = "independent-reference"
    method_name: str = "Independent reference scheduler"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.method_name, "method_name")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, Any]:
        utility_by_candidate = _utility_by_candidate(observation.utility_estimates)
        ranked = tuple(
            sorted(
                (candidate for candidate in observation.candidates if candidate.candidate_kind == "reference_action"),
                key=lambda candidate: _utility_sort_key(candidate, utility_by_candidate),
            )
        )
        selected = _select_with_budget(ranked, action_budget, allowed_kinds=("reference_action",))
        return _decision_payload(self, observation, selected, "independent_reference", utility_by_candidate)


@dataclass(frozen=True)
class BandwidthGreedyBaseline:
    """Greedy baseline that fills the action budget with the smallest candidates."""

    method_id: str = "bandwidth-greedy"
    method_name: str = "Bandwidth-greedy"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.method_name, "method_name")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, Any]:
        utility_by_candidate = _utility_by_candidate(observation.utility_estimates)
        ranked = tuple(sorted(observation.candidates, key=lambda candidate: _bandwidth_sort_key(candidate, utility_by_candidate)))
        selected = _select_with_budget(ranked, action_budget, allowed_kinds=ALL_CANDIDATE_KINDS)
        return _decision_payload(self, observation, selected, "bandwidth_greedy", utility_by_candidate)


@dataclass(frozen=True)
class DeadlineGreedyBaseline:
    """Greedy baseline that prioritizes candidates with the nearest deadlines."""

    method_id: str = "deadline-greedy"
    method_name: str = "Deadline-greedy"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.method_name, "method_name")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, Any]:
        utility_by_candidate = _utility_by_candidate(observation.utility_estimates)
        ranked = tuple(sorted(observation.candidates, key=lambda candidate: _deadline_sort_key(candidate, utility_by_candidate)))
        selected = _select_with_budget(ranked, action_budget, allowed_kinds=ALL_CANDIDATE_KINDS)
        return _decision_payload(self, observation, selected, "deadline_greedy", utility_by_candidate)


@dataclass(frozen=True)
class QualityMaxDeadlineUnawareBaseline:
    """Quality-maximizing baseline that ignores candidate deadlines."""

    method_id: str = "quality-max-deadline-unaware"
    method_name: str = "Quality-max deadline-unaware"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.method_name, "method_name")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def plan_schedule(self, observation: SchedulingObservation, action_budget: ActionBudget) -> dict[str, Any]:
        utility_by_candidate = _utility_by_candidate(observation.utility_estimates)
        ranked = tuple(sorted(observation.candidates, key=lambda candidate: _quality_sort_key(candidate, utility_by_candidate)))
        selected = _select_with_budget(ranked, action_budget, allowed_kinds=ALL_CANDIDATE_KINDS)
        return _decision_payload(self, observation, selected, "quality_max_deadline_unaware", utility_by_candidate)


def minimum_baselines() -> tuple[CAGSFixedReferenceBaseline, SVQGaussianOnlyABRBaseline, ReferenceOnlyAfterBaseBaseline]:
    """Return the minimum status-quo/canonical baseline set."""

    return (
        CAGSFixedReferenceBaseline(),
        SVQGaussianOnlyABRBaseline(),
        ReferenceOnlyAfterBaseBaseline(),
    )


def simple_baselines() -> tuple[
    FixedReferenceCadenceBaseline,
    IndependentGaussianSchedulerBaseline,
    IndependentReferenceSchedulerBaseline,
    BandwidthGreedyBaseline,
    DeadlineGreedyBaseline,
    QualityMaxDeadlineUnawareBaseline,
]:
    """Return the simple greedy/cadence/independent baseline set."""

    return (
        FixedReferenceCadenceBaseline(cadence_ms=1000),
        IndependentGaussianSchedulerBaseline(),
        IndependentReferenceSchedulerBaseline(),
        BandwidthGreedyBaseline(),
        DeadlineGreedyBaseline(),
        QualityMaxDeadlineUnawareBaseline(),
    )


def _select_with_budget(
    candidates: tuple[CandidateObject, ...],
    action_budget: ActionBudget,
    *,
    allowed_kinds: tuple[str, ...],
) -> tuple[CandidateObject, ...]:
    selected: list[CandidateObject] = []
    selected_object_ids: set[str] = set()
    selected_bytes = 0
    max_selected_candidates = action_budget.max_selected_candidates or action_budget.max_selected_objects
    for candidate in candidates:
        if candidate.candidate_kind not in allowed_kinds:
            continue
        if candidate.object_id in selected_object_ids:
            continue
        if len(selected) >= max_selected_candidates:
            break
        if len(selected_object_ids) >= action_budget.max_selected_objects:
            break
        if selected_bytes + candidate.size_bytes > action_budget.max_selected_bytes:
            continue
        selected.append(candidate)
        selected_object_ids.add(candidate.object_id)
        selected_bytes += candidate.size_bytes
    return tuple(selected)


def _decision_payload(
    method: Any,
    observation: SchedulingObservation,
    selected: tuple[CandidateObject, ...],
    policy: str,
    utility_by_candidate: Mapping[str, CandidateUtilityEstimate] | None = None,
) -> dict[str, Any]:
    utility_by_candidate = utility_by_candidate or {}
    expected_utility = sum(
        utility_by_candidate[candidate.candidate_id].expected_utility
        for candidate in selected
        if candidate.candidate_id in utility_by_candidate
    )
    return {
        "selected_candidate_ids": [candidate.candidate_id for candidate in selected],
        "expected_utility": expected_utility if utility_by_candidate else None,
        "metadata": {
            "baseline": {
                "method_id": method.method_id,
                "method_name": method.method_name,
                "policy": policy,
                "selected_candidate_kinds": [candidate.candidate_kind for candidate in selected],
                "parameters": _to_payload(method.metadata),
            }
        },
    }


def _filter_resolution(candidates: tuple[CandidateObject, ...], fixed_resolution: str | None) -> tuple[CandidateObject, ...]:
    if fixed_resolution is None:
        return candidates
    return tuple(candidate for candidate in candidates if _resolution_label(candidate) == fixed_resolution)


def _sorted_candidates(
    candidates: tuple[CandidateObject, ...],
    *,
    kind_preference: tuple[str, ...],
) -> tuple[CandidateObject, ...]:
    preference = {candidate_kind: index for index, candidate_kind in enumerate(kind_preference)}
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                preference.get(candidate.candidate_kind, len(preference)),
                candidate.object_id,
                candidate.layer,
                candidate.resolution.pixel_count,
                candidate.lookahead_ms,
                candidate.expiration_ms,
                candidate.retransmit_priority,
                candidate.candidate_id,
            ),
        )
    )


def _utility_sort_key(
    candidate: CandidateObject,
    utility_by_candidate: Mapping[str, CandidateUtilityEstimate],
) -> tuple[float, int, str, int, int, int, str]:
    utility = utility_by_candidate.get(candidate.candidate_id)
    expected_utility = utility.expected_utility if utility is not None else 0.0
    return (
        -expected_utility,
        _gaussian_kind_rank(candidate.candidate_kind),
        candidate.object_id,
        -candidate.layer,
        -candidate.resolution.pixel_count,
        candidate.lookahead_ms,
        candidate.candidate_id,
    )


def _bandwidth_sort_key(
    candidate: CandidateObject,
    utility_by_candidate: Mapping[str, CandidateUtilityEstimate],
) -> tuple[int, float, int, str, int, str]:
    return (
        candidate.size_bytes,
        -_candidate_quality(candidate, utility_by_candidate),
        _gaussian_kind_rank(candidate.candidate_kind),
        candidate.object_id,
        candidate.lookahead_ms,
        candidate.candidate_id,
    )


def _deadline_sort_key(
    candidate: CandidateObject,
    utility_by_candidate: Mapping[str, CandidateUtilityEstimate],
) -> tuple[int, int, float, int, str]:
    return (
        candidate.deadline_ms,
        candidate.expiration_ms,
        -_candidate_quality(candidate, utility_by_candidate),
        candidate.size_bytes,
        candidate.candidate_id,
    )


def _quality_sort_key(
    candidate: CandidateObject,
    utility_by_candidate: Mapping[str, CandidateUtilityEstimate],
) -> tuple[float, int, int, int, str]:
    return (
        -_candidate_quality(candidate, utility_by_candidate),
        _gaussian_kind_rank(candidate.candidate_kind),
        -candidate.layer,
        candidate.size_bytes,
        candidate.candidate_id,
    )


def _candidate_quality(
    candidate: CandidateObject,
    utility_by_candidate: Mapping[str, CandidateUtilityEstimate],
) -> float:
    estimate = utility_by_candidate.get(candidate.candidate_id)
    if estimate is not None:
        return estimate.visible_qoe_gain
    substrate = candidate.metadata.get("substrate")
    if isinstance(substrate, MappingABC):
        visible_quality = substrate.get("visible_quality")
        if isinstance(visible_quality, (int, float)) and not isinstance(visible_quality, bool):
            return float(visible_quality)
    return min(1.0, 0.1 + 0.05 * candidate.layer + candidate.resolution.megapixels / 10.0)


def _gaussian_kind_rank(candidate_kind: str) -> int:
    if candidate_kind == "gaussian_base":
        return 0
    if candidate_kind == "gaussian_enhancement":
        return 1
    if candidate_kind == "tile":
        return 2
    return 3


def _utility_by_candidate(
    estimates: tuple[CandidateUtilityEstimate, ...],
) -> dict[str, CandidateUtilityEstimate]:
    return {estimate.candidate_id: estimate for estimate in estimates}


def _active_base_object_ids(observation: SchedulingObservation) -> set[str]:
    active_statuses = {status.value for status in ACTIVE_BASE_STATUSES}
    return {
        state.reference_id
        for state in observation.lifecycle_states
        if state.status.value in active_statuses
    }


def _resolution_label(candidate: CandidateObject) -> str:
    resolution = candidate.resolution
    if resolution.width_px == 3840 and resolution.height_px == 2160:
        return "4k"
    if resolution.width_px == 1920 and resolution.height_px == 1080:
        return "1080p"
    if resolution.width_px == 1280 and resolution.height_px == 720:
        return "720p"
    if resolution.width_px == 854 and resolution.height_px == 480:
        return "480p"
    return f"{resolution.width_px}x{resolution.height_px}"


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise BaselineError(f"{field_name} must be a mapping.")
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
        raise BaselineError(f"{field_name} must be a non-empty string.")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise BaselineError(f"{field_name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise BaselineError(f"{field_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise BaselineError(f"{field_name} must be positive.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise BaselineError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise BaselineError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise BaselineError(f"{field_name} must be non-negative.")
    return parsed
