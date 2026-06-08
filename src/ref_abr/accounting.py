"""Component timing and resource accounting."""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.candidates import CandidateObject, CandidateSet
from ref_abr.config import stable_config_id
from ref_abr.devices import DeviceBudgets, DeviceProfile
from ref_abr.substrate import SubstrateQuery, SubstrateValue, SubstrateValueProvider


class AccountingError(ValueError):
    """Raised when timing or resource accounting inputs are invalid."""


@dataclass(frozen=True)
class ResourceAccountingConfig:
    """Controls for converting substrate timings into resource accounts."""

    queue_ms: float = 0.0
    bandwidth_bps: int | None = None
    decode_fraction_of_restoration: float = 0.35
    memory_mb_per_megapixel: float = 128.0
    memory_layer_multiplier: float = 0.25
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "queue_ms", _non_negative_float(self.queue_ms, "queue_ms"))
        if self.bandwidth_bps is not None:
            object.__setattr__(self, "bandwidth_bps", _positive_int(self.bandwidth_bps, "bandwidth_bps"))
        decode_fraction = _unit_interval(self.decode_fraction_of_restoration, "decode_fraction_of_restoration")
        object.__setattr__(self, "decode_fraction_of_restoration", decode_fraction)
        object.__setattr__(self, "memory_mb_per_megapixel", _positive_float(self.memory_mb_per_megapixel, "memory_mb_per_megapixel"))
        object.__setattr__(self, "memory_layer_multiplier", _non_negative_float(self.memory_layer_multiplier, "memory_layer_multiplier"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "queue_ms": self.queue_ms,
            "bandwidth_bps": self.bandwidth_bps,
            "decode_fraction_of_restoration": self.decode_fraction_of_restoration,
            "memory_mb_per_megapixel": self.memory_mb_per_megapixel,
            "memory_layer_multiplier": self.memory_layer_multiplier,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class ComponentTimingAccount:
    """Per-component timing account in milliseconds."""

    server_generation_ms: float
    queue_ms: float
    transfer_ms: float
    decode_ms: float
    restore_ms: float
    render_ms: float

    def __post_init__(self) -> None:
        for field_name in (
            "server_generation_ms",
            "queue_ms",
            "transfer_ms",
            "decode_ms",
            "restore_ms",
            "render_ms",
        ):
            object.__setattr__(self, field_name, _non_negative_float(getattr(self, field_name), field_name))

    @property
    def total_ms(self) -> float:
        return (
            self.server_generation_ms
            + self.queue_ms
            + self.transfer_ms
            + self.decode_ms
            + self.restore_ms
            + self.render_ms
        )

    def as_payload(self) -> dict[str, float]:
        return {
            "server_generation_ms": self.server_generation_ms,
            "queue_ms": self.queue_ms,
            "transfer_ms": self.transfer_ms,
            "decode_ms": self.decode_ms,
            "restore_ms": self.restore_ms,
            "render_ms": self.render_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True)
class ResourceUtilization:
    """Utilization ratios against device and transport budgets."""

    server_generation: float
    queue: float
    transfer_time: float
    decode: float
    restore: float
    render: float
    memory: float
    bandwidth: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("server_generation", "queue", "transfer_time", "decode", "restore", "render", "memory"):
            object.__setattr__(self, field_name, _non_negative_float(getattr(self, field_name), field_name))
        if self.bandwidth is not None:
            object.__setattr__(self, "bandwidth", _non_negative_float(self.bandwidth, "bandwidth"))

    @property
    def max_component(self) -> float:
        values = [
            self.server_generation,
            self.queue,
            self.transfer_time,
            self.decode,
            self.restore,
            self.render,
            self.memory,
        ]
        if self.bandwidth is not None:
            values.append(self.bandwidth)
        return max(values)

    def as_payload(self) -> dict[str, float | None]:
        return {
            "server_generation": self.server_generation,
            "queue": self.queue,
            "transfer_time": self.transfer_time,
            "decode": self.decode,
            "restore": self.restore,
            "render": self.render,
            "memory": self.memory,
            "bandwidth": self.bandwidth,
            "max_component": self.max_component,
        }


@dataclass(frozen=True)
class CandidateResourceAccount:
    """Resource accounting result for one candidate."""

    account_id: str
    candidate_id: str
    object_id: str
    provider_id: str
    device_profile_id: str
    timing: ComponentTimingAccount
    utilization: ResourceUtilization
    memory_mb: float
    bandwidth_bps: int | None
    transfer_bytes: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.account_id, "account_id")
        _require_non_empty(self.candidate_id, "candidate_id")
        _require_non_empty(self.object_id, "object_id")
        _require_non_empty(self.provider_id, "provider_id")
        _require_non_empty(self.device_profile_id, "device_profile_id")
        if not isinstance(self.timing, ComponentTimingAccount):
            raise AccountingError("timing must be a ComponentTimingAccount record.")
        if not isinstance(self.utilization, ResourceUtilization):
            raise AccountingError("utilization must be a ResourceUtilization record.")
        object.__setattr__(self, "memory_mb", _non_negative_float(self.memory_mb, "memory_mb"))
        if self.bandwidth_bps is not None:
            object.__setattr__(self, "bandwidth_bps", _positive_int(self.bandwidth_bps, "bandwidth_bps"))
        object.__setattr__(self, "transfer_bytes", _non_negative_int(self.transfer_bytes, "transfer_bytes"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "candidate_id": self.candidate_id,
            "object_id": self.object_id,
            "provider_id": self.provider_id,
            "device_profile_id": self.device_profile_id,
            "timing": self.timing.as_payload(),
            "utilization": self.utilization.as_payload(),
            "memory_mb": self.memory_mb,
            "bandwidth_bps": self.bandwidth_bps,
            "transfer_bytes": self.transfer_bytes,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class ResourceAccountingSummary:
    """Aggregate resource accounting for a candidate set."""

    summary_id: str
    accounts: tuple[CandidateResourceAccount, ...]
    total_timing: ComponentTimingAccount
    peak_memory_mb: float
    total_transfer_bytes: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.summary_id, "summary_id")
        accounts = tuple(self.accounts)
        for account in accounts:
            if not isinstance(account, CandidateResourceAccount):
                raise AccountingError("accounts must contain CandidateResourceAccount records.")
        if not isinstance(self.total_timing, ComponentTimingAccount):
            raise AccountingError("total_timing must be a ComponentTimingAccount record.")
        object.__setattr__(self, "accounts", accounts)
        object.__setattr__(self, "peak_memory_mb", _non_negative_float(self.peak_memory_mb, "peak_memory_mb"))
        object.__setattr__(self, "total_transfer_bytes", _non_negative_int(self.total_transfer_bytes, "total_transfer_bytes"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "accounts": [account.as_payload() for account in self.accounts],
            "total_timing": self.total_timing.as_payload(),
            "peak_memory_mb": self.peak_memory_mb,
            "total_transfer_bytes": self.total_transfer_bytes,
            "metadata": _to_payload(self.metadata),
        }


def account_candidate_resources(
    candidate: CandidateObject,
    *,
    substrate_provider: SubstrateValueProvider,
    device_profile: DeviceProfile,
    config: ResourceAccountingConfig | None = None,
    substrate_value: SubstrateValue | None = None,
) -> CandidateResourceAccount:
    """Account component timings and resource utilization for one candidate."""

    if not isinstance(candidate, CandidateObject):
        raise AccountingError("candidate must be a CandidateObject record.")
    if not isinstance(device_profile, DeviceProfile):
        raise AccountingError("device_profile must be a DeviceProfile record.")
    if not callable(getattr(substrate_provider, "evaluate", None)):
        raise AccountingError("substrate_provider must expose evaluate(query).")
    accounting_config = config or ResourceAccountingConfig()
    if not isinstance(accounting_config, ResourceAccountingConfig):
        raise AccountingError("config must be a ResourceAccountingConfig record.")
    substrate = substrate_value or substrate_provider.evaluate(_query_for_candidate(candidate))
    if not isinstance(substrate, SubstrateValue):
        raise AccountingError("substrate_provider.evaluate must return a SubstrateValue record.")
    timing = _timing_account(candidate, substrate, accounting_config)
    memory_mb = _candidate_memory_mb(candidate, accounting_config)
    utilization = _resource_utilization(timing, memory_mb, candidate, device_profile.budgets, accounting_config)
    payload = {
        "candidate_id": candidate.candidate_id,
        "provider_id": substrate.provider_id,
        "device_profile_id": device_profile.profile_id,
        "timing": timing.as_payload(),
        "utilization": utilization.as_payload(),
        "memory_mb": memory_mb,
        "transfer_bytes": candidate.size_bytes,
        "config": accounting_config.as_payload(),
    }
    return CandidateResourceAccount(
        account_id=f"resource-account-{stable_config_id(payload)}",
        candidate_id=candidate.candidate_id,
        object_id=candidate.object_id,
        provider_id=substrate.provider_id,
        device_profile_id=device_profile.profile_id,
        timing=timing,
        utilization=utilization,
        memory_mb=memory_mb,
        bandwidth_bps=accounting_config.bandwidth_bps,
        transfer_bytes=candidate.size_bytes,
        metadata={
            "accounting": {
                "candidate_kind": candidate.candidate_kind,
                "device_class": device_profile.device_class,
                "device_budgets": device_profile.budgets.as_payload(),
                "substrate": substrate.as_payload(),
                "config": accounting_config.as_payload(),
            }
        },
    )


def account_candidate_set_resources(
    candidate_set: CandidateSet,
    *,
    substrate_provider: SubstrateValueProvider,
    device_profile: DeviceProfile,
    config: ResourceAccountingConfig | None = None,
) -> ResourceAccountingSummary:
    """Account resource use for every candidate in a candidate set."""

    if not isinstance(candidate_set, CandidateSet):
        raise AccountingError("candidate_set must be a CandidateSet record.")
    accounts = tuple(
        account_candidate_resources(
            candidate,
            substrate_provider=substrate_provider,
            device_profile=device_profile,
            config=config,
        )
        for candidate in candidate_set.candidates
    )
    total_timing = ComponentTimingAccount(
        server_generation_ms=sum(account.timing.server_generation_ms for account in accounts),
        queue_ms=sum(account.timing.queue_ms for account in accounts),
        transfer_ms=sum(account.timing.transfer_ms for account in accounts),
        decode_ms=sum(account.timing.decode_ms for account in accounts),
        restore_ms=sum(account.timing.restore_ms for account in accounts),
        render_ms=sum(account.timing.render_ms for account in accounts),
    )
    peak_memory_mb = max((account.memory_mb for account in accounts), default=0.0)
    total_transfer_bytes = sum(account.transfer_bytes for account in accounts)
    payload = {
        "candidate_set_id": candidate_set.candidate_set_id,
        "accounts": [account.account_id for account in accounts],
        "total_timing": total_timing.as_payload(),
        "peak_memory_mb": peak_memory_mb,
        "total_transfer_bytes": total_transfer_bytes,
    }
    return ResourceAccountingSummary(
        summary_id=f"resource-summary-{stable_config_id(payload)}",
        accounts=accounts,
        total_timing=total_timing,
        peak_memory_mb=peak_memory_mb,
        total_transfer_bytes=total_transfer_bytes,
        metadata={
            "accounting": {
                "candidate_set_id": candidate_set.candidate_set_id,
                "candidate_count": len(accounts),
                "device_profile_id": device_profile.profile_id,
                "provider_id": getattr(substrate_provider, "provider_id", "unknown"),
            }
        },
    )


def _query_for_candidate(candidate: CandidateObject) -> SubstrateQuery:
    return SubstrateQuery(
        layer=candidate.layer,
        ref_resolution=candidate.resolution,
        fov_deg=candidate.fov_deg,
        view_mismatch_deg=0.0,
        freshness_ms=candidate.lookahead_ms,
        metadata={
            "candidate_id": candidate.candidate_id,
            "candidate_kind": candidate.candidate_kind,
        },
    )


def _timing_account(
    candidate: CandidateObject,
    substrate: SubstrateValue,
    config: ResourceAccountingConfig,
) -> ComponentTimingAccount:
    timing = substrate.component_timing
    decode_ms = timing.restoration_ms * config.decode_fraction_of_restoration
    restore_ms = timing.restoration_ms - decode_ms
    transfer_ms = timing.transfer_ms
    if config.bandwidth_bps is not None:
        network_transfer_ms = candidate.size_bytes * 8_000.0 / config.bandwidth_bps
        transfer_ms = max(transfer_ms, network_transfer_ms)
    return ComponentTimingAccount(
        server_generation_ms=timing.generation_ms,
        queue_ms=config.queue_ms,
        transfer_ms=transfer_ms,
        decode_ms=decode_ms,
        restore_ms=restore_ms,
        render_ms=timing.render_ms,
    )


def _resource_utilization(
    timing: ComponentTimingAccount,
    memory_mb: float,
    candidate: CandidateObject,
    budgets: DeviceBudgets,
    config: ResourceAccountingConfig,
) -> ResourceUtilization:
    decode_budget_ms = max(1.0, budgets.restoration_ms * config.decode_fraction_of_restoration)
    restore_budget_ms = max(1.0, budgets.restoration_ms - decode_budget_ms)
    bandwidth_utilization = None
    if config.bandwidth_bps is not None:
        transfer_seconds = max(0.001, timing.transfer_ms / 1000.0)
        effective_bps = candidate.size_bytes * 8.0 / transfer_seconds
        bandwidth_utilization = effective_bps / config.bandwidth_bps
    return ResourceUtilization(
        server_generation=_ratio(timing.server_generation_ms, budgets.generation_ms),
        queue=_ratio(timing.queue_ms, budgets.transfer_ms),
        transfer_time=_ratio(timing.transfer_ms, budgets.transfer_ms),
        decode=_ratio(timing.decode_ms, decode_budget_ms),
        restore=_ratio(timing.restore_ms, restore_budget_ms),
        render=_ratio(timing.render_ms, budgets.render_ms),
        memory=_ratio(memory_mb, budgets.memory_mb),
        bandwidth=bandwidth_utilization,
    )


def _candidate_memory_mb(candidate: CandidateObject, config: ResourceAccountingConfig) -> float:
    layer_multiplier = 1.0 + config.memory_layer_multiplier * candidate.layer
    tile_multiplier = 1.0
    if candidate.tile is not None:
        tile_multiplier = 1.0 / max(1, candidate.tile.rows * candidate.tile.columns)
    resolution_memory = candidate.resolution.megapixels * config.memory_mb_per_megapixel * layer_multiplier * tile_multiplier
    encoded_memory = candidate.size_bytes / (1024.0 * 1024.0)
    return max(resolution_memory, encoded_memory)


def _ratio(value: float, budget: float) -> float:
    return value / max(1.0, budget)


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise AccountingError(f"{field_name} must be a mapping.")
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
        raise AccountingError(f"{field_name} must be a non-empty string.")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise AccountingError(f"{field_name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise AccountingError(f"{field_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise AccountingError(f"{field_name} must be a positive integer.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise AccountingError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise AccountingError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise AccountingError(f"{field_name} must be a non-negative integer.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise AccountingError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AccountingError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise AccountingError(f"{field_name} must be finite.")
    return parsed


def _positive_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed <= 0:
        raise AccountingError(f"{field_name} must be positive.")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0:
        raise AccountingError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if not 0.0 <= parsed <= 1.0:
        raise AccountingError(f"{field_name} must be between 0 and 1.")
    return parsed


__all__ = [
    "AccountingError",
    "CandidateResourceAccount",
    "ComponentTimingAccount",
    "ResourceAccountingConfig",
    "ResourceAccountingSummary",
    "ResourceUtilization",
    "account_candidate_resources",
    "account_candidate_set_resources",
]
