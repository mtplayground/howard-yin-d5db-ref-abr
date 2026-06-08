from __future__ import annotations

import pytest

from ref_abr.accounting import (
    AccountingError,
    ResourceAccountingConfig,
    account_candidate_resources,
    account_candidate_set_resources,
)
from ref_abr.candidates import CandidateObject, CandidateSet, TileSpec
from ref_abr.devices import DeviceBudgets, DeviceProfile
from ref_abr.substrate import ComponentTiming, SubstrateQuery, SubstrateUncertainty, SubstrateValue


def test_account_candidate_resources_splits_timing_and_utilization() -> None:
    account = account_candidate_resources(
        _candidate("base", size_bytes=1_000_000),
        substrate_provider=FixedProvider(),
        device_profile=_device_profile(),
        config=ResourceAccountingConfig(queue_ms=2, bandwidth_bps=8_000_000, decode_fraction_of_restoration=0.25),
    )

    assert account.timing.server_generation_ms == 10
    assert account.timing.queue_ms == 2
    assert account.timing.transfer_ms == 1000
    assert account.timing.decode_ms == 5
    assert account.timing.restore_ms == 15
    assert account.timing.render_ms == 30
    assert account.timing.total_ms == 1062
    assert account.utilization.server_generation == 0.5
    assert account.utilization.transfer_time == 2.0
    assert account.utilization.decode == 1.0
    assert account.utilization.restore == 1.0
    assert account.utilization.render == 0.5
    assert account.utilization.bandwidth == 1.0
    assert account.transfer_bytes == 1_000_000
    assert account.provider_id == "fixed-substrate"
    assert account.metadata["accounting"]["device_class"] == "edge"


def test_account_candidate_resources_uses_substrate_transfer_without_bandwidth_override() -> None:
    account = account_candidate_resources(
        _candidate("base", size_bytes=1_000_000),
        substrate_provider=FixedProvider(),
        device_profile=_device_profile(),
        config=ResourceAccountingConfig(queue_ms=0, bandwidth_bps=None),
    )

    assert account.timing.transfer_ms == 4
    assert account.utilization.bandwidth is None


def test_accounting_estimates_tile_memory_from_resolution_and_tile_area() -> None:
    full = account_candidate_resources(
        _candidate("full", size_bytes=100_000),
        substrate_provider=FixedProvider(),
        device_profile=_device_profile(),
        config=ResourceAccountingConfig(memory_mb_per_megapixel=100),
    )
    tile = account_candidate_resources(
        _candidate("tile", size_bytes=100_000, tile=TileSpec(row=0, column=0, rows=2, columns=2)),
        substrate_provider=FixedProvider(),
        device_profile=_device_profile(),
        config=ResourceAccountingConfig(memory_mb_per_megapixel=100),
    )

    assert round(full.memory_mb, 2) == 92.16
    assert round(tile.memory_mb, 2) == 23.04
    assert tile.utilization.memory < full.utilization.memory


def test_account_candidate_set_resources_sums_timing_and_bytes() -> None:
    candidate_set = CandidateSet(
        candidate_set_id="set-a",
        decision_time_ms=0,
        candidates=(
            _candidate("base-a", size_bytes=100_000),
            _candidate("base-b", size_bytes=200_000, layer=1),
        ),
    )

    summary = account_candidate_set_resources(
        candidate_set,
        substrate_provider=FixedProvider(),
        device_profile=_device_profile(),
        config=ResourceAccountingConfig(queue_ms=1, bandwidth_bps=8_000_000),
    )

    assert len(summary.accounts) == 2
    assert summary.total_transfer_bytes == 300_000
    assert summary.total_timing.server_generation_ms == 20
    assert summary.total_timing.queue_ms == 2
    assert summary.total_timing.transfer_ms == 300
    assert summary.peak_memory_mb == max(account.memory_mb for account in summary.accounts)
    assert summary.metadata["accounting"]["candidate_count"] == 2


def test_accounting_ids_are_deterministic() -> None:
    kwargs = {
        "candidate": _candidate("base", size_bytes=100_000),
        "substrate_provider": FixedProvider(),
        "device_profile": _device_profile(),
        "config": ResourceAccountingConfig(queue_ms=1),
    }

    first = account_candidate_resources(**kwargs)
    second = account_candidate_resources(**kwargs)

    assert first.account_id == second.account_id


def test_accounting_validation_rejects_bad_inputs() -> None:
    with pytest.raises(AccountingError, match="decode_fraction"):
        ResourceAccountingConfig(decode_fraction_of_restoration=2)
    with pytest.raises(AccountingError, match="bandwidth_bps"):
        ResourceAccountingConfig(bandwidth_bps=0)
    with pytest.raises(AccountingError, match="substrate_provider"):
        account_candidate_resources(
            _candidate("base"),
            substrate_provider=object(),  # type: ignore[arg-type]
            device_profile=_device_profile(),
        )


class FixedProvider:
    provider_id = "fixed-substrate"

    def evaluate(self, query):
        substrate_query = query if isinstance(query, SubstrateQuery) else SubstrateQuery(**query)
        return SubstrateValue(
            provider_id=self.provider_id,
            query=substrate_query,
            visible_quality=0.8,
            component_timing=ComponentTiming(
                generation_ms=10,
                transfer_ms=4,
                restoration_ms=20,
                render_ms=30,
            ),
            uncertainty=SubstrateUncertainty(quality_stddev=0.01, timing_stddev_ms=1, confidence=0.95),
        )


def _candidate(
    candidate_id: str,
    *,
    size_bytes: int = 100_000,
    tile: TileSpec | None = None,
    layer: int = 0,
) -> CandidateObject:
    return CandidateObject(
        candidate_id=candidate_id,
        object_id=f"object-{candidate_id}",
        candidate_kind="tile" if tile is not None else "gaussian_base",
        decision_time_ms=0,
        layer=layer,
        resolution="720p",
        fov_deg=90,
        viewpoint=None,
        lookahead_ms=0,
        expiration_ms=100,
        retransmit_priority=0,
        size_bytes=size_bytes,
        tile=tile,
    )


def _device_profile() -> DeviceProfile:
    return DeviceProfile(
        profile_id="edge-test",
        device_class="edge",
        budgets=DeviceBudgets(
            generation_ms=20,
            transfer_ms=500,
            restoration_ms=20,
            render_ms=60,
            memory_mb=512,
            fps=60,
        ),
    )
