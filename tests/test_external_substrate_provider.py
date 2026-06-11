from __future__ import annotations

import json

import pytest

from ref_abr.providers import (
    ExternalSubstrateProviderConfig,
    ExternalTraceSubstrateProvider,
    external_substrate_provider_from_mapping,
    load_external_substrate_provider,
)
from ref_abr.providers.base import external_measurement_to_substrate_value, query_signature
from ref_abr.substrate import SubstrateError, SubstrateQuery, load_substrate_provider, substrate_provider_from_mapping


def _query() -> dict[str, object]:
    return {
        "layer": 1,
        "ref_resolution": "720p",
        "fov_deg": 90,
        "view_mismatch_deg": 5,
        "freshness_ms": 10,
    }


def _record(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "generation_ms": 1.0,
        "transfer_ms": 2.0,
        "decode_ms": 3.0,
        "restore_ms": 4.0,
        "render_ms": 5.0,
        "size_bytes": 2048,
        "visible_quality": 0.83,
        "dropped_frame": False,
        "deadline_hit": True,
        "provenance": {"backend": "fixture"},
        "record_id": "external-a",
        "backend_id": "trace-fixture",
        "object_id": "object-a",
        "frame_id": "frame-a",
        "candidate_kind": "gaussian_base",
        "metadata": {"query": _query()},
    }
    payload.update(overrides)
    return payload


def test_external_trace_provider_maps_measurement_to_substrate_value() -> None:
    provider = ExternalTraceSubstrateProvider(
        records=[_record()],
        config=ExternalSubstrateProviderConfig(provider_id="external-fixture", timing_stddev_ms=0.25, confidence=0.9),
    )

    value = provider.evaluate(_query())

    assert value.provider_id == "external-fixture"
    assert value.visible_quality == 0.83
    assert value.component_timing.generation_ms == 1.0
    assert value.component_timing.transfer_ms == 2.0
    assert value.component_timing.restoration_ms == 7.0
    assert value.component_timing.render_ms == 5.0
    assert value.uncertainty.timing_stddev_ms == 0.25
    assert value.uncertainty.confidence == 0.9
    assert value.metadata["model"]["kind"] == "external_trace"
    assert value.metadata["model"]["record_id"] == "external-a"
    assert value.metadata["model"]["size_bytes"] == 2048
    assert value.metadata["record_metadata"]["query"] == _query()


def test_external_trace_provider_uses_first_policy_without_query_metadata() -> None:
    provider = external_substrate_provider_from_mapping(
        {
            "backend": "external",
            "provider_id": "first-provider",
            "match_policy": "first",
            "records": [_record(metadata={"note": "no query"})],
        }
    )

    value = provider.evaluate({"layer": 7, "ref_resolution": "480p", "fov_deg": 75, "view_mismatch_deg": 30, "freshness_ms": 0})

    assert value.provider_id == "first-provider"
    assert value.query.layer == 7
    assert value.visible_quality == 0.83
    assert provider.as_payload()["record_count"] == 1


def test_external_trace_provider_rejects_missing_query_match() -> None:
    provider = ExternalTraceSubstrateProvider(
        records=[
            _record(
                metadata={
                    "query": {
                        "layer": 0,
                        "ref_resolution": "480p",
                        "fov_deg": 90,
                        "view_mismatch_deg": 0,
                        "freshness_ms": 0,
                    }
                }
            )
        ]
    )

    with pytest.raises(SubstrateError, match="no record matching query"):
        provider.evaluate(_query())


def test_external_provider_config_selection_from_mapping_and_file(tmp_path) -> None:
    records_path = tmp_path / "measurements.jsonl"
    records_path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    provider_path = tmp_path / "provider.yml"
    provider_path.write_text(
        """
backend: external
provider_id: external-configured
records_path: measurements.jsonl
uncertainty:
  quality_stddev: 0.04
  timing_stddev_ms: 0.7
  confidence: 0.88
metadata:
  experiment: smoke
""",
        encoding="utf-8",
    )

    loaded = load_external_substrate_provider(provider_path)
    selected = load_substrate_provider(provider_path)

    assert isinstance(loaded, ExternalTraceSubstrateProvider)
    assert isinstance(selected, ExternalTraceSubstrateProvider)
    assert loaded.provider_id == "external-configured"
    assert selected.evaluate(_query()).visible_quality == 0.83
    assert selected.evaluate(_query()).uncertainty.quality_stddev == 0.04
    assert selected.as_payload()["metadata"]["experiment"] == "smoke"


def test_substrate_provider_from_mapping_selects_external_backend() -> None:
    provider = substrate_provider_from_mapping({"backend": "external-trace", "records": [_record()]})

    assert isinstance(provider, ExternalTraceSubstrateProvider)
    assert provider.evaluate(_query()).metadata["model"]["backend_id"] == "trace-fixture"


def test_external_measurement_to_substrate_value_requires_valid_types() -> None:
    provider = ExternalTraceSubstrateProvider(records=[_record()])
    record = provider.records[0]
    query = SubstrateQuery(layer=1, ref_resolution="720p", fov_deg=90, view_mismatch_deg=5, freshness_ms=10)

    value = external_measurement_to_substrate_value(record, query)

    assert query_signature(query) == query_signature(_query())
    assert value.component_timing.total_ms == 15.0
    with pytest.raises(SubstrateError, match="record must be"):
        external_measurement_to_substrate_value(object(), query)  # type: ignore[arg-type]
