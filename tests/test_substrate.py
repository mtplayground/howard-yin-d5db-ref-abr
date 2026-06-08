from __future__ import annotations

import pytest

from ref_abr.substrate import (
    ParametricSubstrateCoefficients,
    ParametricSubstrateValueProvider,
    ReferenceResolution,
    SubstrateError,
    SubstrateQuery,
    SubstrateValueProvider,
    coerce_ref_resolution,
    load_parametric_substrate_provider,
)


def test_default_parametric_provider_evaluates_query_contract() -> None:
    provider: SubstrateValueProvider = ParametricSubstrateValueProvider()
    value = provider.evaluate(
        SubstrateQuery(
            layer=2,
            ref_resolution="1080p",
            fov_deg=90,
            view_mismatch_deg=5,
            freshness_ms=100,
        )
    )

    assert value.provider_id == "parametric-substrate-default"
    assert 0 <= value.visible_quality <= 1
    assert value.query.ref_resolution == ReferenceResolution(1920, 1080)
    assert value.component_timing.generation_ms > 0
    assert value.component_timing.transfer_ms > 0
    assert value.component_timing.restoration_ms > 0
    assert value.component_timing.render_ms > 0
    assert value.component_timing.total_ms == pytest.approx(
        value.component_timing.generation_ms
        + value.component_timing.transfer_ms
        + value.component_timing.restoration_ms
        + value.component_timing.render_ms
    )
    assert value.uncertainty.quality_stddev > 0
    assert value.uncertainty.timing_stddev_ms > 0
    assert 0 <= value.uncertainty.confidence <= 1
    assert value.metadata["model"]["kind"] == "parametric"
    assert value.metadata["model"]["coefficient_id"]


def test_quality_degrades_with_mismatch_and_freshness_while_uncertainty_grows() -> None:
    provider = ParametricSubstrateValueProvider()
    fresh = provider.evaluate(
        {"layer": 1, "ref_resolution": "1280x720", "fov_deg": 85, "view_mismatch_deg": 0, "freshness_ms": 0}
    )
    stale_mismatched = provider.evaluate(
        {"layer": 1, "ref_resolution": "1280x720", "fov_deg": 85, "view_mismatch_deg": 45, "freshness_ms": 2000}
    )

    assert fresh.visible_quality > stale_mismatched.visible_quality
    assert stale_mismatched.uncertainty.quality_stddev > fresh.uncertainty.quality_stddev


def test_resolution_and_layer_increase_component_timing() -> None:
    provider = ParametricSubstrateValueProvider()
    small = provider.evaluate(
        {"layer": 0, "ref_resolution": "720p", "fov_deg": 90, "view_mismatch_deg": 0, "freshness_ms": 0}
    )
    large = provider.evaluate(
        {"layer": 3, "ref_resolution": "4k", "fov_deg": 90, "view_mismatch_deg": 0, "freshness_ms": 0}
    )

    assert large.component_timing.total_ms > small.component_timing.total_ms
    assert large.visible_quality >= small.visible_quality


def test_load_parametric_provider_from_yaml_with_coefficients(tmp_path) -> None:
    provider_path = tmp_path / "substrate.yml"
    provider_path.write_text(
        """
provider_id: calibrated-parametric
backend: parametric
coefficients:
  base_quality: 0.7
  transfer_resolution_ms: 3
  timing_uncertainty_ms: 2
metadata:
  calibration_set: smoke
""",
        encoding="utf-8",
    )

    provider = load_parametric_substrate_provider(provider_path)
    value = provider.evaluate(
        {
            "layer_index": 0,
            "resolution": { "width": 1920, "height": 1080 },
            "fov": 90,
            "mismatch_deg": 0,
            "age_ms": 0,
        }
    )

    assert provider.provider_id == "calibrated-parametric"
    assert provider.coefficients.base_quality == 0.7
    assert provider.metadata["calibration_set"] == "smoke"
    assert provider.metadata["provenance"]["source_uri"] == str(provider_path)
    assert value.uncertainty.timing_stddev_ms == pytest.approx(2.5)


def test_coefficients_and_provider_outputs_are_deterministic() -> None:
    coefficients = ParametricSubstrateCoefficients.from_mapping({"base_quality": "0.6"})
    first_provider = ParametricSubstrateValueProvider(coefficients=coefficients, provider_id="p")
    second_provider = ParametricSubstrateValueProvider(coefficients=coefficients, provider_id="p")
    query = {"layer": 1, "ref_width_px": 1920, "ref_height_px": 1080, "fov_deg": 90, "view_mismatch_deg": 1, "freshness_ms": 10}

    assert first_provider.evaluate(query).as_payload() == second_provider.evaluate(query).as_payload()


def test_resolution_coercion_accepts_labels_tuples_and_mappings() -> None:
    assert coerce_ref_resolution("1080p") == ReferenceResolution(1920, 1080)
    assert coerce_ref_resolution((640, 480)) == ReferenceResolution(640, 480)
    assert coerce_ref_resolution({"w": "320", "h": "240"}) == ReferenceResolution(320, 240)


def test_invalid_substrate_inputs_raise_clear_errors() -> None:
    provider = ParametricSubstrateValueProvider()

    with pytest.raises(SubstrateError, match="query.layer"):
        provider.evaluate({"ref_resolution": "720p", "fov_deg": 90, "view_mismatch_deg": 0, "freshness_ms": 0})

    with pytest.raises(SubstrateError, match="fov_deg"):
        provider.evaluate({"layer": 0, "ref_resolution": "720p", "fov_deg": 200, "view_mismatch_deg": 0, "freshness_ms": 0})

    with pytest.raises(SubstrateError, match="ref_resolution"):
        provider.evaluate({"layer": 0, "ref_resolution": "bad", "fov_deg": 90, "view_mismatch_deg": 0, "freshness_ms": 0})

    with pytest.raises(SubstrateError, match="unknown field"):
        ParametricSubstrateCoefficients.from_mapping({"not_a_coefficient": 1})
