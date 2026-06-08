from __future__ import annotations

import pytest

from ref_abr.substrate import (
    EmpiricalSubstrateValueProvider,
    ParametricSubstrateCoefficients,
    ParametricSubstrateValueProvider,
    ReferenceResolution,
    SubstrateError,
    SubstrateQuery,
    SubstrateValueProvider,
    coerce_ref_resolution,
    empirical_substrate_provider_from_mapping,
    load_empirical_substrate_provider,
    load_parametric_substrate_provider,
    load_substrate_provider,
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


def test_load_empirical_provider_from_yaml_and_interpolate(tmp_path) -> None:
    provider_path = tmp_path / "empirical.yml"
    provider_path.write_text(
        """
provider_id: measured-lut
backend: empirical
interpolation: linear
max_neighbors: 2
table_id: table-a
table:
  rows:
    - layer: 0
      ref_resolution: 720p
      fov_deg: 90
      view_mismatch_deg: 0
      freshness_ms: 0
      visible_quality: 0.8
      component_timing:
        generation_ms: 1
        transfer_ms: 2
        restoration_ms: 3
        render_ms: 4
      uncertainty:
        quality_stddev: 0.02
        timing_stddev_ms: 0.5
        confidence: 0.95
    - layer: 0
      ref_resolution: 720p
      fov_deg: 90
      view_mismatch_deg: 40
      freshness_ms: 0
      visible_quality: 0.4
      generation_ms: 5
      transfer_ms: 6
      restoration_ms: 7
      render_ms: 8
      quality_stddev: 0.08
      timing_stddev_ms: 1.5
      confidence: 0.75
""",
        encoding="utf-8",
    )

    provider = load_empirical_substrate_provider(provider_path)
    value = provider.evaluate(
        {"layer": 0, "ref_resolution": "720p", "fov_deg": 90, "view_mismatch_deg": 20, "freshness_ms": 0}
    )

    assert isinstance(provider, EmpiricalSubstrateValueProvider)
    assert provider.provider_id == "measured-lut"
    assert provider.interpolation == "inverse_distance"
    assert provider.table.table_id == "table-a"
    assert value.provider_id == "measured-lut"
    assert value.visible_quality == pytest.approx(0.6)
    assert value.component_timing.generation_ms == pytest.approx(3)
    assert value.component_timing.render_ms == pytest.approx(6)
    assert value.uncertainty.quality_stddev == pytest.approx(0.05)
    assert value.uncertainty.confidence == pytest.approx(0.85)
    assert value.metadata["model"]["kind"] == "empirical_lookup"
    assert value.metadata["model"]["neighbor_count"] == 2


def test_empirical_provider_exact_match_uses_exact_row() -> None:
    provider = empirical_substrate_provider_from_mapping(
        {
            "backend": "lookup-table",
            "interpolation": "inverse_distance",
            "rows": [
                {
                    "query": {
                        "layer": 1,
                        "ref_width_px": 1920,
                        "ref_height_px": 1080,
                        "fov_deg": 90,
                        "view_mismatch_deg": 0,
                        "freshness_ms": 0,
                    },
                    "quality": 0.9,
                    "generation_ms": 10,
                    "transfer_ms": 20,
                    "restoration_ms": 30,
                    "render_ms": 40,
                },
                {
                    "layer": 1,
                    "ref_resolution": "1080p",
                    "fov_deg": 90,
                    "view_mismatch_deg": 30,
                    "freshness_ms": 0,
                    "visible_quality": 0.1,
                    "generation_ms": 100,
                    "transfer_ms": 100,
                    "restoration_ms": 100,
                    "render_ms": 100,
                },
            ],
        }
    )

    value = provider.evaluate(
        {"layer": 1, "ref_resolution": "1080p", "fov_deg": 90, "view_mismatch_deg": 0, "freshness_ms": 0}
    )

    assert value.visible_quality == 0.9
    assert value.component_timing.total_ms == 100
    assert value.uncertainty.confidence == 1.0
    assert value.metadata["model"]["interpolation"] == "inverse_distance"
    assert value.metadata["model"]["neighbor_count"] == 1


def test_load_substrate_provider_selects_backend_per_run(tmp_path) -> None:
    empirical_path = tmp_path / "provider.yml"
    empirical_path.write_text(
        """
backend: empirical
rows:
  - layer: 0
    ref_resolution: 480p
    fov_deg: 90
    view_mismatch_deg: 0
    freshness_ms: 0
    visible_quality: 0.5
    generation_ms: 1
    transfer_ms: 1
    restoration_ms: 1
    render_ms: 1
""",
        encoding="utf-8",
    )

    provider = load_substrate_provider(empirical_path)

    assert isinstance(provider, EmpiricalSubstrateValueProvider)
    assert provider.evaluate({"layer": 0, "ref_resolution": "480p", "fov_deg": 90, "view_mismatch_deg": 0, "freshness_ms": 0}).visible_quality == 0.5

    direct_provider = load_empirical_substrate_provider(empirical_path)
    assert isinstance(direct_provider, EmpiricalSubstrateValueProvider)


def test_empirical_nearest_mode_uses_closest_row() -> None:
    provider = empirical_substrate_provider_from_mapping(
        {
            "backend": "empirical",
            "interpolation": "nearest",
            "rows": [
                {
                    "layer": 0,
                    "ref_resolution": "720p",
                    "fov_deg": 90,
                    "view_mismatch_deg": 0,
                    "freshness_ms": 0,
                    "visible_quality": 0.9,
                    "generation_ms": 1,
                    "transfer_ms": 1,
                    "restoration_ms": 1,
                    "render_ms": 1,
                },
                {
                    "layer": 0,
                    "ref_resolution": "720p",
                    "fov_deg": 90,
                    "view_mismatch_deg": 60,
                    "freshness_ms": 0,
                    "visible_quality": 0.2,
                    "generation_ms": 9,
                    "transfer_ms": 9,
                    "restoration_ms": 9,
                    "render_ms": 9,
                },
            ],
        }
    )

    value = provider.evaluate(
        {"layer": 0, "ref_resolution": "720p", "fov_deg": 90, "view_mismatch_deg": 55, "freshness_ms": 0}
    )

    assert value.visible_quality == 0.2
    assert value.component_timing.generation_ms == 9
    assert value.metadata["model"]["interpolation"] == "nearest"


def test_invalid_empirical_provider_inputs_raise_clear_errors() -> None:
    with pytest.raises(SubstrateError, match="table.rows"):
        empirical_substrate_provider_from_mapping({"backend": "empirical", "rows": []})

    with pytest.raises(SubstrateError, match="visible_quality"):
        empirical_substrate_provider_from_mapping(
            {
                "backend": "empirical",
                "rows": [
                    {
                        "layer": 0,
                        "ref_resolution": "720p",
                        "fov_deg": 90,
                        "view_mismatch_deg": 0,
                        "freshness_ms": 0,
                        "generation_ms": 1,
                        "transfer_ms": 1,
                        "restoration_ms": 1,
                        "render_ms": 1,
                    }
                ],
            }
        )

    with pytest.raises(SubstrateError, match="interpolation"):
        empirical_substrate_provider_from_mapping(
            {
                "backend": "empirical",
                "interpolation": "spline",
                "rows": [
                    {
                        "layer": 0,
                        "ref_resolution": "720p",
                        "fov_deg": 90,
                        "view_mismatch_deg": 0,
                        "freshness_ms": 0,
                        "visible_quality": 1,
                        "generation_ms": 1,
                        "transfer_ms": 1,
                        "restoration_ms": 1,
                        "render_ms": 1,
                    }
                ],
            }
        )
