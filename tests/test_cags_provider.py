from __future__ import annotations

from dataclasses import dataclass

import pytest

from ref_abr.domain import ExternalMeasurementRecord
from ref_abr.providers import CAGSAdapterConfig, CAGSAdapterError, CAGSBackendAdapter, normalize_cags_candidate_kind
from ref_abr.providers.cags import CAGSStageOutput, cags_stage_outputs_to_record, resolve_cags_module


class FakeCAGS:
    def __init__(self, artifact_path: str) -> None:
        self.artifact_path = artifact_path
        self.calls: list[tuple[str, dict[str, object]]] = []

    def encode(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("encode", dict(kwargs)))
        return {"encode_ms": 11.0, "compressed_dir": self.artifact_path}

    def decode(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("decode", dict(kwargs)))
        return {"decode_ms": 7.0}

    def pickup(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("pickup", dict(kwargs)))
        return {"pickup_ms": 5.0}

    def render(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("render", dict(kwargs)))
        return {"render_ms": 13.0, "psnr": 34.0, "lpips": 0.2}


def _query() -> dict[str, object]:
    return {
        "layer": 0,
        "ref_resolution": "720p",
        "fov_deg": 90,
        "view_mismatch_deg": 0,
        "freshness_ms": 0,
    }


def test_cags_adapter_invokes_all_stages_and_emits_external_measurement(tmp_path) -> None:
    compressed_dir = tmp_path / "compressed"
    compressed_dir.mkdir()
    (compressed_dir / "base.ply").write_bytes(b"ply")
    (compressed_dir / "enhancement.drc").write_bytes(b"drc-data")
    fake = FakeCAGS(str(compressed_dir))
    config = CAGSAdapterConfig(
        source_uri="file://scene",
        output_dir=str(compressed_dir),
        candidate_kind="base",
        object_id="object-a",
        frame_id="frame-a",
        query=_query(),
        encode_args={"quality": "fast"},
        render_args={"camera": "front"},
        transfer_ms=2.0,
        metadata={"run": "smoke"},
    )
    adapter = CAGSBackendAdapter(config=config, provider_id="cags-fixture")

    record = adapter.run_measurement(cags_module=fake)

    assert isinstance(record, ExternalMeasurementRecord)
    assert record.backend_id == "cags-fixture"
    assert record.candidate_kind == "gaussian_base"
    assert record.object_id == "object-a"
    assert record.generation_ms == 11.0
    assert record.transfer_ms == 2.0
    assert record.decode_ms == 7.0
    assert record.restore_ms == 5.0
    assert record.render_ms == 13.0
    assert record.size_bytes == len(b"ply") + len(b"drc-data")
    assert record.visible_quality == pytest.approx((0.7 + 0.8) / 2)
    assert record.metadata["query"] == _query()
    assert record.metadata["adapter"]["run"] == "smoke"
    assert [name for name, _ in fake.calls] == ["encode", "decode", "pickup", "render"]
    assert fake.calls[0][1]["source_uri"] == "file://scene"
    assert fake.calls[0][1]["output_dir"] == str(compressed_dir)
    assert fake.calls[0][1]["candidate_kind"] == "gaussian_base"
    assert fake.calls[0][1]["quality"] == "fast"
    assert fake.calls[3][1]["camera"] == "front"


def test_cags_adapter_can_wrap_record_as_external_substrate_provider(tmp_path) -> None:
    artifact = tmp_path / "artifact.ply"
    artifact.write_bytes(b"12345")

    class QualityCAGS(FakeCAGS):
        def encode(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(("encode", dict(kwargs)))
            return {"generation_ms": 1.0, "ply_path": str(artifact)}

        def render(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(("render", dict(kwargs)))
            return {"render_ms": 4.0, "visible_quality": 0.91}

    adapter = CAGSBackendAdapter(
        config=CAGSAdapterConfig(candidate_kind="tile", query=_query()),
        provider_id="cags-provider",
    )
    provider = adapter.run_provider(cags_module=QualityCAGS(str(artifact)))
    value = provider.evaluate(_query())

    assert provider.provider_id == "cags-provider"
    assert value.visible_quality == 0.91
    assert value.component_timing.total_ms == pytest.approx(1.0 + 7.0 + 5.0 + 4.0)
    assert value.metadata["model"]["candidate_kind"] == "tile"


def test_cags_stage_outputs_to_record_uses_explicit_size_and_quality() -> None:
    record = cags_stage_outputs_to_record(
        encode=CAGSStageOutput("encode", {"generation_ms": 3.0, "encoded_bytes": 99}, 300.0),
        decode=CAGSStageOutput("decode", {"decode_ms": 4.0}, 400.0),
        pickup=CAGSStageOutput("pickup", {"restore_ms": 5.0}, 500.0),
        render=CAGSStageOutput("render", {"render_ms": 6.0, "visible_quality": 0.77}, 600.0),
        config=CAGSAdapterConfig(candidate_kind="reference"),
    )

    assert record.candidate_kind == "reference_action"
    assert record.size_bytes == 99
    assert record.visible_quality == 0.77
    assert record.generation_ms == 3.0


def test_cags_candidate_kind_validation_and_optional_import_error() -> None:
    assert normalize_cags_candidate_kind("enhancement") == "gaussian_enhancement"
    assert normalize_cags_candidate_kind("viewport-tile") == "tile"

    with pytest.raises(CAGSAdapterError, match="candidate_kind"):
        CAGSAdapterConfig(candidate_kind="not-real")

    with pytest.raises(CAGSAdapterError, match="Optional CAGS dependency"):
        resolve_cags_module("definitely_missing_cags_module_for_ref_abr")


def test_cags_adapter_rejects_missing_stage_callable() -> None:
    @dataclass
    class MissingRender:
        def encode(self, **kwargs: object) -> dict[str, object]:
            return {}

        def decode(self, **kwargs: object) -> dict[str, object]:
            return {}

        def pickup(self, **kwargs: object) -> dict[str, object]:
            return {}

    with pytest.raises(CAGSAdapterError, match="cags.render"):
        CAGSBackendAdapter().run_measurement(cags_module=MissingRender())
