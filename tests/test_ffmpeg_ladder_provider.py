from __future__ import annotations

from pathlib import Path

import pytest

from ref_abr.domain import ExternalMeasurementRecord
from ref_abr.providers import (
    FFMpegLadderConfig,
    FFMpegLadderProviderError,
    FFMpegLadderRung,
    FFMpegLadderTraceProvider,
    resolve_ffmpeg_module,
)
from ref_abr.providers.ffmpeg_ladder import ffmpeg_ladder_result_to_record


class FakeFFmpeg:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def input(self, source_uri: str, **kwargs: object) -> "FakeStream":
        self.calls.append(("input", (source_uri,), dict(kwargs)))
        return FakeStream(self)


class FakeStream:
    def __init__(self, module: FakeFFmpeg) -> None:
        self.module = module

    def filter(self, name: str, *args: object) -> "FakeStream":
        self.module.calls.append(("filter", (name, *args), {}))
        return self

    def output(self, output_path: str, **kwargs: object) -> "FakeCommand":
        self.module.calls.append(("output", (output_path,), dict(kwargs)))
        return FakeCommand(self.module, output_path)


class FakeCommand:
    def __init__(self, module: FakeFFmpeg, output_path: str) -> None:
        self.module = module
        self.output_path = output_path

    def overwrite_output(self) -> "FakeCommand":
        self.module.calls.append(("overwrite_output", (), {}))
        return self

    def run(self, **kwargs: object) -> tuple[bytes, bytes]:
        self.module.calls.append(("run", (), dict(kwargs)))
        Path(self.output_path).write_bytes(f"encoded:{Path(self.output_path).name}".encode("utf-8"))
        return (b"stdout", b"stderr")


def _query() -> dict[str, object]:
    return {
        "layer": 0,
        "ref_resolution": "720p",
        "fov_deg": 90,
        "view_mismatch_deg": 0,
        "freshness_ms": 0,
    }


def test_ffmpeg_ladder_provider_runs_scaled_bitrate_rungs(tmp_path) -> None:
    fake = FakeFFmpeg()
    config = FFMpegLadderConfig(
        source_uri="input.mp4",
        output_dir=str(tmp_path),
        input_args={"ss": "00:00:01"},
        global_output_args={"vcodec": "libx264"},
        rungs=[
            {
                "rung_id": "720p",
                "width_px": 1280,
                "height_px": 720,
                "bitrate": "2500k",
                "query": _query(),
                "object_id": "video-720",
                "metadata": {"tier": "mid"},
            },
            FFMpegLadderRung(rung_id="480p", width_px=854, height_px=480, bitrate="1200k"),
        ],
        visible_quality=0.88,
        metadata={"run": "smoke"},
    )
    provider = FFMpegLadderTraceProvider(config=config, provider_id="ffmpeg-fixture")

    records = provider.profile_records(ffmpeg_module=fake)

    assert len(records) == 2
    first = records[0]
    assert isinstance(first, ExternalMeasurementRecord)
    assert first.backend_id == "ffmpeg-fixture"
    assert first.object_id == "video-720"
    assert first.candidate_kind == "reference_action"
    assert first.artifact_uri.endswith("input_720p.mp4")
    assert first.generation_ms >= 0
    assert first.decode_ms == 0
    assert first.size_bytes == len(b"encoded:input_720p.mp4")
    assert first.visible_quality == 0.88
    assert first.provenance["bitrate"] == "2500k"
    assert first.metadata["query"] == _query()
    assert first.metadata["ffmpeg"]["run_result"] == ("stdout", "stderr")
    assert first.metadata["provider"]["run"] == "smoke"
    assert ("input", ("input.mp4",), {"ss": "00:00:01"}) in fake.calls
    assert ("filter", ("scale", 1280, 720), {}) in fake.calls
    assert any(call[0] == "output" and call[2]["video_bitrate"] == "2500k" and call[2]["vcodec"] == "libx264" for call in fake.calls)
    assert any(call[0] == "run" and call[2] == {"capture_stdout": True, "capture_stderr": True} for call in fake.calls)


def test_ffmpeg_ladder_provider_wraps_as_external_substrate_provider(tmp_path) -> None:
    config = FFMpegLadderConfig(
        source_uri="input.mp4",
        output_dir=str(tmp_path),
        rungs=[{"rung_id": "q", "width": 640, "height": 360, "query": _query()}],
        visible_quality=0.5,
    )
    provider = FFMpegLadderTraceProvider(config=config, provider_id="ffmpeg-provider")

    substrate_provider = provider.run_provider(ffmpeg_module=FakeFFmpeg())
    value = substrate_provider.evaluate(_query())

    assert substrate_provider.provider_id == "ffmpeg-provider"
    assert value.visible_quality == 0.5
    assert value.component_timing.generation_ms == value.metadata["model"]["decode_ms"] + value.component_timing.generation_ms
    assert value.metadata["model"]["candidate_kind"] == "reference_action"


def test_ffmpeg_ladder_result_to_record_uses_explicit_output_path(tmp_path) -> None:
    output_path = tmp_path / "custom.mp4"
    output_path.write_bytes(b"1234")
    config = FFMpegLadderConfig(source_uri="input.mp4", rungs=[FFMpegLadderRung("custom", output_path=str(output_path))])
    rung = config.rungs[0]

    record = ffmpeg_ladder_result_to_record(
        config=config,
        rung=rung,
        output_path=output_path,
        provider_id="manual",
        generation_ms=12.0,
        run_result={"ok": True},
    )

    assert record.artifact_uri == str(output_path)
    assert record.size_bytes == 4
    assert record.generation_ms == 12.0
    assert record.metadata["ffmpeg"]["run_result"] == {"ok": True}


def test_ffmpeg_ladder_provider_validation_and_optional_import_error(tmp_path) -> None:
    with pytest.raises(FFMpegLadderProviderError, match="rungs"):
        FFMpegLadderConfig(source_uri="input.mp4", rungs=[])

    with pytest.raises(FFMpegLadderProviderError, match="output_dir"):
        FFMpegLadderConfig(source_uri="input.mp4", rungs=[{"rung_id": "x"}])

    with pytest.raises(FFMpegLadderProviderError, match="width_px must be positive"):
        FFMpegLadderRung("bad", width_px=0)

    with pytest.raises(FFMpegLadderProviderError, match="Optional ffmpeg-python dependency"):
        resolve_ffmpeg_module("definitely_missing_ffmpeg_python_for_ref_abr")


def test_ffmpeg_ladder_provider_rejects_bad_module(tmp_path) -> None:
    config = FFMpegLadderConfig(
        source_uri="input.mp4",
        output_dir=str(tmp_path),
        rungs=[{"rung_id": "x", "width": 320, "height": 180}],
    )

    with pytest.raises(FFMpegLadderProviderError, match="ffmpeg.input"):
        FFMpegLadderTraceProvider(config=config).profile_records(ffmpeg_module=object())
