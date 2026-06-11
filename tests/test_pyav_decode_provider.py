from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ref_abr.domain import ExternalMeasurementRecord
from ref_abr.providers import PyAVDecodeProfilerConfig, PyAVDecodeProfilerError, PyAVDecodeProfilerProvider
from ref_abr.providers.pyav_decode import pyav_decode_profile_to_record, resolve_pyav_module


@dataclass
class FakeCodecContext:
    name: str = "h264"
    width: int = 1280
    height: int = 720


@dataclass
class FakeStream:
    index: int = 0
    type: str = "video"
    frames: int = 3
    duration: int = 300
    time_base: str = "1/30"
    average_rate: str = "30"
    codec_context: FakeCodecContext = field(default_factory=FakeCodecContext)


@dataclass
class FakeFrame:
    pts: int
    width: int = 1280
    height: int = 720
    time: float = 0.0
    dts: int | None = None
    format: str = "yuv420p"
    key_frame: bool = False


class FakeStreams:
    def __init__(self, video: list[FakeStream]) -> None:
        self.video = video


class FakeContainer:
    def __init__(self, frames: list[FakeFrame]) -> None:
        self.streams = FakeStreams([FakeStream()])
        self.format = type("FakeFormat", (), {"name": "mp4"})()
        self.duration = 3000
        self.bit_rate = 1000
        self.metadata = {"encoder": "fixture"}
        self.frames = frames
        self.closed = False

    def decode(self, stream: FakeStream):
        assert stream.index == 0
        yield from self.frames

    def close(self) -> None:
        self.closed = True


class FakeAV:
    def __init__(self, frames: list[FakeFrame]) -> None:
        self.container = FakeContainer(frames)
        self.open_calls: list[tuple[str, dict[str, object]]] = []

    def open(self, source_uri: str, **kwargs: object) -> FakeContainer:
        self.open_calls.append((source_uri, dict(kwargs)))
        return self.container


def _query() -> dict[str, object]:
    return {
        "layer": 0,
        "ref_resolution": "720p",
        "fov_deg": 90,
        "view_mismatch_deg": 0,
        "freshness_ms": 0,
    }


def test_pyav_decode_profiler_emits_external_measurement_record(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake-video")
    fake_av = FakeAV([FakeFrame(pts=0, time=0.0, key_frame=True), FakeFrame(pts=1, time=1 / 30)])
    config = PyAVDecodeProfilerConfig(
        source_uri=str(video_path),
        max_frames=1,
        object_id="video-segment-a",
        frame_id="frame-a",
        query=_query(),
        open_args={"metadata_errors": "ignore"},
        metadata={"run": "smoke"},
    )
    provider = PyAVDecodeProfilerProvider(config=config, provider_id="pyav-fixture")

    records = provider.profile_records(av_module=fake_av)

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, ExternalMeasurementRecord)
    assert record.backend_id == "pyav-fixture"
    assert record.object_id == "video-segment-a"
    assert record.frame_id == "frame-a"
    assert record.candidate_kind == "reference_action"
    assert record.size_bytes == len(b"fake-video")
    assert record.decode_ms >= 0
    assert record.generation_ms == 0
    assert record.render_ms == 0
    assert record.visible_quality == 1.0
    assert record.dropped_frame is False
    assert record.metadata["query"] == _query()
    assert record.metadata["profiler"]["run"] == "smoke"
    assert record.metadata["pyav"]["frames_decoded"] == 1
    assert record.metadata["pyav"]["stream"]["codec"] == "h264"
    assert record.metadata["pyav"]["container"]["format"] == "mp4"
    assert fake_av.open_calls == [(str(video_path), {"metadata_errors": "ignore"})]
    assert fake_av.container.closed is True


def test_pyav_decode_profiler_wraps_records_as_external_provider(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"12")
    provider = PyAVDecodeProfilerProvider(
        config=PyAVDecodeProfilerConfig(source_uri=str(video_path), query=_query(), visible_quality=0.5),
        provider_id="pyav-provider",
    )

    substrate_provider = provider.run_provider(av_module=FakeAV([FakeFrame(pts=0)]))
    value = substrate_provider.evaluate(_query())

    assert substrate_provider.provider_id == "pyav-provider"
    assert value.visible_quality == 0.5
    assert value.component_timing.restoration_ms == value.metadata["model"]["decode_ms"]
    assert value.metadata["model"]["size_bytes"] == 2


def test_pyav_decode_empty_decode_marks_dropped_frame(tmp_path) -> None:
    video_path = tmp_path / "empty.mp4"
    video_path.write_bytes(b"")

    record = PyAVDecodeProfilerProvider(
        config=PyAVDecodeProfilerConfig(source_uri=str(video_path), dropped_frame=False)
    ).profile_records(av_module=FakeAV([]))[0]

    assert record.dropped_frame is True
    assert record.metadata["pyav"]["frames_decoded"] == 0


def test_pyav_decode_profile_to_record_accepts_container_metadata() -> None:
    config = PyAVDecodeProfilerConfig(source_uri="s3://bucket/video.mp4", candidate_kind=None)
    record = pyav_decode_profile_to_record(
        config=config,
        provider_id="manual",
        stream=FakeStream(),
        frames=(FakeFrame(pts=1),),
        decode_ms=12.0,
        size_bytes=0,
        container_metadata={"format": "mov"},
    )

    assert record.artifact_uri == "s3://bucket/video.mp4"
    assert record.candidate_kind is None
    assert record.decode_ms == 12.0
    assert record.metadata["pyav"]["container"]["format"] == "mov"


def test_pyav_decode_profiler_rejects_missing_optional_dependency_and_bad_stream() -> None:
    with pytest.raises(PyAVDecodeProfilerError, match="Optional PyAV dependency"):
        resolve_pyav_module("definitely_missing_pyav_for_ref_abr")

    fake_av = FakeAV([FakeFrame(pts=0)])
    with pytest.raises(PyAVDecodeProfilerError, match="Video stream index"):
        PyAVDecodeProfilerProvider(
            config=PyAVDecodeProfilerConfig(source_uri="memory.mp4", stream_index=9)
        ).profile_records(av_module=fake_av)

    with pytest.raises(PyAVDecodeProfilerError, match="max_frames must be positive"):
        PyAVDecodeProfilerConfig(source_uri="memory.mp4", max_frames=0)
