"""Optional PyAV-backed decode profiling provider."""

from __future__ import annotations

import importlib
import time
from collections.abc import Iterable, Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping
from urllib.parse import urlparse

from ref_abr.config import stable_config_id
from ref_abr.domain import ExternalMeasurementRecord
from ref_abr.providers.base import ExternalSubstrateProviderConfig, ExternalTraceSubstrateProvider
from ref_abr.substrate import SubstrateError


class PyAVDecodeProfilerError(SubstrateError):
    """Raised when the optional PyAV decode profiler cannot emit records."""


@dataclass(frozen=True)
class PyAVDecodeProfilerConfig:
    """Configuration for one PyAV decode profiling run."""

    source_uri: str
    stream_index: int = 0
    max_frames: int | None = None
    object_id: str | None = None
    frame_id: str | None = None
    candidate_kind: str | None = "reference_action"
    query: Mapping[str, Any] | None = None
    visible_quality: float = 1.0
    deadline_hit: bool = True
    dropped_frame: bool = False
    open_args: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_uri", _non_empty_string(self.source_uri, "source_uri"))
        object.__setattr__(self, "stream_index", _non_negative_int(self.stream_index, "stream_index"))
        if self.max_frames is not None:
            object.__setattr__(self, "max_frames", _positive_int(self.max_frames, "max_frames"))
        for field_name in ("object_id", "frame_id", "candidate_kind"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _non_empty_string(value, field_name))
        object.__setattr__(self, "visible_quality", _unit_interval(self.visible_quality, "visible_quality"))
        if not isinstance(self.deadline_hit, bool):
            raise PyAVDecodeProfilerError("deadline_hit must be a boolean.")
        if not isinstance(self.dropped_frame, bool):
            raise PyAVDecodeProfilerError("dropped_frame must be a boolean.")
        object.__setattr__(self, "open_args", _plain_mapping(self.open_args, "open_args"))
        object.__setattr__(self, "metadata", _plain_mapping(self.metadata, "metadata"))
        if self.query is not None:
            object.__setattr__(self, "query", _plain_mapping(self.query, "query"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "source_uri": self.source_uri,
            "stream_index": self.stream_index,
            "max_frames": self.max_frames,
            "object_id": self.object_id,
            "frame_id": self.frame_id,
            "candidate_kind": self.candidate_kind,
            "query": _to_payload(self.query),
            "visible_quality": self.visible_quality,
            "deadline_hit": self.deadline_hit,
            "dropped_frame": self.dropped_frame,
            "open_args": _to_payload(self.open_args),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class PyAVDecodeProfilerProvider:
    """Decode profiler that delegates all codec work to optional PyAV."""

    config: PyAVDecodeProfilerConfig
    module_name: str = "av"
    provider_id: str = "pyav-decode-profiler"

    def __post_init__(self) -> None:
        if not isinstance(self.config, PyAVDecodeProfilerConfig):
            raise PyAVDecodeProfilerError("config must be a PyAVDecodeProfilerConfig record.")
        object.__setattr__(self, "module_name", _non_empty_string(self.module_name, "module_name"))
        object.__setattr__(self, "provider_id", _non_empty_string(self.provider_id, "provider_id"))

    def profile_records(self, av_module: Any | None = None) -> tuple[ExternalMeasurementRecord, ...]:
        """Decode frames through PyAV and emit one aggregate external-measurement record."""

        module = av_module if av_module is not None else resolve_pyav_module(self.module_name)
        container = _open_container(module, self.config.source_uri, self.config.open_args)
        try:
            stream = _select_video_stream(container, self.config.stream_index)
            start = time.perf_counter()
            frames = tuple(_iter_decoded_frames(container, stream, self.config.max_frames))
            decode_ms = (time.perf_counter() - start) * 1000.0
            return (
                pyav_decode_profile_to_record(
                    config=self.config,
                    provider_id=self.provider_id,
                    stream=stream,
                    frames=frames,
                    decode_ms=decode_ms,
                    size_bytes=_source_size_bytes(self.config.source_uri),
                    container_metadata=_container_metadata(container),
                ),
            )
        finally:
            close = getattr(container, "close", None)
            if callable(close):
                close()

    def run_provider(self, av_module: Any | None = None) -> ExternalTraceSubstrateProvider:
        """Profile decode and wrap the resulting record as an external substrate provider."""

        records = self.profile_records(av_module=av_module)
        return ExternalTraceSubstrateProvider(
            records=records,
            config=ExternalSubstrateProviderConfig(
                provider_id=self.provider_id,
                match_policy="query" if self.config.query else "first",
                metadata={"source": "pyav", "profiler": self.as_payload()},
            ),
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "module_name": self.module_name,
            "config": self.config.as_payload(),
        }


def resolve_pyav_module(module_name: str = "av") -> ModuleType:
    """Import PyAV lazily so the default ref_abr install has no PyAV dependency."""

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise PyAVDecodeProfilerError(
            f"Optional PyAV dependency {module_name!r} is not installed; install the video extra or provide av_module."
        ) from exc


def pyav_decode_profile_to_record(
    *,
    config: PyAVDecodeProfilerConfig,
    provider_id: str,
    stream: Any,
    frames: tuple[Any, ...],
    decode_ms: float,
    size_bytes: int,
    container_metadata: Mapping[str, Any],
) -> ExternalMeasurementRecord:
    """Convert PyAV frame/container observations into an external-measurement record."""

    decode_ms = _non_negative_float(decode_ms, "decode_ms")
    size_bytes = _non_negative_int(size_bytes, "size_bytes")
    frame_metadata = [_frame_metadata(frame, index) for index, frame in enumerate(frames)]
    stream_metadata = _stream_metadata(stream)
    metadata: dict[str, Any] = {
        "pyav": {
            "container": _to_payload(container_metadata),
            "stream": _to_payload(stream_metadata),
            "frames_decoded": len(frames),
            "frames": frame_metadata,
        },
        "profiler": _to_payload(config.metadata),
    }
    if config.query is not None:
        metadata["query"] = _to_payload(config.query)
    return ExternalMeasurementRecord(
        record_id=f"pyav-{stable_config_id({'config': config.as_payload(), 'decode_ms': decode_ms, 'frames': frame_metadata})}",
        backend_id=provider_id,
        object_id=config.object_id,
        frame_id=config.frame_id,
        candidate_kind=config.candidate_kind,
        artifact_uri=config.source_uri,
        generation_ms=0.0,
        transfer_ms=0.0,
        decode_ms=decode_ms,
        restore_ms=0.0,
        render_ms=0.0,
        size_bytes=size_bytes,
        visible_quality=config.visible_quality,
        dropped_frame=config.dropped_frame or len(frames) == 0,
        deadline_hit=config.deadline_hit,
        provenance={
            "backend": "pyav",
            "provider_id": provider_id,
            "source_uri": config.source_uri,
            "stream_index": config.stream_index,
            "max_frames": config.max_frames,
        },
        metadata=metadata,
    )


def _open_container(module: Any, source_uri: str, open_args: Mapping[str, Any]) -> Any:
    open_fn = getattr(module, "open", None)
    if not callable(open_fn):
        raise PyAVDecodeProfilerError("PyAV module does not expose callable av.open.")
    try:
        return open_fn(source_uri, **dict(open_args))
    except Exception as exc:
        raise PyAVDecodeProfilerError(f"av.open failed for {source_uri!r}: {exc}") from exc


def _select_video_stream(container: Any, stream_index: int) -> Any:
    streams = getattr(container, "streams", None)
    video_streams = getattr(streams, "video", None)
    if video_streams is None:
        video_streams = [stream for stream in streams if getattr(stream, "type", None) == "video"] if streams is not None else []
    try:
        stream = tuple(video_streams)[stream_index]
    except IndexError as exc:
        raise PyAVDecodeProfilerError(f"Video stream index {stream_index} is not available.") from exc
    return stream


def _iter_decoded_frames(container: Any, stream: Any, max_frames: int | None) -> Iterable[Any]:
    decode = getattr(container, "decode", None)
    if not callable(decode):
        raise PyAVDecodeProfilerError("PyAV container does not expose callable decode.")
    try:
        iterator = decode(stream)
        for index, frame in enumerate(iterator):
            if max_frames is not None and index >= max_frames:
                break
            yield frame
    except Exception as exc:
        raise PyAVDecodeProfilerError(f"PyAV decode failed: {exc}") from exc


def _source_size_bytes(source_uri: str) -> int:
    parsed = urlparse(source_uri)
    path: Path | None
    if parsed.scheme == "file":
        path = Path(parsed.path)
    elif parsed.scheme:
        path = None
    else:
        path = Path(source_uri)
    if path is not None and path.exists() and path.is_file():
        return path.stat().st_size
    return 0


def _container_metadata(container: Any) -> dict[str, Any]:
    return {
        "format": getattr(getattr(container, "format", None), "name", None),
        "duration": getattr(container, "duration", None),
        "bit_rate": getattr(container, "bit_rate", None),
        "metadata": _safe_mapping(getattr(container, "metadata", {})),
    }


def _stream_metadata(stream: Any) -> dict[str, Any]:
    codec_context = getattr(stream, "codec_context", None)
    return {
        "index": getattr(stream, "index", None),
        "type": getattr(stream, "type", None),
        "frames": getattr(stream, "frames", None),
        "duration": getattr(stream, "duration", None),
        "time_base": str(getattr(stream, "time_base", "")) or None,
        "average_rate": str(getattr(stream, "average_rate", "")) or None,
        "width": getattr(codec_context, "width", getattr(stream, "width", None)),
        "height": getattr(codec_context, "height", getattr(stream, "height", None)),
        "codec": getattr(codec_context, "name", None),
    }


def _frame_metadata(frame: Any, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "pts": getattr(frame, "pts", None),
        "dts": getattr(frame, "dts", None),
        "time": getattr(frame, "time", None),
        "width": getattr(frame, "width", None),
        "height": getattr(frame, "height", None),
        "format": str(getattr(frame, "format", "")) or None,
        "key_frame": getattr(frame, "key_frame", None),
    }


def _safe_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, MappingABC):
        return {str(key): _plain_json_value(nested, f"metadata.{key}") for key, nested in value.items()}
    return {}


def _plain_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, MappingABC):
        raise PyAVDecodeProfilerError(f"{field_name} must be a mapping.")
    return {str(key): _plain_json_value(nested, f"{field_name}.{key}") for key, nested in value.items()}


def _plain_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _plain_json_value(nested, f"{field_name}.{key}") for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item, field_name) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not _is_finite(value):
            raise PyAVDecodeProfilerError(f"{field_name} must be finite.")
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _to_payload(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {key: _to_payload(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_to_payload(item) for item in value]
    if isinstance(value, list):
        return [_to_payload(item) for item in value]
    return value


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PyAVDecodeProfilerError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _non_negative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise PyAVDecodeProfilerError(f"{field_name} must be numeric.")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise PyAVDecodeProfilerError(f"{field_name} must be numeric.") from exc
    if not _is_finite(parsed):
        raise PyAVDecodeProfilerError(f"{field_name} must be finite.")
    if parsed < 0:
        raise PyAVDecodeProfilerError(f"{field_name} must be non-negative.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise PyAVDecodeProfilerError(f"{field_name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PyAVDecodeProfilerError(f"{field_name} must be an integer.") from exc
    if parsed < 0:
        raise PyAVDecodeProfilerError(f"{field_name} must be non-negative.")
    return parsed


def _positive_int(value: Any, field_name: str) -> int:
    parsed = _non_negative_int(value, field_name)
    if parsed <= 0:
        raise PyAVDecodeProfilerError(f"{field_name} must be positive.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name)
    if parsed > 1.0:
        raise PyAVDecodeProfilerError(f"{field_name} must be between 0 and 1.")
    return parsed


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


__all__ = [
    "PyAVDecodeProfilerConfig",
    "PyAVDecodeProfilerError",
    "PyAVDecodeProfilerProvider",
    "pyav_decode_profile_to_record",
    "resolve_pyav_module",
]
