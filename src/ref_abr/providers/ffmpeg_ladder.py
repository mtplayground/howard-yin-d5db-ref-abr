"""Optional ffmpeg-python bitrate-ladder trace provider."""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from ref_abr.config import stable_config_id
from ref_abr.domain import ExternalMeasurementRecord
from ref_abr.providers.base import ExternalSubstrateProviderConfig, ExternalTraceSubstrateProvider
from ref_abr.substrate import SubstrateError


class FFMpegLadderProviderError(SubstrateError):
    """Raised when ffmpeg-python cannot produce ladder trace records."""


@dataclass(frozen=True)
class FFMpegLadderRung:
    """One scale/bitrate output in an offline transcode ladder."""

    rung_id: str
    width_px: int | None = None
    height_px: int | None = None
    bitrate: str | None = None
    output_path: str | None = None
    candidate_kind: str | None = "reference_action"
    object_id: str | None = None
    frame_id: str | None = None
    query: Mapping[str, Any] | None = None
    output_args: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rung_id", _safe_name(_non_empty_string(self.rung_id, "rung_id")))
        if self.width_px is not None:
            object.__setattr__(self, "width_px", _positive_int(self.width_px, "width_px"))
        if self.height_px is not None:
            object.__setattr__(self, "height_px", _positive_int(self.height_px, "height_px"))
        for field_name in ("bitrate", "output_path", "candidate_kind", "object_id", "frame_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _non_empty_string(value, field_name))
        object.__setattr__(self, "output_args", _plain_mapping(self.output_args, "output_args"))
        object.__setattr__(self, "metadata", _plain_mapping(self.metadata, "metadata"))
        if self.query is not None:
            object.__setattr__(self, "query", _plain_mapping(self.query, "query"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "rung_id": self.rung_id,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "bitrate": self.bitrate,
            "output_path": self.output_path,
            "candidate_kind": self.candidate_kind,
            "object_id": self.object_id,
            "frame_id": self.frame_id,
            "query": _to_payload(self.query),
            "output_args": _to_payload(self.output_args),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class FFMpegLadderConfig:
    """Configuration for scripting ffmpeg-python over a bitrate ladder."""

    source_uri: str
    rungs: Sequence[FFMpegLadderRung | Mapping[str, Any]]
    output_dir: str | None = None
    input_args: Mapping[str, Any] = field(default_factory=dict)
    global_output_args: Mapping[str, Any] = field(default_factory=dict)
    overwrite: bool = True
    visible_quality: float = 1.0
    deadline_hit: bool = True
    dropped_frame: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_uri", _non_empty_string(self.source_uri, "source_uri"))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", _non_empty_string(self.output_dir, "output_dir"))
        rungs = tuple(_coerce_rung(rung, index) for index, rung in enumerate(self.rungs))
        if not rungs:
            raise FFMpegLadderProviderError("rungs must contain at least one ladder rung.")
        if self.output_dir is None and any(rung.output_path is None for rung in rungs):
            raise FFMpegLadderProviderError("output_dir is required when any rung omits output_path.")
        object.__setattr__(self, "rungs", rungs)
        object.__setattr__(self, "input_args", _plain_mapping(self.input_args, "input_args"))
        object.__setattr__(self, "global_output_args", _plain_mapping(self.global_output_args, "global_output_args"))
        object.__setattr__(self, "visible_quality", _unit_interval(self.visible_quality, "visible_quality"))
        if not isinstance(self.overwrite, bool):
            raise FFMpegLadderProviderError("overwrite must be a boolean.")
        if not isinstance(self.deadline_hit, bool):
            raise FFMpegLadderProviderError("deadline_hit must be a boolean.")
        if not isinstance(self.dropped_frame, bool):
            raise FFMpegLadderProviderError("dropped_frame must be a boolean.")
        object.__setattr__(self, "metadata", _plain_mapping(self.metadata, "metadata"))

    def output_path_for(self, rung: FFMpegLadderRung) -> Path:
        if rung.output_path is not None:
            return Path(rung.output_path)
        if self.output_dir is None:
            raise FFMpegLadderProviderError("output_dir is required when rung.output_path is omitted.")
        return Path(self.output_dir) / f"{Path(self.source_uri).stem}_{rung.rung_id}.mp4"

    def as_payload(self) -> dict[str, Any]:
        return {
            "source_uri": self.source_uri,
            "output_dir": self.output_dir,
            "rungs": [rung.as_payload() for rung in self.rungs],
            "input_args": _to_payload(self.input_args),
            "global_output_args": _to_payload(self.global_output_args),
            "overwrite": self.overwrite,
            "visible_quality": self.visible_quality,
            "deadline_hit": self.deadline_hit,
            "dropped_frame": self.dropped_frame,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class FFMpegLadderTraceProvider:
    """Run ffmpeg-python transcodes and emit external-measurement records."""

    config: FFMpegLadderConfig
    module_name: str = "ffmpeg"
    provider_id: str = "ffmpeg-ladder-trace"

    def __post_init__(self) -> None:
        if not isinstance(self.config, FFMpegLadderConfig):
            raise FFMpegLadderProviderError("config must be an FFMpegLadderConfig record.")
        object.__setattr__(self, "module_name", _non_empty_string(self.module_name, "module_name"))
        object.__setattr__(self, "provider_id", _non_empty_string(self.provider_id, "provider_id"))

    def profile_records(self, ffmpeg_module: Any | None = None) -> tuple[ExternalMeasurementRecord, ...]:
        """Run all configured ladder rungs through ffmpeg-python."""

        module = ffmpeg_module if ffmpeg_module is not None else resolve_ffmpeg_module(self.module_name)
        return tuple(self._run_rung(module, rung) for rung in self.config.rungs)

    def run_provider(self, ffmpeg_module: Any | None = None) -> ExternalTraceSubstrateProvider:
        """Run ffmpeg and wrap emitted measurements as an external substrate provider."""

        records = self.profile_records(ffmpeg_module=ffmpeg_module)
        return ExternalTraceSubstrateProvider(
            records=records,
            config=ExternalSubstrateProviderConfig(
                provider_id=self.provider_id,
                match_policy="query" if any(rung.query for rung in self.config.rungs) else "first",
                metadata={"source": "ffmpeg-python", "ladder": self.as_payload()},
            ),
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "module_name": self.module_name,
            "config": self.config.as_payload(),
        }

    def _run_rung(self, module: Any, rung: FFMpegLadderRung) -> ExternalMeasurementRecord:
        output_path = self.config.output_path_for(rung)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stream = _ffmpeg_input(module, self.config.source_uri, self.config.input_args)
        if rung.width_px is not None or rung.height_px is not None:
            width = rung.width_px if rung.width_px is not None else -2
            height = rung.height_px if rung.height_px is not None else -2
            stream = _ffmpeg_filter(stream, "scale", width, height)
        output_kwargs = dict(self.config.global_output_args)
        output_kwargs.update(rung.output_args)
        if rung.bitrate is not None and not any(key in output_kwargs for key in ("video_bitrate", "b:v", "b")):
            output_kwargs["video_bitrate"] = rung.bitrate
        command = _ffmpeg_output(stream, str(output_path), output_kwargs)
        if self.config.overwrite:
            overwrite = getattr(command, "overwrite_output", None)
            if callable(overwrite):
                command = overwrite()
        start = time.perf_counter()
        try:
            run = getattr(command, "run", None)
            if not callable(run):
                raise FFMpegLadderProviderError("ffmpeg output command does not expose callable run.")
            result = run(capture_stdout=True, capture_stderr=True)
        except FFMpegLadderProviderError:
            raise
        except Exception as exc:
            raise FFMpegLadderProviderError(f"ffmpeg-python transcode failed for rung {rung.rung_id!r}: {exc}") from exc
        generation_ms = (time.perf_counter() - start) * 1000.0
        return ffmpeg_ladder_result_to_record(
            config=self.config,
            rung=rung,
            output_path=output_path,
            provider_id=self.provider_id,
            generation_ms=generation_ms,
            run_result=result,
        )


def resolve_ffmpeg_module(module_name: str = "ffmpeg") -> ModuleType:
    """Import ffmpeg-python lazily so the default install has no video dependency."""

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise FFMpegLadderProviderError(
            f"Optional ffmpeg-python dependency {module_name!r} is not installed; install the video extra or provide ffmpeg_module."
        ) from exc


def ffmpeg_ladder_result_to_record(
    *,
    config: FFMpegLadderConfig,
    rung: FFMpegLadderRung,
    output_path: Path,
    provider_id: str,
    generation_ms: float,
    run_result: Any,
) -> ExternalMeasurementRecord:
    """Convert one ffmpeg-python rung run into an external-measurement record."""

    generation_ms = _non_negative_float(generation_ms, "generation_ms")
    size_bytes = output_path.stat().st_size if output_path.exists() and output_path.is_file() else 0
    metadata: dict[str, Any] = {
        "ffmpeg": {
            "rung": rung.as_payload(),
            "output_path": str(output_path),
            "run_result": _run_result_payload(run_result),
        },
        "provider": _to_payload(config.metadata),
    }
    if rung.query is not None:
        metadata["query"] = _to_payload(rung.query)
    return ExternalMeasurementRecord(
        record_id=f"ffmpeg-{stable_config_id({'config': config.as_payload(), 'rung': rung.as_payload(), 'output_path': str(output_path), 'generation_ms': generation_ms})}",
        backend_id=provider_id,
        object_id=rung.object_id,
        frame_id=rung.frame_id,
        candidate_kind=rung.candidate_kind,
        artifact_uri=str(output_path),
        generation_ms=generation_ms,
        transfer_ms=0.0,
        decode_ms=0.0,
        restore_ms=0.0,
        render_ms=0.0,
        size_bytes=size_bytes,
        visible_quality=config.visible_quality,
        dropped_frame=config.dropped_frame,
        deadline_hit=config.deadline_hit,
        provenance={
            "backend": "ffmpeg-python",
            "provider_id": provider_id,
            "source_uri": config.source_uri,
            "rung_id": rung.rung_id,
            "bitrate": rung.bitrate,
            "width_px": rung.width_px,
            "height_px": rung.height_px,
        },
        metadata=metadata,
    )


def _ffmpeg_input(module: Any, source_uri: str, input_args: Mapping[str, Any]) -> Any:
    input_fn = getattr(module, "input", None)
    if not callable(input_fn):
        raise FFMpegLadderProviderError("ffmpeg module does not expose callable ffmpeg.input.")
    try:
        return input_fn(source_uri, **dict(input_args))
    except Exception as exc:
        raise FFMpegLadderProviderError(f"ffmpeg.input failed for {source_uri!r}: {exc}") from exc


def _ffmpeg_filter(stream: Any, name: str, *args: Any) -> Any:
    filter_fn = getattr(stream, "filter", None)
    if not callable(filter_fn):
        raise FFMpegLadderProviderError("ffmpeg stream does not expose callable filter.")
    try:
        return filter_fn(name, *args)
    except Exception as exc:
        raise FFMpegLadderProviderError(f"ffmpeg filter {name!r} failed: {exc}") from exc


def _ffmpeg_output(stream: Any, output_path: str, output_kwargs: Mapping[str, Any]) -> Any:
    output_fn = getattr(stream, "output", None)
    if not callable(output_fn):
        raise FFMpegLadderProviderError("ffmpeg stream does not expose callable output.")
    try:
        return output_fn(output_path, **dict(output_kwargs))
    except Exception as exc:
        raise FFMpegLadderProviderError(f"ffmpeg output setup failed for {output_path!r}: {exc}") from exc


def _run_result_payload(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, tuple):
        return [_bytes_to_text(item) for item in result]
    if isinstance(result, MappingABC):
        return {str(key): _plain_json_value(value, f"run_result.{key}") for key, value in result.items()}
    return _bytes_to_text(result)


def _bytes_to_text(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return _plain_json_value(value, "run_result")


def _coerce_rung(rung: FFMpegLadderRung | Mapping[str, Any], index: int) -> FFMpegLadderRung:
    if isinstance(rung, FFMpegLadderRung):
        return rung
    if isinstance(rung, MappingABC):
        try:
            return FFMpegLadderRung(
                rung_id=rung.get("rung_id") or rung.get("id") or rung.get("name") or f"rung-{index}",
                width_px=rung.get("width_px") or rung.get("width"),
                height_px=rung.get("height_px") or rung.get("height"),
                bitrate=rung.get("bitrate") or rung.get("video_bitrate"),
                output_path=rung.get("output_path") or rung.get("path"),
                candidate_kind=rung.get("candidate_kind", "reference_action"),
                object_id=rung.get("object_id"),
                frame_id=rung.get("frame_id"),
                query=rung.get("query"),
                output_args=rung.get("output_args", {}),
                metadata=rung.get("metadata", {}),
            )
        except TypeError as exc:
            raise FFMpegLadderProviderError(f"rungs[{index}] has malformed fields: {exc}") from exc
    raise FFMpegLadderProviderError(f"rungs[{index}] must be a mapping or FFMpegLadderRung.")


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())


def _plain_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, MappingABC):
        raise FFMpegLadderProviderError(f"{field_name} must be a mapping.")
    return {str(key): _plain_json_value(nested, f"{field_name}.{key}") for key, nested in value.items()}


def _plain_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _plain_json_value(nested, f"{field_name}.{key}") for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item, field_name) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not _is_finite(value):
            raise FFMpegLadderProviderError(f"{field_name} must be finite.")
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
        raise FFMpegLadderProviderError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _non_negative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise FFMpegLadderProviderError(f"{field_name} must be numeric.")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise FFMpegLadderProviderError(f"{field_name} must be numeric.") from exc
    if not _is_finite(parsed):
        raise FFMpegLadderProviderError(f"{field_name} must be finite.")
    if parsed < 0:
        raise FFMpegLadderProviderError(f"{field_name} must be non-negative.")
    return parsed


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise FFMpegLadderProviderError(f"{field_name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise FFMpegLadderProviderError(f"{field_name} must be an integer.") from exc
    if parsed <= 0:
        raise FFMpegLadderProviderError(f"{field_name} must be positive.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name)
    if parsed > 1.0:
        raise FFMpegLadderProviderError(f"{field_name} must be between 0 and 1.")
    return parsed


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


__all__ = [
    "FFMpegLadderConfig",
    "FFMpegLadderProviderError",
    "FFMpegLadderRung",
    "FFMpegLadderTraceProvider",
    "ffmpeg_ladder_result_to_record",
    "resolve_ffmpeg_module",
]
