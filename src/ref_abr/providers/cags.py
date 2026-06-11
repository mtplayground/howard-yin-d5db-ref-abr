"""Thin optional adapter for CAGS 3DGS codec/render measurements."""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from ref_abr.config import stable_config_id
from ref_abr.domain import ExternalMeasurementRecord
from ref_abr.providers.base import ExternalSubstrateProviderConfig, ExternalTraceSubstrateProvider
from ref_abr.substrate import SubstrateError


CAGS_CANDIDATE_KINDS: tuple[str, ...] = ("gaussian_base", "gaussian_enhancement", "tile", "reference_action")


class CAGSAdapterError(SubstrateError):
    """Raised when the optional CAGS adapter cannot produce a valid measurement."""


@dataclass(frozen=True)
class CAGSAdapterConfig:
    """Configuration for one CAGS encode/decode/pickup/render measurement run."""

    source_uri: str | None = None
    output_dir: str | None = None
    candidate_kind: str = "gaussian_base"
    object_id: str | None = None
    frame_id: str | None = None
    query: Mapping[str, Any] | None = None
    encode_args: Mapping[str, Any] = field(default_factory=dict)
    decode_args: Mapping[str, Any] = field(default_factory=dict)
    pickup_args: Mapping[str, Any] = field(default_factory=dict)
    render_args: Mapping[str, Any] = field(default_factory=dict)
    transfer_ms: float = 0.0
    deadline_hit: bool = True
    dropped_frame: bool = False
    default_visible_quality: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("source_uri", "output_dir", "object_id", "frame_id"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise CAGSAdapterError(f"{field_name} must be a non-empty string when provided.")
        object.__setattr__(self, "candidate_kind", normalize_cags_candidate_kind(self.candidate_kind))
        for field_name in ("encode_args", "decode_args", "pickup_args", "render_args", "metadata"):
            object.__setattr__(self, field_name, _plain_mapping(getattr(self, field_name), field_name))
        if self.query is not None:
            object.__setattr__(self, "query", _plain_mapping(self.query, "query"))
        object.__setattr__(self, "transfer_ms", _non_negative_float(self.transfer_ms, "transfer_ms"))
        object.__setattr__(self, "default_visible_quality", _unit_interval(self.default_visible_quality, "default_visible_quality"))
        if not isinstance(self.deadline_hit, bool):
            raise CAGSAdapterError("deadline_hit must be a boolean.")
        if not isinstance(self.dropped_frame, bool):
            raise CAGSAdapterError("dropped_frame must be a boolean.")

    def as_payload(self) -> dict[str, Any]:
        return {
            "source_uri": self.source_uri,
            "output_dir": self.output_dir,
            "candidate_kind": self.candidate_kind,
            "object_id": self.object_id,
            "frame_id": self.frame_id,
            "query": _to_payload(self.query),
            "encode_args": _to_payload(self.encode_args),
            "decode_args": _to_payload(self.decode_args),
            "pickup_args": _to_payload(self.pickup_args),
            "render_args": _to_payload(self.render_args),
            "transfer_ms": self.transfer_ms,
            "deadline_hit": self.deadline_hit,
            "dropped_frame": self.dropped_frame,
            "default_visible_quality": self.default_visible_quality,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class CAGSBackendAdapter:
    """Invoke CAGS as an optional dependency and emit external-measurement records."""

    config: CAGSAdapterConfig = field(default_factory=CAGSAdapterConfig)
    module_name: str = "cags"
    provider_id: str = "cags-3dgs-backend"

    def __post_init__(self) -> None:
        if not isinstance(self.config, CAGSAdapterConfig):
            raise CAGSAdapterError("config must be a CAGSAdapterConfig record.")
        if not isinstance(self.module_name, str) or not self.module_name.strip():
            raise CAGSAdapterError("module_name must be a non-empty string.")
        if not isinstance(self.provider_id, str) or not self.provider_id.strip():
            raise CAGSAdapterError("provider_id must be a non-empty string.")
        object.__setattr__(self, "module_name", self.module_name.strip())
        object.__setattr__(self, "provider_id", self.provider_id.strip())

    def run_measurement(self, cags_module: Any | None = None) -> ExternalMeasurementRecord:
        """Run encode/decode/pickup/render and return one normalized record."""

        module = cags_module if cags_module is not None else resolve_cags_module(self.module_name)
        encode = _invoke_cags_stage(module, "encode", self._stage_args("encode", self.config.encode_args))
        decode = _invoke_cags_stage(module, "decode", self._stage_args("decode", self.config.decode_args))
        pickup = _invoke_cags_stage(module, "pickup", self._stage_args("pickup", self.config.pickup_args))
        render = _invoke_cags_stage(module, "render", self._stage_args("render", self.config.render_args))
        return cags_stage_outputs_to_record(
            encode=encode,
            decode=decode,
            pickup=pickup,
            render=render,
            config=self.config,
            provider_id=self.provider_id,
        )

    def run_provider(self, cags_module: Any | None = None) -> ExternalTraceSubstrateProvider:
        """Run CAGS and wrap the resulting measurement as an external substrate provider."""

        record = self.run_measurement(cags_module=cags_module)
        return ExternalTraceSubstrateProvider(
            records=(record,),
            config=ExternalSubstrateProviderConfig(
                provider_id=self.provider_id,
                match_policy="query" if self.config.query else "first",
                metadata={"source": "cags", "adapter": self.as_payload()},
            ),
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "module_name": self.module_name,
            "config": self.config.as_payload(),
        }

    def _stage_args(self, stage: str, explicit_args: Mapping[str, Any]) -> dict[str, Any]:
        args = dict(explicit_args)
        if self.config.source_uri and not any(key in args for key in ("source_uri", "source_path", "input_path", "scene_path")):
            args["source_uri"] = self.config.source_uri
        if self.config.output_dir and not any(key in args for key in ("output_dir", "output_path", "work_dir")):
            args["output_dir"] = self.config.output_dir
        args.setdefault("candidate_kind", self.config.candidate_kind)
        args.setdefault("stage", stage)
        return args


@dataclass(frozen=True)
class CAGSStageOutput:
    """Normalized return payload and elapsed wall-clock time for one CAGS stage."""

    stage: str
    payload: Mapping[str, Any]
    elapsed_ms: float

    def __post_init__(self) -> None:
        if self.stage not in {"encode", "decode", "pickup", "render"}:
            raise CAGSAdapterError("stage must be encode, decode, pickup, or render.")
        object.__setattr__(self, "payload", _plain_mapping(self.payload, "payload"))
        object.__setattr__(self, "elapsed_ms", _non_negative_float(self.elapsed_ms, "elapsed_ms"))

    def as_payload(self) -> dict[str, Any]:
        return {"stage": self.stage, "payload": _to_payload(self.payload), "elapsed_ms": self.elapsed_ms}


def resolve_cags_module(module_name: str = "cags") -> ModuleType:
    """Import CAGS lazily so the default ref_abr install has no CAGS dependency."""

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise CAGSAdapterError(
            f"Optional CAGS dependency {module_name!r} is not installed; install the cags extra or provide cags_module."
        ) from exc


def cags_stage_outputs_to_record(
    *,
    encode: CAGSStageOutput,
    decode: CAGSStageOutput,
    pickup: CAGSStageOutput,
    render: CAGSStageOutput,
    config: CAGSAdapterConfig,
    provider_id: str = "cags-3dgs-backend",
) -> ExternalMeasurementRecord:
    """Convert CAGS stage results into one external-measurement record."""

    generation_ms = _stage_timing_ms(encode, ("generation_ms", "encode_ms", "encoding_ms"))
    decode_ms = _stage_timing_ms(decode, ("decode_ms", "decoding_ms"))
    restore_ms = _stage_timing_ms(pickup, ("restore_ms", "pickup_ms", "restoration_ms"))
    render_ms = _stage_timing_ms(render, ("render_ms", "rendering_ms"))
    size_bytes = _artifact_size_bytes(config, encode, decode, pickup, render)
    visible_quality = _visible_quality_from_render(render.payload, config.default_visible_quality)
    dropped_frame = _bool_from_outputs(config.dropped_frame, "dropped_frame", encode, decode, pickup, render)
    deadline_hit = _bool_from_outputs(config.deadline_hit, "deadline_hit", encode, decode, pickup, render)
    record_id = _record_id(config, encode, decode, pickup, render)
    metadata: dict[str, Any] = {
        "cags": {
            "stages": {
                "encode": encode.as_payload(),
                "decode": decode.as_payload(),
                "pickup": pickup.as_payload(),
                "render": render.as_payload(),
            },
            "psnr": _first_present(render.payload, ("psnr", "render_psnr", "psnr_db")),
            "lpips": _first_present(render.payload, ("lpips", "render_lpips")),
        },
        "adapter": _to_payload(config.metadata),
    }
    if config.query is not None:
        metadata["query"] = _to_payload(config.query)

    return ExternalMeasurementRecord(
        record_id=record_id,
        backend_id=provider_id,
        object_id=config.object_id,
        frame_id=config.frame_id,
        candidate_kind=config.candidate_kind,
        artifact_uri=_artifact_uri(config, encode, decode, pickup, render),
        generation_ms=generation_ms,
        transfer_ms=config.transfer_ms,
        decode_ms=decode_ms,
        restore_ms=restore_ms,
        render_ms=render_ms,
        size_bytes=size_bytes,
        visible_quality=visible_quality,
        dropped_frame=dropped_frame,
        deadline_hit=deadline_hit,
        provenance={
            "backend": "cags",
            "provider_id": provider_id,
            "source_uri": config.source_uri,
            "output_dir": config.output_dir,
        },
        metadata=metadata,
    )


def normalize_cags_candidate_kind(value: str) -> str:
    """Map CAGS-facing labels to ref_abr candidate kinds."""

    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "base": "gaussian_base",
        "gaussian": "gaussian_base",
        "gaussian_base": "gaussian_base",
        "enhancement": "gaussian_enhancement",
        "gaussian_enhancement": "gaussian_enhancement",
        "tile": "tile",
        "viewport_tile": "tile",
        "reference": "reference_action",
        "reference_action": "reference_action",
        "ref": "reference_action",
    }
    candidate_kind = aliases.get(normalized, normalized)
    if candidate_kind not in CAGS_CANDIDATE_KINDS:
        valid = ", ".join(CAGS_CANDIDATE_KINDS)
        raise CAGSAdapterError(f"candidate_kind must be one of: {valid}.")
    return candidate_kind


def _invoke_cags_stage(module: Any, stage: str, kwargs: Mapping[str, Any]) -> CAGSStageOutput:
    fn = getattr(module, stage, None)
    if not callable(fn):
        raise CAGSAdapterError(f"CAGS module does not expose callable cags.{stage}.")
    start = time.perf_counter()
    try:
        result = fn(**dict(kwargs))
    except TypeError as exc:
        raise CAGSAdapterError(f"cags.{stage} rejected adapter arguments {sorted(kwargs)}: {exc}") from exc
    except Exception as exc:
        raise CAGSAdapterError(f"cags.{stage} failed: {exc}") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return CAGSStageOutput(stage=stage, payload=_result_mapping(result, stage), elapsed_ms=elapsed_ms)


def _result_mapping(result: Any, stage: str) -> Mapping[str, Any]:
    if result is None:
        return {}
    if isinstance(result, MappingABC):
        return dict(result)
    as_payload = getattr(result, "as_payload", None)
    if callable(as_payload):
        payload = as_payload()
        if isinstance(payload, MappingABC):
            return dict(payload)
    if hasattr(result, "__dict__"):
        return {key: value for key, value in vars(result).items() if not key.startswith("_")}
    raise CAGSAdapterError(f"cags.{stage} must return a mapping-like result.")


def _stage_timing_ms(output: CAGSStageOutput, keys: tuple[str, ...]) -> float:
    value = _first_present(output.payload, (*keys, "elapsed_ms", "duration_ms", "time_ms", "timing_ms"))
    return output.elapsed_ms if value is None else _non_negative_float(value, f"{output.stage}.timing_ms")


def _artifact_size_bytes(config: CAGSAdapterConfig, *outputs: CAGSStageOutput) -> int:
    explicit_size = _first_present_many(tuple(output.payload for output in outputs), ("size_bytes", "encoded_bytes", "compressed_bytes"))
    if explicit_size is not None:
        return _non_negative_int(explicit_size, "size_bytes")
    paths = _artifact_paths(config, *outputs)
    if not paths:
        return 0
    files: dict[str, Path] = {}
    for path in paths:
        for file_path in _artifact_files(path):
            files[str(file_path.resolve())] = file_path
    return sum(file_path.stat().st_size for file_path in files.values())


def _artifact_paths(config: CAGSAdapterConfig, *outputs: CAGSStageOutput) -> tuple[Path, ...]:
    paths: list[Path] = []
    for output in outputs:
        for key in (
            "compressed_dir",
            "compressed_path",
            "output_dir",
            "output_path",
            "artifact_path",
            "ply_path",
            "drc_path",
            "path",
        ):
            value = output.payload.get(key)
            if isinstance(value, str) and value:
                paths.append(Path(value))
    if config.output_dir:
        paths.append(Path(config.output_dir))
    deduped: dict[str, Path] = {}
    for path in paths:
        deduped[str(path)] = path
    return tuple(deduped.values())


def _artifact_files(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        return ()
    if path.is_file():
        return (path,)
    if path.is_dir():
        files: list[Path] = []
        for child in path.rglob("*"):
            if child.is_file() and (child.suffix.lower() in {"", ".ply", ".drc"} or child.parent == path):
                files.append(child)
        return tuple(files)
    return ()


def _visible_quality_from_render(payload: Mapping[str, Any], default_quality: float) -> float:
    explicit = _first_present(payload, ("visible_quality", "quality", "quality_score"))
    if explicit is not None:
        return _unit_interval(explicit, "visible_quality")
    psnr = _first_present(payload, ("psnr", "render_psnr", "psnr_db"))
    lpips = _first_present(payload, ("lpips", "render_lpips"))
    if psnr is None and lpips is None:
        return default_quality
    components: list[float] = []
    if psnr is not None:
        components.append(max(0.0, min(1.0, (_non_negative_float(psnr, "psnr") - 20.0) / 20.0)))
    if lpips is not None:
        components.append(max(0.0, min(1.0, 1.0 - _non_negative_float(lpips, "lpips"))))
    return sum(components) / len(components)


def _bool_from_outputs(default: bool, key: str, *outputs: CAGSStageOutput) -> bool:
    value = _first_present_many(tuple(output.payload for output in outputs), (key,))
    if value is None:
        return default
    if not isinstance(value, bool):
        raise CAGSAdapterError(f"{key} must be a boolean.")
    return value


def _artifact_uri(config: CAGSAdapterConfig, *outputs: CAGSStageOutput) -> str | None:
    value = _first_present_many(
        tuple(output.payload for output in outputs),
        ("artifact_uri", "artifact_path", "compressed_dir", "compressed_path", "output_path", "ply_path", "drc_path"),
    )
    if value is not None:
        return str(value)
    return config.output_dir


def _record_id(config: CAGSAdapterConfig, *outputs: CAGSStageOutput) -> str:
    explicit = _first_present_many(tuple(output.payload for output in outputs), ("record_id", "measurement_id", "id"))
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    return f"cags-{stable_config_id({'config': config.as_payload(), 'outputs': [output.as_payload() for output in outputs]})}"


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _first_present_many(mappings: tuple[Mapping[str, Any], ...], keys: tuple[str, ...]) -> Any:
    for mapping in mappings:
        value = _first_present(mapping, keys)
        if value is not None:
            return value
    return None


def _plain_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, MappingABC):
        raise CAGSAdapterError(f"{field_name} must be a mapping.")
    return {str(key): _plain_json_value(nested, f"{field_name}.{key}") for key, nested in value.items()}


def _plain_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _plain_json_value(nested, f"{field_name}.{key}") for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item, field_name) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not _is_finite(value):
            raise CAGSAdapterError(f"{field_name} must be finite.")
        return value
    if isinstance(value, Path):
        return str(value)
    raise CAGSAdapterError(f"{field_name} contains unsupported value type {type(value).__name__}.")


def _to_payload(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {key: _to_payload(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_to_payload(item) for item in value]
    if isinstance(value, list):
        return [_to_payload(item) for item in value]
    return value


def _non_negative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise CAGSAdapterError(f"{field_name} must be numeric.")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise CAGSAdapterError(f"{field_name} must be numeric.") from exc
    if not _is_finite(parsed):
        raise CAGSAdapterError(f"{field_name} must be finite.")
    if parsed < 0:
        raise CAGSAdapterError(f"{field_name} must be non-negative.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise CAGSAdapterError(f"{field_name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CAGSAdapterError(f"{field_name} must be an integer.") from exc
    if parsed < 0:
        raise CAGSAdapterError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name)
    if parsed > 1.0:
        raise CAGSAdapterError(f"{field_name} must be between 0 and 1.")
    return parsed


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


__all__ = [
    "CAGSAdapterConfig",
    "CAGSAdapterError",
    "CAGSBackendAdapter",
    "CAGSStageOutput",
    "CAGS_CANDIDATE_KINDS",
    "cags_stage_outputs_to_record",
    "normalize_cags_candidate_kind",
    "resolve_cags_module",
]
