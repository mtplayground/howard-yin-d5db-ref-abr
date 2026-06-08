"""Candidate object generation for scheduling decisions."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Mapping

from ref_abr.config import stable_config_id
from ref_abr.domain import MediaObject, MediaType, WorkloadManifest
from ref_abr.substrate import ReferenceResolution, SubstrateValueProvider, coerce_ref_resolution
from ref_abr.viewport import ViewportPose


class CandidateError(ValueError):
    """Raised when candidate generation inputs are invalid."""


CANDIDATE_KINDS: tuple[str, ...] = (
    "gaussian_base",
    "gaussian_enhancement",
    "tile",
    "reference_action",
)


@dataclass(frozen=True)
class Viewpoint:
    """Normalized 6-DoF viewpoint for a candidate."""

    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("x_m", "y_m", "z_m", "yaw_deg", "pitch_deg", "roll_deg"):
            object.__setattr__(self, field_name, _finite_float(getattr(self, field_name), field_name))

    def as_payload(self) -> dict[str, float]:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "z_m": self.z_m,
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
        }


@dataclass(frozen=True)
class TileSpec:
    """Grid tile coordinate for tile candidates."""

    row: int
    column: int
    rows: int
    columns: int

    def __post_init__(self) -> None:
        rows = _positive_int(self.rows, "tile.rows")
        columns = _positive_int(self.columns, "tile.columns")
        row = _non_negative_int(self.row, "tile.row")
        column = _non_negative_int(self.column, "tile.column")
        if row >= rows:
            raise CandidateError("tile.row must be less than tile.rows.")
        if column >= columns:
            raise CandidateError("tile.column must be less than tile.columns.")
        object.__setattr__(self, "row", row)
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)

    def as_payload(self) -> dict[str, int]:
        return {"row": self.row, "column": self.column, "rows": self.rows, "columns": self.columns}


@dataclass(frozen=True)
class DecisionEpoch:
    """Inputs that define one candidate-generation decision epoch."""

    decision_time_ms: int
    frame_id: str | None = None
    viewpoint: Viewpoint | ViewportPose | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_time_ms", _non_negative_int(self.decision_time_ms, "decision_time_ms"))
        if self.frame_id is not None:
            _require_non_empty(self.frame_id, "frame_id")
        object.__setattr__(self, "viewpoint", coerce_viewpoint(self.viewpoint))

    def as_payload(self) -> dict[str, Any]:
        return {
            "decision_time_ms": self.decision_time_ms,
            "frame_id": self.frame_id,
            "viewpoint": self.viewpoint.as_payload(),
        }


@dataclass(frozen=True)
class CandidateGenerationSpec:
    """Discrete controls for feasible candidate generation."""

    resolutions: tuple[ReferenceResolution | str | tuple[int, int] | Mapping[str, Any], ...] = ("720p", "1080p")
    fov_degrees: tuple[float, ...] = (90.0,)
    lookahead_ms: tuple[int, ...] = (0, 100)
    expiration_ms: tuple[int, ...] = (500,)
    retransmit_priorities: tuple[int, ...] = (0, 1)
    enhancement_layers: tuple[int, ...] = (1,)
    tile_rows: int = 1
    tile_columns: int = 1
    include_gaussian_base: bool = True
    include_gaussian_enhancement: bool = True
    include_tiles: bool = True
    include_reference_actions: bool = True
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        resolutions = tuple(coerce_ref_resolution(resolution) for resolution in self.resolutions)
        if not resolutions:
            raise CandidateError("resolutions must contain at least one resolution.")
        object.__setattr__(self, "resolutions", resolutions)
        object.__setattr__(self, "fov_degrees", _float_tuple(self.fov_degrees, "fov_degrees", positive=True))
        object.__setattr__(self, "lookahead_ms", _int_tuple(self.lookahead_ms, "lookahead_ms", non_negative=True))
        object.__setattr__(self, "expiration_ms", _int_tuple(self.expiration_ms, "expiration_ms", positive=True))
        if not any(expiration_ms >= lookahead_ms for lookahead_ms, expiration_ms in itertools.product(self.lookahead_ms, self.expiration_ms)):
            raise CandidateError("expiration_ms must include at least one value greater than or equal to lookahead_ms.")
        object.__setattr__(
            self,
            "retransmit_priorities",
            _int_tuple(self.retransmit_priorities, "retransmit_priorities", non_negative=True),
        )
        object.__setattr__(self, "enhancement_layers", _int_tuple(self.enhancement_layers, "enhancement_layers", positive=True))
        object.__setattr__(self, "tile_rows", _positive_int(self.tile_rows, "tile_rows"))
        object.__setattr__(self, "tile_columns", _positive_int(self.tile_columns, "tile_columns"))
        if self.max_candidates is not None:
            object.__setattr__(self, "max_candidates", _positive_int(self.max_candidates, "max_candidates"))
        for field_name in (
            "include_gaussian_base",
            "include_gaussian_enhancement",
            "include_tiles",
            "include_reference_actions",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise CandidateError(f"{field_name} must be boolean.")

    def as_payload(self) -> dict[str, Any]:
        return {
            "resolutions": [resolution.as_payload() for resolution in self.resolutions],
            "fov_degrees": list(self.fov_degrees),
            "lookahead_ms": list(self.lookahead_ms),
            "expiration_ms": list(self.expiration_ms),
            "retransmit_priorities": list(self.retransmit_priorities),
            "enhancement_layers": list(self.enhancement_layers),
            "tile_rows": self.tile_rows,
            "tile_columns": self.tile_columns,
            "include_gaussian_base": self.include_gaussian_base,
            "include_gaussian_enhancement": self.include_gaussian_enhancement,
            "include_tiles": self.include_tiles,
            "include_reference_actions": self.include_reference_actions,
            "max_candidates": self.max_candidates,
        }


@dataclass(frozen=True)
class CandidateObject:
    """Feasible action candidate for a media or reference object."""

    candidate_id: str
    object_id: str
    candidate_kind: str
    decision_time_ms: int
    layer: int
    resolution: ReferenceResolution | str | tuple[int, int] | Mapping[str, Any]
    fov_deg: float
    viewpoint: Viewpoint | ViewportPose | Mapping[str, Any] | None
    lookahead_ms: int
    expiration_ms: int
    retransmit_priority: int
    size_bytes: int
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    tile: TileSpec | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.candidate_id, "candidate_id")
        _require_non_empty(self.object_id, "object_id")
        if self.candidate_kind not in CANDIDATE_KINDS:
            raise CandidateError(f"candidate_kind must be one of: {', '.join(CANDIDATE_KINDS)}.")
        object.__setattr__(self, "decision_time_ms", _non_negative_int(self.decision_time_ms, "decision_time_ms"))
        object.__setattr__(self, "layer", _non_negative_int(self.layer, "layer"))
        object.__setattr__(self, "resolution", coerce_ref_resolution(self.resolution))
        object.__setattr__(self, "fov_deg", _positive_float(self.fov_deg, "fov_deg"))
        object.__setattr__(self, "viewpoint", coerce_viewpoint(self.viewpoint))
        object.__setattr__(self, "lookahead_ms", _non_negative_int(self.lookahead_ms, "lookahead_ms"))
        object.__setattr__(self, "expiration_ms", _positive_int(self.expiration_ms, "expiration_ms"))
        if self.expiration_ms < self.lookahead_ms:
            raise CandidateError("expiration_ms must be greater than or equal to lookahead_ms.")
        object.__setattr__(self, "retransmit_priority", _non_negative_int(self.retransmit_priority, "retransmit_priority"))
        object.__setattr__(self, "size_bytes", _non_negative_int(self.size_bytes, "size_bytes"))
        object.__setattr__(self, "dependencies", _string_tuple(self.dependencies, "dependencies"))
        if self.tile is not None and not isinstance(self.tile, TileSpec):
            raise CandidateError("tile must be a TileSpec when provided.")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def deadline_ms(self) -> int:
        return self.decision_time_ms + self.expiration_ms

    def as_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "object_id": self.object_id,
            "candidate_kind": self.candidate_kind,
            "decision_time_ms": self.decision_time_ms,
            "layer": self.layer,
            "resolution": self.resolution.as_payload(),
            "fov_deg": self.fov_deg,
            "viewpoint": self.viewpoint.as_payload(),
            "lookahead_ms": self.lookahead_ms,
            "expiration_ms": self.expiration_ms,
            "deadline_ms": self.deadline_ms,
            "retransmit_priority": self.retransmit_priority,
            "size_bytes": self.size_bytes,
            "dependencies": list(self.dependencies),
            "tile": self.tile.as_payload() if self.tile is not None else None,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class CandidateSet:
    """Candidate generation result for one decision epoch."""

    candidate_set_id: str
    decision_time_ms: int
    candidates: tuple[CandidateObject, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.candidate_set_id, "candidate_set_id")
        object.__setattr__(self, "decision_time_ms", _non_negative_int(self.decision_time_ms, "decision_time_ms"))
        candidates = tuple(self.candidates)
        if not candidates:
            raise CandidateError("candidates must contain at least one candidate.")
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        duplicates = sorted({candidate_id for candidate_id in candidate_ids if candidate_ids.count(candidate_id) > 1})
        if duplicates:
            raise CandidateError(f"candidates must not contain duplicate candidate_id values: {', '.join(duplicates)}.")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "candidate_set_id": self.candidate_set_id,
            "decision_time_ms": self.decision_time_ms,
            "candidates": [candidate.as_payload() for candidate in self.candidates],
            "metadata": _to_payload(self.metadata),
        }


def generate_candidate_objects(
    workload: WorkloadManifest,
    epoch: DecisionEpoch | Mapping[str, Any],
    *,
    spec: CandidateGenerationSpec | None = None,
    substrate_provider: SubstrateValueProvider | None = None,
) -> CandidateSet:
    """Generate feasible candidates for one decision epoch."""

    if not isinstance(workload, WorkloadManifest):
        raise CandidateError("workload must be a WorkloadManifest record.")
    decision_epoch = epoch if isinstance(epoch, DecisionEpoch) else DecisionEpoch(**_require_mapping(epoch, "epoch"))
    generation_spec = spec or CandidateGenerationSpec()
    candidates: list[CandidateObject] = []
    for media_object in sorted(workload.media_objects, key=lambda media: media.object_id):
        candidates.extend(_candidates_for_media(media_object, decision_epoch, generation_spec, substrate_provider))
    if generation_spec.max_candidates is not None:
        candidates = candidates[: generation_spec.max_candidates]
    if not candidates:
        raise CandidateError("No feasible candidates were generated for the decision epoch.")
    metadata = {
        "provenance": {
            "workload_manifest_id": workload.manifest_id,
            "config_id": workload.config_id,
            "split": workload.split,
            "spec": generation_spec.as_payload(),
            "candidate_count": len(candidates),
        }
    }
    set_payload = {
        "decision_time_ms": decision_epoch.decision_time_ms,
        "candidates": [candidate.as_payload() for candidate in candidates],
    }
    return CandidateSet(
        candidate_set_id=f"candidate-set-{stable_config_id(set_payload)}",
        decision_time_ms=decision_epoch.decision_time_ms,
        candidates=tuple(candidates),
        metadata=metadata,
    )


def candidate_generation_spec_from_mapping(values: Mapping[str, Any]) -> CandidateGenerationSpec:
    """Build a CandidateGenerationSpec from a config mapping."""

    mapping = _require_mapping(values, "candidate_generation")
    defaults = CandidateGenerationSpec()
    return CandidateGenerationSpec(
        resolutions=_tuple_value(mapping.get("resolutions", defaults.resolutions)),
        fov_degrees=_tuple_value(mapping.get("fov_degrees", defaults.fov_degrees)),
        lookahead_ms=_tuple_value(mapping.get("lookahead_ms", defaults.lookahead_ms)),
        expiration_ms=_tuple_value(mapping.get("expiration_ms", defaults.expiration_ms)),
        retransmit_priorities=_tuple_value(mapping.get("retransmit_priorities", defaults.retransmit_priorities)),
        enhancement_layers=_tuple_value(mapping.get("enhancement_layers", defaults.enhancement_layers)),
        tile_rows=mapping.get("tile_rows", defaults.tile_rows),
        tile_columns=mapping.get("tile_columns", defaults.tile_columns),
        include_gaussian_base=_bool_value(
            mapping.get("include_gaussian_base", defaults.include_gaussian_base),
            "include_gaussian_base",
        ),
        include_gaussian_enhancement=_bool_value(
            mapping.get("include_gaussian_enhancement", defaults.include_gaussian_enhancement),
            "include_gaussian_enhancement",
        ),
        include_tiles=_bool_value(mapping.get("include_tiles", defaults.include_tiles), "include_tiles"),
        include_reference_actions=_bool_value(
            mapping.get("include_reference_actions", defaults.include_reference_actions),
            "include_reference_actions",
        ),
        max_candidates=mapping.get("max_candidates", defaults.max_candidates),
    )


def coerce_viewpoint(value: Viewpoint | ViewportPose | Mapping[str, Any] | None) -> Viewpoint:
    """Normalize a viewpoint from existing records or mappings."""

    if value is None:
        return Viewpoint()
    if isinstance(value, Viewpoint):
        return value
    if isinstance(value, ViewportPose):
        return Viewpoint(
            x_m=value.x_m,
            y_m=value.y_m,
            z_m=value.z_m,
            yaw_deg=value.yaw_deg,
            pitch_deg=value.pitch_deg,
            roll_deg=value.roll_deg,
        )
    mapping = _require_mapping(value, "viewpoint")
    return Viewpoint(
        x_m=_first_present(mapping, ("x_m", "x")) or 0.0,
        y_m=_first_present(mapping, ("y_m", "y")) or 0.0,
        z_m=_first_present(mapping, ("z_m", "z")) or 0.0,
        yaw_deg=_first_present(mapping, ("yaw_deg", "yaw")) or 0.0,
        pitch_deg=_first_present(mapping, ("pitch_deg", "pitch")) or 0.0,
        roll_deg=_first_present(mapping, ("roll_deg", "roll")) or 0.0,
    )


def _candidates_for_media(
    media_object: MediaObject,
    epoch: DecisionEpoch,
    spec: CandidateGenerationSpec,
    substrate_provider: SubstrateValueProvider | None,
) -> list[CandidateObject]:
    media_type = media_object.media_type
    candidates: list[CandidateObject] = []
    if media_type == MediaType.GAUSSIAN_SPLAT:
        if spec.include_gaussian_base:
            candidates.extend(
                _candidate_grid(
                    media_object,
                    epoch,
                    spec,
                    substrate_provider,
                    candidate_kind="gaussian_base",
                    layer=0,
                    tile=None,
                )
            )
        if spec.include_gaussian_enhancement:
            for layer in spec.enhancement_layers:
                candidates.extend(
                    _candidate_grid(
                        media_object,
                        epoch,
                        spec,
                        substrate_provider,
                        candidate_kind="gaussian_enhancement",
                        layer=layer,
                        tile=None,
                    )
                )
        if spec.include_tiles:
            for tile in _tiles(spec):
                candidates.extend(
                    _candidate_grid(
                        media_object,
                        epoch,
                        spec,
                        substrate_provider,
                        candidate_kind="tile",
                        layer=0,
                        tile=tile,
                    )
                )
    if spec.include_reference_actions:
        candidates.extend(
            _candidate_grid(
                media_object,
                epoch,
                spec,
                substrate_provider,
                candidate_kind="reference_action",
                layer=0,
                tile=None,
            )
        )
    return candidates


def _candidate_grid(
    media_object: MediaObject,
    epoch: DecisionEpoch,
    spec: CandidateGenerationSpec,
    substrate_provider: SubstrateValueProvider | None,
    *,
    candidate_kind: str,
    layer: int,
    tile: TileSpec | None,
) -> list[CandidateObject]:
    candidates: list[CandidateObject] = []
    for resolution, fov_deg, lookahead_ms, expiration_ms, priority in itertools.product(
        spec.resolutions,
        spec.fov_degrees,
        spec.lookahead_ms,
        spec.expiration_ms,
        spec.retransmit_priorities,
    ):
        if expiration_ms < lookahead_ms:
            continue
        metadata = _candidate_metadata(media_object, candidate_kind, tile)
        if substrate_provider is not None:
            substrate_value = substrate_provider.evaluate(
                {
                    "layer": layer,
                    "ref_resolution": resolution,
                    "fov_deg": fov_deg,
                    "view_mismatch_deg": 0.0,
                    "freshness_ms": lookahead_ms,
                }
            )
            metadata["substrate"] = substrate_value.as_payload()
        payload = {
            "object_id": media_object.object_id,
            "candidate_kind": candidate_kind,
            "decision_time_ms": epoch.decision_time_ms,
            "layer": layer,
            "resolution": resolution.as_payload(),
            "fov_deg": fov_deg,
            "viewpoint": epoch.viewpoint.as_payload(),
            "lookahead_ms": lookahead_ms,
            "expiration_ms": expiration_ms,
            "retransmit_priority": priority,
            "tile": tile.as_payload() if tile is not None else None,
        }
        candidates.append(
            CandidateObject(
                candidate_id=f"candidate-{stable_config_id(payload)}",
                object_id=media_object.object_id,
                candidate_kind=candidate_kind,
                decision_time_ms=epoch.decision_time_ms,
                layer=layer,
                resolution=resolution,
                fov_deg=fov_deg,
                viewpoint=epoch.viewpoint,
                lookahead_ms=lookahead_ms,
                expiration_ms=expiration_ms,
                retransmit_priority=priority,
                size_bytes=_candidate_size_bytes(media_object, tile),
                dependencies=media_object.dependencies,
                tile=tile,
                metadata=metadata,
            )
        )
    return candidates


def _candidate_metadata(media_object: MediaObject, candidate_kind: str, tile: TileSpec | None) -> dict[str, Any]:
    media_payload = media_object.as_payload()
    return {
        "provenance": {
            "media_type": media_payload["media_type"],
            "candidate_kind": candidate_kind,
            "source_uri": media_payload["uri"],
            "tile": tile.as_payload() if tile is not None else None,
        }
    }


def _tiles(spec: CandidateGenerationSpec) -> tuple[TileSpec, ...]:
    return tuple(
        TileSpec(row=row, column=column, rows=spec.tile_rows, columns=spec.tile_columns)
        for row, column in itertools.product(range(spec.tile_rows), range(spec.tile_columns))
    )


def _candidate_size_bytes(media_object: MediaObject, tile: TileSpec | None) -> int:
    if tile is None:
        return media_object.size_bytes
    tile_count = tile.rows * tile.columns
    return int(math.ceil(media_object.size_bytes / tile_count))


def _tuple_value(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _float_tuple(values: Any, field_name: str, *, positive: bool = False) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise CandidateError(f"{field_name} must be a tuple.")
    if not values:
        raise CandidateError(f"{field_name} must contain at least one value.")
    parsed_values = tuple(_finite_float(value, field_name) for value in values)
    if positive and any(value <= 0 for value in parsed_values):
        raise CandidateError(f"{field_name} values must be positive.")
    return parsed_values


def _int_tuple(values: Any, field_name: str, *, non_negative: bool = False, positive: bool = False) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise CandidateError(f"{field_name} must be a tuple.")
    if not values:
        raise CandidateError(f"{field_name} must contain at least one value.")
    parsed_values = tuple(_positive_int(value, field_name) if positive else _non_negative_int(value, field_name) for value in values)
    if non_negative and any(value < 0 for value in parsed_values):
        raise CandidateError(f"{field_name} values must be non-negative.")
    return parsed_values


def _string_tuple(values: Any, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (_require_non_empty(values, field_name),)
    if not isinstance(values, tuple):
        values = tuple(values)
    result = tuple(_require_non_empty(value, field_name) for value in values)
    return result


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise CandidateError(f"{field_name} must be a positive integer.")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise CandidateError(f"{field_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise CandidateError(f"{field_name} must be a positive integer.")
    return parsed


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise CandidateError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise CandidateError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise CandidateError(f"{field_name} must be a non-negative integer.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise CandidateError(f"{field_name} must be numeric.")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise CandidateError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise CandidateError(f"{field_name} must be finite.")
    return parsed


def _positive_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed <= 0:
        raise CandidateError(f"{field_name} must be positive.")
    return parsed


def _bool_value(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise CandidateError(f"{field_name} must be boolean.")


def _plain_json_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, MappingABC):
        raise CandidateError(f"{field_name} must be a mapping.")
    return {
        str(key): _plain_json_value(nested, f"{field_name}.{key}")
        for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
    }


def _plain_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, MappingABC):
        return _plain_json_mapping(value, field_name)
    if isinstance(value, list | tuple):
        return [_plain_json_value(item, field_name) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise CandidateError(f"{field_name} must be finite.")
        return value
    raise CandidateError(f"{field_name} contains unsupported value type {type(value).__name__}.")


def _to_payload(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {key: _to_payload(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_to_payload(item) for item in value]
    if isinstance(value, list):
        return [_to_payload(item) for item in value]
    return value


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise CandidateError(f"{path} must be a mapping.")
    return value


__all__ = [
    "CANDIDATE_KINDS",
    "CandidateError",
    "CandidateGenerationSpec",
    "CandidateObject",
    "CandidateSet",
    "DecisionEpoch",
    "TileSpec",
    "Viewpoint",
    "candidate_generation_spec_from_mapping",
    "coerce_viewpoint",
    "generate_candidate_objects",
]
