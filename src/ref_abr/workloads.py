"""Scene and sequence workload metadata normalization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from ref_abr.config import ConfigError, load_config_file, stable_config_id
from ref_abr.domain import DomainError, MediaObject, MediaType, WorkloadManifest
from ref_abr.schema import SchemaError, validate_record_payload


class WorkloadError(ValueError):
    """Raised when scene or sequence metadata cannot be normalized."""


ASSET_KEYS: tuple[str, ...] = (
    "media_objects",
    "objects",
    "assets",
    "gaussians",
    "references",
    "segments",
    "frames",
)


def load_workload_manifest(
    metadata_path: str | Path,
    *,
    config_id: str,
    split: str,
    seed: int,
    dataset_base_path: str | Path | None = None,
) -> WorkloadManifest:
    """Load scene or sequence metadata from disk and assemble a workload manifest."""

    path = Path(metadata_path)
    try:
        raw_metadata = load_config_file(path)
    except ConfigError as exc:
        raise WorkloadError(str(exc)) from exc
    return assemble_workload_manifest(
        raw_metadata,
        config_id=config_id,
        split=split,
        seed=seed,
        source_uri=str(path),
        dataset_base_path=dataset_base_path,
    )


def assemble_workload_manifest(
    metadata: Mapping[str, Any],
    *,
    config_id: str,
    split: str,
    seed: int,
    source_uri: str | None = None,
    dataset_base_path: str | Path | None = None,
) -> WorkloadManifest:
    """Normalize scene/sequence metadata into a validated workload manifest."""

    raw_metadata = _require_mapping(metadata, "metadata")
    source_format = _detect_source_format(raw_metadata)
    dataset_name = _string_or_none(_first_present(raw_metadata, ("dataset_name", "dataset", "name")))
    base_path = Path(dataset_base_path) if dataset_base_path is not None else None

    media_objects: list[MediaObject] = []
    scene_ids: set[str] = set()
    sequence_ids: set[str] = set()
    for sequence_index, context in enumerate(_iter_sequence_contexts(raw_metadata)):
        scene_id = context["scene_id"] or "scene"
        sequence_id = context["sequence_id"] or f"sequence-{sequence_index}"
        scene_ids.add(scene_id)
        sequence_ids.add(sequence_id)
        media_objects.extend(
            _media_objects_for_sequence(
                context=context,
                sequence_index=sequence_index,
                source_format=source_format,
                dataset_name=dataset_name,
                base_path=base_path,
            )
        )

    if not media_objects:
        raise WorkloadError("Workload metadata did not contain any media objects, assets, frames, or full-scene paths.")

    media_objects = _deduplicate_media_object_ids(media_objects)
    metadata_payload = {
        "provenance": {
            "source_format": source_format,
            "source_uri": source_uri,
            "dataset_name": dataset_name,
            "scene_ids": sorted(scene_ids),
            "sequence_ids": sorted(sequence_ids),
            "media_object_count": len(media_objects),
        }
    }
    manifest_id = _manifest_id(
        config_id=config_id,
        split=split,
        seed=seed,
        source_uri=source_uri,
        source_format=source_format,
        media_objects=media_objects,
    )
    try:
        manifest = WorkloadManifest(
            manifest_id=manifest_id,
            config_id=config_id,
            split=split,
            seed=seed,
            media_objects=tuple(media_objects),
            source_uri=source_uri,
            metadata=metadata_payload,
        )
        validate_record_payload("workload_manifest", manifest.as_payload())
    except (DomainError, SchemaError) as exc:
        raise WorkloadError(f"Normalized workload manifest is invalid: {exc}") from exc
    return manifest


def _iter_sequence_contexts(metadata: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    root = metadata.get("cags") if isinstance(metadata.get("cags"), MappingABC) else metadata
    root_scene_id = _string_or_none(_first_present(root, ("scene_id", "scene", "scene_name")))
    root_sequence_id = _string_or_none(_first_present(root, ("sequence_id", "sequence", "sequence_name", "take")))

    scenes = root.get("scenes")
    if scenes is not None:
        if not isinstance(scenes, list):
            raise WorkloadError("metadata.scenes must be a list when present.")
        for scene_index, scene in enumerate(scenes):
            scene_node = _require_mapping(scene, f"metadata.scenes[{scene_index}]")
            scene_id = _string_or_none(_first_present(scene_node, ("scene_id", "scene", "name", "id"))) or root_scene_id
            sequences = _first_present(scene_node, ("sequences", "captures", "takes"))
            if sequences is None:
                yield {"scene_id": scene_id, "sequence_id": root_sequence_id, "node": scene_node}
                continue
            if not isinstance(sequences, list):
                raise WorkloadError(f"metadata.scenes[{scene_index}].sequences must be a list when present.")
            for sequence_index, sequence in enumerate(sequences):
                sequence_node = _require_mapping(sequence, f"metadata.scenes[{scene_index}].sequences[{sequence_index}]")
                sequence_id = _string_or_none(
                    _first_present(sequence_node, ("sequence_id", "sequence", "name", "id", "take"))
                )
                yield {"scene_id": scene_id, "sequence_id": sequence_id, "node": sequence_node}
        return

    sequences = root.get("sequences")
    if sequences is not None:
        if not isinstance(sequences, list):
            raise WorkloadError("metadata.sequences must be a list when present.")
        for sequence_index, sequence in enumerate(sequences):
            sequence_node = _require_mapping(sequence, f"metadata.sequences[{sequence_index}]")
            scene_id = (
                _string_or_none(_first_present(sequence_node, ("scene_id", "scene", "scene_name")))
                or root_scene_id
            )
            sequence_id = _string_or_none(_first_present(sequence_node, ("sequence_id", "sequence", "name", "id", "take")))
            yield {"scene_id": scene_id, "sequence_id": sequence_id, "node": sequence_node}
        return

    yield {"scene_id": root_scene_id, "sequence_id": root_sequence_id, "node": root}


def _media_objects_for_sequence(
    *,
    context: Mapping[str, Any],
    sequence_index: int,
    source_format: str,
    dataset_name: str | None,
    base_path: Path | None,
) -> list[MediaObject]:
    node = _require_mapping(context["node"], "sequence")
    scene_id = context["scene_id"] or "scene"
    sequence_id = context["sequence_id"] or f"sequence-{sequence_index}"
    media_objects: list[MediaObject] = []

    for asset_key in ASSET_KEYS:
        raw_assets = node.get(asset_key)
        if raw_assets is None:
            continue
        assets = raw_assets if isinstance(raw_assets, list) else [raw_assets]
        for asset_index, raw_asset in enumerate(assets):
            asset_path = f"{scene_id}.{sequence_id}.{asset_key}[{asset_index}]"
            asset = _require_mapping(raw_asset, asset_path)
            media_objects.append(
                _media_object_from_asset(
                    asset=asset,
                    asset_key=asset_key,
                    asset_index=asset_index,
                    scene_id=scene_id,
                    sequence_id=sequence_id,
                    source_format=source_format,
                    dataset_name=dataset_name,
                    base_path=base_path,
                )
            )

    if not media_objects:
        full_scene_asset = _single_full_scene_asset(node)
        if full_scene_asset is not None:
            media_objects.append(
                _media_object_from_asset(
                    asset=full_scene_asset,
                    asset_key="full_scene",
                    asset_index=0,
                    scene_id=scene_id,
                    sequence_id=sequence_id,
                    source_format=source_format,
                    dataset_name=dataset_name,
                    base_path=base_path,
                )
            )
    return media_objects


def _media_object_from_asset(
    *,
    asset: Mapping[str, Any],
    asset_key: str,
    asset_index: int,
    scene_id: str,
    sequence_id: str,
    source_format: str,
    dataset_name: str | None,
    base_path: Path | None,
) -> MediaObject:
    object_id = _string_or_none(
        _first_present(asset, ("object_id", "id", "name", "reference_id", "frame_id", "asset_id"))
    )
    if object_id is None:
        object_id = f"{scene_id}:{sequence_id}:{asset_key}:{asset_index}"
    uri_value = _first_present(asset, ("uri", "path", "file", "filename", "relative_path"))
    if uri_value is None:
        raise WorkloadError(f"{scene_id}.{sequence_id}.{asset_key}[{asset_index}] is missing uri/path/file.")
    uri = _normalize_uri(str(uri_value), base_path)
    media_type = _coerce_media_type(asset, asset_key, uri)
    size_bytes = _coerce_non_negative_int(
        _first_present(asset, ("size_bytes", "bytes", "file_size_bytes", "size")),
        default=0,
        field_name=f"{object_id}.size_bytes",
    )
    duration_ms = _optional_non_negative_int(
        _first_present(asset, ("duration_ms", "frame_duration_ms", "segment_duration_ms")),
        field_name=f"{object_id}.duration_ms",
    )
    dependencies = _string_tuple(_first_present(asset, ("dependencies", "depends_on", "requires")), object_id)
    metadata = {
        "provenance": {
            "source_format": source_format,
            "dataset_name": dataset_name,
            "scene_id": scene_id,
            "sequence_id": sequence_id,
            "asset_key": asset_key,
            "asset_index": asset_index,
            "original_id": _string_or_none(_first_present(asset, ("object_id", "id", "name", "reference_id", "frame_id"))),
            "size_bytes_provided": _first_present(asset, ("size_bytes", "bytes", "file_size_bytes", "size")) is not None,
        }
    }
    try:
        return MediaObject(
            object_id=object_id,
            uri=uri,
            media_type=media_type,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            dependencies=dependencies,
            metadata=metadata,
        )
    except DomainError as exc:
        raise WorkloadError(f"Invalid media object {object_id!r}: {exc}") from exc


def _single_full_scene_asset(node: Mapping[str, Any]) -> dict[str, Any] | None:
    uri_value = _first_present(node, ("uri", "path", "file", "filename", "relative_path", "asset_path"))
    if uri_value is None:
        return None
    object_id = _string_or_none(_first_present(node, ("object_id", "id", "name", "scene_id", "scene")))
    return {
        "object_id": object_id or "full-scene",
        "uri": uri_value,
        "media_type": _first_present(node, ("media_type", "type", "kind")),
        "size_bytes": _first_present(node, ("size_bytes", "bytes", "file_size_bytes", "size")),
        "duration_ms": _first_present(node, ("duration_ms", "frame_duration_ms", "segment_duration_ms")),
        "dependencies": _first_present(node, ("dependencies", "depends_on", "requires")),
    }


def _deduplicate_media_object_ids(media_objects: list[MediaObject]) -> list[MediaObject]:
    counts: dict[str, int] = {}
    deduplicated: list[MediaObject] = []
    for media_object in media_objects:
        count = counts.get(media_object.object_id, 0)
        counts[media_object.object_id] = count + 1
        if count == 0:
            deduplicated.append(media_object)
            continue
        deduplicated.append(
            MediaObject(
                object_id=f"{media_object.object_id}#{count + 1}",
                uri=media_object.uri,
                media_type=media_object.media_type,
                size_bytes=media_object.size_bytes,
                duration_ms=media_object.duration_ms,
                dependencies=media_object.dependencies,
                metadata=media_object.as_payload()["metadata"],
            )
        )
    return deduplicated


def _manifest_id(
    *,
    config_id: str,
    split: str,
    seed: int,
    source_uri: str | None,
    source_format: str,
    media_objects: list[MediaObject],
) -> str:
    payload = {
        "config_id": config_id,
        "split": split,
        "seed": seed,
        "source_uri": source_uri,
        "source_format": source_format,
        "media_objects": [media_object.as_payload() for media_object in media_objects],
    }
    return f"workload-{stable_config_id(payload)}"


def _detect_source_format(metadata: Mapping[str, Any]) -> str:
    explicit = _string_or_none(
        _first_present(metadata, ("format", "source_format", "dataset_type", "style", "schema"))
    )
    if explicit:
        normalized = explicit.strip().lower().replace("_", "-")
        if "cags" in normalized:
            return "cags"
        if "n3dv" in normalized:
            return "n3dv"
        if "human" in normalized:
            return "human"
        if "full" in normalized and "scene" in normalized:
            return "full-scene"
        if "dynamic" in normalized or "3dgs" in normalized:
            return "dynamic-3dgs"
        return normalized
    if "cags" in metadata or "cags_version" in metadata:
        return "cags"
    dataset_name = str(_first_present(metadata, ("dataset_name", "dataset", "name")) or "").lower()
    if "n3dv" in dataset_name:
        return "n3dv"
    if "human" in dataset_name:
        return "human"
    if metadata.get("full_scene") is True:
        return "full-scene"
    return "generic-sequence"


def _coerce_media_type(asset: Mapping[str, Any], asset_key: str, uri: str) -> str:
    raw_type = _string_or_none(_first_present(asset, ("media_type", "type", "kind", "asset_type")))
    if raw_type:
        normalized = raw_type.lower().replace("-", "_")
        aliases = {
            "gaussian": MediaType.GAUSSIAN_SPLAT.value,
            "gaussian_splats": MediaType.GAUSSIAN_SPLAT.value,
            "3dgs": MediaType.GAUSSIAN_SPLAT.value,
            "segment": MediaType.VIDEO_SEGMENT.value,
            "frame": MediaType.VIDEO_SEGMENT.value,
        }
        return aliases.get(normalized, normalized)
    if asset_key == "frames":
        return MediaType.VIDEO_SEGMENT.value
    suffix = Path(urlparse(uri).path).suffix.lower()
    if suffix in {".mp4", ".mov", ".mkv", ".webm"}:
        return MediaType.VIDEO_SEGMENT.value
    if suffix in {".obj", ".glb", ".gltf"}:
        return MediaType.MESH.value
    if suffix in {".png", ".jpg", ".jpeg", ".ktx", ".exr"}:
        return MediaType.TEXTURE.value
    if suffix in {".json", ".yaml", ".yml", ".toml"}:
        return MediaType.METADATA.value
    return MediaType.GAUSSIAN_SPLAT.value


def _normalize_uri(value: str, base_path: Path | None) -> str:
    parsed = urlparse(value)
    if parsed.scheme:
        return value
    path = Path(value)
    if path.is_absolute():
        return path.as_posix()
    if base_path is not None:
        return (base_path / path).as_posix()
    return path.as_posix()


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _coerce_non_negative_int(value: Any, *, default: int, field_name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise WorkloadError(f"{field_name} must be a non-negative integer.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isdecimal():
        parsed = int(value)
    else:
        raise WorkloadError(f"{field_name} must be a non-negative integer.")
    if parsed < 0:
        raise WorkloadError(f"{field_name} must be a non-negative integer.")
    return parsed


def _optional_non_negative_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _coerce_non_negative_int(value, default=0, field_name=field_name)


def _string_tuple(value: Any, object_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        values: list[str] = []
        for index, item in enumerate(value):
            item_string = _string_or_none(item)
            if item_string is None:
                raise WorkloadError(f"{object_id}.dependencies[{index}] must be a string.")
            values.append(item_string)
        return tuple(values)
    raise WorkloadError(f"{object_id}.dependencies must be a string or list of strings.")


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise WorkloadError(f"{path} must be a mapping.")
    return value


__all__ = [
    "WorkloadError",
    "assemble_workload_manifest",
    "load_workload_manifest",
]
