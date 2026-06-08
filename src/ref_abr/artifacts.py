"""Raw-first artifact export for replay and metric pipelines."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from ref_abr.accounting import CandidateResourceAccount, ComponentTimingAccount, ResourceAccountingSummary
from ref_abr.candidates import CandidateObject
from ref_abr.domain import ControllerState, FrameOutcome, ScheduleDecision
from ref_abr.lifecycle import ReferenceLifecycleEvent
from ref_abr.schema import DOMAIN_SCHEMA_VERSION, RECORD_TYPE_FIELD, SCHEMA_VERSION_FIELD, stamp_record


ArtifactFormat = Literal["jsonl", "json"]
RawExportRecord = (
    CandidateObject
    | ControllerState
    | ScheduleDecision
    | ReferenceLifecycleEvent
    | FrameOutcome
    | CandidateResourceAccount
    | ResourceAccountingSummary
    | ComponentTimingAccount
)


class ArtifactExportError(ValueError):
    """Raised when raw artifact export inputs are invalid."""


@dataclass(frozen=True)
class ArtifactProvenance:
    """Stable provenance fields attached to every raw exported record."""

    run_id: str
    config_id: str | None = None
    split: str | None = None
    method_id: str | None = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, "run_id")
        for field_name in ("config_id", "split", "method_id", "source"):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_empty(value, field_name)
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_id": self.config_id,
            "split": self.split,
            "method_id": self.method_id,
            "source": self.source,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class RawArtifactFile:
    """Manifest entry for one exported raw artifact file."""

    artifact_name: str
    record_type: str
    path: str
    format: ArtifactFormat
    record_count: int
    sha256: str

    def __post_init__(self) -> None:
        _require_non_empty(self.artifact_name, "artifact_name")
        _require_non_empty(self.record_type, "record_type")
        _require_non_empty(self.path, "path")
        if self.format not in {"jsonl", "json"}:
            raise ArtifactExportError("format must be one of: jsonl, json.")
        object.__setattr__(self, "record_count", _non_negative_int(self.record_count, "record_count"))
        _require_non_empty(self.sha256, "sha256")

    def as_payload(self) -> dict[str, Any]:
        return {
            "artifact_name": self.artifact_name,
            "record_type": self.record_type,
            "path": self.path,
            "format": self.format,
            "record_count": self.record_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class RawArtifactManifest:
    """Manifest for a raw-first artifact export."""

    export_id: str
    output_root: str
    files: tuple[RawArtifactFile, ...]
    provenance: ArtifactProvenance
    schema_version: int = DOMAIN_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.export_id, "export_id")
        _require_non_empty(self.output_root, "output_root")
        if self.schema_version != DOMAIN_SCHEMA_VERSION:
            raise ArtifactExportError(f"schema_version must be {DOMAIN_SCHEMA_VERSION}.")
        files = tuple(self.files)
        for artifact_file in files:
            if not isinstance(artifact_file, RawArtifactFile):
                raise ArtifactExportError("files must contain RawArtifactFile records.")
        if not isinstance(self.provenance, ArtifactProvenance):
            raise ArtifactExportError("provenance must be an ArtifactProvenance record.")
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_type": "raw_artifact_manifest",
            "export_id": self.export_id,
            "output_root": self.output_root,
            "files": [artifact_file.as_payload() for artifact_file in self.files],
            "provenance": self.provenance.as_payload(),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class RawArtifactExportConfig:
    """Controls for raw artifact file layout and encoding."""

    output_format: ArtifactFormat = "jsonl"
    include_empty_files: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output_format not in {"jsonl", "json"}:
            raise ArtifactExportError("output_format must be one of: jsonl, json.")
        if not isinstance(self.include_empty_files, bool):
            raise ArtifactExportError("include_empty_files must be boolean.")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "output_format": self.output_format,
            "include_empty_files": self.include_empty_files,
            "metadata": _to_payload(self.metadata),
        }


def export_raw_artifacts(
    output_root: str | Path,
    *,
    provenance: ArtifactProvenance,
    object_candidates: Sequence[CandidateObject] = (),
    controller_states: Sequence[ControllerState] = (),
    decisions: Sequence[ScheduleDecision] = (),
    lifecycle_events: Sequence[ReferenceLifecycleEvent] = (),
    frame_outcomes: Sequence[FrameOutcome] = (),
    timing_records: Sequence[CandidateResourceAccount | ResourceAccountingSummary | ComponentTimingAccount] = (),
    config: RawArtifactExportConfig | None = None,
) -> RawArtifactManifest:
    """Write raw artifact files before aggregation and return a manifest."""

    if not isinstance(provenance, ArtifactProvenance):
        raise ArtifactExportError("provenance must be an ArtifactProvenance record.")
    export_config = config or RawArtifactExportConfig()
    if not isinstance(export_config, RawArtifactExportConfig):
        raise ArtifactExportError("config must be a RawArtifactExportConfig record.")
    root = Path(output_root)
    if not str(root):
        raise ArtifactExportError("output_root must be non-empty.")
    root.mkdir(parents=True, exist_ok=True)

    collections = (
        ("object_candidates", "candidate_object", _coerce_records(object_candidates, CandidateObject, "object_candidates")),
        ("controller_states", "controller_state", _coerce_records(controller_states, ControllerState, "controller_states")),
        ("decisions", "schedule_decision", _coerce_records(decisions, ScheduleDecision, "decisions")),
        ("lifecycle_events", "reference_lifecycle_event", _coerce_records(lifecycle_events, ReferenceLifecycleEvent, "lifecycle_events")),
        ("frame_outcomes", "frame_outcome", _coerce_records(frame_outcomes, FrameOutcome, "frame_outcomes")),
        ("timing_records", "timing_record", _coerce_timing_records(timing_records)),
    )

    files: list[RawArtifactFile] = []
    for artifact_name, record_type, records in collections:
        if not records and not export_config.include_empty_files:
            continue
        envelopes = tuple(
            raw_artifact_envelope(
                record,
                provenance=provenance,
                artifact_name=artifact_name,
                sequence_index=index,
            )
            for index, record in enumerate(records)
        )
        artifact_path = root / f"{artifact_name}.{export_config.output_format}"
        content = _encode_records(envelopes, export_config.output_format)
        _write_text_atomic(artifact_path, content)
        files.append(
            RawArtifactFile(
                artifact_name=artifact_name,
                record_type=record_type,
                path=artifact_path.as_posix(),
                format=export_config.output_format,
                record_count=len(envelopes),
                sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        )

    manifest_payload = {
        "output_root": root.as_posix(),
        "files": [artifact_file.as_payload() for artifact_file in files],
        "provenance": provenance.as_payload(),
        "config": export_config.as_payload(),
    }
    manifest = RawArtifactManifest(
        export_id=f"raw-artifact-export-{_stable_payload_hash(manifest_payload)}",
        output_root=root.as_posix(),
        files=tuple(files),
        provenance=provenance,
        metadata={"export_config": export_config.as_payload()},
    )
    manifest_content = _json_line(manifest.as_payload())
    _write_text_atomic(root / "manifest.json", manifest_content)
    return manifest


def raw_artifact_envelope(
    record: RawExportRecord,
    *,
    provenance: ArtifactProvenance,
    artifact_name: str,
    sequence_index: int,
) -> dict[str, Any]:
    """Build one versioned raw artifact envelope."""

    if not isinstance(provenance, ArtifactProvenance):
        raise ArtifactExportError("provenance must be an ArtifactProvenance record.")
    _require_non_empty(artifact_name, "artifact_name")
    sequence = _non_negative_int(sequence_index, "sequence_index")
    record_type = infer_raw_record_type(record)
    stamped_payload = _stamp_raw_payload(record, record_type)
    return {
        SCHEMA_VERSION_FIELD: DOMAIN_SCHEMA_VERSION,
        RECORD_TYPE_FIELD: record_type,
        "artifact_name": artifact_name,
        "sequence_index": sequence,
        "provenance": provenance.as_payload(),
        "payload": stamped_payload,
    }


def infer_raw_record_type(record: object) -> str:
    """Infer the raw artifact record type for supported export records."""

    if isinstance(record, CandidateObject):
        return "candidate_object"
    if isinstance(record, ControllerState):
        return "controller_state"
    if isinstance(record, ScheduleDecision):
        return "schedule_decision"
    if isinstance(record, ReferenceLifecycleEvent):
        return "reference_lifecycle_event"
    if isinstance(record, FrameOutcome):
        return "frame_outcome"
    if isinstance(record, CandidateResourceAccount):
        return "candidate_resource_account"
    if isinstance(record, ResourceAccountingSummary):
        return "resource_accounting_summary"
    if isinstance(record, ComponentTimingAccount):
        return "component_timing_account"
    raise ArtifactExportError(f"Unsupported raw artifact record type {type(record).__name__}.")


def _stamp_raw_payload(record: RawExportRecord, record_type: str) -> dict[str, Any]:
    if record_type in {"controller_state", "schedule_decision", "frame_outcome"}:
        return stamp_record(record, record_type=record_type)
    as_payload = getattr(record, "as_payload", None)
    if not callable(as_payload):
        raise ArtifactExportError(f"{type(record).__name__} is not serializable as a raw artifact.")
    payload = as_payload()
    if not isinstance(payload, MappingABC):
        raise ArtifactExportError(f"{type(record).__name__}.as_payload() must return a mapping.")
    return {
        SCHEMA_VERSION_FIELD: DOMAIN_SCHEMA_VERSION,
        RECORD_TYPE_FIELD: record_type,
        **_plain_json_mapping(payload, "payload"),
    }


def _coerce_records(records: Sequence[Any], record_type: type, field_name: str) -> tuple[Any, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ArtifactExportError(f"{field_name} must be a sequence.")
    coerced = tuple(records)
    for record in coerced:
        if not isinstance(record, record_type):
            raise ArtifactExportError(f"{field_name} must contain {record_type.__name__} records.")
    return coerced


def _coerce_timing_records(records: Sequence[Any]) -> tuple[CandidateResourceAccount | ResourceAccountingSummary | ComponentTimingAccount, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ArtifactExportError("timing_records must be a sequence.")
    allowed = (CandidateResourceAccount, ResourceAccountingSummary, ComponentTimingAccount)
    coerced = tuple(records)
    for record in coerced:
        if not isinstance(record, allowed):
            raise ArtifactExportError("timing_records must contain timing/accounting records.")
    return coerced


def _encode_records(records: tuple[Mapping[str, Any], ...], output_format: ArtifactFormat) -> str:
    if output_format == "jsonl":
        if not records:
            return ""
        return "".join(f"{_json_line(record)}\n" for record in records)
    if output_format == "json":
        return _json_line(list(records))
    raise ArtifactExportError("output_format must be one of: jsonl, json.")


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise ArtifactExportError(f"Failed to write artifact {path}: {exc}") from exc


def _json_line(value: Any) -> str:
    return json.dumps(_to_payload(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_payload_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_line(value).encode("utf-8")).hexdigest()[:16]


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise ArtifactExportError(f"{field_name} must be a mapping.")
    return {str(key): _to_payload(item) for key, item in value.items()}


def _to_payload(value: Any) -> Any:
    if hasattr(value, "as_payload"):
        return value.as_payload()
    if isinstance(value, MappingABC):
        return {str(key): _to_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_payload(item) for item in value]
    if isinstance(value, list):
        return [_to_payload(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactExportError(f"{field_name} must be a non-empty string.")


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ArtifactExportError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactExportError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise ArtifactExportError(f"{field_name} must be a non-negative integer.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ArtifactExportError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactExportError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise ArtifactExportError(f"{field_name} must be finite.")
    return parsed


__all__ = [
    "ArtifactExportError",
    "ArtifactFormat",
    "ArtifactProvenance",
    "RawArtifactExportConfig",
    "RawArtifactFile",
    "RawArtifactManifest",
    "export_raw_artifacts",
    "infer_raw_record_type",
    "raw_artifact_envelope",
]
