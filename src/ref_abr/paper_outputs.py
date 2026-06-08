"""Derive named paper outputs from existing raw and metric artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from ref_abr.config import stable_config_id


PaperOutputFormat = Literal["json", "jsonl"]

DEFAULT_PAPER_OUTPUT_SPECS: tuple[Mapping[str, Any], ...] = (
    {"output_name": "substitution_surface", "source_filenames": ("substitution_surface_summary.json", "substitution_surface.jsonl")},
    {"output_name": "lifecycle_matrix", "source_filenames": ("lifecycle_matrix.jsonl",)},
    {"output_name": "screening_table", "source_filenames": ("candidate_method_selection_summary.json", "candidate_method_selection.jsonl")},
    {"output_name": "main_qoe_table", "source_filenames": ("main_qoe_table.json", "full_system_qoe_summary.json")},
    {"output_name": "quality_deadline_pareto", "source_filenames": ("quality_deadline_pareto.json", "full_system_qoe_summary.json")},
    {"output_name": "deadline_hit_qoe_cdf", "source_filenames": ("deadline_hit_qoe_cdf.json", "full_system_qoe_summary.json")},
    {"output_name": "ablation_table", "source_filenames": ("paired_ablation_table.json", "mechanism_attribution_summary.json")},
    {"output_name": "stress_matrix", "source_filenames": ("coupled_stress_matrix.json", "coupled_stress_summary.json")},
    {"output_name": "traceability", "source_filenames": ("claim_artifact_traceability.json", "reproducibility_evidence_summary.json")},
    {"output_name": "tolerance_checks", "source_filenames": ("tolerance_checks.json", "reproducibility_evidence_summary.json"), "required": False},
    {"output_name": "calibration_manifest", "source_filenames": ("utility_lifecycle_calibration.json",), "required": False},
    {"output_name": "frozen_method_manifest", "source_filenames": ("frozen_method_manifest.json", "freeze_method_manifest.json"), "required": False},
)
DERIVED_OUTPUT_VERSION = "paper_outputs_v1"


class PaperOutputError(ValueError):
    """Raised when paper outputs cannot be derived from existing artifacts."""


@dataclass(frozen=True)
class PaperOutputSpec:
    """Mapping from source artifact filename(s) to one named paper output."""

    output_name: str
    source_filenames: Sequence[str]
    required: bool = True
    output_format: PaperOutputFormat = "json"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.output_name, "output_name")
        object.__setattr__(self, "source_filenames", _string_tuple(self.source_filenames, "source_filenames"))
        if not isinstance(self.required, bool):
            raise PaperOutputError("required must be boolean.")
        if self.output_format not in {"json", "jsonl"}:
            raise PaperOutputError("output_format must be one of: json, jsonl.")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "output_name": self.output_name,
            "source_filenames": list(self.source_filenames),
            "required": self.required,
            "output_format": self.output_format,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class PaperOutputRecord:
    """One materialized paper output and its source provenance."""

    output_name: str
    source_path: str
    output_path: str
    output_format: PaperOutputFormat
    row_count: int
    sha256: str
    derived_from_artifacts: bool = True
    validation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.output_name, "output_name")
        _require_non_empty(self.source_path, "source_path")
        _require_non_empty(self.output_path, "output_path")
        if self.output_format not in {"json", "jsonl"}:
            raise PaperOutputError("output_format must be one of: json, jsonl.")
        object.__setattr__(self, "row_count", _non_negative_int(self.row_count, "row_count"))
        _require_non_empty(self.sha256, "sha256")
        if not isinstance(self.derived_from_artifacts, bool):
            raise PaperOutputError("derived_from_artifacts must be boolean.")
        object.__setattr__(self, "validation", _plain_json_mapping(self.validation, "validation"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "output_name": self.output_name,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "output_format": self.output_format,
            "row_count": self.row_count,
            "sha256": self.sha256,
            "derived_from_artifacts": self.derived_from_artifacts,
            "validation": _to_payload(self.validation),
        }


@dataclass(frozen=True)
class PaperOutputConfig:
    """Controls for deriving named paper outputs from artifact roots."""

    artifact_roots: Sequence[str | Path]
    output_root: str | Path
    output_specs: Sequence[PaperOutputSpec | Mapping[str, Any]] = DEFAULT_PAPER_OUTPUT_SPECS
    enforce_tolerances: bool = True
    require_traceability: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_roots", _path_tuple(self.artifact_roots, "artifact_roots"))
        if not str(self.output_root):
            raise PaperOutputError("output_root must be non-empty.")
        object.__setattr__(self, "output_root", Path(self.output_root).as_posix())
        object.__setattr__(self, "output_specs", _paper_output_specs_tuple(self.output_specs))
        if not isinstance(self.enforce_tolerances, bool):
            raise PaperOutputError("enforce_tolerances must be boolean.")
        if not isinstance(self.require_traceability, bool):
            raise PaperOutputError("require_traceability must be boolean.")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def derivation_id(self) -> str:
        return f"paper-output-derivation-{stable_config_id(self.stable_payload())}"

    def stable_payload(self) -> dict[str, Any]:
        return {
            "artifact_roots": list(self.artifact_roots),
            "output_specs": [spec.as_payload() for spec in self.output_specs],
            "enforce_tolerances": self.enforce_tolerances,
            "require_traceability": self.require_traceability,
            "metadata": _to_payload(self.metadata),
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "derivation_id": self.derivation_id,
            **self.stable_payload(),
            "output_root": self.output_root,
        }


@dataclass(frozen=True)
class PaperOutputResult:
    """Complete derivation result for paper output inputs."""

    derivation_id: str
    config: PaperOutputConfig
    outputs: tuple[PaperOutputRecord, ...]
    missing_outputs: tuple[str, ...] = ()
    validation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.derivation_id, "derivation_id")
        if not isinstance(self.config, PaperOutputConfig):
            raise PaperOutputError("config must be a PaperOutputConfig record.")
        outputs = tuple(self.outputs)
        for output in outputs:
            if not isinstance(output, PaperOutputRecord):
                raise PaperOutputError("outputs must contain PaperOutputRecord records.")
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "missing_outputs", _string_tuple(self.missing_outputs, "missing_outputs", allow_empty=True))
        object.__setattr__(self, "validation", _plain_json_mapping(self.validation, "validation"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "derivation_id": self.derivation_id,
            "version": DERIVED_OUTPUT_VERSION,
            "config": self.config.as_payload(),
            "outputs": [output.as_payload() for output in self.outputs],
            "missing_outputs": list(self.missing_outputs),
            "validation": _to_payload(self.validation),
        }


def derive_paper_outputs(config: PaperOutputConfig | Mapping[str, Any]) -> PaperOutputResult:
    """Map existing raw and metric artifacts to named paper outputs without rerunning experiments."""

    config = _coerce_config(config)
    discovered = _discover_artifacts(config.artifact_roots)
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[PaperOutputRecord] = []
    missing: list[str] = []

    for spec in config.output_specs:
        source_path = _resolve_source_path(spec, discovered)
        if source_path is None:
            if spec.required:
                missing.append(spec.output_name)
            continue
        payload = _derive_output_payload(spec.output_name, source_path)
        validation = _validate_output_payload(
            spec.output_name,
            payload,
            enforce_tolerances=config.enforce_tolerances,
            require_traceability=config.require_traceability,
        )
        output_path = output_root / f"{spec.output_name}.{spec.output_format}"
        content = _encode_output(payload, spec.output_format)
        _write_text_atomic(output_path, content)
        records.append(
            PaperOutputRecord(
                output_name=spec.output_name,
                source_path=source_path.as_posix(),
                output_path=output_path.as_posix(),
                output_format=spec.output_format,
                row_count=_row_count(payload),
                sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                validation=validation,
            )
        )

    if missing:
        raise PaperOutputError(f"Missing required paper outputs: {', '.join(sorted(missing))}.")
    result = PaperOutputResult(
        derivation_id=config.derivation_id,
        config=config,
        outputs=tuple(records),
        missing_outputs=tuple(missing),
        validation={
            "derived_without_rerun": True,
            "artifact_root_count": len(config.artifact_roots),
            "output_count": len(records),
            "output_names": [record.output_name for record in records],
        },
    )
    manifest_path = output_root / "paper_outputs_manifest.json"
    _write_text_atomic(manifest_path, json.dumps(result.as_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return result


def load_paper_output_manifest(path: str | Path) -> PaperOutputResult:
    """Load a previously written paper output manifest."""

    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "paper_outputs_manifest.json"
    if not manifest_path.exists():
        raise PaperOutputError(f"Paper output manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperOutputError(f"Could not load paper output manifest {manifest_path}: {exc}") from exc
    return _result_from_payload(_mapping(payload, "paper_output_manifest"))


def _derive_output_payload(output_name: str, source_path: Path) -> Any:
    payload = _load_artifact_payload(source_path)
    if output_name == "substitution_surface":
        return _extract_or_aggregate(payload, preferred_keys=("substitution_surface", "outcomes"), aggregate_kind="substitution")
    if output_name == "screening_table":
        return _extract_or_aggregate(payload, preferred_keys=("screening_table", "outcomes"), aggregate_kind="screening")
    key_by_name = {
        "main_qoe_table": "main_qoe_table",
        "quality_deadline_pareto": "quality_deadline_pareto",
        "deadline_hit_qoe_cdf": "deadline_hit_qoe_cdf",
        "ablation_table": "paired_ablation_table",
        "stress_matrix": "coupled_stress_matrix",
        "traceability": "claim_artifact_traceability",
        "tolerance_checks": "tolerance_checks",
    }
    key = key_by_name.get(output_name)
    if key is not None and isinstance(payload, MappingABC) and key in payload:
        return _to_payload(payload[key])
    return payload


def _extract_or_aggregate(payload: Any, *, preferred_keys: Sequence[str], aggregate_kind: str) -> Any:
    if isinstance(payload, MappingABC):
        for key in preferred_keys:
            if key in payload:
                candidate = payload[key]
                if key == "outcomes":
                    return _aggregate_outcomes(candidate, aggregate_kind)
                return _to_payload(candidate)
    if isinstance(payload, list) and aggregate_kind in {"substitution", "screening"}:
        return _aggregate_outcomes(payload, aggregate_kind)
    return payload


def _aggregate_outcomes(outcomes: Any, aggregate_kind: str) -> list[dict[str, Any]]:
    if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)):
        raise PaperOutputError(f"{aggregate_kind} outcomes must be a sequence.")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for outcome in outcomes:
        row = _mapping(outcome, f"{aggregate_kind}.outcome")
        method_id = str(row.get("method_id", "unknown"))
        grouped.setdefault(method_id, []).append(row)
    rows: list[dict[str, Any]] = []
    for method_id, method_outcomes in sorted(grouped.items()):
        if aggregate_kind == "substitution":
            rows.append(
                {
                    "method_id": method_id,
                    "substitution_gain": _mean(_numbers(method_outcomes, "substitution_gain")),
                    "utility_score": _mean(_numbers(method_outcomes, "utility_score")),
                    "budget_pressure": _mean(_numbers(method_outcomes, "budget_pressure")),
                    "sample_count": len(method_outcomes),
                }
            )
        else:
            rows.append(
                {
                    "method_id": method_id,
                    "method_selection_quality": _mean(_numbers(method_outcomes, "quality_score")),
                    "method_selection_deadline_score": _mean(_numbers(method_outcomes, "deadline_score")),
                    "method_selection_resource_efficiency": _mean(_numbers(method_outcomes, "resource_efficiency")),
                    "method_selection_runtime_ms": _mean(_numbers(method_outcomes, "runtime_ms")),
                    "method_selection_interpretability": _mean(_numbers(method_outcomes, "interpretability_score")),
                    "sample_count": len(method_outcomes),
                }
            )
    return rows


def _validate_output_payload(
    output_name: str,
    payload: Any,
    *,
    enforce_tolerances: bool,
    require_traceability: bool,
) -> dict[str, Any]:
    row_count = _row_count(payload)
    if row_count == 0:
        raise PaperOutputError(f"Derived paper output {output_name} is empty.")
    validation: dict[str, Any] = {"row_count": row_count}
    if output_name == "tolerance_checks":
        rows = _rows(payload, output_name)
        failures = [row for row in rows if row.get("tolerance_pass") is False]
        validation["tolerance_failures"] = len(failures)
        if enforce_tolerances and failures:
            raise PaperOutputError("Tolerance checks contain failures.")
    if output_name == "traceability":
        rows = _rows(payload, output_name)
        missing = [row for row in rows if not row.get("traceable", bool(row.get("artifact_ids")))]
        validation["traceability_failures"] = len(missing)
        if require_traceability and missing:
            raise PaperOutputError("Traceability output contains untraceable claims.")
    return validation


def _discover_artifacts(artifact_roots: Sequence[str]) -> dict[str, list[Path]]:
    discovered: dict[str, list[Path]] = {}
    for root_value in artifact_roots:
        root = Path(root_value)
        if not root.exists():
            raise PaperOutputError(f"Artifact root does not exist: {root}")
        if root.is_file():
            discovered.setdefault(root.name, []).append(root)
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".jsonl"}:
                discovered.setdefault(path.name, []).append(path)
    return discovered


def _resolve_source_path(spec: PaperOutputSpec, discovered: Mapping[str, Sequence[Path]]) -> Path | None:
    for filename in spec.source_filenames:
        paths = tuple(discovered.get(filename, ()))
        if paths:
            return sorted(paths, key=lambda path: path.as_posix())[0]
    return None


def _load_artifact_payload(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PaperOutputError(f"Could not read artifact {path}: {exc}") from exc
    if path.suffix == ".jsonl":
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PaperOutputError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
        return rows
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise PaperOutputError(f"Invalid JSON in {path}: {exc}") from exc


def _encode_output(payload: Any, output_format: PaperOutputFormat) -> str:
    normalized = _to_payload(payload)
    if output_format == "json":
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    rows = _rows(normalized, "jsonl_output")
    return "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows)


def _rows(payload: Any, field_name: str) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [_mapping(row, field_name) for row in payload]
    if isinstance(payload, MappingABC):
        return [_mapping(payload, field_name)]
    raise PaperOutputError(f"{field_name} must be a mapping or list of mappings.")


def _row_count(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, MappingABC):
        for key in ("rows", "outcomes", "outputs"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
        return 1
    return 1 if payload is not None else 0


def _result_from_payload(payload: Mapping[str, Any]) -> PaperOutputResult:
    raw_config = _mapping(payload.get("config"), "paper_output_manifest.config")
    config = PaperOutputConfig(
        artifact_roots=raw_config.get("artifact_roots", ()),
        output_root=raw_config.get("output_root", ""),
        output_specs=raw_config.get("output_specs", DEFAULT_PAPER_OUTPUT_SPECS),
        enforce_tolerances=bool(raw_config.get("enforce_tolerances", True)),
        require_traceability=bool(raw_config.get("require_traceability", True)),
        metadata=_mapping(raw_config.get("metadata", {}), "paper_output_manifest.config.metadata"),
    )
    outputs = tuple(PaperOutputRecord(**_mapping(item, "paper_output_manifest.outputs")) for item in _sequence(payload.get("outputs"), "outputs"))
    return PaperOutputResult(
        derivation_id=str(payload.get("derivation_id")),
        config=config,
        outputs=outputs,
        missing_outputs=tuple(str(item) for item in payload.get("missing_outputs", ())),
        validation=_mapping(payload.get("validation", {}), "paper_output_manifest.validation"),
    )


def _coerce_config(value: PaperOutputConfig | Mapping[str, Any]) -> PaperOutputConfig:
    if isinstance(value, PaperOutputConfig):
        return value
    if not isinstance(value, MappingABC):
        raise PaperOutputError("config must be a PaperOutputConfig or mapping.")
    try:
        return PaperOutputConfig(**value)
    except TypeError as exc:
        raise PaperOutputError(f"Malformed paper output config: {exc}") from exc


def _paper_output_specs_tuple(values: Sequence[PaperOutputSpec | Mapping[str, Any]]) -> tuple[PaperOutputSpec, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PaperOutputError("output_specs must be a sequence.")
    parsed: list[PaperOutputSpec] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, PaperOutputSpec):
            spec = value
        elif isinstance(value, MappingABC):
            try:
                spec = PaperOutputSpec(**value)
            except TypeError as exc:
                raise PaperOutputError(f"Malformed output spec: {exc}") from exc
        else:
            raise PaperOutputError("output_specs entries must be PaperOutputSpec records or mappings.")
        if spec.output_name in seen:
            raise PaperOutputError("output_specs must not contain duplicate output_name values.")
        parsed.append(spec)
        seen.add(spec.output_name)
    if not parsed:
        raise PaperOutputError("output_specs must not be empty.")
    return tuple(parsed)


def _path_tuple(values: Sequence[str | Path], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Path)) or not isinstance(values, Sequence):
        raise PaperOutputError(f"{field_name} must be a sequence of paths.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        path = Path(value).as_posix()
        _require_non_empty(path, field_name)
        if path not in seen:
            parsed.append(path)
            seen.add(path)
    if not parsed:
        raise PaperOutputError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _string_tuple(values: Sequence[str], field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PaperOutputError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    if not parsed and not allow_empty:
        raise PaperOutputError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise PaperOutputError(f"{field_name} must be a mapping.")
    return {str(key): _to_payload(item) for key, item in value.items()}


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise PaperOutputError(f"{field_name} must be a mapping.")
    return value


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PaperOutputError(f"{field_name} must be a sequence.")
    return value


def _numbers(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[float, ...]:
    values: list[float] = []
    for row in rows:
        if key in row:
            values.append(_finite_float(row[key], key))
    return tuple(values)


def _mean(values: Sequence[float]) -> float:
    parsed = tuple(_finite_float(value, "mean.value") for value in values)
    if not parsed:
        return 0.0
    return sum(parsed) / len(parsed)


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise PaperOutputError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PaperOutputError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise PaperOutputError(f"{field_name} must be non-negative.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise PaperOutputError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PaperOutputError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise PaperOutputError(f"{field_name} must be finite.")
    return parsed


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PaperOutputError(f"{field_name} must be a non-empty string.")


def _to_payload(value: Any) -> Any:
    if hasattr(value, "as_payload"):
        return value.as_payload()
    if isinstance(value, MappingABC):
        return {str(key): _to_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_payload(item) for item in value]
    if isinstance(value, list):
        return [_to_payload(item) for item in value]
    return value


def _write_text_atomic(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise PaperOutputError(f"Failed to write paper output {path}: {exc}") from exc


__all__ = [
    "DEFAULT_PAPER_OUTPUT_SPECS",
    "DERIVED_OUTPUT_VERSION",
    "PaperOutputConfig",
    "PaperOutputError",
    "PaperOutputRecord",
    "PaperOutputResult",
    "PaperOutputSpec",
    "derive_paper_outputs",
    "load_paper_output_manifest",
]
