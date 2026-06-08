"""Generic harness runner framework over raw and metric artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from ref_abr.artifacts import RawArtifactFile, RawArtifactManifest
from ref_abr.config import stable_config_id
from ref_abr.domain import MetricRecord
from ref_abr.schema import DOMAIN_SCHEMA_VERSION, RECORD_TYPE_FIELD, SCHEMA_VERSION_FIELD, materialize_record


HarnessRunMode = Literal["plan_only", "metrics_only", "full"]
HarnessRunStatus = Literal["planned", "loaded", "executed"]
HarnessExecutor = Callable[["HarnessRunSpec"], "HarnessRunResult"]


class HarnessError(ValueError):
    """Raised when harness configuration, artifacts, or comparisons are invalid."""


@dataclass(frozen=True)
class HarnessRunSpec:
    """One method/workload/seed run with fixed variables pinned."""

    run_id: str
    method_id: str
    workload_id: str
    seed: int
    run_mode: HarnessRunMode
    fixed_variables: Mapping[str, Any] = field(default_factory=dict)
    raw_artifact_root: str | None = None
    metric_artifact_root: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, "run_id")
        _require_non_empty(self.method_id, "method_id")
        _require_non_empty(self.workload_id, "workload_id")
        object.__setattr__(self, "seed", _non_negative_int(self.seed, "seed"))
        if self.run_mode not in {"plan_only", "metrics_only", "full"}:
            raise HarnessError("run_mode must be one of: plan_only, metrics_only, full.")
        for field_name in ("raw_artifact_root", "metric_artifact_root"):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_empty(value, field_name)
        object.__setattr__(self, "fixed_variables", _plain_json_mapping(self.fixed_variables, "fixed_variables"))
        object.__setattr__(self, "tags", _string_mapping(self.tags, "tags"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def run_key(self) -> str:
        return run_key(self.method_id, self.workload_id, self.seed)

    def as_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_key": self.run_key,
            "method_id": self.method_id,
            "workload_id": self.workload_id,
            "seed": self.seed,
            "run_mode": self.run_mode,
            "fixed_variables": _to_payload(self.fixed_variables),
            "raw_artifact_root": self.raw_artifact_root,
            "metric_artifact_root": self.metric_artifact_root,
            "tags": dict(self.tags),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class HarnessRunResult:
    """Artifacts and metrics observed for one harness run."""

    spec: HarnessRunSpec
    status: HarnessRunStatus
    raw_artifacts: RawArtifactManifest | None = None
    metric_artifacts: RawArtifactManifest | None = None
    metrics: Sequence[MetricRecord] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, HarnessRunSpec):
            raise HarnessError("spec must be a HarnessRunSpec record.")
        if self.status not in {"planned", "loaded", "executed"}:
            raise HarnessError("status must be one of: planned, loaded, executed.")
        if self.raw_artifacts is not None and not isinstance(self.raw_artifacts, RawArtifactManifest):
            raise HarnessError("raw_artifacts must be a RawArtifactManifest record.")
        if self.metric_artifacts is not None and not isinstance(self.metric_artifacts, RawArtifactManifest):
            raise HarnessError("metric_artifacts must be a RawArtifactManifest record.")
        object.__setattr__(self, "metrics", _coerce_metrics(self.metrics, "metrics"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "spec": self.spec.as_payload(),
            "status": self.status,
            "raw_artifacts": self.raw_artifacts.as_payload() if self.raw_artifacts is not None else None,
            "metric_artifacts": self.metric_artifacts.as_payload() if self.metric_artifacts is not None else None,
            "metrics": [metric.as_payload() for metric in self.metrics],
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class HarnessComparisonSummary:
    """Comparison rows over matched method/workload/seed metric tuples."""

    summary_id: str
    baseline_method_id: str
    metric_names: tuple[str, ...]
    group_keys: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]
    missing_pairs: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.summary_id, "summary_id")
        _require_non_empty(self.baseline_method_id, "baseline_method_id")
        object.__setattr__(self, "metric_names", _string_tuple(self.metric_names, "metric_names", allow_empty=True))
        object.__setattr__(self, "group_keys", _string_tuple(self.group_keys, "group_keys"))
        object.__setattr__(self, "rows", tuple(_plain_json_mapping(row, "rows") for row in self.rows))
        object.__setattr__(self, "missing_pairs", tuple(_plain_json_mapping(row, "missing_pairs") for row in self.missing_pairs))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "baseline_method_id": self.baseline_method_id,
            "metric_names": list(self.metric_names),
            "group_keys": list(self.group_keys),
            "rows": [_to_payload(row) for row in self.rows],
            "missing_pairs": [_to_payload(row) for row in self.missing_pairs],
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class HarnessConfig:
    """Configuration for expanding and running a generic harness."""

    harness_name: str
    methods: Sequence[str]
    workloads: Sequence[str]
    seeds: Sequence[int]
    run_mode: HarnessRunMode = "metrics_only"
    baseline_method_id: str | None = None
    fixed_variables: Mapping[str, Any] = field(default_factory=dict)
    comparison_metric_names: Sequence[str] = ()
    comparison_group_keys: Sequence[str] = ("workload_id", "seed", "metric_name")
    raw_artifact_roots: Mapping[str, str | Path] = field(default_factory=dict)
    metric_artifact_roots: Mapping[str, str | Path] = field(default_factory=dict)
    output_root: str | Path | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.harness_name, "harness_name")
        object.__setattr__(self, "methods", _string_tuple(self.methods, "methods"))
        object.__setattr__(self, "workloads", _string_tuple(self.workloads, "workloads"))
        object.__setattr__(self, "seeds", _int_tuple(self.seeds, "seeds"))
        if self.run_mode not in {"plan_only", "metrics_only", "full"}:
            raise HarnessError("run_mode must be one of: plan_only, metrics_only, full.")
        baseline = self.baseline_method_id or self.methods[0]
        _require_non_empty(baseline, "baseline_method_id")
        if baseline not in self.methods:
            raise HarnessError("baseline_method_id must be one of the configured methods.")
        object.__setattr__(self, "baseline_method_id", baseline)
        object.__setattr__(self, "fixed_variables", _plain_json_mapping(self.fixed_variables, "fixed_variables"))
        object.__setattr__(
            self,
            "comparison_metric_names",
            _string_tuple(self.comparison_metric_names, "comparison_metric_names", allow_empty=True),
        )
        object.__setattr__(self, "comparison_group_keys", _string_tuple(self.comparison_group_keys, "comparison_group_keys"))
        object.__setattr__(self, "raw_artifact_roots", _path_mapping(self.raw_artifact_roots, "raw_artifact_roots"))
        object.__setattr__(self, "metric_artifact_roots", _path_mapping(self.metric_artifact_roots, "metric_artifact_roots"))
        if self.output_root is not None:
            object.__setattr__(self, "output_root", Path(self.output_root).as_posix())
        object.__setattr__(self, "tags", _string_mapping(self.tags, "tags"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def harness_id(self) -> str:
        return f"harness-{stable_config_id(self.stable_payload())}"

    def stable_payload(self) -> dict[str, Any]:
        return {
            "harness_name": self.harness_name,
            "methods": list(self.methods),
            "workloads": list(self.workloads),
            "seeds": list(self.seeds),
            "run_mode": self.run_mode,
            "baseline_method_id": self.baseline_method_id,
            "fixed_variables": _to_payload(self.fixed_variables),
            "comparison_metric_names": list(self.comparison_metric_names),
            "comparison_group_keys": list(self.comparison_group_keys),
            "tags": dict(self.tags),
            "metadata": _to_payload(self.metadata),
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            **self.stable_payload(),
            "raw_artifact_roots": dict(self.raw_artifact_roots),
            "metric_artifact_roots": dict(self.metric_artifact_roots),
            "output_root": self.output_root,
        }


@dataclass(frozen=True)
class HarnessResult:
    """Complete result of a generic harness run."""

    harness_id: str
    config: HarnessConfig
    run_specs: tuple[HarnessRunSpec, ...]
    run_results: tuple[HarnessRunResult, ...]
    comparison_summary: HarnessComparisonSummary

    def __post_init__(self) -> None:
        _require_non_empty(self.harness_id, "harness_id")
        if not isinstance(self.config, HarnessConfig):
            raise HarnessError("config must be a HarnessConfig record.")
        run_specs = tuple(self.run_specs)
        run_results = tuple(self.run_results)
        for spec in run_specs:
            if not isinstance(spec, HarnessRunSpec):
                raise HarnessError("run_specs must contain HarnessRunSpec records.")
        for result in run_results:
            if not isinstance(result, HarnessRunResult):
                raise HarnessError("run_results must contain HarnessRunResult records.")
        if len(run_specs) != len(run_results):
            raise HarnessError("run_specs and run_results must have the same length.")
        if not isinstance(self.comparison_summary, HarnessComparisonSummary):
            raise HarnessError("comparison_summary must be a HarnessComparisonSummary record.")
        object.__setattr__(self, "run_specs", run_specs)
        object.__setattr__(self, "run_results", run_results)

    def as_payload(self) -> dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            "config": self.config.as_payload(),
            "run_specs": [spec.as_payload() for spec in self.run_specs],
            "run_results": [result.as_payload() for result in self.run_results],
            "comparison_summary": self.comparison_summary.as_payload(),
        }


def run_harness(config: HarnessConfig, *, executor: HarnessExecutor | None = None) -> HarnessResult:
    """Expand a harness matrix, execute or load runs, and summarize comparisons."""

    if not isinstance(config, HarnessConfig):
        raise HarnessError("config must be a HarnessConfig record.")
    specs = build_harness_run_specs(config)
    results = tuple(_run_one_spec(spec, executor=executor) for spec in specs)
    summary = compare_harness_results(
        results,
        baseline_method_id=config.baseline_method_id or config.methods[0],
        metric_names=config.comparison_metric_names,
        group_keys=config.comparison_group_keys,
        metadata={
            "harness_id": config.harness_id,
            "harness_name": config.harness_name,
            "fixed_variables": config.fixed_variables,
        },
    )
    harness_result = HarnessResult(
        harness_id=config.harness_id,
        config=config,
        run_specs=specs,
        run_results=results,
        comparison_summary=summary,
    )
    if config.output_root is not None:
        export_harness_result(config.output_root, harness_result)
    return harness_result


def build_harness_run_specs(config: HarnessConfig) -> tuple[HarnessRunSpec, ...]:
    """Return deterministic method/workload/seed run specs for a harness."""

    if not isinstance(config, HarnessConfig):
        raise HarnessError("config must be a HarnessConfig record.")
    specs: list[HarnessRunSpec] = []
    for workload_id in config.workloads:
        for seed in config.seeds:
            for method_id in config.methods:
                key = run_key(method_id, workload_id, seed)
                raw_root = config.raw_artifact_roots.get(key)
                metric_root = config.metric_artifact_roots.get(key)
                payload = {
                    "harness_id": config.harness_id,
                    "run_key": key,
                    "fixed_variables": config.fixed_variables,
                    "tags": config.tags,
                }
                specs.append(
                    HarnessRunSpec(
                        run_id=f"harness-run-{stable_config_id(payload)}",
                        method_id=method_id,
                        workload_id=workload_id,
                        seed=seed,
                        run_mode=config.run_mode,
                        fixed_variables=config.fixed_variables,
                        raw_artifact_root=raw_root,
                        metric_artifact_root=metric_root,
                        tags=config.tags,
                        metadata={"harness_id": config.harness_id, **config.metadata},
                    )
                )
    return tuple(specs)


def compare_harness_results(
    results: Sequence[HarnessRunResult],
    *,
    baseline_method_id: str,
    metric_names: Sequence[str] = (),
    group_keys: Sequence[str] = ("workload_id", "seed", "metric_name"),
    metadata: Mapping[str, Any] | None = None,
) -> HarnessComparisonSummary:
    """Build baseline-vs-method comparison rows over matched metric tuples."""

    run_results = _coerce_results(results)
    _require_non_empty(baseline_method_id, "baseline_method_id")
    groups = _string_tuple(group_keys, "group_keys")
    configured_metric_names = _string_tuple(metric_names, "metric_names", allow_empty=True) if metric_names else _observed_metric_names(run_results)
    metric_index = _index_metrics(run_results, configured_metric_names, groups)
    rows: list[Mapping[str, Any]] = []
    missing: list[Mapping[str, Any]] = []
    comparison_groups = sorted({key for key, values in metric_index.items() if baseline_method_id in values})
    for group in comparison_groups:
        baseline_metric = metric_index[group][baseline_method_id]
        for method_id, metric in sorted(metric_index[group].items()):
            if method_id == baseline_method_id:
                continue
            rows.append(
                {
                    **_group_payload(group, groups),
                    "baseline_method_id": baseline_method_id,
                    "method_id": method_id,
                    "baseline_value": baseline_metric.value,
                    "method_value": metric.value,
                    "delta": metric.value - baseline_metric.value,
                    "unit": metric.unit,
                    "baseline_metric_id": baseline_metric.metadata.get("metric_id"),
                    "method_metric_id": metric.metadata.get("metric_id"),
                }
            )
    for group, values in sorted(metric_index.items()):
        if baseline_method_id not in values:
            missing.append({**_group_payload(group, groups), "missing": "baseline", "baseline_method_id": baseline_method_id})
            continue
        for result in run_results:
            if result.spec.method_id not in values and _result_matches_group(result, group, groups):
                missing.append({**_group_payload(group, groups), "missing": "method", "method_id": result.spec.method_id})

    payload = {
        "baseline_method_id": baseline_method_id,
        "metric_names": configured_metric_names,
        "group_keys": groups,
        "rows": rows,
        "missing_pairs": missing,
        "metadata": _plain_json_mapping(metadata, "metadata"),
    }
    return HarnessComparisonSummary(
        summary_id=f"harness-comparison-{stable_config_id(payload)}",
        baseline_method_id=baseline_method_id,
        metric_names=configured_metric_names,
        group_keys=groups,
        rows=tuple(rows),
        missing_pairs=tuple(missing),
        metadata=_plain_json_mapping(metadata, "metadata"),
    )


def export_harness_result(output_root: str | Path, result: HarnessResult) -> Path:
    """Write a deterministic JSON harness summary and return its path."""

    if not isinstance(result, HarnessResult):
        raise HarnessError("result must be a HarnessResult record.")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "harness_result.json"
    content = json.dumps(result.as_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    temporary_path = path.with_name(".harness_result.json.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise HarnessError(f"Failed to write harness result {path}: {exc}") from exc
    return path


def run_key(method_id: str, workload_id: str, seed: int) -> str:
    """Return the stable lookup key for run artifact roots."""

    _require_non_empty(method_id, "method_id")
    _require_non_empty(workload_id, "workload_id")
    return f"{method_id}__{workload_id}__seed-{_non_negative_int(seed, 'seed')}"


def _run_one_spec(spec: HarnessRunSpec, *, executor: HarnessExecutor | None) -> HarnessRunResult:
    if spec.run_mode == "plan_only":
        return HarnessRunResult(spec=spec, status="planned")
    if executor is not None:
        result = executor(spec)
        if not isinstance(result, HarnessRunResult):
            raise HarnessError("executor must return a HarnessRunResult record.")
        if result.spec != spec:
            raise HarnessError("executor returned a HarnessRunResult for a different spec.")
        return result
    raw_manifest = _load_manifest_root(spec.raw_artifact_root) if spec.raw_artifact_root is not None else None
    metric_manifest = _load_manifest_root(spec.metric_artifact_root) if spec.metric_artifact_root is not None else None
    metrics = _load_metric_records(spec.metric_artifact_root) if spec.metric_artifact_root is not None else ()
    return HarnessRunResult(
        spec=spec,
        status="loaded",
        raw_artifacts=raw_manifest,
        metric_artifacts=metric_manifest,
        metrics=metrics,
        metadata={"artifact_loading": "metric_artifact_root" if spec.metric_artifact_root is not None else "no_metric_artifact_root"},
    )


def _load_manifest_root(root: str | Path) -> RawArtifactManifest:
    manifest_path = Path(root) / "manifest.json"
    if not manifest_path.exists():
        raise HarnessError(f"Artifact manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"Could not read artifact manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, MappingABC):
        raise HarnessError(f"Artifact manifest {manifest_path} must be a mapping.")
    files = tuple(
        RawArtifactFile(
            artifact_name=file_payload["artifact_name"],
            record_type=file_payload["record_type"],
            path=file_payload["path"],
            format=file_payload["format"],
            record_count=file_payload["record_count"],
            sha256=file_payload["sha256"],
        )
        for file_payload in _sequence_of_mappings(payload.get("files", ()), "manifest.files")
    )
    provenance_payload = payload.get("provenance")
    if not isinstance(provenance_payload, MappingABC):
        raise HarnessError("manifest.provenance must be a mapping.")
    from ref_abr.artifacts import ArtifactProvenance

    return RawArtifactManifest(
        export_id=payload["export_id"],
        output_root=payload["output_root"],
        files=files,
        provenance=ArtifactProvenance(
            run_id=provenance_payload["run_id"],
            config_id=provenance_payload.get("config_id"),
            split=provenance_payload.get("split"),
            method_id=provenance_payload.get("method_id"),
            source=provenance_payload.get("source"),
            metadata=provenance_payload.get("metadata", {}),
        ),
        schema_version=payload.get(SCHEMA_VERSION_FIELD, DOMAIN_SCHEMA_VERSION),
        metadata=payload.get("metadata", {}),
    )


def _load_metric_records(root: str | Path) -> tuple[MetricRecord, ...]:
    manifest = _load_manifest_root(root)
    metric_files = tuple(file for file in manifest.files if file.artifact_name == "metric_records")
    metrics: list[MetricRecord] = []
    for metric_file in metric_files:
        path = Path(metric_file.path)
        if not path.exists():
            path = Path(root) / path.name
        metrics.extend(_read_metric_file(path))
    return tuple(metrics)


def _read_metric_file(path: Path) -> tuple[MetricRecord, ...]:
    if not path.exists():
        raise HarnessError(f"Metric artifact file does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HarnessError(f"Could not read metric artifact file {path}: {exc}") from exc
    if not text.strip():
        return ()
    try:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            parsed = json.loads(text)
            rows = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError as exc:
        raise HarnessError(f"Could not parse metric artifact file {path}: {exc}") from exc
    metrics: list[MetricRecord] = []
    for row in rows:
        if not isinstance(row, MappingABC):
            raise HarnessError(f"Metric artifact rows in {path} must be mappings.")
        payload = row.get("payload") if isinstance(row.get("payload"), MappingABC) else row
        record = materialize_record(payload, expected_record_type="metric_record")
        if not isinstance(record, MetricRecord):
            raise HarnessError(f"Metric artifact row in {path} did not materialize to MetricRecord.")
        metrics.append(record)
    return tuple(metrics)


def _index_metrics(
    results: tuple[HarnessRunResult, ...],
    metric_names: tuple[str, ...],
    group_keys: tuple[str, ...],
) -> dict[tuple[str, ...], dict[str, MetricRecord]]:
    indexed: dict[tuple[str, ...], dict[str, MetricRecord]] = {}
    allowed = set(metric_names)
    for result in results:
        for metric in result.metrics:
            if metric.metric_name not in allowed:
                continue
            key = tuple(_comparison_value(result, metric, group_key) for group_key in group_keys)
            indexed.setdefault(key, {})[result.spec.method_id] = metric
    return indexed


def _comparison_value(result: HarnessRunResult, metric: MetricRecord, group_key: str) -> str:
    if group_key == "method_id":
        return result.spec.method_id
    if group_key == "workload_id":
        return result.spec.workload_id
    if group_key == "seed":
        return str(result.spec.seed)
    if group_key == "metric_name":
        return metric.metric_name
    if group_key == "frame_id":
        return metric.frame_id or ""
    if group_key == "split":
        return metric.split or ""
    if group_key.startswith("tag."):
        return metric.tags.get(group_key.removeprefix("tag."), "")
    if group_key.startswith("fixed."):
        value = result.spec.fixed_variables.get(group_key.removeprefix("fixed."))
        return "" if value is None else str(value)
    if group_key.startswith("metadata."):
        value = _nested_lookup(metric.metadata, group_key.removeprefix("metadata."))
        return "" if value is None else str(value)
    raise HarnessError(f"Unsupported comparison group key {group_key!r}.")


def _result_matches_group(result: HarnessRunResult, group: tuple[str, ...], group_keys: tuple[str, ...]) -> bool:
    for expected, group_key in zip(group, group_keys, strict=True):
        if group_key == "method_id":
            continue
        if group_key == "workload_id" and result.spec.workload_id != expected:
            return False
        if group_key == "seed" and str(result.spec.seed) != expected:
            return False
    return True


def _observed_metric_names(results: tuple[HarnessRunResult, ...]) -> tuple[str, ...]:
    names = sorted({metric.metric_name for result in results for metric in result.metrics})
    return tuple(names)


def _group_payload(group: tuple[str, ...], group_keys: tuple[str, ...]) -> dict[str, str]:
    return {name: value for name, value in zip(group_keys, group, strict=True)}


def _coerce_results(results: Sequence[HarnessRunResult]) -> tuple[HarnessRunResult, ...]:
    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise HarnessError("results must be a sequence of HarnessRunResult records.")
    coerced = tuple(results)
    for result in coerced:
        if not isinstance(result, HarnessRunResult):
            raise HarnessError("results must contain HarnessRunResult records.")
    return coerced


def _coerce_metrics(metrics: Sequence[MetricRecord], field_name: str) -> tuple[MetricRecord, ...]:
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence):
        raise HarnessError(f"{field_name} must be a sequence of MetricRecord records.")
    coerced = tuple(metrics)
    for metric in coerced:
        if not isinstance(metric, MetricRecord):
            raise HarnessError(f"{field_name} must contain MetricRecord records.")
    return coerced


def _sequence_of_mappings(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HarnessError(f"{field_name} must be a sequence of mappings.")
    rows = tuple(value)
    for row in rows:
        if not isinstance(row, MappingABC):
            raise HarnessError(f"{field_name} must contain mappings.")
    return rows


def _path_mapping(value: Mapping[str, str | Path], field_name: str) -> Mapping[str, str]:
    if not isinstance(value, MappingABC):
        raise HarnessError(f"{field_name} must be a mapping.")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        _require_non_empty(str(key), f"{field_name} key")
        parsed[str(key)] = Path(item).as_posix()
    return {key: parsed[key] for key in sorted(parsed)}


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise HarnessError(f"{field_name} must be a mapping.")
    return {str(key): _to_payload(item) for key, item in value.items()}


def _string_mapping(value: Mapping[str, str], field_name: str) -> Mapping[str, str]:
    if not isinstance(value, MappingABC):
        raise HarnessError(f"{field_name} must be a mapping.")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        _require_non_empty(str(key), f"{field_name} key")
        _require_non_empty(item, f"{field_name}.{key}")
        parsed[str(key)] = item
    return {key: parsed[key] for key in sorted(parsed)}


def _string_tuple(values: Sequence[str], field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise HarnessError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    if not parsed and not allow_empty:
        raise HarnessError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _int_tuple(values: Sequence[int], field_name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise HarnessError(f"{field_name} must be a sequence of integers.")
    parsed: list[int] = []
    seen: set[int] = set()
    for value in values:
        parsed_value = _non_negative_int(value, field_name)
        if parsed_value not in seen:
            parsed.append(parsed_value)
            seen.add(parsed_value)
    if not parsed:
        raise HarnessError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _nested_lookup(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    cursor: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(cursor, MappingABC):
            return None
        cursor = cursor.get(part)
    return cursor


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
        raise HarnessError(f"{field_name} must be a non-empty string.")


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise HarnessError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HarnessError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise HarnessError(f"{field_name} must be a non-negative integer.")
    return parsed


__all__ = [
    "HarnessComparisonSummary",
    "HarnessConfig",
    "HarnessError",
    "HarnessExecutor",
    "HarnessResult",
    "HarnessRunMode",
    "HarnessRunResult",
    "HarnessRunSpec",
    "build_harness_run_specs",
    "compare_harness_results",
    "export_harness_result",
    "run_harness",
    "run_key",
]
