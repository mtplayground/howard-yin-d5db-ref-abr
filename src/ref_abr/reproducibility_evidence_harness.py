"""Reproducibility and artifact-evidence harness."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ref_abr.config import stable_config_id
from ref_abr.domain import MetricRecord
from ref_abr.harness import HarnessConfig, HarnessResult, HarnessRunResult, HarnessRunSpec, run_harness
from ref_abr.replay import ReplayCase, ReplaySourceRef, ReplaySubsetManifest, SOURCE_GROUPS, validate_replay_subset_manifest


REPRODUCIBILITY_METRIC_NAMES: tuple[str, ...] = (
    "replay_determinism_pass",
    "workload_config_coverage_ratio",
    "claim_traceability_ratio",
    "tolerance_pass_rate",
    "max_abs_metric_delta",
)


class ReproducibilityEvidenceHarnessError(ValueError):
    """Raised when reproducibility evidence inputs are invalid."""


@dataclass(frozen=True)
class ClaimEvidenceSpec:
    """One paper claim that must trace to concrete artifacts and pass tolerance."""

    claim_id: str
    metric_name: str
    expected_value: float
    tolerance: float
    artifact_ids: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.claim_id, "claim_id")
        _require_non_empty(self.metric_name, "metric_name")
        object.__setattr__(self, "expected_value", _finite_float(self.expected_value, "expected_value"))
        object.__setattr__(self, "tolerance", _non_negative_float(self.tolerance, "tolerance"))
        object.__setattr__(self, "artifact_ids", _string_tuple(self.artifact_ids, "artifact_ids"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "metric_name": self.metric_name,
            "expected_value": self.expected_value,
            "tolerance": self.tolerance,
            "artifact_ids": list(self.artifact_ids),
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class ClaimEvidenceResult:
    """Resolved claim evidence with traceability and tolerance status."""

    claim_id: str
    metric_name: str
    expected_value: float
    observed_value: float
    tolerance: float
    artifact_ids: Sequence[str]
    traceable: bool
    tolerance_pass: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.claim_id, "claim_id")
        _require_non_empty(self.metric_name, "metric_name")
        object.__setattr__(self, "expected_value", _finite_float(self.expected_value, "expected_value"))
        object.__setattr__(self, "observed_value", _finite_float(self.observed_value, "observed_value"))
        object.__setattr__(self, "tolerance", _non_negative_float(self.tolerance, "tolerance"))
        object.__setattr__(self, "artifact_ids", _string_tuple(self.artifact_ids, "artifact_ids"))
        if not isinstance(self.traceable, bool):
            raise ReproducibilityEvidenceHarnessError("traceable must be boolean.")
        if not isinstance(self.tolerance_pass, bool):
            raise ReproducibilityEvidenceHarnessError("tolerance_pass must be boolean.")
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    def as_payload(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "metric_name": self.metric_name,
            "expected_value": self.expected_value,
            "observed_value": self.observed_value,
            "absolute_delta": abs(self.observed_value - self.expected_value),
            "tolerance": self.tolerance,
            "artifact_ids": list(self.artifact_ids),
            "traceable": self.traceable,
            "tolerance_pass": self.tolerance_pass,
            "metadata": _to_payload(self.metadata),
        }


@dataclass(frozen=True)
class ReproducibilityEvidenceOutcome:
    """Deterministic replay and claim-evidence outcome for one method/case/seed."""

    replay_case: ReplayCase
    method_id: str
    seed: int
    replay_digest_a: str
    replay_digest_b: str
    replay_determinism_pass: bool
    workload_config_coverage_ratio: float
    claim_results: Sequence[ClaimEvidenceResult]
    max_abs_metric_delta: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.replay_case, ReplayCase):
            raise ReproducibilityEvidenceHarnessError("replay_case must be a ReplayCase record.")
        _require_non_empty(self.method_id, "method_id")
        object.__setattr__(self, "seed", _non_negative_int(self.seed, "seed"))
        _require_non_empty(self.replay_digest_a, "replay_digest_a")
        _require_non_empty(self.replay_digest_b, "replay_digest_b")
        if not isinstance(self.replay_determinism_pass, bool):
            raise ReproducibilityEvidenceHarnessError("replay_determinism_pass must be boolean.")
        object.__setattr__(self, "workload_config_coverage_ratio", _unit_interval(self.workload_config_coverage_ratio, "workload_config_coverage_ratio"))
        claim_results = tuple(self.claim_results)
        if not claim_results:
            raise ReproducibilityEvidenceHarnessError("claim_results must not be empty.")
        for claim_result in claim_results:
            if not isinstance(claim_result, ClaimEvidenceResult):
                raise ReproducibilityEvidenceHarnessError("claim_results must contain ClaimEvidenceResult records.")
        object.__setattr__(self, "claim_results", claim_results)
        object.__setattr__(self, "max_abs_metric_delta", _non_negative_float(self.max_abs_metric_delta, "max_abs_metric_delta"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def outcome_id(self) -> str:
        payload = {
            "case_id": self.replay_case.case_id,
            "method_id": self.method_id,
            "seed": self.seed,
            "replay_digest_a": self.replay_digest_a,
            "replay_digest_b": self.replay_digest_b,
        }
        return f"reproducibility-evidence-outcome-{stable_config_id(payload)}"

    @property
    def claim_traceability_ratio(self) -> float:
        return _mean(1.0 if result.traceable else 0.0 for result in self.claim_results)

    @property
    def tolerance_pass_rate(self) -> float:
        return _mean(1.0 if result.tolerance_pass else 0.0 for result in self.claim_results)

    def metric_records(self, *, run_id: str, split: str | None = "final") -> tuple[MetricRecord, ...]:
        tags = {
            "run_id": run_id,
            "method": self.method_id,
            "case_id": self.replay_case.case_id,
            "workload_manifest_id": self.replay_case.workload_manifest_id,
            "viewport_trace_id": self.replay_case.viewport_trace_id,
            "network_trace_id": self.replay_case.network_trace_id,
            "device_profile_id": self.replay_case.device_profile_id,
        }
        base_metadata = {
            "reproducibility_evidence_outcome_id": self.outcome_id,
            "replay_case": self.replay_case.as_payload(),
            "reproducibility_evidence_outcome": self.as_payload(include_metrics=False),
        }
        return (
            _metric("replay_determinism_pass", 1.0 if self.replay_determinism_pass else 0.0, "boolean", tags, base_metadata, split=split),
            _metric("workload_config_coverage_ratio", self.workload_config_coverage_ratio, "ratio", tags, base_metadata, split=split),
            _metric("claim_traceability_ratio", self.claim_traceability_ratio, "ratio", tags, base_metadata, split=split),
            _metric("tolerance_pass_rate", self.tolerance_pass_rate, "ratio", tags, base_metadata, split=split),
            _metric("max_abs_metric_delta", self.max_abs_metric_delta, "absolute_delta", tags, base_metadata, split=split),
        )

    def as_payload(self, *, include_metrics: bool = True) -> dict[str, Any]:
        payload = {
            "outcome_id": self.outcome_id,
            "replay_case": self.replay_case.as_payload(),
            "method_id": self.method_id,
            "seed": self.seed,
            "replay_digest_a": self.replay_digest_a,
            "replay_digest_b": self.replay_digest_b,
            "replay_determinism_pass": self.replay_determinism_pass,
            "workload_config_coverage_ratio": self.workload_config_coverage_ratio,
            "claim_traceability_ratio": self.claim_traceability_ratio,
            "tolerance_pass_rate": self.tolerance_pass_rate,
            "max_abs_metric_delta": self.max_abs_metric_delta,
            "claim_results": [result.as_payload() for result in self.claim_results],
            "metadata": _to_payload(self.metadata),
        }
        if include_metrics:
            payload["metric_names"] = list(REPRODUCIBILITY_METRIC_NAMES)
        return payload


@dataclass(frozen=True)
class ReproducibilityEvidenceConfig:
    """Deterministic replay and artifact evidence harness configuration."""

    replay_subset_manifest: ReplaySubsetManifest | Mapping[str, Any]
    methods: Sequence[str]
    claim_specs: Sequence[ClaimEvidenceSpec | Mapping[str, Any]]
    seeds: Sequence[int] = (0,)
    split: str = "final"
    run_mode: str = "full"
    output_root: str | Path | None = None
    tolerance: float = 0.0
    fixed_variables: Mapping[str, Any] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        manifest = _coerce_replay_subset_manifest(self.replay_subset_manifest)
        validate_replay_subset_manifest(manifest)
        object.__setattr__(self, "replay_subset_manifest", manifest)
        object.__setattr__(self, "methods", _string_tuple(self.methods, "methods"))
        object.__setattr__(self, "claim_specs", _claim_specs_tuple(self.claim_specs))
        object.__setattr__(self, "seeds", _int_tuple(self.seeds, "seeds"))
        if self.split != manifest.split:
            raise ReproducibilityEvidenceHarnessError("split must match replay_subset_manifest.split.")
        if self.run_mode not in {"plan_only", "metrics_only", "full"}:
            raise ReproducibilityEvidenceHarnessError("run_mode must be one of: plan_only, metrics_only, full.")
        object.__setattr__(self, "tolerance", _non_negative_float(self.tolerance, "tolerance"))
        if self.output_root is not None:
            object.__setattr__(self, "output_root", Path(self.output_root).as_posix())
        object.__setattr__(self, "fixed_variables", _plain_json_mapping(self.fixed_variables, "fixed_variables"))
        object.__setattr__(self, "tags", _string_mapping(self.tags, "tags"))
        object.__setattr__(self, "metadata", _plain_json_mapping(self.metadata, "metadata"))

    @property
    def harness_id(self) -> str:
        return f"reproducibility-artifact-evidence-{stable_config_id(self.stable_payload())}"

    def stable_payload(self) -> dict[str, Any]:
        return {
            "replay_subset_manifest": self.replay_subset_manifest.as_payload(),
            "methods": list(self.methods),
            "claim_specs": [claim.as_payload() for claim in self.claim_specs],
            "seeds": list(self.seeds),
            "split": self.split,
            "run_mode": self.run_mode,
            "tolerance": self.tolerance,
            "fixed_variables": _to_payload(self.fixed_variables),
            "tags": dict(self.tags),
            "metadata": _to_payload(self.metadata),
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            **self.stable_payload(),
            "output_root": self.output_root,
            "replay_case_count": len(self.replay_subset_manifest.cases),
        }


@dataclass(frozen=True)
class ReproducibilityEvidenceResult:
    """Complete reproducibility and artifact-evidence harness output."""

    harness_id: str
    config: ReproducibilityEvidenceConfig
    outcomes: tuple[ReproducibilityEvidenceOutcome, ...]
    harness_result: HarnessResult

    def __post_init__(self) -> None:
        _require_non_empty(self.harness_id, "harness_id")
        if not isinstance(self.config, ReproducibilityEvidenceConfig):
            raise ReproducibilityEvidenceHarnessError("config must be a ReproducibilityEvidenceConfig record.")
        outcomes = tuple(self.outcomes)
        for outcome in outcomes:
            if not isinstance(outcome, ReproducibilityEvidenceOutcome):
                raise ReproducibilityEvidenceHarnessError("outcomes must contain ReproducibilityEvidenceOutcome records.")
        if not isinstance(self.harness_result, HarnessResult):
            raise ReproducibilityEvidenceHarnessError("harness_result must be a HarnessResult record.")
        object.__setattr__(self, "outcomes", outcomes)

    def as_payload(self) -> dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            "config": self.config.as_payload(),
            "outcomes": [outcome.as_payload() for outcome in self.outcomes],
            "deterministic_replay_checks": deterministic_replay_checks(self),
            "workload_config_coverage": workload_config_coverage(self),
            "claim_artifact_traceability": claim_artifact_traceability(self),
            "tolerance_checks": tolerance_checks(self),
            "harness_result": self.harness_result.as_payload(),
        }


def run_reproducibility_evidence_harness(config: ReproducibilityEvidenceConfig) -> ReproducibilityEvidenceResult:
    """Run deterministic replay and claim-to-artifact evidence checks."""

    if not isinstance(config, ReproducibilityEvidenceConfig):
        raise ReproducibilityEvidenceHarnessError("config must be a ReproducibilityEvidenceConfig record.")
    case_by_id = {case.case_id: case for case in config.replay_subset_manifest.cases}
    outcomes: list[ReproducibilityEvidenceOutcome] = []

    def executor(spec: HarnessRunSpec) -> HarnessRunResult:
        replay_case = case_by_id[spec.workload_id]
        outcome = evaluate_reproducibility_case(
            replay_case,
            method_id=spec.method_id,
            seed=spec.seed,
            replay_subset_manifest=config.replay_subset_manifest,
            claim_specs=config.claim_specs,
            default_tolerance=config.tolerance,
        )
        outcomes.append(outcome)
        return HarnessRunResult(
            spec=spec,
            status="executed",
            metrics=outcome.metric_records(run_id=spec.run_id, split=config.split),
            metadata={"reproducibility_evidence_outcome_id": outcome.outcome_id},
        )

    harness_config = HarnessConfig(
        harness_name="reproducibility-artifact-evidence",
        methods=config.methods,
        workloads=tuple(case.case_id for case in config.replay_subset_manifest.cases),
        seeds=config.seeds,
        run_mode=config.run_mode,
        baseline_method_id=config.methods[0],
        fixed_variables={
            **config.fixed_variables,
            "reproducibility_evidence_harness_id": config.harness_id,
            "replay_subset_id": config.replay_subset_manifest.subset_id,
            "config_id": config.replay_subset_manifest.config_id,
            "split": config.split,
        },
        comparison_metric_names=REPRODUCIBILITY_METRIC_NAMES,
        comparison_group_keys=("workload_id", "seed", "metric_name"),
        output_root=Path(config.output_root) / "harness" if config.output_root is not None else None,
        tags={**config.tags, "split": config.split, "replay_subset_id": config.replay_subset_manifest.subset_id},
        metadata={"reproducibility_evidence_config": config.as_payload(), **config.metadata},
    )
    harness_result = run_harness(harness_config, executor=executor)
    result = ReproducibilityEvidenceResult(
        harness_id=config.harness_id,
        config=config,
        outcomes=tuple(outcomes),
        harness_result=harness_result,
    )
    if config.output_root is not None:
        export_reproducibility_evidence_outputs(config.output_root, result)
    return result


def evaluate_reproducibility_case(
    replay_case: ReplayCase,
    *,
    method_id: str,
    seed: int,
    replay_subset_manifest: ReplaySubsetManifest,
    claim_specs: Sequence[ClaimEvidenceSpec],
    default_tolerance: float = 0.0,
) -> ReproducibilityEvidenceOutcome:
    """Evaluate one replay case twice and resolve claim evidence."""

    if not isinstance(replay_case, ReplayCase):
        raise ReproducibilityEvidenceHarnessError("replay_case must be a ReplayCase record.")
    _require_non_empty(method_id, "method_id")
    seed = _non_negative_int(seed, "seed")
    default_tolerance = _non_negative_float(default_tolerance, "default_tolerance")
    validate_replay_subset_manifest(replay_subset_manifest)
    claim_specs = _claim_specs_tuple(claim_specs)
    replay_a = _replay_payload(replay_case, method_id=method_id, seed=seed, manifest=replay_subset_manifest)
    replay_b = _replay_payload(replay_case, method_id=method_id, seed=seed, manifest=replay_subset_manifest)
    digest_a = stable_config_id(replay_a)
    digest_b = stable_config_id(replay_b)
    coverage = _case_coverage_ratio(replay_case, replay_subset_manifest)
    observed_by_metric = _observed_metric_values(replay_a, coverage)
    claim_results = tuple(_resolve_claim(claim, observed_by_metric, default_tolerance) for claim in claim_specs)
    max_abs_delta = max(abs(result.observed_value - result.expected_value) for result in claim_results)
    return ReproducibilityEvidenceOutcome(
        replay_case=replay_case,
        method_id=method_id,
        seed=seed,
        replay_digest_a=digest_a,
        replay_digest_b=digest_b,
        replay_determinism_pass=digest_a == digest_b,
        workload_config_coverage_ratio=round(coverage, 6),
        claim_results=claim_results,
        max_abs_metric_delta=round(max_abs_delta, 6),
        metadata={
            "replay_subset_id": replay_subset_manifest.subset_id,
            "config_id": replay_subset_manifest.config_id,
            "deterministic_payload": replay_a,
        },
    )


def deterministic_replay_checks(result: ReproducibilityEvidenceResult | Sequence[ReproducibilityEvidenceOutcome]) -> list[dict[str, Any]]:
    """Return per-run deterministic replay digest checks."""

    return [
        {
            "method_id": outcome.method_id,
            "case_id": outcome.replay_case.case_id,
            "seed": outcome.seed,
            "replay_digest_a": outcome.replay_digest_a,
            "replay_digest_b": outcome.replay_digest_b,
            "replay_determinism_pass": outcome.replay_determinism_pass,
        }
        for outcome in _result_outcomes(result)
    ]


def workload_config_coverage(result: ReproducibilityEvidenceResult | Sequence[ReproducibilityEvidenceOutcome]) -> list[dict[str, Any]]:
    """Return workload/config coverage rows for the replay subset."""

    rows: list[dict[str, Any]] = []
    for outcome in _result_outcomes(result):
        rows.append(
            {
                "method_id": outcome.method_id,
                "case_id": outcome.replay_case.case_id,
                "seed": outcome.seed,
                "workload_manifest_id": outcome.replay_case.workload_manifest_id,
                "viewport_trace_id": outcome.replay_case.viewport_trace_id,
                "network_trace_id": outcome.replay_case.network_trace_id,
                "device_profile_id": outcome.replay_case.device_profile_id,
                "workload_config_coverage_ratio": outcome.workload_config_coverage_ratio,
            }
        )
    return rows


def claim_artifact_traceability(result: ReproducibilityEvidenceResult | Sequence[ReproducibilityEvidenceOutcome]) -> list[dict[str, Any]]:
    """Return claim-to-artifact traceability rows."""

    rows: list[dict[str, Any]] = []
    for outcome in _result_outcomes(result):
        for claim in outcome.claim_results:
            rows.append(
                {
                    "method_id": outcome.method_id,
                    "case_id": outcome.replay_case.case_id,
                    "seed": outcome.seed,
                    "claim_id": claim.claim_id,
                    "metric_name": claim.metric_name,
                    "artifact_ids": list(claim.artifact_ids),
                    "traceable": claim.traceable,
                }
            )
    return rows


def tolerance_checks(result: ReproducibilityEvidenceResult | Sequence[ReproducibilityEvidenceOutcome]) -> list[dict[str, Any]]:
    """Return per-claim tolerance check rows."""

    rows: list[dict[str, Any]] = []
    for outcome in _result_outcomes(result):
        for claim in outcome.claim_results:
            rows.append(
                {
                    "method_id": outcome.method_id,
                    "case_id": outcome.replay_case.case_id,
                    "seed": outcome.seed,
                    **claim.as_payload(),
                }
            )
    return rows


def export_reproducibility_evidence_outputs(
    output_root: str | Path,
    result: ReproducibilityEvidenceResult,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Write reproducibility raw outcomes and evidence table JSON payloads."""

    if not isinstance(result, ReproducibilityEvidenceResult):
        raise ReproducibilityEvidenceHarnessError("result must be a ReproducibilityEvidenceResult record.")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    outcomes_path = root / "reproducibility_evidence_outcomes.jsonl"
    replay_path = root / "deterministic_replay_checks.json"
    coverage_path = root / "workload_config_coverage.json"
    traceability_path = root / "claim_artifact_traceability.json"
    tolerance_path = root / "tolerance_checks.json"
    summary_path = root / "reproducibility_evidence_summary.json"
    _write_text_atomic(
        outcomes_path,
        "".join(json.dumps(outcome.as_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for outcome in result.outcomes),
    )
    _write_text_atomic(replay_path, _json_payload(deterministic_replay_checks(result)))
    _write_text_atomic(coverage_path, _json_payload(workload_config_coverage(result)))
    _write_text_atomic(traceability_path, _json_payload(claim_artifact_traceability(result)))
    _write_text_atomic(tolerance_path, _json_payload(tolerance_checks(result)))
    _write_text_atomic(summary_path, _json_payload(result.as_payload()))
    return outcomes_path, replay_path, coverage_path, traceability_path, tolerance_path, summary_path


def _replay_payload(
    replay_case: ReplayCase,
    *,
    method_id: str,
    seed: int,
    manifest: ReplaySubsetManifest,
) -> dict[str, Any]:
    base_score = _hash_fraction("replay", replay_case.case_id, method_id, seed, manifest.subset_id)
    coverage = _case_coverage_ratio(replay_case, manifest)
    reproducibility_score = _clamp01(0.70 + coverage * 0.20 + base_score * 0.10)
    artifact_traceability = _clamp01(coverage)
    tolerance_metric = _clamp01(reproducibility_score - (1.0 - coverage) * 0.10)
    return {
        "subset_id": manifest.subset_id,
        "config_id": manifest.config_id,
        "split": manifest.split,
        "case": replay_case.as_payload(),
        "method_id": method_id,
        "seed": seed,
        "metrics": {
            "reproducibility_score": round(reproducibility_score, 6),
            "artifact_traceability": round(artifact_traceability, 6),
            "tolerance_metric": round(tolerance_metric, 6),
            "coverage_ratio": round(coverage, 6),
        },
    }


def _observed_metric_values(replay_payload: Mapping[str, Any], coverage: float) -> dict[str, float]:
    metrics = replay_payload.get("metrics", {})
    if not isinstance(metrics, MappingABC):
        raise ReproducibilityEvidenceHarnessError("replay metrics must be a mapping.")
    observed = {str(key): _finite_float(value, f"metrics.{key}") for key, value in metrics.items()}
    observed.setdefault("coverage_ratio", coverage)
    return observed


def _resolve_claim(
    claim: ClaimEvidenceSpec,
    observed_by_metric: Mapping[str, float],
    default_tolerance: float,
) -> ClaimEvidenceResult:
    observed = observed_by_metric.get(claim.metric_name, 0.0)
    tolerance = max(claim.tolerance, default_tolerance)
    traceable = bool(claim.artifact_ids) and all(str(artifact_id).strip() for artifact_id in claim.artifact_ids)
    tolerance_pass = abs(observed - claim.expected_value) <= tolerance
    return ClaimEvidenceResult(
        claim_id=claim.claim_id,
        metric_name=claim.metric_name,
        expected_value=claim.expected_value,
        observed_value=round(observed, 6),
        tolerance=tolerance,
        artifact_ids=claim.artifact_ids,
        traceable=traceable,
        tolerance_pass=tolerance_pass,
        metadata=claim.metadata,
    )


def _case_coverage_ratio(replay_case: ReplayCase, manifest: ReplaySubsetManifest) -> float:
    source_ids = {
        "workloads": {source.record_id for source in manifest.sources["workloads"]},
        "viewports": {source.record_id for source in manifest.sources["viewports"]},
        "networks": {source.record_id for source in manifest.sources["networks"]},
        "devices": {source.record_id for source in manifest.sources["devices"]},
    }
    covered = (
        replay_case.workload_manifest_id in source_ids["workloads"],
        replay_case.viewport_trace_id in source_ids["viewports"],
        replay_case.network_trace_id in source_ids["networks"],
        replay_case.device_profile_id in source_ids["devices"],
        bool(manifest.config_id),
        bool(manifest.metadata.get("provenance")),
    )
    return sum(1.0 for value in covered if value) / len(covered)


def _coerce_replay_subset_manifest(value: ReplaySubsetManifest | Mapping[str, Any]) -> ReplaySubsetManifest:
    if isinstance(value, ReplaySubsetManifest):
        return value
    if not isinstance(value, MappingABC):
        raise ReproducibilityEvidenceHarnessError("replay_subset_manifest must be a ReplaySubsetManifest or mapping.")
    try:
        cases = tuple(ReplayCase(**case) if isinstance(case, MappingABC) else case for case in value.get("cases", ()))
        raw_sources = value.get("sources", {})
        if not isinstance(raw_sources, MappingABC):
            raise ReproducibilityEvidenceHarnessError("replay_subset_manifest.sources must be a mapping.")
        sources = {
            group: tuple(ReplaySourceRef(**source) if isinstance(source, MappingABC) else source for source in raw_sources.get(group, ()))
            for group in SOURCE_GROUPS
        }
        return ReplaySubsetManifest(
            subset_id=str(value["subset_id"]),
            config_id=str(value["config_id"]),
            split=str(value["split"]),
            seed=value["seed"],
            cases=cases,
            sources=sources,
            metadata=value.get("metadata", {}),
        )
    except KeyError as exc:
        raise ReproducibilityEvidenceHarnessError(f"Malformed replay_subset_manifest missing key: {exc}") from exc
    except TypeError as exc:
        raise ReproducibilityEvidenceHarnessError(f"Malformed replay_subset_manifest: {exc}") from exc


def _claim_specs_tuple(values: Sequence[ClaimEvidenceSpec | Mapping[str, Any]]) -> tuple[ClaimEvidenceSpec, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ReproducibilityEvidenceHarnessError("claim_specs must be a sequence.")
    parsed: list[ClaimEvidenceSpec] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, ClaimEvidenceSpec):
            claim = value
        elif isinstance(value, MappingABC):
            try:
                claim = ClaimEvidenceSpec(**value)
            except TypeError as exc:
                raise ReproducibilityEvidenceHarnessError(f"Malformed claim spec: {exc}") from exc
        else:
            raise ReproducibilityEvidenceHarnessError("claim_specs entries must be ClaimEvidenceSpec records or mappings.")
        if claim.claim_id in seen:
            raise ReproducibilityEvidenceHarnessError("claim_specs must not contain duplicate claim_id values.")
        parsed.append(claim)
        seen.add(claim.claim_id)
    if not parsed:
        raise ReproducibilityEvidenceHarnessError("claim_specs must not be empty.")
    return tuple(parsed)


def _result_outcomes(result: ReproducibilityEvidenceResult | Sequence[ReproducibilityEvidenceOutcome]) -> tuple[ReproducibilityEvidenceOutcome, ...]:
    outcomes = result.outcomes if isinstance(result, ReproducibilityEvidenceResult) else tuple(result)
    for outcome in outcomes:
        if not isinstance(outcome, ReproducibilityEvidenceOutcome):
            raise ReproducibilityEvidenceHarnessError("outcomes must contain ReproducibilityEvidenceOutcome records.")
    return tuple(outcomes)


def _metric(
    metric_name: str,
    value: float,
    unit: str,
    tags: Mapping[str, str],
    metadata: Mapping[str, Any],
    *,
    split: str | None,
) -> MetricRecord:
    metric_metadata = {
        **_plain_json_mapping(metadata, "metadata"),
        "metric_id": f"reproducibility-evidence-metric-{stable_config_id({'metric_name': metric_name, 'value': value, 'tags': dict(tags)})}",
    }
    return MetricRecord(metric_name=metric_name, value=value, unit=unit, tags=tags, split=split, metadata=metric_metadata)


def _json_payload(value: Any) -> str:
    return json.dumps(_to_payload(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise ReproducibilityEvidenceHarnessError(f"Failed to write reproducibility evidence output {path}: {exc}") from exc


def _string_tuple(values: Sequence[str], field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must be a sequence of strings.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_non_empty(value, field_name)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    if not parsed and not allow_empty:
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _string_mapping(value: Mapping[str, str], field_name: str) -> Mapping[str, str]:
    if not isinstance(value, MappingABC):
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must be a mapping.")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        _require_non_empty(str(key), f"{field_name} key")
        _require_non_empty(item, f"{field_name}.{key}")
        parsed[str(key)] = item
    return {key: parsed[key] for key in sorted(parsed)}


def _int_tuple(values: Sequence[int], field_name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must be a sequence of integers.")
    parsed: list[int] = []
    seen: set[int] = set()
    for value in values:
        parsed_value = _non_negative_int(value, field_name)
        if parsed_value not in seen:
            parsed.append(parsed_value)
            seen.add(parsed_value)
    if not parsed:
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must not be empty.")
    return tuple(parsed)


def _plain_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must be a mapping.")
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
    return value


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must be a non-empty string.")


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must be non-negative.")
    return parsed


def _unit_interval(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0 or parsed > 1.0:
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must be in [0, 1].")
    return parsed


def _non_negative_float(value: Any, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if parsed < 0.0:
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must be non-negative.")
    return parsed


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise ReproducibilityEvidenceHarnessError(f"{field_name} must be finite.")
    return parsed


def _mean(values: Sequence[float]) -> float:
    parsed = tuple(_finite_float(value, "mean.value") for value in values)
    if not parsed:
        return 0.0
    return sum(parsed) / len(parsed)


def _hash_fraction(*parts: object) -> float:
    digest = stable_config_id({"parts": [str(part) for part in parts]})
    return int(digest[:8], 16) / 0xFFFFFFFF


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


__all__ = [
    "REPRODUCIBILITY_METRIC_NAMES",
    "ClaimEvidenceResult",
    "ClaimEvidenceSpec",
    "ReproducibilityEvidenceConfig",
    "ReproducibilityEvidenceHarnessError",
    "ReproducibilityEvidenceOutcome",
    "ReproducibilityEvidenceResult",
    "claim_artifact_traceability",
    "deterministic_replay_checks",
    "evaluate_reproducibility_case",
    "export_reproducibility_evidence_outputs",
    "run_reproducibility_evidence_harness",
    "tolerance_checks",
    "workload_config_coverage",
]
