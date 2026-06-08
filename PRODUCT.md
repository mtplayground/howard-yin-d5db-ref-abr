# howard-yin-d5db-ref-abr Product Contract

## What This Project Is

`howard-yin-d5db-ref-abr` is a Python reference ABR experiment toolkit for deadline-aware reference streaming research. It models workloads, viewport and network traces, candidate objects, scheduling decisions, reference lifecycle events, resource accounting, metric computation, and paper-output derivation in a deterministic, schema-stamped pipeline.

The package exposes a `ref-abr` CLI and importable `ref_abr` modules. It is intentionally lightweight: runtime dependencies are `click` and `PyYAML`; tests use `pytest`.

## What It Does

- Loads and normalizes workload, viewport, network, and device-profile data into typed domain records.
- Generates candidate Gaussian/reference/tile objects from workloads and substrate-value models.
- Runs scheduler/controller interfaces through shared observation and action budgets.
- Provides baseline and candidate methods, including simple greedy/cadence schedulers, canonical ABR adaptations, diagnostics, deadline-aware allocation, robust MPC, virtual-queue control, and RefABR ablation variants.
- Simulates scheduling with decision epochs, deadline tracking, lifecycle transitions, transport-aware expiration/retransmission, timing/resource accounting, and frame outcome evaluation.
- Exports raw-first artifacts before aggregation with provenance, source references, schema versions, and manifest hashes.
- Computes quality, deadline-QoE, useful-resource, lifecycle, resource-cost, stability, viewport-prediction, and paired statistical-confidence metrics.
- Runs generic and specialized harnesses for substitution, lifecycle deadlines, candidate selection, full-system QoE, mechanism attribution, stress robustness, and reproducibility evidence.
- Freezes selected methods into manifest records and derives named paper-output inputs from existing artifacts without rerunning experiments.

## Architecture And Conventions

- Domain records are dataclasses in `src/ref_abr/domain.py`; schema stamping and validation live in `src/ref_abr/schema.py`.
- Raw artifacts use JSON/JSONL envelopes with `schema_version`, `record_type`, `provenance`, and `payload` fields.
- Metric records are exported as raw artifacts through the same manifest mechanism.
- Harnesses operate over method/workload/seed matrices and compare methods against a configured baseline using metric records.
- CLI entrypoint verbs are registered centrally. `compute_metrics` and `freeze_method` have concrete handlers; other verbs currently dispatch through the pending-entrypoint skeleton unless called through their module APIs.
- Determinism is a design goal: stable IDs, sorted payloads, seeded confidence intervals, and repeatable mini-run tests are part of the contract.

## Verification Snapshot

The merged test suite covers data loading, schema validation, method interfaces, lifecycle transitions, schedulers, baselines, metrics, artifact export, harnesses, paper-output derivation, CLI smoke behavior, and deterministic end-to-end mini-runs.
