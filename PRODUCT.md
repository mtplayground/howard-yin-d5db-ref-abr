# howard-yin-d5db-ref-abr Product Contract

## What This Project Is

`howard-yin-d5db-ref-abr` is a Python reference ABR experiment toolkit for deadline-aware reference streaming research. It models workloads, viewport and network traces, candidate objects, scheduling decisions, reference lifecycle events, resource accounting, metric computation, external measurement traces, and paper-output derivation in a deterministic, schema-stamped pipeline.

The package exposes a `ref-abr` CLI and importable `ref_abr` modules. It is intentionally lightweight by default: core runtime dependencies are only `click` and `PyYAML`; codec, transcode, decode, and render integrations are optional adapters.

## What It Does

- Loads and normalizes workload, viewport, network, device-profile, replay, and external-measurement data into typed records.
- Defines a versioned `external_measurement` schema for JSON/JSONL/YAML traces with timing, size, quality, dropped-frame, deadline-hit, artifact, and provenance fields.
- Generates candidate Gaussian/reference/tile objects from workloads and substrate-value models.
- Supports swappable substrate backends: parametric default, empirical lookup table, and external trace-backed providers.
- Provides optional thin adapters for external tooling:
  - CAGS calls `cags.encode` / `decode` / `pickup` / `render` and maps timing, compressed artifact sizes, PSNR, and LPIPS to external records.
  - PyAV profiles real container/frame decode timings and source sizes without implementing a decoder.
  - ffmpeg-python scripts offline transcode/scale/bitrate-ladder runs and records encode timing and output size without implementing an encoder.
- Runs scheduler/controller interfaces through shared observation and action budgets.
- Provides baseline and candidate methods, including greedy/cadence schedulers, canonical ABR adaptations, diagnostics, deadline-aware allocation, robust MPC, virtual-queue control, and RefABR ablation variants.
- Simulates scheduling with decision epochs, deadline tracking, lifecycle transitions, transport-aware expiration/retransmission, timing/resource accounting, and frame outcome evaluation.
- Exports raw-first artifacts with provenance, source references, schema versions, and manifest hashes.
- Computes quality, deadline-QoE, useful-resource, lifecycle, resource-cost, stability, viewport-prediction, and paired statistical-confidence metrics.
- Runs generic and specialized harnesses for substitution, lifecycle deadlines, candidate selection, full-system QoE, mechanism attribution, stress robustness, reproducibility evidence, and offline external-trace checks.
- Freezes selected methods into manifest records and derives named paper-output inputs from existing artifacts without rerunning experiments.

## Architecture And Conventions

- Domain records are dataclasses in `src/ref_abr/domain.py`; schema stamping and validation live in `src/ref_abr/schema.py`.
- External measurement loading/validation lives in `src/ref_abr/external_measurements.py`; optional adapters live under `src/ref_abr/providers/`.
- Raw artifacts use JSON/JSONL envelopes with `schema_version`, `record_type`, `provenance`, and `payload` fields.
- Metric records are exported as raw artifacts through the same manifest mechanism.
- Harnesses operate over method/workload/seed matrices and compare methods against a configured baseline using metric records.
- CLI entrypoint verbs are registered centrally. `compute_metrics` and `freeze_method` have concrete handlers; other verbs currently dispatch through the pending-entrypoint skeleton unless called through their module APIs.
- Default dependencies stay small. Optional extras are `video` for PyAV/ffmpeg-python, `render` for permissive Apache-2.0 render tooling, `cags` as a source-install selector, and `test` for pytest.
- The project does not vendor or reimplement codec, decoder, encoder, or renderer algorithms. External tools are invoked as optional dependencies and normalized into trace records.
- Prefer permissive external tooling. PyAV is BSD-3-Clause, ffmpeg-python and gsplat/nerfstudio-family tooling are Apache-2.0; the graphdeco-inria Gaussian Splatting reference implementation is documented as reference-only / not treated as a permissive runtime dependency.
- Determinism is a design goal: stable IDs, sorted payloads, seeded confidence intervals, fixture-based offline external traces, and repeatable mini-run tests are part of the contract.

## Verification Snapshot

The merged test suite covers data loading, schema validation, external measurement records, mocked CAGS/PyAV/ffmpeg adapters, external trace provider mapping, method interfaces, lifecycle transitions, schedulers, baselines, metrics, artifact export, harnesses, paper-output derivation, CLI smoke behavior, deterministic end-to-end mini-runs, and offline external-trace harness-to-metrics checks.
