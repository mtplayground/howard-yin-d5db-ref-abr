# External backend install and selection guide

`howard-yin-d5db-ref-abr` keeps codec, decode, transcode, and render implementations outside the core package. Optional providers call external libraries and convert their outputs into validated `external_measurement` records.

## Optional dependency extras

| Extra | Installs | Intended adapters |
| --- | --- | --- |
| `test` | `pytest` | repository test suite |
| `video` | `av`, `ffmpeg-python` | PyAV decode profiler and ffmpeg-python ladder provider |
| `render` | `gsplat` | permissive Apache-2.0 render/profiling workflows |
| `cags` | no pinned package | CAGS source installs selected by the experiment |

The default dependency list remains only `click` and `PyYAML`.

## CAGS source install

Install the CAGS module from the CAGS repository/revision used by your experiment, then install this package with its CAGS selector extra:

```bash
pip install 'cags @ git+https://github.com/<cags-owner>/<cags-repo>.git@<commit-or-tag>'
pip install -e '.[cags]'
```

The adapter imports `cags` lazily and expects these callables at runtime:

- `cags.encode(**kwargs)`
- `cags.decode(**kwargs)`
- `cags.pickup(**kwargs)`
- `cags.render(**kwargs)`

No CAGS implementation code is vendored into this repository.

## Backend selection

Use `load_substrate_provider(...)` with a provider config. The `backend` field chooses the implementation.

### Parametric default

```yaml
backend: parametric
provider_id: default-parametric
coefficients:
  base_quality: 0.55
```

### Empirical lookup table

```yaml
backend: empirical
provider_id: measured-lut
interpolation: linear
rows:
  - layer: 0
    ref_resolution: 720p
    fov_deg: 90
    view_mismatch_deg: 0
    freshness_ms: 0
    visible_quality: 0.8
    generation_ms: 1
    transfer_ms: 2
    restoration_ms: 3
    render_ms: 4
```

### External trace

```yaml
backend: external
provider_id: external-measurements
records_path: measurements.jsonl
match_policy: query
```

Each external-measurement record may include `metadata.query` so the provider can match a `SubstrateQuery` exactly. Use `match_policy: first` only for single-record smoke tests or profiling runs where every query should map to the same measurement.

## Adapter output conventions

- CAGS maps timing, `.ply` / `.drc` / compressed-directory sizes, PSNR, and LPIPS to external-measurement records for `gaussian_base`, `gaussian_enhancement`, `tile`, or `reference_action`.
- PyAV records real decode wall time (`decode_ms`) and source container size (`size_bytes`), plus stream and frame metadata.
- ffmpeg-python records wall-clock transcode time as `generation_ms` and output file size for each ladder rung.

## License policy

Prefer permissive backends for reproducible artifacts:

- PyAV: BSD-3-Clause.
- ffmpeg-python: Apache-2.0.
- gsplat / nerfstudio-family render tooling: Apache-2.0.

The graphdeco-inria 3D Gaussian Splatting reference implementation is not treated as a permissive runtime dependency in this project. Treat it as reference-only unless your project has separately cleared its license terms.
