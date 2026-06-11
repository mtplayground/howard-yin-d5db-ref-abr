# howard-yin-d5db-ref-abr

`howard-yin-d5db-ref-abr` is a lightweight Python reference ABR experiment toolkit for deadline-aware reference streaming research. The default install intentionally depends only on `click` and `PyYAML`; codec, decoder, transcoder, and renderer integrations are optional adapters that delegate to external open-source projects.

## Install

Core package:

```bash
pip install -e .
```

Test tools:

```bash
pip install -e '.[test]'
pytest
```

Video profiling adapters for PyAV and ffmpeg-python:

```bash
pip install -e '.[video]'
```

Render/profiling helpers for Apache-2.0 Gaussian-splat tooling:

```bash
pip install -e '.[render]'
```

CAGS is expected as a separate source install because deployments may use different forks or revisions. Install the project-provided CAGS repository first, then install this package:

```bash
pip install 'cags @ git+https://github.com/<cags-owner>/<cags-repo>.git@<commit-or-tag>'
pip install -e '.[cags]'
```

The `cags` extra is present so environments can consistently request CAGS support, but the actual CAGS module is resolved at runtime as an optional dependency. The adapters fail loudly with an install hint if `cags`, `av`, or `ffmpeg` is requested but unavailable.

## External measurement backends

The external backend path is trace-first: adapters invoke external libraries and emit versioned external-measurement records. `ref_abr` then validates those records and maps them into `SubstrateValue` / `ComponentTiming` for existing scheduling and metric code.

Supported backend selectors:

- `backend: parametric` — default analytic substrate model.
- `backend: empirical` — lookup-table substrate model.
- `backend: external` / `external-trace` — validated JSON/JSONL/YAML measurement records.

Example external trace provider config:

```yaml
backend: external
provider_id: measured-trace
records_path: measurements.jsonl
match_policy: query
uncertainty:
  quality_stddev: 0.02
  timing_stddev_ms: 0.5
  confidence: 0.95
```

Load it from Python:

```python
from ref_abr.substrate import load_substrate_provider

provider = load_substrate_provider("provider.yml")
value = provider.evaluate({
    "layer": 0,
    "ref_resolution": "720p",
    "fov_deg": 90,
    "view_mismatch_deg": 0,
    "freshness_ms": 0,
})
```

## Optional adapter usage

### CAGS 3DGS codec/render adapter

The CAGS adapter calls `cags.encode`, `cags.decode`, `cags.pickup`, and `cags.render`; it does not copy or reimplement any CAGS algorithm code.

```python
from ref_abr.providers import CAGSAdapterConfig, CAGSBackendAdapter

adapter = CAGSBackendAdapter(
    config=CAGSAdapterConfig(
        source_uri="file://scene",
        output_dir="artifacts/cags/scene-a",
        candidate_kind="gaussian_base",
        query={
            "layer": 0,
            "ref_resolution": "720p",
            "fov_deg": 90,
            "view_mismatch_deg": 0,
            "freshness_ms": 0,
        },
    )
)
record = adapter.run_measurement()
```

CAGS outputs such as timing fields, compressed `.ply` / `.drc` files, PSNR, and LPIPS are normalized into external-measurement records mapped to `gaussian_base`, `gaussian_enhancement`, `tile`, or `reference_action`.

### PyAV decode profiler

```python
from ref_abr.providers import PyAVDecodeProfilerConfig, PyAVDecodeProfilerProvider

provider = PyAVDecodeProfilerProvider(
    config=PyAVDecodeProfilerConfig(
        source_uri="segment.mp4",
        max_frames=30,
        query={
            "layer": 0,
            "ref_resolution": "720p",
            "fov_deg": 90,
            "view_mismatch_deg": 0,
            "freshness_ms": 0,
        },
    )
)
records = provider.profile_records()
```

PyAV performs the real container/frame decode. The adapter records `decode_ms`, `size_bytes`, stream metadata, and frame metadata.

### ffmpeg-python ladder provider

```python
from ref_abr.providers import FFMpegLadderConfig, FFMpegLadderTraceProvider

provider = FFMpegLadderTraceProvider(
    config=FFMpegLadderConfig(
        source_uri="input.mp4",
        output_dir="artifacts/ladder",
        rungs=[
            {"rung_id": "720p", "width_px": 1280, "height_px": 720, "bitrate": "2500k"},
            {"rung_id": "480p", "width_px": 854, "height_px": 480, "bitrate": "1200k"},
        ],
    )
)
records = provider.profile_records()
```

ffmpeg-python scripts the real ffmpeg transcode/scale work. The adapter records encode timing and generated output size for each ladder rung.

## Licensing notes

- PyAV is BSD-3-Clause.
- ffmpeg-python is Apache-2.0.
- gsplat / nerfstudio-family render tooling is Apache-2.0 and is the preferred permissive render integration path.
- CAGS should be installed from a repository/revision whose license is compatible with your experiment or deployment.
- The original graphdeco-inria 3D Gaussian Splatting reference implementation is not treated as a permissive dependency here; use it as reference-only unless your project has separately cleared its license terms.

