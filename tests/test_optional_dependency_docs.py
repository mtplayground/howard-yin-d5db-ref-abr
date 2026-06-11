from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_default_dependencies_stay_lightweight_and_backend_extras_exist() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["dependencies"] == ["click==8.1.7", "PyYAML==6.0.2"]
    extras = pyproject["project"]["optional-dependencies"]
    assert set(extras) >= {"cags", "video", "render", "test"}
    assert extras["cags"] == []
    assert "av>=12.0.0" in extras["video"]
    assert "ffmpeg-python>=0.2.0" in extras["video"]
    assert "gsplat>=1.4.0" in extras["render"]
    assert extras["test"] == ["pytest==8.2.2"]


def test_external_backend_docs_cover_install_selection_and_license_notes() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "external_backends.md").read_text(encoding="utf-8")
    combined = f"{readme}\n{guide}"

    assert "pip install 'cags @ git+" in combined
    assert "backend: parametric" in combined
    assert "backend: empirical" in combined
    assert "backend: external" in combined
    assert "PyAV" in combined
    assert "ffmpeg-python" in combined
    assert "Apache-2.0" in combined
    assert "graphdeco-inria" in combined
    assert "reference-only" in combined
