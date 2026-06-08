from __future__ import annotations

import json

import pytest

from ref_abr.domain import MediaType, WorkloadManifest
from ref_abr.workloads import WorkloadError, assemble_workload_manifest, load_workload_manifest


def test_load_cags_scene_sequences_into_workload_manifest(tmp_path) -> None:
    metadata_path = tmp_path / "cags.yaml"
    metadata_path.write_text(
        """
cags_version: 1
dataset_name: dynamic-3dgs
scenes:
  - scene_id: coffee
    sequences:
      - sequence_id: train-000
        assets:
          - object_id: splat-000
            path: coffee/train/000.ply
            size_bytes: 100
            media_type: gaussian_splat
          - object_id: mesh-000
            path: coffee/train/mesh.obj
            size: "200"
            dependencies: [splat-000]
""",
        encoding="utf-8",
    )

    manifest = load_workload_manifest(
        metadata_path,
        config_id="cfg-a",
        split="train",
        seed=7,
        dataset_base_path=tmp_path / "dataset",
    )

    assert isinstance(manifest, WorkloadManifest)
    assert manifest.config_id == "cfg-a"
    assert manifest.split == "train"
    assert manifest.source_uri == str(metadata_path)
    assert [media.object_id for media in manifest.media_objects] == ["splat-000", "mesh-000"]
    assert manifest.media_objects[0].uri.endswith("dataset/coffee/train/000.ply")
    assert manifest.media_objects[1].media_type == MediaType.MESH
    provenance = manifest.as_payload()["metadata"]["provenance"]
    assert provenance["source_format"] == "cags"
    assert provenance["scene_ids"] == ["coffee"]


def test_n3dv_style_sequences_with_frames_normalize_to_video_segments() -> None:
    manifest = assemble_workload_manifest(
        {
            "dataset": "n3dv",
            "sequences": [
                {
                    "scene": "flame_salmon",
                    "name": "v001",
                    "frames": [
                        {"frame_id": "0001", "file": "frames/0001.mp4", "bytes": 12, "duration_ms": 33},
                        {"frame_id": "0002", "file": "frames/0002.mp4", "bytes": 13, "duration_ms": 33},
                    ],
                }
            ],
        },
        config_id="cfg-a",
        split="calibration",
        seed=11,
        source_uri="memory://n3dv",
    )

    assert [media.media_type for media in manifest.media_objects] == [
        MediaType.VIDEO_SEGMENT,
        MediaType.VIDEO_SEGMENT,
    ]
    assert manifest.media_objects[0].metadata["provenance"]["source_format"] == "n3dv"
    assert manifest.media_objects[0].duration_ms == 33
    assert manifest.as_payload()["metadata"]["provenance"]["sequence_ids"] == ["v001"]


def test_full_scene_single_path_is_normalized() -> None:
    manifest = assemble_workload_manifest(
        {
            "format": "full-scene",
            "scene_id": "longdress",
            "path": "longdress/full_scene.ply",
            "size_bytes": 99,
        },
        config_id="cfg-a",
        split="final",
        seed=5,
    )

    assert len(manifest.media_objects) == 1
    assert manifest.media_objects[0].object_id == "longdress"
    assert manifest.media_objects[0].media_type == MediaType.GAUSSIAN_SPLAT
    assert manifest.metadata["provenance"]["source_format"] == "full-scene"


def test_human_metadata_duplicate_ids_are_made_unique() -> None:
    manifest = assemble_workload_manifest(
        {
            "dataset_name": "human",
            "scene": "actor01",
            "objects": [
                {"id": "body", "uri": "body_000.ply", "size_bytes": 10},
                {"id": "body", "uri": "body_001.ply", "size_bytes": 11},
            ],
        },
        config_id="cfg-a",
        split="train",
        seed=1,
    )

    assert [media.object_id for media in manifest.media_objects] == ["body", "body#2"]
    assert manifest.metadata["provenance"]["source_format"] == "human"


def test_manifest_id_is_stable_for_equivalent_metadata() -> None:
    first = assemble_workload_manifest(
        {
            "dataset": "n3dv",
            "sequences": [{"scene": "s", "name": "q", "assets": [{"path": "a.ply", "size_bytes": 1}]}],
        },
        config_id="cfg-a",
        split="train",
        seed=1,
    )
    second = assemble_workload_manifest(
        {
            "sequences": [{"assets": [{"size_bytes": 1, "path": "a.ply"}], "name": "q", "scene": "s"}],
            "dataset": "n3dv",
        },
        config_id="cfg-a",
        split="train",
        seed=1,
    )

    assert first.manifest_id == second.manifest_id


def test_malformed_metadata_raises_clear_error(tmp_path) -> None:
    metadata_path = tmp_path / "bad.json"
    metadata_path.write_text(json.dumps({"sequences": "not-a-list"}), encoding="utf-8")

    with pytest.raises(WorkloadError, match="metadata.sequences must be a list"):
        load_workload_manifest(metadata_path, config_id="cfg-a", split="train", seed=1)

    with pytest.raises(WorkloadError, match="did not contain any media objects"):
        assemble_workload_manifest({"dataset": "empty"}, config_id="cfg-a", split="train", seed=1)
