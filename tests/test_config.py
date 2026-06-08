from __future__ import annotations

import json

import pytest

from ref_abr.config import ConfigError, apply_overrides, resolve_config


def test_resolve_config_produces_stable_id_independent_of_file_path(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "nested" / "second.json"
    second.parent.mkdir()
    first.write_text(
        json.dumps(
            {
                "seed": 11,
                "split": "final",
                "splits": {
                    "train": "train-a",
                    "calibration": {"identity": "cal-a"},
                    "final": "final-a",
                },
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "splits": {
                    "final": "final-a",
                    "calibration": {"identity": "cal-a"},
                    "train": "train-a",
                },
                "split": "final",
                "seed": 11,
            }
        ),
        encoding="utf-8",
    )

    first_resolved = resolve_config(first)
    second_resolved = resolve_config(second)

    assert first_resolved.config_id == second_resolved.config_id
    assert first_resolved.active_split == "final"
    assert first_resolved.splits["train"].identity == "train-a"
    assert first_resolved.splits["calibration"].seed == second_resolved.splits["calibration"].seed
    assert first_resolved.as_payload()["source_path"] == str(first)


def test_overrides_are_applied_before_seed_and_split_resolution(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
seed:
  base: 1
split: train
splits:
  train: train-a
  calibration: cal-a
  final:
    identity: final-a
""",
        encoding="utf-8",
    )

    resolved = resolve_config(
        config,
        overrides={
            "seed.base": "42",
            "splits.final.identity": "final-b",
            "splits.final.seed": "99",
        },
        split="final",
    )

    assert resolved.seed.value == 42
    assert resolved.active_split == "final"
    assert resolved.splits["final"].identity == "final-b"
    assert resolved.splits["final"].seed == 99


def test_invalid_seed_is_rejected(tmp_path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("seed = -1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="seed must be between"):
        resolve_config(config)


def test_apply_overrides_rejects_non_mapping_conflict() -> None:
    with pytest.raises(ConfigError, match="conflicts with non-mapping"):
        apply_overrides({"seed": 1}, {"seed.base": "2"})
