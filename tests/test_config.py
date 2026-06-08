from __future__ import annotations

import json
from pathlib import Path

import pytest

from ref_abr.config import (
    REQUIRED_ENV_KEYS,
    ConfigError,
    apply_overrides,
    load_env_config,
    load_env_file,
    resolve_config,
)


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


def test_load_env_config_uses_defaults() -> None:
    env_config = load_env_config(environ={})

    assert env_config.as_payload() == {
        "artifact_output_root": "artifacts",
        "dataset_base_path": "data/datasets",
        "trace_base_path": "data/traces",
        "default_run_name": "default",
        "default_seed": 0,
        "default_split": "train",
        "max_workers": 1,
        "overwrite_outputs": False,
        "log_level": "INFO",
    }


def test_load_env_config_reads_env_file_and_environment_override(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
REF_ABR_ARTIFACT_ROOT=/tmp/artifacts
REF_ABR_DATASET_BASE_PATH=/tmp/datasets
REF_ABR_TRACE_BASE_PATH=/tmp/traces
REF_ABR_DEFAULT_RUN_NAME='nightly'
REF_ABR_DEFAULT_SEED=123
REF_ABR_DEFAULT_SPLIT=calibration
REF_ABR_MAX_WORKERS=4
REF_ABR_OVERWRITE_OUTPUTS=yes
REF_ABR_LOG_LEVEL=debug
""",
        encoding="utf-8",
    )

    env_config = load_env_config(
        environ={"REF_ABR_DEFAULT_SEED": "321"},
        env_file=env_file,
    )

    assert env_config.artifact_output_root.as_posix() == "/tmp/artifacts"
    assert env_config.dataset_base_path.as_posix() == "/tmp/datasets"
    assert env_config.trace_base_path.as_posix() == "/tmp/traces"
    assert env_config.default_run_name == "nightly"
    assert env_config.default_seed == 321
    assert env_config.default_split == "calibration"
    assert env_config.max_workers == 4
    assert env_config.overwrite_outputs is True
    assert env_config.log_level == "DEBUG"


def test_load_env_config_rejects_invalid_values() -> None:
    with pytest.raises(ConfigError, match="REF_ABR_DEFAULT_SPLIT"):
        load_env_config(environ={"REF_ABR_DEFAULT_SPLIT": "holdout"})

    with pytest.raises(ConfigError, match="REF_ABR_MAX_WORKERS"):
        load_env_config(environ={"REF_ABR_MAX_WORKERS": "0"})

    with pytest.raises(ConfigError, match="REF_ABR_OVERWRITE_OUTPUTS"):
        load_env_config(environ={"REF_ABR_OVERWRITE_OUTPUTS": "sometimes"})


def test_load_env_file_rejects_malformed_lines(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("REF_ABR_DEFAULT_SEED\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="KEY=VALUE"):
        load_env_file(env_file)


def test_env_example_lists_all_required_keys() -> None:
    example_text = (Path(__file__).parents[1] / ".env.example").read_text(encoding="utf-8")

    for key in REQUIRED_ENV_KEYS:
        assert f"{key}=" in example_text
