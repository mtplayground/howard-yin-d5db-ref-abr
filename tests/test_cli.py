from __future__ import annotations

import json

from click.testing import CliRunner

from ref_abr.cli import main
from ref_abr.entrypoints import ENTRYPOINT_VERBS


def test_root_help_lists_all_entrypoint_verbs() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    for verb in ENTRYPOINT_VERBS:
        assert verb in result.output


def test_entrypoint_dispatch_emits_structured_json() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("config.yml", "w", encoding="utf-8") as config_file:
            config_file.write(
                """
seed: 3
splits:
  train: train-a
  calibration:
    identity: cal-a
    seed: 19
  final: final-a
"""
            )

        result = runner.invoke(
            main,
            [
                "prepare_workload",
                "--config",
                "config.yml",
                "--output-dir",
                "artifacts",
                "--set",
                "seed=7",
                "--split",
                "calibration",
                "--dry-run",
                "--json",
            ],
        )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "pending"
    assert payload["payload"]["verb"] == "prepare_workload"
    assert payload["payload"]["config"] == "config.yml"
    assert payload["payload"]["output_dir"] == "artifacts"
    assert payload["payload"]["overrides"] == {"seed": "7"}
    assert payload["payload"]["split"] == "calibration"
    assert payload["payload"]["dry_run"] is True
    assert payload["payload"]["resolved_config"]["active_split"] == "calibration"
    assert payload["payload"]["resolved_config"]["seed"] == {"value": 7}
    assert payload["payload"]["resolved_config"]["splits"]["calibration"] == {
        "name": "calibration",
        "identity": "cal-a",
        "seed": 19,
    }
    assert len(payload["payload"]["resolved_config"]["config_id"]) == 16


def test_entrypoint_without_config_still_dispatches() -> None:
    result = CliRunner().invoke(
        main,
        [
            "prepare_workload",
            "--output-dir",
            "artifacts",
            "--set",
            "seed=7",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "pending"
    assert payload["payload"] == {
        "verb": "prepare_workload",
        "config": None,
        "output_dir": "artifacts",
        "overrides": {"seed": "7"},
        "dry_run": True,
    }


def test_invalid_override_is_click_error() -> None:
    result = CliRunner().invoke(main, ["prepare_workload", "--set", "seed"])

    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output
