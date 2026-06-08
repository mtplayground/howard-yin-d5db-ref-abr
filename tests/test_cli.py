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
    result = CliRunner().invoke(
        main,
        [
            "prepare_workload",
            "--config",
            "config.yml",
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
        "config": "config.yml",
        "output_dir": "artifacts",
        "overrides": {"seed": "7"},
        "dry_run": True,
    }


def test_invalid_override_is_click_error() -> None:
    result = CliRunner().invoke(main, ["prepare_workload", "--set", "seed"])

    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output
