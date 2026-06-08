"""Click command line interface for the reference ABR toolkit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from ref_abr import __version__
from ref_abr.entrypoints import (
    ENTRYPOINT_VERBS,
    EntrypointInvocation,
    EntrypointRegistry,
    build_default_registry,
    parse_overrides,
)


class CliState:
    """Mutable Click context object containing process-local services."""

    def __init__(self, registry: EntrypointRegistry | None = None) -> None:
        self.registry = registry or build_default_registry()


def _format_result(result_payload: dict[str, Any], as_json: bool) -> str:
    if as_json:
        return json.dumps(result_payload, sort_keys=True)
    return result_payload["message"]


def _dispatch_command(
    ctx: click.Context,
    verb: str,
    config: str | None,
    output_dir: str | None,
    overrides: tuple[str, ...],
    dry_run: bool,
    as_json: bool,
) -> None:
    state = ctx.find_object(CliState)
    if state is None:
        state = CliState()
        ctx.obj = state

    try:
        parsed_overrides = parse_overrides(overrides)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--set") from exc

    invocation = EntrypointInvocation(
        verb=verb,
        config=Path(config) if config is not None else None,
        output_dir=Path(output_dir) if output_dir is not None else None,
        overrides=parsed_overrides,
        dry_run=dry_run,
    )
    result = state.registry.dispatch(invocation)
    click.echo(_format_result(result.as_payload(), as_json))


def _entrypoint_command(verb: str) -> click.Command:
    @click.command(name=verb)
    @click.option(
        "--config",
        "config",
        type=click.Path(dir_okay=False, path_type=str),
        help="Path to an entrypoint configuration file.",
    )
    @click.option(
        "--output-dir",
        "output_dir",
        type=click.Path(file_okay=False, path_type=str),
        help="Directory where the entrypoint should write artifacts.",
    )
    @click.option(
        "--set",
        "overrides",
        multiple=True,
        metavar="KEY=VALUE",
        help="Configuration override. May be provided more than once.",
    )
    @click.option(
        "--dry-run",
        is_flag=True,
        help="Resolve and validate routing without writing artifacts.",
    )
    @click.option(
        "--json",
        "as_json",
        is_flag=True,
        help="Emit the structured entrypoint result as JSON.",
    )
    @click.pass_context
    def command(
        ctx: click.Context,
        config: str | None,
        output_dir: str | None,
        overrides: tuple[str, ...],
        dry_run: bool,
        as_json: bool,
    ) -> None:
        _dispatch_command(ctx, verb, config, output_dir, overrides, dry_run, as_json)

    command.help = f"Resolve and dispatch the {verb} entrypoint."
    return command


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="ref-abr")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Reference ABR experiment toolkit."""

    ctx.obj = ctx.obj or CliState()


for _verb in ENTRYPOINT_VERBS:
    main.add_command(_entrypoint_command(_verb))


__all__ = ["CliState", "main"]
