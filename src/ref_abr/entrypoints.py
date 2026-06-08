"""Uniform entrypoint registry used by the Click CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


EntrypointHandler = Callable[["EntrypointInvocation"], "EntrypointResult"]


ENTRYPOINT_VERBS: tuple[str, ...] = (
    "prepare_workload",
    "normalize_viewport_trace",
    "normalize_network_trace",
    "assemble_replay_subset",
    "plan_schedule",
    "compute_metrics",
    "freeze_method",
    "derive_paper_outputs",
)


@dataclass(frozen=True)
class EntrypointInvocation:
    """Request passed from the CLI to an entrypoint handler."""

    verb: str
    config: Path | None = None
    output_dir: Path | None = None
    overrides: Mapping[str, str] = field(default_factory=dict)
    split: str | None = None
    resolved_config: Mapping[str, Any] | None = None
    dry_run: bool = False

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "verb": self.verb,
            "config": str(self.config) if self.config is not None else None,
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
            "overrides": dict(self.overrides),
            "dry_run": self.dry_run,
        }
        if self.split is not None:
            payload["split"] = self.split
        if self.resolved_config is not None:
            payload["resolved_config"] = dict(self.resolved_config)
        return payload


@dataclass(frozen=True)
class EntrypointResult:
    """Structured result returned by an entrypoint handler."""

    status: str
    message: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "payload": dict(self.payload),
        }


class EntrypointRegistry:
    """Resolve known entrypoint verbs to callable handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, EntrypointHandler] = {}

    @property
    def verbs(self) -> tuple[str, ...]:
        return ENTRYPOINT_VERBS

    @property
    def handlers(self) -> Mapping[str, EntrypointHandler]:
        return MappingProxyType(self._handlers)

    def register(self, verb: str, handler: EntrypointHandler) -> None:
        if verb not in ENTRYPOINT_VERBS:
            valid = ", ".join(ENTRYPOINT_VERBS)
            raise ValueError(f"Unknown entrypoint verb '{verb}'. Expected one of: {valid}.")
        self._handlers[verb] = handler

    def resolve(self, verb: str) -> EntrypointHandler:
        try:
            return self._handlers[verb]
        except KeyError as exc:
            valid = ", ".join(sorted(self._handlers))
            raise KeyError(f"Entrypoint '{verb}' is not registered. Registered verbs: {valid}.") from exc

    def dispatch(self, invocation: EntrypointInvocation) -> EntrypointResult:
        handler = self.resolve(invocation.verb)
        return handler(invocation)


def parse_overrides(values: Sequence[str]) -> dict[str, str]:
    """Parse repeated KEY=VALUE arguments into a deterministic mapping."""

    overrides: dict[str, str] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if not separator or not key:
            raise ValueError(f"Override '{raw}' must use KEY=VALUE format.")
        overrides[key] = value
    return dict(sorted(overrides.items()))


def pending_entrypoint_handler(invocation: EntrypointInvocation) -> EntrypointResult:
    """Return a valid skeleton response for entrypoints implemented by later issues."""

    return EntrypointResult(
        status="pending",
        message=f"Entrypoint '{invocation.verb}' resolved and dispatched; implementation is scheduled for a later issue.",
        payload=invocation.as_payload(),
    )


def build_default_registry() -> EntrypointRegistry:
    registry = EntrypointRegistry()
    for verb in ENTRYPOINT_VERBS:
        registry.register(verb, pending_entrypoint_handler)
    return registry
