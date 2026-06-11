"""Optional and trace-backed substrate provider adapters."""

from ref_abr.providers.base import (
    ExternalSubstrateProviderConfig,
    ExternalTraceSubstrateProvider,
    external_substrate_provider_from_mapping,
    load_external_substrate_provider,
)
from ref_abr.providers.cags import (
    CAGSAdapterConfig,
    CAGSAdapterError,
    CAGSBackendAdapter,
    CAGSStageOutput,
    normalize_cags_candidate_kind,
    resolve_cags_module,
)

__all__ = [
    "CAGSAdapterConfig",
    "CAGSAdapterError",
    "CAGSBackendAdapter",
    "CAGSStageOutput",
    "ExternalSubstrateProviderConfig",
    "ExternalTraceSubstrateProvider",
    "external_substrate_provider_from_mapping",
    "load_external_substrate_provider",
    "normalize_cags_candidate_kind",
    "resolve_cags_module",
]
