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
from ref_abr.providers.pyav_decode import (
    PyAVDecodeProfilerConfig,
    PyAVDecodeProfilerError,
    PyAVDecodeProfilerProvider,
    resolve_pyav_module,
)

__all__ = [
    "CAGSAdapterConfig",
    "CAGSAdapterError",
    "CAGSBackendAdapter",
    "CAGSStageOutput",
    "ExternalSubstrateProviderConfig",
    "ExternalTraceSubstrateProvider",
    "PyAVDecodeProfilerConfig",
    "PyAVDecodeProfilerError",
    "PyAVDecodeProfilerProvider",
    "external_substrate_provider_from_mapping",
    "load_external_substrate_provider",
    "normalize_cags_candidate_kind",
    "resolve_cags_module",
    "resolve_pyav_module",
]
