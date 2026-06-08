from __future__ import annotations

import pytest

from ref_abr.network import (
    NetworkBoundaryConfig,
    NetworkError,
    generate_synthetic_boundary_traces,
    load_network_trace,
    normalize_network_trace,
)


def test_normalize_mobile_network_trace_units_and_sorting() -> None:
    trace = normalize_network_trace(
        {
            "network_type": "LTE",
            "samples": [
                {"time_s": 0.2, "throughput_mbps": 4.5, "rtt_ms": 40, "loss_percent": 2},
                {"time_ms": 0, "kbps": 1200, "latency_ms": 30, "packet_loss": 0.1, "jitter": 3},
            ],
        },
        trace_id="net-a",
    )

    assert trace.trace_id == "net-a"
    assert [sample.timestamp_ms for sample in trace.samples] == [0, 200]
    assert trace.samples[0].throughput_bps == 1_200_000
    assert trace.samples[0].network_type == "lte"
    assert trace.samples[1].throughput_bps == 4_500_000
    assert trace.samples[1].packet_loss == 0.02


def test_load_broadband_trace_from_yaml(tmp_path) -> None:
    trace_path = tmp_path / "network.yml"
    trace_path.write_text(
        """
access: broadband
rows:
  - timestamp_ms: 0
    bytes_per_s: 1000
    ping_ms: 12
  - timestamp_ms: 100
    bandwidth_bps: 9000
    jitter_ms: 1
""",
        encoding="utf-8",
    )

    trace = load_network_trace(trace_path)

    assert trace.source_uri == str(trace_path)
    assert trace.samples[0].throughput_bps == 8000
    assert trace.samples[0].latency_ms == 12
    assert trace.samples[1].network_type == "broadband"
    assert trace.metadata["provenance"]["input_format"] == "rows"


def test_generate_synthetic_boundary_traces() -> None:
    traces = generate_synthetic_boundary_traces(
        NetworkBoundaryConfig(
            duration_ms=200,
            interval_ms=100,
            baseline_bps=5_000,
            low_bps=1_000,
            high_bps=9_000,
            threshold_bps=4_000,
            jitter_fraction=0.1,
        )
    )

    assert [trace.metadata["synthetic"]["kind"] for trace in traces] == [
        "step",
        "outage",
        "oscillation",
        "burst",
        "jitter",
        "threshold-near",
    ]
    assert [sample.throughput_bps for sample in traces[0].samples] == [1_000, 9_000, 9_000]
    assert traces[1].samples[1].throughput_bps == 0
    assert traces[1].samples[1].packet_loss == 1.0
    assert [sample.throughput_bps for sample in traces[2].samples] == [9_000, 1_000, 9_000]
    assert traces[3].samples[1].throughput_bps == 9_000
    assert [sample.throughput_bps for sample in traces[4].samples] == [5_500, 4_500, 5_500]
    assert [sample.throughput_bps for sample in traces[5].samples] == [4_200, 3_800, 4_200]


def test_invalid_network_traces_raise_clear_errors() -> None:
    with pytest.raises(NetworkError, match="non-empty samples"):
        normalize_network_trace({"samples": []})

    with pytest.raises(NetworkError, match="missing throughput"):
        normalize_network_trace({"samples": [{"timestamp_ms": 0}]})

    with pytest.raises(NetworkError, match="packet_loss"):
        normalize_network_trace({"samples": [{"timestamp_ms": 0, "mbps": 1, "packet_loss": 2}]})

    with pytest.raises(NetworkError, match="network_type"):
        normalize_network_trace({"network_type": "satellite", "samples": [{"timestamp_ms": 0, "mbps": 1}]})


def test_trace_id_is_stable_for_equivalent_network_content() -> None:
    first = normalize_network_trace({"samples": [{"timestamp_ms": 0, "mbps": 1, "latency_ms": 1}]})
    second = normalize_network_trace({"samples": [{"latency_ms": 1, "throughput_bps": 1_000_000, "timestamp_ms": 0}]})

    assert first.trace_id == second.trace_id
