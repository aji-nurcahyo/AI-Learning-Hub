"""Inference Latency and Throughput Benchmark (Localhost CPU).

Measures actual runtime performance on CPU:
1. In-memory direct model.predict() latency (500 single-sample calls, 20 warm-up discarded).
2. Live Localhost (127.0.0.1) REST API end-to-end HTTP POST latency (100 calls, 10 warm-up).
Reports Mean, Median (P50), P95, P99 latency (ms), and Throughput (req/s).
"""

import json
import time
import urllib.request
import numpy as np

from model_serving_api import start_local_serving_server


def summarize_latencies(latencies_ms: np.ndarray, total_wall_s: float) -> dict:
    return {
        "mean_ms": round(float(np.mean(latencies_ms)), 3),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "throughput_rps": round(float(len(latencies_ms) / max(1e-9, total_wall_s)), 1),
    }


def main():
    print("=" * 75)
    print("CHAPTER 36: INFERENCE LATENCY & THROUGHPUT BENCHMARK (CPU / 127.0.0.1)")
    print("=" * 75)

    server, port, state = start_local_serving_server(port=0)
    base_url = f"http://127.0.0.1:{port}"
    sample_vec = np.array([[0.5, -1.1, 0.2, 0.8, -0.3, 0.4]], dtype=np.float64)

    try:
        # 1. Direct In-Memory Model Inference Benchmark
        for _ in range(20):  # warm-up
            state.model.predict_proba(sample_vec)

        mem_latencies = []
        t_wall0 = time.perf_counter()
        for _ in range(500):
            t0 = time.perf_counter()
            state.model.predict_proba(sample_vec)
            mem_latencies.append((time.perf_counter() - t0) * 1000.0)
        t_wall_mem = time.perf_counter() - t_wall0
        mem_stats = summarize_latencies(np.array(mem_latencies), t_wall_mem)

        # 2. End-to-End Localhost REST API Latency Benchmark
        payload_bytes = json.dumps({"features": [0.5, -1.1, 0.2, 0.8, -0.3, 0.4]}).encode("utf-8")

        def send_once():
            req = urllib.request.Request(
                f"{base_url}/predict",
                data=payload_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.read()

        for _ in range(10):  # HTTP warm-up
            send_once()

        http_latencies = []
        t_wall0 = time.perf_counter()
        for _ in range(100):
            t0 = time.perf_counter()
            send_once()
            http_latencies.append((time.perf_counter() - t0) * 1000.0)
        t_wall_http = time.perf_counter() - t_wall0
        http_stats = summarize_latencies(np.array(http_latencies), t_wall_http)

        print("1. Direct In-Memory Python Inference (N=500, 20 warm-up excluded):")
        print(f"   Mean: {mem_stats['mean_ms']} ms | P50: {mem_stats['p50_ms']} ms | P95: {mem_stats['p95_ms']} ms | P99: {mem_stats['p99_ms']} ms")
        print(f"   Throughput: {mem_stats['throughput_rps']} req/s\n")

        print("2. End-to-End Localhost REST API HTTP POST (N=100, 10 warm-up excluded):")
        print(f"   Mean: {http_stats['mean_ms']} ms | P50: {http_stats['p50_ms']} ms | P95: {http_stats['p95_ms']} ms | P99: {http_stats['p99_ms']} ms")
        print(f"   Throughput: {http_stats['throughput_rps']} req/s\n")

        print("Caveat: Measurements reflect single-host Windows CPU loopback (127.0.0.1)")
        print("and cannot be extrapolated directly to multi-tenant cloud production servers.")
    finally:
        server.shutdown()
        server.server_close()

    print("\n[SUCCESS] Inference latency benchmark completed.")


if __name__ == "__main__":
    main()

