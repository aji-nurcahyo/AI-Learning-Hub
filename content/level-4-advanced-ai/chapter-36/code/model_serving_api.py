"""Local Educational Model Serving REST API (127.0.0.1 Only).

Implements a lightweight, dependency-free local HTTP REST API using Python's
standard library http.server, strictly bound to localhost (127.0.0.1):
- GET  /health      : Readiness status and active model version.
- GET  /model/info  : Model metadata, input schema, and checksum provenance.
- POST /predict     : Schema-validated inference returning prediction, prob, latency_ms.
- GET  /metrics     : Operational telemetry (requests, errors, mean & P95 latency).

Security Guarantees:
- Binds strictly to 127.0.0.1 (never 0.0.0.0).
- Verifies SHA-256 artifact checksum before deserializing model.
- Rejects malformed payloads with HTTP 400 without exposing stack traces.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
import urllib.request
from typing import Any, Dict, List, Tuple

import numpy as np

from reproducible_training import train_and_save_pipeline
from model_artifact_validation import safe_load_trusted_artifact
from request_validation import RequestValidationError, validate_predict_payload


class ModelServiceState:
    """Holds loaded trusted model and in-memory telemetry metrics."""

    def __init__(self, model: Any, metadata: Dict[str, Any]):
        self.model = model
        self.metadata = metadata
        self.lock = threading.Lock()
        self.total_requests = 0
        self.error_requests = 0
        self.latencies_ms: List[float] = []

    def record_request(self, latency_ms: float, is_error: bool = False):
        with self.lock:
            self.total_requests += 1
            if is_error:
                self.error_requests += 1
            else:
                self.latencies_ms.append(latency_ms)
                if len(self.latencies_ms) > 1000:
                    self.latencies_ms.pop(0)

    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            if self.latencies_ms:
                arr = np.array(self.latencies_ms)
                mean_ms = round(float(np.mean(arr)), 3)
                p95_ms = round(float(np.percentile(arr, 95)), 3)
            else:
                mean_ms, p95_ms = 0.0, 0.0
            return {
                "total_requests": self.total_requests,
                "error_requests": self.error_requests,
                "mean_latency_ms": mean_ms,
                "p95_latency_ms": p95_ms,
            }


def make_handler(service_state: ModelServiceState):
    class LocalServingHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress noisy stdout logging during automated tests
            pass

        def _send_json(self, status_code: int, body: Dict[str, Any]):
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {
                    "status": "healthy",
                    "bind_host": "127.0.0.1",
                    "model_version": service_state.metadata["version"],
                })
            elif self.path == "/model/info":
                self._send_json(200, service_state.metadata)
            elif self.path == "/metrics":
                self._send_json(200, service_state.get_metrics())
            else:
                self._send_json(404, {"error": "Endpoint not found"})

        def do_POST(self):
            if self.path != "/predict":
                self._send_json(404, {"error": "Endpoint not found"})
                return

            t0 = time.perf_counter()
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                if content_len <= 0 or content_len > 65536:
                    raise RequestValidationError("Invalid or excessive Content-Length.")
                raw_body = self.rfile.read(content_len)
                payload = json.loads(raw_body.decode("utf-8"))
                features = validate_predict_payload(payload, expected_dim=6)

                x_arr = np.array([features], dtype=np.float64)
                pred_class = int(service_state.model.predict(x_arr)[0])
                prob_pos = float(service_state.model.predict_proba(x_arr)[0, 1])

                latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                service_state.record_request(latency_ms, is_error=False)

                self._send_json(200, {
                    "prediction": pred_class,
                    "probability_positive": round(prob_pos, 4),
                    "model_version": service_state.metadata["version"],
                    "latency_ms": latency_ms,
                })
            except (RequestValidationError, json.JSONDecodeError) as e:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                service_state.record_request(latency_ms, is_error=True)
                self._send_json(400, {"error": f"Bad Request: {str(e)}"})
            except Exception:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                service_state.record_request(latency_ms, is_error=True)
                self._send_json(500, {"error": "Internal inference error"})

    return LocalServingHandler


def start_local_serving_server(port: int = 0) -> Tuple[ThreadingHTTPServer, int, ModelServiceState]:
    """Trains/verifies trusted local model and starts 127.0.0.1 HTTP server on an ephemeral port."""
    train_res = train_and_save_pipeline(
        model_type="logistic_regression",
        C=1.0,
        seed=42,
        artifact_filename="serving_model_v1.pkl",
    )
    model = safe_load_trusted_artifact(train_res["artifact_path"], train_res["checksum_sha256"])

    metadata = {
        "model_id": "telemetry_anomaly_cls",
        "version": "v1.0.0",
        "algorithm": "LogisticRegression",
        "dataset_version": train_res["dataset_version"],
        "checksum_sha256": train_res["checksum_sha256"],
        "expected_feature_dim": 6,
        "local_only": True,
    }

    state = ModelServiceState(model, metadata)
    handler_cls = make_handler(state)
    # STRICT BIND TO 127.0.0.1
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    actual_port = server.server_address[1]

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, actual_port, state


def main():
    print("=" * 75)
    print("CHAPTER 36: LOCAL MODEL SERVING REST API (127.0.0.1)")
    print("=" * 75)

    server, port, _ = start_local_serving_server(port=0)
    base_url = f"http://127.0.0.1:{port}"
    print(f"Started local educational serving API strictly at: {base_url}")

    try:
        with urllib.request.urlopen(f"{base_url}/health") as resp:
            health = json.loads(resp.read().decode("utf-8"))
        print(f"GET /health     -> {health}")

        req = urllib.request.Request(
            f"{base_url}/predict",
            data=json.dumps({"features": [0.5, -1.1, 0.2, 0.8, -0.3, 0.4]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            pred = json.loads(resp.read().decode("utf-8"))
        print(f"POST /predict   -> {pred}")
    finally:
        server.shutdown()
        server.server_close()
        print("\nLocal server shut down cleanly.")

    print("[SUCCESS] Local model serving API operational on 127.0.0.1.")


if __name__ == "__main__":
    main()

