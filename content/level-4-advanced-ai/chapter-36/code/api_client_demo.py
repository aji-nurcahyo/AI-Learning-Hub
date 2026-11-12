"""Local REST API Integration Test Client.

Exercises full lifecycle against a live localhost server (127.0.0.1):
1. Verifies GET /health and GET /model/info endpoints.
2. Submits valid inference requests and checks predictions & version tags.
3. Submits invalid schema payloads and asserts HTTP 400 Bad Request rejection.
4. Inspects GET /metrics latency and error counters.
5. Shuts down the server gracefully after tests complete.
"""

import json
import urllib.error
import urllib.request

from model_serving_api import start_local_serving_server


def http_get_json(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def http_post_json(url: str, body: dict):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def main():
    print("=" * 75)
    print("CHAPTER 36: LOCAL REST API INTEGRATION TEST SUITE")
    print("=" * 75)

    server, port, _ = start_local_serving_server(port=0)
    base_url = f"http://127.0.0.1:{port}"
    print(f"Target Localhost Endpoint: {base_url}\n")

    try:
        # 1. Health Endpoint
        code, health = http_get_json(f"{base_url}/health")
        assert code == 200 and health["status"] == "healthy"
        print(f"[PASS] GET /health      -> HTTP {code} | Version={health['model_version']}")

        # 2. Model Info Endpoint
        code, info = http_get_json(f"{base_url}/model/info")
        assert code == 200 and info["expected_feature_dim"] == 6
        print(f"[PASS] GET /model/info  -> HTTP {code} | Alg={info['algorithm']} | LocalOnly={info['local_only']}")

        # 3. Valid Predict Request
        code, pred = http_post_json(f"{base_url}/predict", {"features": [1.2, -0.8, 0.4, -0.5, 0.1, 0.9]})
        assert code == 200 and "prediction" in pred and pred["model_version"] == "v1.0.0"
        print(f"[PASS] POST /predict    -> HTTP {code} | Pred={pred['prediction']}, Prob={pred['probability_positive']}, Latency={pred['latency_ms']}ms")

        # 4. Invalid Schema Rejection (wrong dimension)
        code, err = http_post_json(f"{base_url}/predict", {"features": [1.0, 2.0]})
        assert code == 400 and "error" in err
        print(f"[PASS] Invalid Schema   -> HTTP {code} (Blocked cleanly: {err['error']})")

        # 5. Metrics Endpoint
        code, metrics = http_get_json(f"{base_url}/metrics")
        assert code == 200 and metrics["total_requests"] == 2 and metrics["error_requests"] == 1
        print(f"[PASS] GET /metrics     -> HTTP {code} | TotalReq={metrics['total_requests']}, Errors={metrics['error_requests']}, MeanLatency={metrics['mean_latency_ms']}ms")

    finally:
        server.shutdown()
        server.server_close()
        print("\nServer stopped cleanly.")

    print("[SUCCESS] All REST API integration assertions passed.")


if __name__ == "__main__":
    main()

