"""Candidate ML Comparison, Latency/Size Resource Profiling & SHA-256 Artifact Saving."""

import hashlib
import os
import pickle
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from config import CapstoneConfig
from preprocessing import prepare_splits
from baseline_model import evaluate_classifier


def benchmark_latency_ms(model, X_sample: np.ndarray, warmup: int = 30, runs: int = 300) -> dict:
    single = X_sample[:1]
    for _ in range(warmup):
        model.predict_proba(single)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict_proba(single)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {"mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95))}


def compare_and_save_best(cfg: CapstoneConfig = CapstoneConfig()) -> dict:
    data = prepare_splits(cfg)
    candidates = {
        "LogisticRegression": LogisticRegression(random_state=cfg.seed),
        "RandomForest(n=50,d=8)": RandomForestClassifier(n_estimators=50, max_depth=8, random_state=cfg.seed),
        "MLP(32,16)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400, random_state=cfg.seed),
    }

    summary = {}
    best_name, best_model, best_f1 = None, None, -1.0
    for name, mdl in candidates.items():
        t_fit = time.perf_counter()
        mdl.fit(data.X_train, data.y_train)
        fit_s = time.perf_counter() - t_fit
        val_metrics = evaluate_classifier(mdl, data.X_val, data.y_val)
        lat = benchmark_latency_ms(mdl, data.X_val)
        size_kb = len(pickle.dumps({"scaler": data.scaler, "model": mdl})) / 1024.0
        summary[name] = {
            "val_metrics": val_metrics,
            "latency_ms": lat,
            "bundle_size_kb": round(size_kb, 2),
            "train_time_s": round(fit_s, 3),
        }
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_name = name
            best_model = mdl

    os.makedirs(cfg.exp_dir, exist_ok=True)
    bundle_path = os.path.join(cfg.exp_dir, "capstone_sensor_model.pkl")
    payload_bytes = pickle.dumps({"model_name": best_name, "scaler": data.scaler, "model": best_model})
    with open(bundle_path, "wb") as f:
        f.write(payload_bytes)
    sha256_hex = hashlib.sha256(payload_bytes).hexdigest()
    sha_path = os.path.join(cfg.exp_dir, "capstone_model_sha256.txt")
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(sha256_hex)

    return {"best_model_name": best_name, "candidates": summary, "artifact_path": bundle_path, "sha256": sha256_hex}


def main():
    res = compare_and_save_best()
    print("=" * 84)
    print("CHAPTER 40 CAPSTONE: CANDIDATE MODEL BENCHMARK (VALIDATION + CPU RESOURCES)")
    print("=" * 84)
    for name, info in res["candidates"].items():
        vm = info["val_metrics"]
        lat = info["latency_ms"]
        print(
            f"- {name:24s} | Val F1={vm['f1']:.4f} | AUC={vm['roc_auc']:.4f} | "
            f"Size={info['bundle_size_kb']:6.2f} KB | LatMean={lat['mean_ms']:.4f} ms (P95={lat['p95_ms']:.4f} ms)"
        )
    print(f"\nSelected Best Model (Validation F1): {res['best_model_name']}")
    print(f"Saved Signed Artifact (SHA-256={res['sha256'][:16]}...) to: {res['artifact_path']}")


if __name__ == "__main__":
    main()

