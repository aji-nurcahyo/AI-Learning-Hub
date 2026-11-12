"""Held-Out Test Evaluation, Permutation Explainability & Error Analysis."""

import hashlib
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from config import CapstoneConfig
from preprocessing import prepare_splits
from baseline_model import evaluate_classifier


def load_verified_bundle(cfg: CapstoneConfig = CapstoneConfig()):
    bundle_path = os.path.join(cfg.exp_dir, "capstone_sensor_model.pkl")
    sha_path = os.path.join(cfg.exp_dir, "capstone_model_sha256.txt")
    with open(sha_path, "r", encoding="utf-8") as f:
        expected_hash = f.read().strip()
    with open(bundle_path, "rb") as f:
        raw = f.read()
    actual_hash = hashlib.sha256(raw).hexdigest()
    assert actual_hash == expected_hash, "SHA-256 verification failed! Refusing pickle deserialization."
    return pickle.loads(raw)


def run_final_test_evaluation(cfg: CapstoneConfig = CapstoneConfig()) -> dict:
    data = prepare_splits(cfg)
    bundle = load_verified_bundle(cfg)
    model = bundle["model"]

    test_metrics = evaluate_classifier(model, data.X_test, data.y_test)
    preds = model.predict(data.X_test)
    probs = model.predict_proba(data.X_test)[:, 1]
    cm = confusion_matrix(data.y_test, preds)

    perm = permutation_importance(model, data.X_test, data.y_test, n_repeats=15, random_state=cfg.seed)
    importances = {feat: round(float(val), 4) for feat, val in zip(cfg.feature_names, perm.importances_mean)}

    mis_idx = np.where(preds != data.y_test)[0]
    error_examples = []
    for i in mis_idx[:3]:
        error_examples.append({
            "test_index": int(i),
            "true_label": int(data.y_test[i]),
            "pred_label": int(preds[i]),
            "pred_prob": round(float(probs[i]), 4),
            "temperature_c": round(float(data.X_test_raw[i, 0]), 2),
            "vibration_rms": round(float(data.X_test_raw[i, 1]), 2),
        })

    # Plot Confusion Matrix + Feature Importance side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Pred Normal", "Pred Fault"])
    axes[0].set_yticklabels(["True Normal", "True Fault"])
    axes[0].set_title(f"Held-Out Test Confusion Matrix\n({bundle['model_name']})")
    for r in range(2):
        for c in range(2):
            axes[0].text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=12, fontweight="bold")

    feats = list(importances.keys())
    vals = [importances[k] for k in feats]
    axes[1].barh(feats, vals, color="#2563eb")
    axes[1].set_xlabel("Permutation Importance (Accuracy Drop)")
    axes[1].set_title("Explainability: Held-Out Feature Importance")
    axes[1].invert_yaxis()
    fig.tight_layout()
    chart_path = os.path.join(cfg.exp_dir, "capstone_evaluation_chart.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    return {
        "model_name": bundle["model_name"],
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist(),
        "feature_importance": importances,
        "error_count": int(len(mis_idx)),
        "sample_errors": error_examples,
        "chart_path": chart_path,
    }


def main():
    res = run_final_test_evaluation()
    tm = res["test_metrics"]
    print("=" * 78)
    print(f"CHAPTER 40 CAPSTONE: FINAL HELD-OUT TEST EVALUATION ({res['model_name']})")
    print("=" * 78)
    print(
        f"Test Metrics (N=300 unseen) -> Acc={tm['accuracy']*100:.2f}% | Prec={tm['precision']:.4f} | "
        f"Rec={tm['recall']:.4f} | F1={tm['f1']:.4f} | ROC-AUC={tm['roc_auc']:.4f}"
    )
    print(f"Confusion Matrix (TN, FP / FN, TP) : {res['confusion_matrix']} | Total Errors={res['error_count']}")
    print(f"Permutation Importance             : {res['feature_importance']}")
    print(f"Sample Misclassified Cases         : {res['sample_errors']}")


if __name__ == "__main__":
    main()

