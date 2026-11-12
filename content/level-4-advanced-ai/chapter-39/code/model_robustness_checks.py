"""Model Robustness, Out-of-Distribution Shift & Human Escalation Fallback."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def predict_with_human_escalation(clf, X_in: np.ndarray, confidence_threshold: float = 0.70, train_mean=None, train_std=None):
    # Impute missing NaNs defensively before inference (safe fallback)
    X_clean = np.copy(X_in)
    nan_mask = np.isnan(X_clean)
    if nan_mask.any() and train_mean is not None:
        X_clean = np.where(nan_mask, train_mean, X_clean)

    probs = clf.predict_proba(X_clean)
    max_conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)

    # Check both low confidence and extreme z-score OOD inputs (> 4 std from training mean)
    ood_flags = np.zeros(len(X_clean), dtype=bool)
    if train_mean is not None and train_std is not None:
        z_max = np.abs((X_clean - train_mean) / np.maximum(train_std, 1e-6)).max(axis=1)
        ood_flags = z_max > 4.0

    escalate = (max_conf < confidence_threshold) | ood_flags
    return preds, max_conf, escalate


def main():
    print("=" * 78)
    print("CHAPTER 39: MODEL ROBUSTNESS, SHIFT & HUMAN ESCALATION FALLBACK")
    print("=" * 78)
    print("Caveat: Synthetic robustness stress-test does NOT prove production safety.\n")

    X, y = make_classification(n_samples=800, n_features=8, n_informative=5, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    train_mean = X_tr.mean(axis=0)
    train_std = X_tr.std(axis=0)

    clf = RandomForestClassifier(n_estimators=40, random_state=42)
    clf.fit(X_tr, y_tr)

    rng = np.random.default_rng(42)
    scenarios = {
        "Clean In-Distribution Test": X_te,
        "Noisy Shift (sigma=1.2)": X_te + rng.normal(0.0, 1.2, size=X_te.shape),
        "Missing Features (30% NaN)": np.where(rng.random(X_te.shape) < 0.30, np.nan, X_te),
        "Extreme OOD Stress (+6.0 shift)": X_te + 6.0,
    }

    for name, X_s in scenarios.items():
        preds, conf, esc = predict_with_human_escalation(clf, X_s, 0.70, train_mean, train_std)
        acc = accuracy_score(y_te, preds)
        esc_rate = 100.0 * esc.mean()
        print(f"- {name:32s} | Acc={acc*100:6.2f}% | MeanConf={conf.mean():.3f} | Human Escalation={esc_rate:5.1f}%")

    _, _, esc_clean = predict_with_human_escalation(clf, scenarios["Clean In-Distribution Test"], 0.70, train_mean, train_std)
    _, _, esc_ood = predict_with_human_escalation(clf, scenarios["Extreme OOD Stress (+6.0 shift)"], 0.70, train_mean, train_std)
    assert esc_ood.mean() > esc_clean.mean()
    print("\n[SUCCESS] Robustness degradation & OOD Human Escalation fallback verified.")


if __name__ == "__main__":
    main()

