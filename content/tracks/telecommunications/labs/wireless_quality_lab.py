"""Track D (AI for Telecommunications) Applied Lab: Wireless Network Quality Classification.

Explicitly distinguishes SNR (signal vs thermal noise, excluding interference)
from SINR (signal vs interference + noise), alongside RSRP, RSRQ, Latency,
Throughput, and Packet Loss on a synthetic radio telemetry dataset.
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_synthetic_radio_telemetry(n_samples: int = 800, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Physical radio metrics (synthetic ranges consistent with LTE/5G NR monitoring)
    rsrp_dbm = rng.normal(-92.0, 9.0, size=n_samples)               # Reference Signal Received Power
    rsrq_db = np.clip(-11.0 + 0.18 * (rsrp_dbm + 92.0) + rng.normal(0, 1.5, n_samples), -20.0, -3.0)
    snr_db = np.clip(16.0 + 0.35 * (rsrp_dbm + 92.0) + rng.normal(0, 2.0, n_samples), -2.0, 32.0)
    interference_db_penalty = rng.uniform(1.0, 9.5, size=n_samples)  # Adjacent-cell load lowers SINR below SNR
    sinr_db = snr_db - interference_db_penalty                       # Distinct from SNR!

    throughput_mbps = np.clip(55.0 + 3.8 * sinr_db + rng.normal(0, 6.0, n_samples), 2.0, 180.0)
    latency_ms = np.clip(28.0 - 0.65 * sinr_db + 0.8 * interference_db_penalty + rng.normal(0, 3.0, n_samples), 8.0, 120.0)
    packet_loss_pct = np.clip(0.4 + np.maximum(0.0, 6.0 - sinr_db) * 0.45 + rng.normal(0, 0.25, n_samples), 0.0, 15.0)

    # Quality degraded alert (1) when SINR drops low or packet loss / latency exceeds SLA threshold
    degraded = ((sinr_db < 6.0) | (packet_loss_pct > 2.0) | (rsrq_db < -14.5)).astype(int)

    X = np.column_stack([rsrp_dbm, rsrq_db, snr_db, sinr_db, latency_ms, throughput_mbps, packet_loss_pct])
    names = ["RSRP (dBm)", "RSRQ (dB)", "SNR (dB)", "SINR (dB)", "Latency (ms)", "Throughput (Mbps)", "Packet Loss (%)"]
    return X, degraded, names


def main():
    X, y, names = generate_synthetic_radio_telemetry(n_samples=800, seed=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    assert np.all(X[:, 3] < X[:, 2]), "SINR must remain distinct from (and <=) SNR under co-channel interference"

    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr_s, y_tr)
    logreg = LogisticRegression(random_state=42).fit(X_tr_s, y_tr)
    rf = RandomForestClassifier(n_estimators=40, random_state=42).fit(X_tr_s, y_tr)

    print("=" * 78)
    print("TRACK D (AI FOR TELECOMMUNICATIONS): WIRELESS QUALITY MONITORING LAB")
    print("=" * 78)
    print(f"Features Checked : {names}")
    print(f"Mean SNR={X[:,2].mean():.2f} dB vs Mean SINR={X[:,3].mean():.2f} dB (Distinct interference gap!)")
    for name, mdl in [("Majority Baseline", dummy), ("LogisticRegression", logreg), ("RandomForest(40)", rf)]:
        preds = mdl.predict(X_te_s)
        print(f"- {name:20s} | Test Acc={accuracy_score(y_te, preds)*100:6.2f}% | Macro-F1={f1_score(y_te, preds, average='macro'):.4f}")
    print("Caveat: Synthetic dataset; no field deployment validation claimed.")
    assert accuracy_score(y_te, rf.predict(X_te_s)) > 0.90


if __name__ == "__main__":
    main()

