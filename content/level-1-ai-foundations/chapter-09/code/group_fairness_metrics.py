"""Group Fairness Metrics Computation.

Menghitung dan memvisualisasikan metrik formal keadilan kelompok:
1. Demographic Parity (Statistical Parity)
2. Equal Opportunity (Kesetaraan TPR)
3. Equalized Odds (Kesetaraan TPR dan FPR)
4. Disparate Impact Ratio (Kaidah 80% EEOC)

Melatih model Logistic Regression baseline untuk mengevaluasi bias inheren.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray
) -> dict:
    """Menghitung metrik performa dan keadilan untuk tiap kelompok demografis."""
    results = {}
    unique_groups = np.unique(group)
    
    for g in unique_groups:
        idx = (group == g)
        y_t_g = y_true[idx]
        y_p_g = y_pred[idx]
        
        # Ukuran kelompok & selection rate
        n = len(y_t_g)
        selection_rate = np.mean(y_p_g)
        
        # Confusion matrix: TN, FP, FN, TP
        tn, fp, fn, tp = confusion_matrix(y_t_g, y_p_g, labels=[0, 1]).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / Sensitivity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # Fall-out
        acc = (tp + tn) / n if n > 0 else 0.0
        
        results[g] = {
            "n": n,
            "accuracy": acc,
            "selection_rate": selection_rate,
            "tpr": tpr,
            "fpr": fpr,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        }
        
    return results


def main() -> None:
    csv_path = Path(__file__).resolve().parent / "data" / "scholarship_historical.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"File {csv_path} tidak ditemukan. Jalankan historical_bias_demo.py terlebih dahulu.")
        
    df = pd.read_csv(csv_path)
    
    # Fitur numerik untuk melatih model klasifikasi
    features = ["test_score", "gpa", "extracurricular_score"]
    X = df[features].values
    y_hist = df["historical_decision"].values
    y_true = df["true_potential"].values
    groups = df["demographic_group"].values
    
    # Partisi train/test split (70:30)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.30, random_state=42, stratify=groups)
    
    # Model dilatih pada label historis masa lalu (unconstrained standard model)
    model = LogisticRegression(random_state=42)
    model.fit(X[train_idx], y_hist[train_idx])
    
    # Prediksi keputusan pada data uji
    y_pred_test = model.predict(X[test_idx])
    test_groups = groups[test_idx]
    y_true_test = y_true[test_idx]
    
    # Hitung metrik keadilan kelompok
    metrics = compute_group_metrics(y_true_test, y_pred_test, test_groups)
    
    print("=== METRIK KEADILAN KELOMPOK (GROUP FAIRNESS AUDIT) ===")
    print(f"Total Sampel Data Uji: {len(test_idx)}")
    print(f"Akurasi Keseluruhan Model: {accuracy_score(y_true_test, y_pred_test)*100:.2f}%\n")
    
    mA = metrics["Kelompok A"]
    mB = metrics["Kelompok B"]
    
    print(f"{'Metrik Evaluasi':<32} | {'Kelompok A (Mayoritas)':<22} | {'Kelompok B (Minoritas)':<22}")
    print("-" * 82)
    print(f"{'Jumlah Sampel (n)':<32} | {mA['n']:<22} | {mB['n']:<22}")
    print(f"{'Tingkat Akurasi':<32} | {mA['accuracy']*100:<21.2f}% | {mB['accuracy']*100:<21.2f}%")
    print(f"{'Tingkat Seleksi (Selection Rate)':<32} | {mA['selection_rate']*100:<21.2f}% | {mB['selection_rate']*100:<21.2f}%")
    print(f"{'True Positive Rate (TPR / Recall)':<32} | {mA['tpr']*100:<21.2f}% | {mB['tpr']*100:<21.2f}%")
    print(f"{'False Positive Rate (FPR)':<32} | {mA['fpr']*100:<21.2f}% | {mB['fpr']*100:<21.2f}%")
    print("-" * 82)
    
    # Selisih Keadilan Formal
    dp_diff = abs(mA["selection_rate"] - mB["selection_rate"])
    di_ratio = mB["selection_rate"] / mA["selection_rate"] if mA["selection_rate"] > 0 else 0
    eo_diff = abs(mA["tpr"] - mB["tpr"])
    fpr_diff = abs(mA["fpr"] - mB["fpr"])
    
    print("\n=== EVALUASI PERBEDAAN METRIK KEADILAN FORMAL ===")
    print(f"1. Demographic Parity Difference (Delta DP) : {dp_diff:.4f} (Ambang batas ideal: < 0.10)")
    print(f"2. Disparate Impact Ratio (Kaidah 80%)      : {di_ratio:.4f} ({'MEMENUHI' if di_ratio >= 0.8 else 'MELANGGAR (DISPARATE IMPACT)'})")
    print(f"3. Equal Opportunity Difference (Delta EO)  : {eo_diff:.4f} (Ambang batas ideal: < 0.10)")
    print(f"4. False Positive Rate Difference (Delta FPR): {fpr_diff:.4f}")
    
    print("\nKESIMPULAN AUDIT:")
    if eo_diff > 0.10 or di_ratio < 0.8:
        print("PERINGATAN: Model terbukti mengalami bias sistemik!")
        print("Kandidat berpotensi tinggi dari Kelompok B memiliki peluang lolos (TPR)")
        print(f"jauh lebih rendah ({mB['tpr']*100:.1f}%) dibanding Kelompok A ({mA['tpr']*100:.1f}%).")
        print("Sistem ini TIDAK LAYAK diimplementasikan secara otomatis tanpa intervensi etis.")


if __name__ == "__main__":
    main()
