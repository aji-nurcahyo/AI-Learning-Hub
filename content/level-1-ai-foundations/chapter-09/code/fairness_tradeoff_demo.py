"""Fairness-Utility Trade-off Demonstration.

Mengeksplorasi penyesuaian ambang batas keputusan (threshold tuning)
untuk menyeimbangkan keadilan kelompok vs akurasi keseluruhan.
Mendemonstrasikan mitigasi pasca-pemrosesan (post-processing mitigation)
untuk menyamakan True Positive Rate (Equal Opportunity) serta
menjelaskan Teorema Ketidakmungkinan Keadilan (Kleinberg et al., 2016).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main() -> None:
    csv_path = Path(__file__).resolve().parent / "data" / "scholarship_historical.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"File {csv_path} tidak ditemukan.")
        
    df = pd.read_csv(csv_path)
    
    features = ["test_score", "gpa", "extracurricular_score"]
    X = df[features].values
    y_hist = df["historical_decision"].values
    y_true = df["true_potential"].values
    groups = df["demographic_group"].values
    
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.30, random_state=42, stratify=groups
    )
    
    model = LogisticRegression(random_state=42)
    model.fit(X[train_idx], y_hist[train_idx])
    
    # Probabilitas prediksi positif
    probs_test = model.predict_proba(X[test_idx])[:, 1]
    y_test = y_true[test_idx]
    g_test = groups[test_idx]
    
    idx_A = (g_test == "Kelompok A")
    idx_B = (g_test == "Kelompok B")
    
    # 1. Model Baseline (Ambang Batas Tunggal Default = 0.50)
    thresh_base = 0.50
    preds_base = (probs_test >= thresh_base).astype(int)
    
    tpr_A_base = np.sum((preds_base[idx_A] == 1) & (y_test[idx_A] == 1)) / np.sum(y_test[idx_A] == 1)
    tpr_B_base = np.sum((preds_base[idx_B] == 1) & (y_test[idx_B] == 1)) / np.sum(y_test[idx_B] == 1)
    acc_base = accuracy_score(y_test, preds_base)
    eo_gap_base = abs(tpr_A_base - tpr_B_base)
    
    # 2. Mitigasi Pasca-Pemrosesan: Mencari ambang batas kelompok B (tau_B)
    # yang menyamakan TPR Kelompok B dengan Kelompok A (Equal Opportunity)
    best_tau_B = thresh_base
    min_eo_gap = eo_gap_base
    best_preds = preds_base.copy()
    
    # Grid search untuk threshold Kelompok B
    for candidate_tau in np.linspace(0.10, 0.70, 61):
        preds_cand = np.zeros_like(preds_base)
        preds_cand[idx_A] = (probs_test[idx_A] >= thresh_base).astype(int)
        preds_cand[idx_B] = (probs_test[idx_B] >= candidate_tau).astype(int)
        
        tpr_B_cand = np.sum((preds_cand[idx_B] == 1) & (y_test[idx_B] == 1)) / np.sum(y_test[idx_B] == 1)
        gap = abs(tpr_A_base - tpr_B_cand)
        
        if gap < min_eo_gap:
            min_eo_gap = gap
            best_tau_B = candidate_tau
            best_preds = preds_cand
            
    # Hitung metrik pasca-mitigasi
    tpr_A_mitigated = tpr_A_base
    tpr_B_mitigated = np.sum((best_preds[idx_B] == 1) & (y_test[idx_B] == 1)) / np.sum(y_test[idx_B] == 1)
    acc_mitigated = accuracy_score(y_test, best_preds)
    
    sr_A_base = np.mean(preds_base[idx_A])
    sr_B_base = np.mean(preds_base[idx_B])
    sr_A_mit = np.mean(best_preds[idx_A])
    sr_B_mit = np.mean(best_preds[idx_B])
    
    print("=== TRADE-OFF KEADILAN VS UTILITAS (FAIRNESS-UTILITY TRADE-OFF) ===")
    print("Skenario: Mitigasi Pasca-Pemrosesan untuk Menyamakan True Positive Rate (Equal Opportunity)")
    print("-" * 80)
    print(f"{'Metrik Sistem':<36} | {'Baseline (Ambang 0.50)':<18} | {'Termitigasi (Ambang Terpisah)':<22}")
    print("-" * 80)
    print(f"{'Ambang Batas Kelompok A (tau_A)':<36} | {thresh_base:<18.2f} | {thresh_base:<22.2f}")
    print(f"{'Ambang Batas Kelompok B (tau_B)':<36} | {thresh_base:<18.2f} | {best_tau_B:<22.2f}")
    print(f"{'Akurasi Global Keseluruhan':<36} | {acc_base*100:<17.2f}% | {acc_mitigated*100:<21.2f}%")
    print(f"{'TPR Kelompok A (Recall A)':<36} | {tpr_A_base*100:<17.2f}% | {tpr_A_mitigated*100:<21.2f}%")
    print(f"{'TPR Kelompok B (Recall B)':<36} | {tpr_B_base*100:<17.2f}% | {tpr_B_mitigated*100:<21.2f}%")
    print(f"{'Selisih Equal Opportunity (Delta EO)':<36} | {eo_gap_base:<18.4f} | {min_eo_gap:<22.4f}")
    print(f"{'Rasio Disparate Impact (Selection B/A)':<36} | {sr_B_base/sr_A_base:<18.4f} | {sr_B_mit/sr_A_mit:<22.4f}")
    print("-" * 80)
    
    print("\nANALISIS HASIL:")
    print(f"1. Dengan menurunkan ambang batas keputusan Kelompok B dari {thresh_base:.2f} ke {best_tau_B:.2f},")
    print(f"   kesenjangan Equal Opportunity (Delta EO) anjlok drastis dari {eo_gap_base:.4f} menjadi {min_eo_gap:.4f}!")
    print(f"2. Biaya Utilitas (Trade-off): Akurasi model hanya turun marjinal dari {acc_base*100:.2f}% ke {acc_mitigated*100:.2f}%,")
    print("   namun menghasilkan sistem seleksi yang jauh lebih adil bagi kandidat minoritas berpotensi tinggi.")
    print("\nCATATAN TEOREMA KLEINBERG ET AL. (2016):")
    print("Secara matematis, jika base rate kedua kelompok berbeda, mustahil memenuhi")
    print("Demographic Parity, Equal Opportunity, dan Predictive Parity secara bersamaan.")
    print("Insinyur AI harus secara sadar memilih kriteria etis mana yang paling relevan bagi konteks kasus.")


if __name__ == "__main__":
    main()

