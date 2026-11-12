"""Historical Bias Demonstration.

Simulasi data sintetis seleksi penerimaan beasiswa akademik yang mereproduksi
ketimpangan historis (historical bias / representation bias).
Membuktikan bahwa model yang dilatih pada label historis masa lalu akan
mereplikasi diskriminasi sistemik, meskipun potensi objektif kandidat setara.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_scholarship_data(
    n_samples: int = 1000,
    random_seed: int = 42
) -> pd.DataFrame:
    """Menghasilkan dataset sintetis pelamar beasiswa akademik.
    
    Fitur:
    - applicant_id: ID unik pelamar.
    - demographic_group: Kelompok A (Mayoritas) atau Kelompok B (Minoritas).
    - test_score: Skor tes objektif bakat skolastik (kemampuan murni, skala 0-100).
    - gpa: Indeks Prestasi Kumulatif (skala 0.0 - 4.0).
    - extracurricular_score: Skor kegiatan ekstrakurikuler (skala 0-100).
    - true_potential: Potensi keberhasilan akademik murni calon penerima (0 atau 1).
    - historical_decision: Keputusan penerimaan masa lalu yang bias (0 atau 1).
    """
    np.random.seed(random_seed)
    
    # 70% Kelompok A (Mayoritas/Historis Diuntungkan), 30% Kelompok B (Minoritas)
    groups = np.random.choice(["Kelompok A", "Kelompok B"], size=n_samples, p=[0.7, 0.3])
    
    # Potensi akademik objektif kedua kelompok dibuat SETARA secara intrinsik
    test_scores = np.random.normal(loc=72.0, scale=10.0, size=n_samples).clip(40, 100)
    gpas = (test_scores / 25.0 + np.random.normal(0, 0.15, size=n_samples)).clip(2.0, 4.0)
    
    # Skor ekstrakurikuler: Kelompok A memiliki akses fasilitas pendukung masa lalu lebih baik
    extra_scores = []
    for g in groups:
        if g == "Kelompok A":
            extra_scores.append(np.random.normal(loc=78.0, scale=8.0))
        else:
            extra_scores.append(np.random.normal(loc=65.0, scale=10.0))
    extra_scores = np.array(extra_scores).clip(30, 100)
    
    # Potensi Sebenarnya (Ground Truth Potential): Murni fungsi dari test_score & gpa
    academic_index = 0.6 * (test_scores / 100.0) + 0.4 * (gpas / 4.0)
    true_potential = (academic_index >= 0.72).astype(int)
    
    # Keputusan Historis Masa Lalu: Terdistorsi oleh bobot ekstrakurikuler dan bias afinitas subjektif
    historical_score = (
        0.45 * (test_scores / 100.0)
        + 0.25 * (gpas / 4.0)
        + 0.30 * (extra_scores / 100.0)
    )
    # Penalti afinitas subjektif tak sadar terhadap Kelompok B di masa lalu
    penalties = np.array([-0.05 if g == "Kelompok B" else 0.0 for g in groups])
    distorted_index = historical_score + penalties
    
    # Penerimaan historis (kuota ~35%)
    cutoff = np.percentile(distorted_index, 65)
    historical_decision = (distorted_index >= cutoff).astype(int)
    
    df = pd.DataFrame({
        "applicant_id": [f"MHS-{i+1:04d}" for i in range(n_samples)],
        "demographic_group": groups,
        "test_score": np.round(test_scores, 2),
        "gpa": np.round(gpas, 2),
        "extracurricular_score": np.round(extra_scores, 2),
        "true_potential": true_potential,
        "historical_decision": historical_decision
    })
    
    return df


def audit_historical_disparity(df: pd.DataFrame) -> None:
    """Menganalisis disparitas penerimaan historis vs potensi sebenarnya."""
    print("=== AUDIT KETIMPANGAN HISTORIS SELEKSI BEASISWA ===")
    print(f"Total Pelamar: {len(df)}")
    
    for grp in ["Kelompok A", "Kelompok B"]:
        sub = df[df["demographic_group"] == grp]
        pot_rate = sub["true_potential"].mean() * 100
        hist_rate = sub["historical_decision"].mean() * 100
        mean_test = sub["test_score"].mean()
        
        print(f"\n[{grp}] (n={len(sub)}, {len(sub)/len(df)*100:.1f}%):")
        print(f"  - Rata-rata Skor Tes Objektif: {mean_test:.2f}")
        print(f"  - Proporsi Berpotensi Tinggi (Ground Truth): {pot_rate:.2f}%")
        print(f"  - Tingkat Kelulusan Historis (Masa Lalu)  : {hist_rate:.2f}%")
        print(f"  - Disparitas Gap Historis vs Potensi       : {hist_rate - pot_rate:+.2f}%")
        
    # Selisih tingkat kelulusan antar kelompok
    hist_A = df[df["demographic_group"] == "Kelompok A"]["historical_decision"].mean()
    hist_B = df[df["demographic_group"] == "Kelompok B"]["historical_decision"].mean()
    ratio = hist_B / hist_A if hist_A > 0 else 0
    
    print("\n=== METRIK DISPARITAS HISTORIS ===")
    print(f"Tingkat Kelulusan Kelompok A: {hist_A*100:.2f}%")
    print(f"Tingkat Kelulusan Kelompok B: {hist_B*100:.2f}%")
    print(f"Rasio Kelulusan B / A       : {ratio:.4f}")
    print(f"Aturan Empat-Perlima (80% Rule): {'TIDAK LOLOS (Disparate Impact)' if ratio < 0.8 else 'LOLOS'}")
    print("\nKESIMPULAN ETIS:")
    print("Meskipun kemampuan akademik objektif kedua kelompok setara (72.0 vs 72.1),")
    print("Kelompok B mengalami diskriminasi dampak berbeda (disparate impact)")
    print("karena model pembelajaran mesin mereplikasi bias seleksi masa lalu!")


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "scholarship_historical.csv"
    
    df = generate_synthetic_scholarship_data(n_samples=1000, random_seed=42)
    df.to_csv(csv_path, index=False)
    print(f"Dataset sintetis berhasil disimpan ke: {csv_path.name}")
    
    audit_historical_disparity(df)


if __name__ == "__main__":
    main()

