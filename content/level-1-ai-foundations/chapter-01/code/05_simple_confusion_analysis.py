"""
05_simple_confusion_analysis.py
Contoh 5: Analisis Kesalahan Sederhana dengan Confusion Counts.

Tujuan:
Membongkar angka akurasi ke dalam 4 kategori kuadran:
- True Positive (TP)
- False Positive (FP) - Alarm Palsu / Tipe I Error
- True Negative (TN)
- False Negative (FN) - Lolos Berbahaya / Tipe II Error

Penting:
Data dalam script ini merupakan data ilustratif terstruktur untuk edukasi,
bukan hasil inferensi dari dataset medis/keamanan riil.
"""

from typing import List, Dict

def compute_confusion_counts(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    """
    Menghitung TP, FP, TN, FN untuk klasifikasi biner.
    Asumsi: 1 = Kelas Positif (misal: Terdeteksi Anomali / Penyakit),
            0 = Kelas Negatif (misal: Normal / Sehat).
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

if __name__ == "__main__":
    print("=== DEMO: CONFUSION COUNTS & ANALISIS ERROR ===")
    
    # Skenario: Deteksi Penipuan Transaksi (1 = Fraud, 0 = Normal)
    # Total 20 transaksi ilustratif
    y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] # 16 Normal, 4 Fraud
    y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0] # Prediksi

    counts = compute_confusion_counts(y_true, y_pred)
    total = len(y_true)
    accuracy = (counts["TP"] + counts["TN"]) / total

    print(f"Total sampel dievaluasi : {total}")
    print(f"Distribusi aktual       : {y_true.count(0)} Normal, {y_true.count(1)} Fraud")
    print("-" * 52)
    print("Tabel Matriks Kontingensi Sederhana (Confusion Matrix):")
    print(f"{'':<18} | {'Prediksi Positif (1)':<20} | {'Prediksi Negatif (0)':<20}")
    print("-" * 64)
    print(f"{'Aktual Positif (1)':<18} | TP = {counts['TP']:<15} | FN = {counts['FN']:<15}")
    print(f"{'Aktual Negatif (0)':<18} | FP = {counts['FP']:<15} | TN = {counts['TN']:<15}")
    print("-" * 64)

    print(f"\nAkurasi Keseluruhan : {accuracy * 100:.1f}%")
    print(f"Rincian Kesalahan:")
    print(f"- False Positive (FP) : {counts['FP']} transaksi normal diblokir keliru (mengganggu pengguna)")
    print(f"- False Negative (FN) : {counts['FN']} transaksi penipuan lolos tak terdeteksi (kerugian finansial)")

    print("\n[Wawasan Kunci]")
    print("Meskipun akurasi mencapai 80%, sistem gagal menangkap 50% kasus penipuan nyata (FN = 2 dari 4)!")
    print("Dalam aplikasi kritis (medis, keamanan, keuangan), biaya dari False Negative")
    print("sering kali jauh lebih fatal daripada sekadar angka akurasi umum.")

