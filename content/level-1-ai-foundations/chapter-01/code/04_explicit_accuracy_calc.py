"""
04_explicit_accuracy_calc.py
Contoh 4: Menghitung Akurasi (Accuracy) Secara Eksplisit.

Tujuan:
Memahami rumus dasar evaluasi model klasifikasi biner/multi-kelas
menggunakan logika Python standar tanpa bergantung pada library eksternal.

Penting:
Data dalam script ini merupakan data sintetis ilustratif untuk tujuan pembelajaran,
bukan hasil pelatihan model riil di lapangan.
"""

from typing import List, Tuple

def calculate_accuracy(y_true: List[int], y_pred: List[int]) -> Tuple[float, int, int]:
    """
    Menghitung akurasi secara eksplisit.
    
    Rumus:
    Akurasi = (Jumlah Prediksi Benar) / (Total Sampel)
    
    Return:
    (accuracy_float, correct_count, total_count)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Panjang y_true dan y_pred harus identik!")
    
    if len(y_true) == 0:
        raise ValueError("Dataset tidak boleh kosong!")

    correct = 0
    total = len(y_true)

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct += 1

    accuracy = correct / total
    return accuracy, correct, total

if __name__ == "__main__":
    print("=== KALKULASI EKSPLISIT METRIK AKURASI ===")
    
    # Label kelas: 1 = Spam, 0 = Ham (Bukan Spam)
    # 10 data ilustratif pengujian:
    y_ground_truth = [1, 0, 0, 1, 0, 0, 1, 1, 0, 1]
    y_model_pred   = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]

    acc, correct, total = calculate_accuracy(y_ground_truth, y_model_pred)

    print(f"\nJumlah total sampel uji : {total}")
    print(f"Data Aktual (Ground Truth): {y_ground_truth}")
    print(f"Prediksi Sistem            : {y_model_pred}")
    print("-" * 50)
    
    # Rincian perbandingan elemen
    for i, (yt, yp) in enumerate(zip(y_ground_truth, y_model_pred), start=1):
        status = "BENAR" if yt == yp else "SALAH"
        print(f"Sampel #{i:02d}: Aktual={yt}, Pred={yp} -> {status}")

    print("-" * 50)
    print(f"Prediksi benar : {correct} dari {total} sampel")
    print(f"Akurasi Sistem : {acc:.4f} ({acc * 100:.1f}%)")
    
    print("\n[Peringatan Evaluasi]")
    print("Akurasi 80% terlihat baik di atas kertas. Namun, jika dalam dunia nyata 99% email adalah Ham,")
    print("model bodoh yang selalu memprediksi 0 akan mendapat akurasi 99% padahal tidak mampu mendeteksi 1 pun spam!")
    print("Inilah sebabnya kita memerlukan metrik evaluasi yang lebih mendalam (Precision, Recall, F1).")

