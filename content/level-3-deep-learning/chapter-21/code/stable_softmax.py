"""
Chapter 21 - Neural Network Fundamentals
Script 5: Stabilisasi Numerik Fungsi Softmax (Stable Softmax)
Membuktikan bahaya floating-point overflow pada formulasi naif Softmax
dan mengimplementasikan trik stabilitas numerik (subtract max trick).
"""

import numpy as np

def naive_softmax(logits):
    """Implementasi naif berisiko overflow pada logits bernilai besar."""
    exp_z = np.exp(logits)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def stable_softmax(logits):
    """
    Softmax stabil numerik dengan mengurangi nilai maksimum logits:
    softmax(z_i) = exp(z_i - C) / sum(exp(z_j - C)), di mana C = max(z).
    Secara matematis nilainya identik karena faktor exp(-C) saling membatalkan.
    """
    z = np.asarray(logits, dtype=float)
    # Kurangi nilai maksimum sepanjang sumbu kelas terakhir
    shift_z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shift_z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def run_softmax_demonstration():
    print("=" * 80)
    print("CHAPTER 21: SOFTMAX & STABILITAS NUMERIK UNTUK MULTICLASS CLASSIFICATION")
    print("=" * 80)
    
    # 1. Kasus Logits Standar (Aman)
    standard_logits = np.array([2.0, 1.0, 0.1])
    probs_standard = stable_softmax(standard_logits)
    print("1. Kasus Logits Normal [2.0, 1.0, 0.1]:")
    print(f"   Distribusi Probabilitas : {np.round(probs_standard, 4)}")
    print(f"   Total Probabilitas      : {np.sum(probs_standard):.4f} (Wajib = 1.0)")
    
    # 2. Kasus Logits Bernilai Sangat Besar (Uji Stres Overflow)
    large_logits = np.array([1000.0, 1001.0, 1002.0])
    print("\n2. Kasus Logits Sangat Besar [1000.0, 1001.0, 1002.0]:")
    
    # Menguji implementasi naif
    try:
        with np.errstate(over='ignore', invalid='ignore'):
            naive_result = naive_softmax(large_logits)
        print(f"   Hasil Naive Softmax     : {naive_result} (Gagal karena Overflow exp(1000) = inf!)")
    except Exception as e:
        print(f"   Naive Softmax Error     : {e}")
        
    # Menguji implementasi stabil
    stable_result = stable_softmax(large_logits)
    print(f"   Hasil Stable Softmax    : {np.round(stable_result, 4)}")
    print(f"   Total Probabilitas      : {np.sum(stable_result):.4f}")
    
    # 3. Kasus Batch (Matrix Logits N x K)
    batch_logits = np.array([
        [2.5, 1.2, 0.3],   # Sampel 1: Dominan Kelas 0
        [0.1, 3.8, 0.4],   # Sampel 2: Dominan Kelas 1
        [1.0, 1.1, 4.5]    # Sampel 3: Dominan Kelas 2
    ])
    batch_probs = stable_softmax(batch_logits)
    print("\n3. Evaluasi Batch 3 Sampel (3 Kelas):")
    for i, (log, prob) in enumerate(zip(batch_logits, batch_probs)):
        pred_class = np.argmax(prob)
        print(f"   Sampel {i+1} Logits: {log} -> Probs: {np.round(prob, 4)} -> Prediksi: Kelas {pred_class}")
        
    print("=" * 80)
    print("BUKTI MATEMATIS IDENTITAS SOFTMAX TERSHIFT:")
    print("exp(z_i - C) / sum_j exp(z_j - C) = [exp(z_i) * exp(-C)] / [sum_j exp(z_j) * exp(-C)]")
    print("                                   = exp(z_i) / sum_j exp(z_j)")
    print("Pengurangan C = max(z) menjamin eksponen terbesar adalah exp(0) = 1, mencegah overflow 100%!")

if __name__ == "__main__":
    run_softmax_demonstration()

