"""
Chapter 22 - Backpropagation & Optimization
Script 14: Pemotongan Gradien (Gradient Clipping)
Mengimplementasikan teknik pencegahan lonjakan gradien tak terkendali (Exploding Gradients):
pemotongan nilai batas elemen (Clip by Value) dan pemotongan norma vektor global (Clip by Global Norm).
"""

import numpy as np

def clip_by_value(grad_vector, min_val=-1.0, max_val=1.0):
    """Memotong setiap komponen skalar gradien ke rentang [min_val, max_val]."""
    return np.clip(grad_vector, min_val, max_val)

def clip_by_norm(grad_vector, max_norm=1.0):
    """
    Memotong norma global vektor gradien:
    jika ||g||_2 > max_norm, maka g = g * (max_norm / ||g||_2).
    Keunggulan: Mempertahankan arah vektor gradien secara sempurna tanpa distorsi sudut.
    """
    norm = np.linalg.norm(grad_vector)
    if norm > max_norm:
        clipped = grad_vector * (max_norm / norm)
        return clipped, norm, max_norm
    return grad_vector, norm, norm

def run_gradient_clipping_demo():
    print("=" * 85)
    print("CHAPTER 22: TEKNIK MITIGASI EXPLODING GRADIENTS (GRADIENT CLIPPING)")
    print("=" * 85)
    
    # Skenario lonjakan gradien ekstrem (misalnya akibat nilai loss tak stabil atau kedalaman jaringan)
    unstable_gradient = np.array([125.0, -840.0, 45.0, -0.05, 320.0])
    original_norm = np.linalg.norm(unstable_gradient)
    
    print(f"Vektor Gradien Meledak (Unstable Grad) : {unstable_gradient}")
    print(f"Norma L2 Gradien Asli (||g||_2)        : {original_norm:.4f} (Sangat berbahaya!)")
    print("-" * 85)
    
    # 1. Clip by Value [-5.0, 5.0]
    clipped_val = clip_by_value(unstable_gradient, -5.0, 5.0)
    norm_val = np.linalg.norm(clipped_val)
    print("1. Pemotongan Nilai Elemen (Clip by Value [-5.0, 5.0]):")
    print(f"   Vektor Terpotong : {clipped_val}")
    print(f"   Norma L2 Baru    : {norm_val:.4f}")
    
    # Cosine similarity untuk mengecek perubahan arah
    cos_sim_val = np.dot(unstable_gradient, clipped_val) / (original_norm * norm_val)
    print(f"   Cosine Sim Arah  : {cos_sim_val:.4f} (Arah vektor terdistorsi akibat pemotongan per elemen)")
    
    # 2. Clip by Global Norm (max_norm = 5.0)
    clipped_norm, _, target_norm = clip_by_norm(unstable_gradient, max_norm=5.0)
    new_norm = np.linalg.norm(clipped_norm)
    print("\n2. Pemotongan Norma Global (Clip by Global Norm [max_norm = 5.0]):")
    print(f"   Vektor Terpotong : {np.round(clipped_norm, 4)}")
    print(f"   Norma L2 Baru    : {new_norm:.4f} (Tepat sesuai batas max_norm)")
    
    cos_sim_norm = np.dot(unstable_gradient, clipped_norm) / (original_norm * new_norm)
    print(f"   Cosine Sim Arah  : {cos_sim_norm:.6f} (Arah vektor 100% PRESERVED, sudut identik!)")
    
    print("=" * 85)
    print("KESIMPULAN REKAYASA:")
    print("Clip by Global Norm adalah standar industri dalam melatih RNN, Transformer, dan Deep MLP")
    print("karena mencegah lonjakan langkah tanpa mengubah arah lintasan pencarian optimum.")

if __name__ == "__main__":
    run_gradient_clipping_demo()

