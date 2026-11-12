"""
lora_parameter_count.py
Analisis komparatif jumlah parameter terbobot dan efisiensi memori LoRA:
- Menghitung parameter base model vs LoRA adapters lintas berbagai dimensi (d) dan rank (r)
- Menghitung persentase penghematan parameter dan memori gradien
- Memberikan panduan praktis penentuan hiperparameter rank (r) dan alpha
"""

def calculate_lora_efficiency(d_in, d_out, rank):
    """Menghitung parameter base, LoRA adapter, dan persentase penghematan."""
    base_params = d_in * d_out
    lora_params = (rank * d_in) + (d_out * rank)
    savings = (1.0 - (lora_params / base_params)) * 100.0
    
    # Estimasi memori float32 (4 byte per parameter)
    # Pada full fine-tuning: Bobot + Gradien + 2 Momen Adam = 4 * 4 byte = 16 byte per parameter
    # Pada LoRA: Base weights frozen (hanya 4 byte, tanpa gradien & momen)
    #            LoRA weights (16 byte per param)
    mem_full_mb = (base_params * 16) / (1024 * 1024)
    mem_lora_mb = (base_params * 4 + lora_params * 16) / (1024 * 1024)
    mem_savings = (1.0 - (mem_lora_mb / mem_full_mb)) * 100.0
    
    return {
        "base_params": base_params,
        "lora_params": lora_params,
        "param_savings": savings,
        "mem_full_mb": mem_full_mb,
        "mem_lora_mb": mem_lora_mb,
        "mem_savings": mem_savings
    }

def main():
    print("=== Analisis Efisiensi Parameter & Memori LoRA ===")
    
    # Studi 1: Pengaruh Nilai Rank (r) pada Lapisan Standar (d_in = 768, d_out = 768)
    print("\n--- Studi 1: Pengaruh Rank (r) pada Lapisan Transformer Standar (768 x 768) ---")
    print(f"{'Rank (r)':<10} {'Param Base':<15} {'Param LoRA':<15} {'Hemat Param (%)':<18} {'Hemat Memori (%)'}")
    print("-" * 75)
    
    ranks = [1, 2, 4, 8, 16, 32, 64]
    for r in ranks:
        res = calculate_lora_efficiency(768, 768, r)
        print(f"r = {r:<6} {res['base_params']:<15,d} {res['lora_params']:<15,d} "
              f"{res['param_savings']:<18.2f} {res['mem_savings']:.2f}%")
        
    # Studi 2: Skala Dimensi Model (r = 8 tetap)
    print("\n--- Studi 2: Skala Dimensi Model (Rank r = 8 Konstan) ---")
    print(f"{'Dimensi (d)':<12} {'Param Base':<15} {'Param LoRA':<15} {'Hemat Param (%)':<18} {'Memori FT vs LoRA'}")
    print("-" * 75)
    
    dimensions = [256, 512, 1024, 2048, 4096]
    for d in dimensions:
        res = calculate_lora_efficiency(d, d, rank=8)
        mem_str = f"{res['mem_full_mb']:.1f} MB -> {res['mem_lora_mb']:.1f} MB"
        print(f"{d}x{d:<8} {res['base_params']:<15,d} {res['lora_params']:<15,d} "
              f"{res['param_savings']:<18.2f} {mem_str}")
        
    print("\nKesimpulan Praktis:")
    print("1. Semakin besar dimensi model dasar, semakin dramatis persentase penghematan LoRA (>99%).")
    print("2. Pada mayoritas tugas klasifikasi dan instruksi, rank r=4 atau r=8 sudah cukup memadai.")
    print("3. Penghematan memori VRAM/RAM berasal dari ketiadaan status optimizer untuk bobot beku.")

if __name__ == "__main__":
    main()

