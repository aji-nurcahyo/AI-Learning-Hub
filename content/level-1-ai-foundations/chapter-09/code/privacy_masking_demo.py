"""Privacy Masking, Pseudonymization & Re-identification Risk Demo.

Mendemonstrasikan teknik penyamaran identitas (pseudonymization / masking)
serta pembuktian analitis mengapa penyamaran sederhana bukanlah anonimisasi mutlak
di bawah payung hukum UU Perlindungan Data Pribadi (UU PDP No. 27/2022) dan GDPR.
Memperagakan bahaya serangan re-identifikasi melalui tautan quasi-identifiers
(studi kasus Latanya Sweeney) serta konsep k-Anonymity.
"""

import hashlib
import pandas as pd


def hash_identifier(val: str, salt: str = "itera_ai_hub_2026") -> str:
    """Melakukan hashing kriptografis SHA-256 dengan salt unik."""
    salted = f"{salt}_{val}".encode("utf-8")
    return hashlib.sha256(salted).hexdigest()[:12]


def mask_string(val: str, visible_start: int = 2, visible_end: int = 2) -> str:
    """Menyamarkan string dengan karakter bintang (masking)."""
    if len(val) <= visible_start + visible_end:
        return "*" * len(val)
    return val[:visible_start] + "*" * (len(val) - visible_start - visible_end) + val[-visible_end:]


def main() -> None:
    # 1. Dataset Rekam Medis / Akademik Mahasiswa Asli (Data Pribadi Spesifik & Umum)
    raw_data = [
        {"nik": "1801021999010001", "nama": "Ahmad Fauzi", "usia": 25, "kelamin": "L", "kode_pos": "35141", "diagnosis": "Anxiety Disorder"},
        {"nik": "1801021999020002", "nama": "Budi Santoso", "usia": 25, "kelamin": "L", "kode_pos": "35141", "diagnosis": "Hypertension"},
        {"nik": "1801022000030003", "nama": "Citra Lestari", "usia": 24, "kelamin": "P", "kode_pos": "35142", "diagnosis": "Diabetes Melitus"},
        {"nik": "1801022001040004", "nama": "Dewi Sartika", "usia": 23, "kelamin": "P", "kode_pos": "35143", "diagnosis": "Major Depression"},
        {"nik": "1801021998050005", "nama": "Eko Prasetyo", "usia": 26, "kelamin": "L", "kode_pos": "35145", "diagnosis": "Cardiac Arrhythmia"}
    ]
    df_raw = pd.DataFrame(raw_data)
    
    print("=== DATASET ASLI (MENGANDUNG DATA PRIBADI SPESIFIK & UMUM) ===")
    print(df_raw.to_string(index=False))
    
    # 2. Penyamaran Sederhana (Pseudonimisasi & Masking NIK / Nama)
    df_pseudonymized = df_raw.copy()
    df_pseudonymized["nik_hash"] = df_pseudonymized["nik"].apply(hash_identifier)
    df_pseudonymized["nama_masked"] = df_pseudonymized["nama"].apply(mask_string)
    
    # Menghapus identitas langsung (Direct Identifiers)
    df_sanitized = df_pseudonymized[["nik_hash", "nama_masked", "usia", "kelamin", "kode_pos", "diagnosis"]]
    
    print("\n=== DATASET TER-PSEUDONIMISASI (IDENTITAS LANGSUNG DISAMARKAN) ===")
    print(df_sanitized.to_string(index=False))
    
    # 3. Analisis Kerentanan Re-Identifikasi (Serangan Tautan Basis Data Terbuka / Voter Registry)
    print("\n=== PEMBUKTIAN KERENTANAN: SERANGAN TAUTAN QUASI-IDENTIFIER ===")
    print("Misalkan seorang penyerang memiliki data publik Daftar Pemilih Tetap (DPT) berikut:")
    
    voter_registry = pd.DataFrame([
        {"nama_asli": "Eko Prasetyo", "usia": 26, "kelamin": "L", "kode_pos": "35145", "alamat": "Jl. Ryacudu No. 12, Sukarame"},
        {"nama_asli": "Ferry Kurniawan", "usia": 30, "kelamin": "L", "kode_pos": "35145", "alamat": "Jl. Terusan Ryacudu No. 5"},
        {"nama_asli": "Gita Gutawa", "usia": 29, "kelamin": "P", "kode_pos": "35141", "alamat": "Jl. Way Halim No. 88"}
    ])
    print(voter_registry.to_string(index=False))
    
    # Melakukan join pada Quasi-Identifiers: ['usia', 'kelamin', 'kode_pos']
    linked = pd.merge(
        df_sanitized,
        voter_registry,
        on=["usia", "kelamin", "kode_pos"],
        how="inner"
    )
    
    print("\nHASIL RE-IDENTIFIKASI REKAM MEDIS SENSITIF:")
    for _, row in linked.iterrows():
        print(f"-> Target Teridentifikasi: {row['nama_asli']}")
        print(f"   Kombinasi Quasi-ID   : Usia={row['usia']}, Kelamin={row['kelamin']}, Kode Pos={row['kode_pos']}")
        print(f"   Diagnosis Rahasia    : {row['diagnosis']}")
        print(f"   Alamat Domisili      : {row['alamat']}\n")
        
    print("=== LANDASAN HUKUM UU PDP (UU NO. 27/2022) & GDPR ===")
    print("1. Pasal 1 angka 1 UU PDP mendefinisikan Data Pribadi sebagai data apa pun")
    print("   yang dapat mengidentifikasi seseorang, baik tersendiri MAUPUN DIKOMBINASIKAN")
    print("   dengan informasi lainnya secara langsung atau tidak langsung.")
    print("2. Pseudonimisasi (Pasal 35) BUKANLAH Anonimisasi Mutlak (Recital 26 GDPR).")
    print("   Kombinasi quasi-identifiers (Usia + Kelamin + Kode Pos) mampu mengidentifikasi")
    print("   87% populasi (studi Latanya Sweeney, 2000).")
    print("3. Solusi Privasi: Terapkan generalisasi k-Anonymity (misal: usia dijadikan rentang 20-29)")
    print("   atau Differential Privacy sebelum membagikan dataset pelatihan AI.")


if __name__ == "__main__":
    main()

