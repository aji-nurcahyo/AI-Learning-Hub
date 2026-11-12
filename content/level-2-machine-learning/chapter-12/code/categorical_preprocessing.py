"""
Script 06: Categorical Preprocessing (One-Hot & Ordinal Encoding)
Chapter 12: ML Workflow, Dataset & Feature Engineering

Mendemonstrasikan penanganan fitur kategorikal:
1. OneHotEncoder dengan parameter handle_unknown='ignore' (mencegah error pada runtime saat ada kategori baru).
2. OrdinalEncoder untuk fitur berkategori bertingkat (misal: LOW < MEDIUM < HIGH).
3. Pengujian ketahanan terhadap kategori asing yang belum pernah dilihat saat pelatihan (unseen category).
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split


def load_telemetry_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_telemetry.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI PRA-PEMROSESAN FITUR KATEGORIKAL")
    print("=" * 75)
    
    df = load_telemetry_data()
    cat_nominal = ["cooling_status"]
    cat_ordinal = ["warning_flag"]
    
    X_train_raw, X_test_raw, _, _ = train_test_split(
        df[cat_nominal + cat_ordinal], df["failure"], test_size=0.20, random_state=42
    )
    
    print("1. DISTRIBUSI KATEGORI PADA DATA LATIH (TRAIN):")
    print(f"   Nominal (cooling_status): {X_train_raw['cooling_status'].value_counts().to_dict()}")
    print(f"   Ordinal (warning_flag)  : {X_train_raw['warning_flag'].value_counts().to_dict()}\n")
    
    # A. One-Hot Encoding pada Nominal (cooling_status)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(X_train_raw[cat_nominal])
    
    train_ohe = ohe.transform(X_train_raw[cat_nominal])
    ohe_cols = ohe.get_feature_names_out(cat_nominal)
    print("2. ONE-HOT ENCODING (NOMINAL):")
    print(f"   Kolom hasil OHE: {list(ohe_cols)}")
    print(f"   Contoh 3 baris pertama hasil OHE Train:\n{pd.DataFrame(train_ohe[:3], columns=ohe_cols)}\n")
    
    # B. Ordinal Encoding pada Ordinal (warning_flag: LOW=0, MEDIUM=1, HIGH=2)
    warning_order = [["LOW", "MEDIUM", "HIGH"]]
    ord_enc = OrdinalEncoder(categories=warning_order, handle_unknown="use_encoded_value", unknown_value=-1)
    ord_enc.fit(X_train_raw[cat_ordinal])
    
    train_ord = ord_enc.transform(X_train_raw[cat_ordinal])
    print("3. ORDINAL ENCODING (HIERARKIS):")
    print(f"   Mapping Kategori: LOW=0, MEDIUM=1, HIGH=2")
    print(f"   Contoh 3 baris pertama hasil Ordinal Train:\n{pd.DataFrame(train_ord[:3], columns=cat_ordinal)}\n")
    
    # C. Simulasi Kejutan: Unseen Categories pada Data Baru / Produksi
    print("4. SIMULASI DATA PRODUKSI DENGAN KATEGORI ASING (UNSEEN CATEGORIES):")
    unseen_df = pd.DataFrame({
        "cooling_status": ["ACTIVE", "EMERGENCY_COOLING", "OFF"],  # EMERGENCY_COOLING tidak ada di train!
        "warning_flag": ["LOW", "CRITICAL_RED", "HIGH"]             # CRITICAL_RED tidak ada di train!
    })
    print("   Data Asing Baru:")
    print(unseen_df)
    
    # Transformasi OHE dengan handle_unknown='ignore' -> menghasilkan semua nol untuk kategori baru
    unseen_ohe = ohe.transform(unseen_df[cat_nominal])
    unseen_ohe_df = pd.DataFrame(unseen_ohe, columns=ohe_cols)
    print("\n   Hasil OHE pada data asing (baris ke-2 menjadi all 0 tanpa crash):")
    print(unseen_ohe_df)
    
    # Transformasi Ordinal dengan unknown_value=-1
    unseen_ord = ord_enc.transform(unseen_df[cat_ordinal])
    unseen_ord_df = pd.DataFrame(unseen_ord, columns=cat_ordinal)
    print("\n   Hasil Ordinal pada data asing (kategori tidak dikenal ditandai -1):")
    print(unseen_ord_df)
    
    print("\n[OK] Sistem pra-pemrosesan terbukti tangguh dan aman dari kebocoran maupun runtime crash!")
    print("=" * 75)


if __name__ == "__main__":
    main()

