"""
Script 03: Group-Based Splitting (Preventing Cross-Group Leakage)
Chapter 12: ML Workflow, Dataset & Feature Engineering

Mendemonstrasikan pemisahan berbasis kelompok (GroupSplit / GroupKFold)
menggunakan entitas 'machine_id'.
Jika data dari mesin yang sama muncul di Train dan Test set, model dapat
menghafal karakteristik spesifik mesin (data leakage / pseudoreplication),
bukan mempelajari pola kegagalan umum.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split


def load_telemetry_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_telemetry.csv"
    return pd.read_csv(data_path)


def demo_naive_vs_group_split(df: pd.DataFrame):
    print("--- 1. NAIVE SPLIT VS GROUP SHUFFLE SPLIT ---")
    groups = df["machine_id"]
    
    # 1. Naive random split
    train_naive, test_naive = train_test_split(df, test_size=0.20, random_state=42)
    mch_train_naive = set(train_naive["machine_id"].unique())
    mch_test_naive = set(test_naive["machine_id"].unique())
    overlap_naive = mch_train_naive.intersection(mch_test_naive)
    
    print("A. Naive Random Split:")
    print(f"   Mesin pada Training Set : {sorted(list(mch_train_naive))}")
    print(f"   Mesin pada Test Set     : {sorted(list(mch_test_naive))}")
    print(f"   Overlap Mesin           : {sorted(list(overlap_naive))} -> ({len(overlap_naive)} mesin bocor ke test!)")
    
    # 2. GroupShuffleSplit (memisahkan mesin secara eksklusif)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    
    train_group = df.iloc[train_idx]
    test_group = df.iloc[test_idx]
    
    mch_train_group = set(train_group["machine_id"].unique())
    mch_test_group = set(test_group["machine_id"].unique())
    overlap_group = mch_train_group.intersection(mch_test_group)
    
    print("\nB. Group-Aware Split (GroupShuffleSplit):")
    print(f"   Mesin pada Training Set : {sorted(list(mch_train_group))} ({len(train_group)} baris)")
    print(f"   Mesin pada Test Set     : {sorted(list(mch_test_group))} ({len(test_group)} baris)")
    print(f"   Overlap Mesin           : {sorted(list(overlap_group))} -> Bebas dari kebocoran kelompok!\n")


def demo_group_kfold(df: pd.DataFrame, n_splits: int = 5):
    print(f"--- 2. {n_splits}-FOLD GROUP K-FOLD CROSS VALIDATION ---")
    groups = df["machine_id"]
    gkf = GroupKFold(n_splits=n_splits)
    
    print(f"{'Fold':<6} | {'Train Machines':<32} | {'Test Machines':<18} | {'Test Samples':<12}")
    print("-" * 75)
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        train_mch = sorted(df.iloc[train_idx]["machine_id"].unique())
        test_mch = sorted(df.iloc[test_idx]["machine_id"].unique())
        
        # Pastikan strictly disjoint
        assert set(train_mch).isdisjoint(set(test_mch)), "Terjadi kebocoran kelompok pada fold!"
        
        print(f"Fold {fold+1:<2} | {str(train_mch):<32} | {str(test_mch):<18} | {len(test_idx):<12}")
        
    print("-" * 75)
    print("Semua fold menjamin bahwa model dievaluasi pada mesin yang BELUM PERNAH dilihat sebelumnya.")


def main():
    print("=" * 75)
    print("DEMONSTRASI GROUP-BASED SPLITTING (MENCEGAH CROSS-ENTITY LEAKAGE)")
    print("=" * 75)
    df = load_telemetry_data()
    demo_naive_vs_group_split(df)
    demo_group_kfold(df)
    print("=" * 75)


if __name__ == "__main__":
    main()

