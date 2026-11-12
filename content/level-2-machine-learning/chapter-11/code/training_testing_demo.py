"""
Chapter 11: Introduction to Machine Learning
Script 5: training_testing_demo.py

Tujuan:
Membuktikan secara empiris mengapa dataset WAJIB dibagi menjadi Training Set dan Test Set:
- Evaluasi model pada data latih saja bersifat bias dan menyesatkan (over-optimistic).
- Generalisasi sejati hanya dapat diukur pada data uji yang belum pernah dilihat model sebelumnya.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def main():
    print("=" * 65)
    print("DEMO: TRAINING VS TESTING SPLIT & GENERALISASI MODEL")
    print("=" * 65)

    # 1. Bangkitkan dataset sintetis 100 sampel 4 fitur
    np.random.seed(42)
    # Kelas 0: distribusi N(0, 1)
    X0 = np.random.normal(loc=0.0, scale=1.0, size=(50, 4))
    y0 = np.zeros(50, dtype=int)
    # Kelas 1: distribusi N(1.5, 1.2)
    X1 = np.random.normal(loc=1.5, scale=1.2, size=(50, 4))
    y1 = np.ones(50, dtype=int)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    # 2. Pembagian Dataset (70% Training, 30% Testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    print(f"Total Sampel Dataset   : {len(X)} sampel")
    print(f"Jumlah Sampel Latih    : {len(X_train)} sampel ({len(X_train)/len(X)*100:.0f}%)")
    print(f"Jumlah Sampel Uji      : {len(X_test)} sampel ({len(X_test)/len(X)*100:.0f}%)")

    # 3. Model 1: k-NN dengan k=1 (Menghafal data latih)
    knn_memorizer = KNeighborsClassifier(n_neighbors=1)
    knn_memorizer.fit(X_train, y_train)

    train_acc_1 = accuracy_score(y_train, knn_memorizer.predict(X_train))
    test_acc_1 = accuracy_score(y_test, knn_memorizer.predict(X_test))

    print("\n--- Kasus A: Model Penghafal (k=1) ---")
    print(f"Akurasi Data Latih (Training Accuracy): {train_acc_1 * 100:.1f}%  <-- Sempurna!")
    print(f"Akurasi Data Uji   (Testing Accuracy) : {test_acc_1 * 100:.1f}%   <-- Turun drastis")

    # 4. Model 2: k-NN dengan k=5 (Generalisasi lebih baik)
    knn_regular = KNeighborsClassifier(n_neighbors=5)
    knn_regular.fit(X_train, y_train)

    train_acc_5 = accuracy_score(y_train, knn_regular.predict(X_train))
    test_acc_5 = accuracy_score(y_test, knn_regular.predict(X_test))

    print("\n--- Kasus B: Model Regular (k=5) ---")
    print(f"Akurasi Data Latih (Training Accuracy): {train_acc_5 * 100:.1f}%")
    print(f"Akurasi Data Uji   (Testing Accuracy) : {test_acc_5 * 100:.1f}%   <-- Lebih stabil")

    print("\n--- Kesimpulan Penting ---")
    print("Akurasi pelatihan 100% sering kali merupakan ilusi penghafalan (memorization).")
    print("Tujuan utama Machine Learning bukanlah menghafal data latih, melainkan")
    print("mampu menggeneralisasi pola ke data masa depan yang belum pernah dilihat.")
    print("=" * 65)


if __name__ == "__main__":
    main()

