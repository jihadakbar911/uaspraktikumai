# ===================================================================
# MODEL 2 (VERSI PENINGKATAN AKURASI)
# ===================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler # Impor untuk scaling
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt # Impor untuk membuat grafik
import numpy as np

print("Memulai proses optimasi Model 2...")

# 1. Muat dataset
try:
    df_user = pd.read_csv('dataset_user_sintetis.csv')
except FileNotFoundError:
    print("❌ ERROR: File 'dataset_user_sintetis.csv' tidak ditemukan.")
    exit()
print("✅ Dataset berhasil dimuat.")

# 2. Pisahkan Fitur (X) dan Label (y)
X = df_user.drop('profil_sampah', axis=1)
y = df_user['profil_sampah']

# 3. FEATURE SCALING: Menyamakan skala semua fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Fitur telah di-scaling.")

# Simpan scaler ini untuk digunakan nanti di website
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler telah disimpan ke 'scaler.pkl'")

# 4. Bagi data yang SUDAH DI-SCALING menjadi data latih dan tes
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("✅ Data berhasil dibagi.")

# 5. MENCARI NILAI 'K' TERBAIK
k_range = range(1, 21) # Kita akan coba k dari 1 sampai 20
accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Cari k dengan akurasi tertinggi
best_k = k_range[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"\nNilai 'k' terbaik ditemukan adalah: {best_k}")
print(f"🎯 Akurasi tertinggi yang dicapai: {best_accuracy * 100:.2f}%")

# Tampilkan grafik untuk melihat performa setiap 'k'
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies, marker='o', linestyle='dashed')
plt.title('Akurasi vs. Nilai K')
plt.xlabel('Nilai K')
plt.ylabel('Akurasi')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# 6. Latih Ulang Model Final dengan 'K' Terbaik
print(f"\nMelatih ulang model final dengan k={best_k}...")
final_model_knn = KNeighborsClassifier(n_neighbors=best_k)
final_model_knn.fit(X_train, y_train)
print("✅ Model final berhasil dilatih.")

# Simpan model final
joblib.dump(final_model_knn, 'model_knn.pkl')
print("✅ Model final telah disimpan kembali ke 'model_knn.pkl'")
print("\nProses Selesai.")