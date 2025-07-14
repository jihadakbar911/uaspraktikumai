# ===================================================================
# MODEL 2 (VERSI BARU DENGAN FITUR UMUM)
# ===================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import numpy as np

print("Memulai proses optimasi Model 2 dengan fitur baru...")

# 1. Muat dataset baru
try:
    df_user = pd.read_csv('dataset_user_sintetis.csv')
except FileNotFoundError:
    print("❌ ERROR: File 'dataset_user_sintetis.csv' tidak ditemukan.")
    exit()
print("✅ Dataset baru berhasil dimuat.")

# 2. Pisahkan Fitur (X) dan Label (y)
X = df_user.drop('profil_sampah', axis=1)
y = df_user['profil_sampah']
print(f"✅ Fitur yang digunakan: {list(X.columns)}")

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler baru telah disimpan ke 'scaler.pkl'")

# 4. Bagi data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Mencari Nilai 'K' Terbaik
k_range = range(1, 21)
accuracies = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

best_k = k_range[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"\nNilai 'k' terbaik ditemukan adalah: {best_k}")
print(f"🎯 Akurasi tertinggi yang dicapai: {best_accuracy * 100:.2f}%")

# Tampilkan grafik performa
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies, marker='o', linestyle='dashed')
plt.title('Akurasi vs. Nilai K (Fitur Baru)')
plt.xlabel('Nilai K')
plt.ylabel('Akurasi')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# 6. Latih Ulang & Simpan Model Final
final_model_knn = KNeighborsClassifier(n_neighbors=best_k)
final_model_knn.fit(X_train, y_train)
joblib.dump(final_model_knn, 'model_knn.pkl')
print("✅ Model final baru telah disimpan ke 'model_knn.pkl'")
print("\nProses Selesai.")
