# ===================================================================
# MODEL 2 (VERSI BARU DENGAN CONFUSION MATRIX)
# ===================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
# Impor library untuk membuat confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
import numpy as np

print("Memulai proses optimasi Model 2...")

# 1. Muat dataset sintetis
try:
    df_user = pd.read_csv('dataset_user_sintetis.csv')
except FileNotFoundError:
    print("❌ ERROR: File 'dataset_user_sintetis.csv' tidak ditemukan.")
    exit()
print("✅ Dataset sintetis berhasil dimuat.")

# 2. Pisahkan Fitur (X) dan Label (y)
X = df_user.drop('profil_sampah', axis=1)
y = df_user['profil_sampah']
# Simpan nama kelas untuk label di grafik nanti
class_names = sorted(y.unique())

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler telah disimpan ke 'scaler.pkl'")

# 4. Bagi data menjadi data latih dan data tes
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("✅ Data berhasil dibagi menjadi data training dan testing.")

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

# 6. Latih Ulang Model Final dengan 'K' Terbaik
print(f"\nMelatih ulang model final dengan k={best_k}...")
final_model_knn = KNeighborsClassifier(n_neighbors=best_k)
final_model_knn.fit(X_train, y_train)
joblib.dump(final_model_knn, 'model_knn.pkl')
print("✅ Model final telah disimpan ke 'model_knn.pkl'")

# ===================================================================
# BAGIAN UTAMA: Membuat dan Menyimpan Grafik Confusion Matrix
# ===================================================================
print("\nMembuat dan menyimpan grafik Confusion Matrix...")
# Lakukan prediksi pada data tes dengan model final
y_pred_final = final_model_knn.predict(X_test)

# Buat confusion matrix
cm = confusion_matrix(y_test, y_pred_final, labels=class_names)

# Tampilkan confusion matrix menggunakan ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Gambar dan kustomisasi plot
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d') # 'd' untuk format angka integer
ax.set_title('Confusion Matrix untuk Model KNN')
plt.xticks(rotation=45)

# Simpan grafik sebagai file gambar di folder static
plt.savefig('static/grafik_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✅ Grafik Confusion Matrix telah disimpan sebagai 'grafik_confusion_matrix.png'.")
# ===================================================================

print("\nProses Selesai.")
