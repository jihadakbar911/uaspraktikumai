# ===================================================================
# MODEL 1: PREDIKSI SAMPAH KOTA BANDUNG DENGAN REGRESOR PENDUDUK
# --- VERSI FINAL SESUAI STRUKTUR DATA ASLI ---
# ===================================================================

# Langkah 1: Impor library yang dibutuhkan
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

print("Memulai proses training Model 1...")

# Langkah 2: Muat kedua dataset
try:
    df_sampah = pd.read_csv('jumlah_capaian_penanganan_sampah_di_kota_bandung.csv')
    df_penduduk = pd.read_csv('jumlah_penduduk_kota_bandung.csv')
except FileNotFoundError:
    print("ERROR: File tidak ditemukan. Pastikan kedua file CSV ada di folder yang sama.")
    exit()

print("Dataset berhasil dimuat.")

# Langkah 3: Agregasi Data Sampah dari Bulanan ke Tahunan
# Kita menjumlahkan 'jumlah_sampah' untuk setiap 'tahun'
print("Mengagregasi data sampah bulanan menjadi tahunan...")
df_sampah_tahunan = df_sampah.groupby('tahun')['jumlah_sampah'].sum().reset_index()

# Langkah 4: Pilih kolom yang relevan
# Kita gunakan data sampah yang sudah menjadi tahunan
df_penduduk_clean = df_penduduk[['tahun', 'jumlah_penduduk']]

# Langkah 5: Gabungkan data sampah tahunan dengan data penduduk
df_gabungan = pd.merge(df_sampah_tahunan, df_penduduk_clean, on='tahun', how='inner')
df_gabungan = df_gabungan.dropna().sort_values('tahun')

print("\n--- Data Gabungan (Tahunan) Siap Olah ---")
print(df_gabungan)

# Langkah 6: Pra-pemrosesan Data untuk Prophet
df_prophet = df_gabungan.rename(columns={
    'tahun': 'ds',
    'jumlah_sampah': 'y'
})

# Mengubah kolom 'ds' ke format datetime
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'].astype(str) + '-12-31')

print("\n--- Data Final Siap untuk Prophet ---")
print(df_prophet.head())

# Langkah 7: Inisialisasi, Tambahkan Regresor, dan Latih Model
model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
model.add_regressor('jumlah_penduduk')
model.fit(df_prophet)
print("\nModel berhasil dilatih dengan regresor jumlah penduduk.")

# Langkah 8: Buat Frame Masa Depan & Isi Data Regresor
future = model.make_future_dataframe(periods=3, freq='Y')

# Estimasi populasi untuk masa depan
# Kita gunakan pertumbuhan rata-rata dari data yang ada
pertumbuhan_penduduk_rata2 = df_prophet['jumlah_penduduk'].diff().mean()
populasi_terakhir = df_prophet['jumlah_penduduk'].iloc[-1]
# Buat list populasi masa depan
populasi_masa_depan = [populasi_terakhir + (pertumbuhan_penduduk_rata2 * i) for i in range(1, 4)]
# Gabungkan dengan data populasi historis
semua_populasi = list(df_prophet['jumlah_penduduk']) + populasi_masa_depan
future['jumlah_penduduk'] = semua_populasi

# Langkah 9: Lakukan Prediksi
forecast = model.predict(future)

# ===================================================================
# LANGKAH 10 (VERSI PERBAIKAN) - Tampilkan Hasil dengan Benar
# ===================================================================

print("\n--- Hasil Prediksi ---")

# Gabungkan data asli (dari 'future') dengan hasil prediksi (dari 'forecast')
# Kita hanya ambil kolom penting dari forecast untuk menghindari nama kolom yang sama
hasil_akhir = pd.merge(future, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

# Sekarang, 'jumlah_penduduk' di tabel ini adalah angka sebenarnya
print(hasil_akhir[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'jumlah_penduduk']].tail())


# Visualisasi tidak perlu diubah dan tetap benar
print("\nMenampilkan grafik...")
fig1 = model.plot(forecast)
plt.title('Prediksi Jumlah Sampah di Kota Bandung', fontsize=16)
plt.xlabel('Tahun', fontsize=12)
plt.ylabel('Jumlah Sampah (Ton)', fontsize=12)
plt.show()

fig2 = model.plot_components(forecast)
plt.show()

print("\nProses Selesai.")