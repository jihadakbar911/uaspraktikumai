# ===================================================================
# GENERATE PROPHET DATA & GRAFIK
# Tujuan: Menjalankan model Prophet sekali saja untuk menghasilkan
#         file gambar grafik (.png) yang akan ditampilkan di web.
# ===================================================================

# 1. Impor library yang dibutuhkan
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import json

print("Memulai proses pra-kalkulasi data dan pembuatan grafik Prophet...")

# 2. Muat kedua dataset
try:
    df_sampah = pd.read_csv('jumlah_capaian_penanganan_sampah_di_kota_bandung.csv')
    df_penduduk = pd.read_csv('jumlah_penduduk_kota_bandung.csv')
    print("✅ Dataset berhasil dimuat.")
except FileNotFoundError:
    print("❌ ERROR: File CSV tidak ditemukan. Pastikan file berada di folder yang sama.")
    exit()

# 3. Agregasi Data Sampah dari Bulanan ke Tahunan
df_sampah_tahunan = df_sampah.groupby('tahun')['jumlah_sampah'].sum().reset_index()
print("✅ Data sampah berhasil diagregasi menjadi tahunan.")

# 4. Pilih kolom yang relevan dan gabungkan dataset
df_penduduk_clean = df_penduduk[['tahun', 'jumlah_penduduk']]
df_gabungan = pd.merge(df_sampah_tahunan, df_penduduk_clean, on='tahun', how='inner').dropna()

# 5. Persiapan data untuk Prophet
df_prophet = df_gabungan.rename(columns={'tahun': 'ds', 'jumlah_sampah': 'y'})
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'].astype(str) + '-12-31')
print("✅ Data telah diformat untuk Prophet.")

# 6. Latih model Prophet dengan regresor penduduk
model_prophet = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
model_prophet.add_regressor('jumlah_penduduk')
model_prophet.fit(df_prophet)
print("✅ Model Prophet berhasil dilatih.")

# 7. Buat frame prediksi masa depan
future = model_prophet.make_future_dataframe(periods=3, freq='YE')
pertumbuhan_penduduk_rata2 = df_prophet['jumlah_penduduk'].diff().mean()
populasi_terakhir = df_prophet['jumlah_penduduk'].iloc[-1]
populasi_masa_depan = [populasi_terakhir + (pertumbuhan_penduduk_rata2 * i) for i in range(1, 4)]
future['jumlah_penduduk'] = list(df_prophet['jumlah_penduduk']) + populasi_masa_depan

# 8. Lakukan prediksi
forecast = model_prophet.predict(future)
print("✅ Prediksi masa depan telah dibuat.")

# 9. Buat dan Simpan Grafik Prediksi sebagai Gambar PNG
print("Membuat dan menyimpan grafik prediksi...")
fig1 = model_prophet.plot(forecast)
plt.title('Prediksi Volume Sampah Kota Bandung (Ton)', fontsize=16)
plt.xlabel('Tahun', fontsize=12)
plt.ylabel('Jumlah Sampah (Ton)', fontsize=12)
plt.grid(True)

# Menyimpan objek grafik (fig1) ke dalam sebuah file gambar di folder static
# dpi=150 membuat resolusi gambar lebih baik
# bbox_inches='tight' memastikan tidak ada bagian yang terpotong
fig1.savefig('static/grafik_prediksi.png', dpi=150, bbox_inches='tight')
print("✅ Grafik telah disimpan sebagai 'grafik_prediksi.png' di dalam folder 'static'.")

# (Opsional) Simpan data mentah ke JSON jika suatu saat dibutuhkan
hasil_prediksi = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
hasil_prediksi['ds'] = hasil_prediksi['ds'].dt.strftime('%Y')
data_to_save = hasil_prediksi.to_dict(orient='records')
with open('static/prophet_forecast.json', 'w') as f:
    json.dump(data_to_save, f)
print("✅ Data prediksi mentah juga disimpan sebagai 'prophet_forecast.json'.")


print("\nProses Selesai.")
