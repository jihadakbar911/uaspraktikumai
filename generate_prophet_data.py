# ===================================================================
# GENERATE PROPHET DATA & SEMUA GRAFIK (PREDIKSI + EVALUASI)
# ===================================================================

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
# Impor untuk membuat plot cross-validation
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt

print("Memulai proses pra-kalkulasi dengan dataset JAKARTA TIMUR dan data PENDUDUK...")

# 1. Muat dataset sampah dan penduduk
try:
    nama_file_sampah = 'Dataset Sampah Jakarta Timur 2019-2024.csv'
    df_sampah = pd.read_csv(nama_file_sampah)
    print(f"✅ Dataset '{nama_file_sampah}' berhasil dimuat.")

    nama_file_penduduk = 'Dataset Penduduk Jakarta Timur 2019-2024.csv'
    df_penduduk = pd.read_csv(nama_file_penduduk)
    print(f"✅ Dataset '{nama_file_penduduk}' berhasil dimuat.")

except FileNotFoundError as e:
    print(f"❌ ERROR: File tidak ditemukan. Pastikan kedua file CSV ada di folder proyek.")
    print(f"   Detail: {e}")
    exit()

# Variabel untuk nama kolom
NAMA_KOLOM_TAHUN_SAMPAH = 'tahun'
NAMA_KOLOM_VOLUME = 'timbulan_sampah_tahunan(ton)'
NAMA_KOLOM_TAHUN_PENDUDUK = 'tahun'
NAMA_KOLOM_PENDUDUK = 'jumlah_penduduk_jakarta_timur'

# 2. Membersihkan dan mempersiapkan data
print("Membersihkan dan menggabungkan data...")
def clean_numeric_advanced(value):
    s_value = str(value)
    if s_value.count('.') > 1:
        parts = s_value.split('.')
        cleaned_value = "".join(parts[:-1]) + "." + parts[-1]
    elif s_value.count('.') == 1:
        if len(s_value.split('.')[1]) != 3:
             cleaned_value = s_value
        else:
             cleaned_value = s_value.replace('.', '')
    else:
        cleaned_value = s_value
    return pd.to_numeric(cleaned_value.replace(',', ''), errors='coerce')

df_sampah[NAMA_KOLOM_VOLUME] = df_sampah[NAMA_KOLOM_VOLUME].apply(clean_numeric_advanced)
df_sampah.dropna(subset=[NAMA_KOLOM_VOLUME], inplace=True)

df_penduduk[NAMA_KOLOM_PENDUDUK] = df_penduduk[NAMA_KOLOM_PENDUDUK].apply(clean_numeric_advanced)
df_penduduk.dropna(subset=[NAMA_KOLOM_PENDUDUK], inplace=True)
print("✅ Data numerik di kedua dataset telah dibersihkan.")

# Gabungkan kedua dataset
df_gabungan = pd.merge(df_sampah, df_penduduk, left_on=NAMA_KOLOM_TAHUN_SAMPAH, right_on=NAMA_KOLOM_TAHUN_PENDUDUK)

# Persiapan data untuk Prophet
df_prophet = df_gabungan.rename(columns={
    NAMA_KOLOM_TAHUN_SAMPAH: 'ds',
    NAMA_KOLOM_VOLUME: 'y',
    NAMA_KOLOM_PENDUDUK: 'jumlah_penduduk'
})
df_prophet = df_prophet[['ds', 'y', 'jumlah_penduduk']]
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'].astype(str) + '-12-31')
print("✅ Data telah digabungkan dan diformat untuk Prophet.")

# 3. Latih model Prophet dengan regresor
model_prophet = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
model_prophet.add_regressor('jumlah_penduduk')
model_prophet.fit(df_prophet)
print("✅ Model Prophet berhasil dilatih.")

# 4. Melakukan Cross-Validation untuk Evaluasi Model
print("\nMemulai proses evaluasi model dengan Cross-Validation...")
try:
    df_cv = cross_validation(model_prophet, initial='1095 days', period='365 days', horizon='365 days', disable_tqdm=True)
    df_p = performance_metrics(df_cv)
    print("\n--- HASIL EVALUASI MODEL PROPHET (Cross-Validation) ---")
    print(df_p.head())
    print("---------------------------------------------------------")
    
    # Membuat dan Menyimpan Grafik Evaluasi
    print("Membuat dan menyimpan grafik evaluasi (cross-validation)...")
    fig_cv = plot_cross_validation_metric(df_cv, metric='mape')
    ax_cv = fig_cv.gca()
    ax_cv.set_title('Evaluasi Model Prophet (MAPE)', fontsize=16)
    ax_cv.set_xlabel('Horizon (Jarak Prediksi)', fontsize=12)
    ax_cv.set_ylabel('MAPE (Error %)', fontsize=12)
    fig_cv.savefig('static/grafik_evaluasi_prophet.png', dpi=150, bbox_inches='tight')
    print("✅ Grafik evaluasi telah disimpan sebagai 'grafik_evaluasi_prophet.png'.")

except Exception as e:
    print(f"\n⚠️ Gagal melakukan Cross-Validation. Kemungkinan data historis terlalu sedikit. Error: {e}")

# 5. Membuat prediksi masa depan untuk grafik utama
tahun_terakhir_data = df_prophet['ds'].dt.year.max()
tahun_untuk_diprediksi = 2026 - tahun_terakhir_data
future = model_prophet.make_future_dataframe(periods=tahun_untuk_diprediksi, freq='YE')
future['jumlah_penduduk'] = list(df_prophet['jumlah_penduduk']) + [df_prophet['jumlah_penduduk'].iloc[-1]] * tahun_untuk_diprediksi # Estimasi sederhana
forecast = model_prophet.predict(future)
print("\n✅ Prediksi masa depan untuk grafik telah dibuat.")

# 6. Menggambar grafik prediksi utama
print("\nMembuat dan menyimpan grafik prediksi utama...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='skyblue', alpha=0.3, label='Uncertainty Interval')
ax.plot(df_prophet['ds'], df_prophet['y'], 'k.', markersize=8, label='Data Asli')
ax.set_title('Prediksi Produksi Sampah Jakarta Timur (Ton)', fontsize=16)
ax.set_xlabel('Tahun', fontsize=12)
ax.set_ylabel('Jumlah Produksi Sampah (Ton)', fontsize=12)
start_date = pd.to_datetime(f"{df_prophet['ds'].dt.year.min()}-01-01")
end_date = pd.to_datetime("2026-12-31")
ax.set_xlim([start_date, end_date])
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
fig.savefig('static/grafik_prediksi.png', dpi=150, bbox_inches='tight')
print("✅ Grafik prediksi utama telah disimpan.")

# Menampilkan hasil prediksi angka untuk tahun 2026
try:
    prediksi_2026 = forecast[forecast['ds'].dt.year == 2026]
    if not prediksi_2026.empty:
        nilai_prediksi = prediksi_2026['yhat'].iloc[0]
        print("\n--- PREDIKSI ANGKA UNTUK TAHUN 2026 ---")
        print(f"Jumlah produksi sampah yang diprediksi untuk 31 Desember 2026 adalah: {nilai_prediksi:,.2f} Ton")
        print("------------------------------------------")
except Exception as e:
    print(f"\nGagal menampilkan prediksi angka: {e}")

print("\nProses Selesai.")
