# ===================================================================
# GENERATE PROPHET DATA, EVALUASI & GRAFIK
# ===================================================================

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
# Impor untuk membuat plot cross-validation
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt

print("Memulai proses pra-kalkulasi dengan dataset KOTA BOGOR...")

try:
    df_sampah = pd.read_csv('jumlah_produksi_sampah_di_kota_bogor.csv')
    print("✅ Dataset Kota Bogor berhasil dimuat.")
except FileNotFoundError:
    print("❌ ERROR: Pastikan file 'jumlah_produksi_sampah_di_kota_bogor.csv' ada di folder yang sama.")
    exit()

# Koreksi data anomali untuk tahun 2024
df_sampah.loc[(df_sampah['tahun'] == 2024) & (df_sampah['jumlah_produksi_sampah'] > 1000), 'jumlah_produksi_sampah'] /= 10
print("✅ Data anomali untuk tahun 2024 telah dikoreksi.")

# Persiapan data untuk Prophet
df_prophet = df_sampah.rename(columns={
    'tahun': 'ds',
    'jumlah_produksi_sampah': 'y'
})
df_prophet = df_prophet[['ds', 'y']]
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'].astype(str) + '-06-30')
print("✅ Data telah diformat untuk Prophet.")

# Latih model Prophet
model_prophet = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
model_prophet.fit(df_prophet)
print("✅ Model Prophet berhasil dilatih.")

# Evaluasi Model dengan Cross-Validation
print("\nMemulai proses evaluasi model dengan Cross-Validation...")
try:
    df_cv = cross_validation(model_prophet, initial='1825 days', period='365 days', horizon='730 days', disable_tqdm=True)
    df_p = performance_metrics(df_cv)
    print("\n--- HASIL EVALUASI MODEL PROPHET ---")
    print(df_p.head())
    print("------------------------------------")
    
    # ===================================================================
    # BAGIAN BARU: Membuat dan Menyimpan Grafik Evaluasi
    # ===================================================================
    print("Membuat dan menyimpan grafik evaluasi (cross-validation)...")
    fig_cv = plot_cross_validation_metric(df_cv, metric='mape')
    ax_cv = fig_cv.gca()
    ax_cv.set_title('Evaluasi Model Prophet (MAPE)', fontsize=16)
    ax_cv.set_xlabel('Horizon (Jarak Prediksi)', fontsize=12)
    ax_cv.set_ylabel('MAPE (Error %)', fontsize=12)
    
    # Simpan grafik evaluasi sebagai file gambar terpisah
    fig_cv.savefig('static/grafik_evaluasi_prophet.png', dpi=150, bbox_inches='tight')
    print("✅ Grafik evaluasi telah disimpan sebagai 'grafik_evaluasi_prophet.png'.")
    # ===================================================================

except Exception as e:
    print(f"\n⚠️ Gagal melakukan Cross-Validation. Kemungkinan data historis terlalu sedikit. Error: {e}")
    print("   Evaluasi model Prophet akan dilewati.")

# Membuat prediksi HANYA sampai tahun 2026
tahun_terakhir_data = df_prophet['ds'].dt.year.max()
tahun_untuk_diprediksi = 2026 - tahun_terakhir_data
future = model_prophet.make_future_dataframe(periods=tahun_untuk_diprediksi, freq='YE')

# Lakukan prediksi
forecast = model_prophet.predict(future)
print("\n✅ Prediksi masa depan untuk grafik telah dibuat.")

# Menggambar grafik dengan plotting manual
print("Membuat dan menyimpan grafik prediksi utama...")
fig1 = model_prophet.plot(forecast)
ax = fig1.gca()

for line in ax.lines:
    if line.get_marker() == '.':
        line.set_marker('none')

ax.plot(df_prophet['ds'], df_prophet['y'], 'k.', label='Data Asli')
ax.set_title('Prediksi Produksi Sampah Kota Bogor (Ton)', fontsize=16)
ax.set_xlabel('Tahun', fontsize=12)
ax.set_ylabel('Jumlah Produksi Sampah (Ton)', fontsize=12)
start_date = pd.to_datetime(f"{df_prophet['ds'].dt.year.min()}-01-01")
end_date = pd.to_datetime("2026-12-31")
ax.set_xlim([start_date, end_date])
ax.grid(True)
ax.legend()

fig1.savefig('static/grafik_prediksi.png', dpi=150, bbox_inches='tight')
print("✅ Grafik prediksi utama yang akurat telah disimpan.")

print("\nProses Selesai.")
