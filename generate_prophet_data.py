# ===================================================================
# GENERATE DATA UNTUK GRAFIK INTERAKTIF (VERSI PERBAIKAN FINAL)
# ===================================================================

import pandas as pd
from prophet import Prophet
import json
import numpy as np # Impor numpy untuk pengecekan NaN yang lebih baik

print("Memulai proses pembuatan data untuk grafik interaktif...")

# 1. Muat dan bersihkan data
try:
    df_sampah = pd.read_csv('Dataset Sampah Jakarta Timur 2019-2024.csv')
    df_penduduk = pd.read_csv('Dataset Penduduk Jakarta Timur 2019-2024.csv')
    print("✅ Dataset Sampah dan Penduduk berhasil dimuat.")
except Exception as e:
    print(f"❌ ERROR memuat dataset: {e}")
    exit()

NAMA_KOLOM_TAHUN_SAMPAH = 'tahun'
NAMA_KOLOM_VOLUME = 'timbulan_sampah_tahunan(ton)'
NAMA_KOLOM_TAHUN_PENDUDUK = 'tahun'
NAMA_KOLOM_PENDUDUK = 'jumlah_penduduk_jakarta_timur'

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
print("✅ Data numerik telah dibersihkan.")

# 2. Gabungkan dan Latih Model Prophet
df_gabungan = pd.merge(df_sampah, df_penduduk, left_on=NAMA_KOLOM_TAHUN_SAMPAH, right_on=NAMA_KOLOM_TAHUN_PENDUDUK)
df_prophet = df_gabungan.rename(columns={
    NAMA_KOLOM_TAHUN_SAMPAH: 'ds',
    NAMA_KOLOM_VOLUME: 'y',
    NAMA_KOLOM_PENDUDUK: 'jumlah_penduduk'
})
df_prophet = df_prophet[['ds', 'y', 'jumlah_penduduk']]
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'].astype(str) + '-12-31')

model_prophet = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
model_prophet.add_regressor('jumlah_penduduk')
model_prophet.fit(df_prophet)
print("✅ Model Prophet berhasil dilatih.")

# 3. Lakukan prediksi hingga 2026
future = model_prophet.make_future_dataframe(periods=2, freq='YE')
future['jumlah_penduduk'] = list(df_prophet['jumlah_penduduk']) + [df_prophet['jumlah_penduduk'].iloc[-1]] * 2
forecast = model_prophet.predict(future)
print("✅ Prediksi masa depan telah dibuat.")

# ===================================================================
# BAGIAN YANG DIPERBAIKI: Menggabungkan dan membersihkan data untuk JSON
# ===================================================================
df_prophet.rename(columns={'y': 'data_asli'}, inplace=True)
df_forecast_clean = forecast[['ds', 'yhat']].rename(columns={'yhat': 'prediksi'})
df_final = pd.merge(df_forecast_clean, df_prophet, on='ds', how='left')

df_final['tahun'] = df_final['ds'].dt.year
df_final = df_final[['tahun', 'data_asli', 'prediksi']]

# Konversi ke list of dictionaries
data_to_save = df_final.to_dict(orient='records')

# PERBAIKAN FINAL: Loop manual untuk mengganti NaN dengan None
cleaned_data_for_json = []
for record in data_to_save:
    cleaned_record = {}
    for key, value in record.items():
        # Cek jika value adalah NaN (baik dari numpy maupun pandas)
        # dan ganti dengan None yang valid di JSON
        if pd.isna(value):
            cleaned_record[key] = None
        else:
            cleaned_record[key] = value
    cleaned_data_for_json.append(cleaned_record)

# Simpan data yang sudah 100% bersih ke file JSON
with open('static/grafik_data.json', 'w') as f:
    json.dump(cleaned_data_for_json, f)

print("✅ Data untuk grafik interaktif telah disimpan sebagai 'grafik_data.json'.")
# ===================================================================

print("\nProses Selesai.")
