# ===================================================================
# APLIKASI WEB FLASK - DENGAN API UNTUK GRAFIK INTERAKTIF
# ===================================================================

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import joblib
from datetime import date
import json
import os
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Memuat model KNN & Scaler
try:
    model_knn = joblib.load('model_knn.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model KNN dan Scaler berhasil dimuat.")
except Exception as e:
    model_knn = None
    scaler = None
    print(f"⚠️ PERINGATAN: Gagal memuat model atau scaler: {e}")

# Route untuk halaman-halaman
@app.route('/')
def home(): return render_template('index.html')

@app.route('/prediksi')
def prediksi(): return render_template('analisis.html')

@app.route('/edukasi')
def edukasi(): return render_template('edukasi.html')

@app.route('/simulasi')
def simulasi(): return render_template('simulasi.html')

@app.route('/prediksi-user')
def prediksi_user():
    return render_template('prediksi_user.html')

from prophet import Prophet
import pandas as pd

@app.route('/api/prediksi-user', methods=['POST'])
def api_prediksi_user():
    try:
        data = request.get_json()
        tahun = int(data['tahun'])
        jumlah_sampah = float(data['jumlah_sampah'])
        jumlah_penduduk = float(data['jumlah_penduduk'])

        # Load data historis dan model Prophet
        with open('static/prophet_forecast.json', 'r') as f:
            data_hist = json.load(f)
        df_hist = pd.DataFrame(data_hist)
        # Gunakan data historis untuk retrain Prophet (agar input user bisa diprediksi di tahun berapapun)
        df_prophet = df_hist.dropna(subset=['data_asli', 'prediksi', 'tahun'])
        df_prophet = df_prophet.rename(columns={'tahun': 'ds', 'data_asli': 'y'})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'].astype(int).astype(str) + '-12-31')
        df_prophet['jumlah_penduduk'] = jumlah_penduduk  # Gunakan input user untuk semua prediksi
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        model.add_regressor('jumlah_penduduk')
        model.fit(df_prophet[['ds', 'y', 'jumlah_penduduk']])

        # Buat dataframe future sesuai tahun yang diminta user
        tahun_terakhir = df_prophet['ds'].dt.year.max()
        tahun_user = tahun
        n_years = tahun_user - tahun_terakhir
        if n_years < 0:
            # Jika tahun user di masa lalu, prediksi ulang dari data historis
            future = df_prophet[df_prophet['ds'].dt.year == tahun_user][['ds', 'jumlah_penduduk']]
        else:
            future = model.make_future_dataframe(periods=n_years, freq='YE')
            future['jumlah_penduduk'] = jumlah_penduduk
        forecast = model.predict(future)
        # Ambil prediksi tahun yang diminta user
        prediksi_tahun = forecast[forecast['ds'].dt.year == tahun_user]['yhat'].values
        prediksi_angka = float(prediksi_tahun[0]) if len(prediksi_tahun) > 0 else None
        # Siapkan data grafik (tahun, prediksi, data_asli)
        grafik = []
        for _, row in forecast.iterrows():
            tahun_row = int(row['ds'].year)
            # Cari data_asli dari data_hist jika ada
            data_asli = None
            for rec in data_hist:
                if rec['tahun'] == tahun_row:
                    data_asli = rec.get('data_asli', None)
                    break
            grafik.append({
                'tahun': tahun_row,
                'prediksi': float(row['yhat']),
                'data_asli': data_asli
            })
        return jsonify({'prediksi': prediksi_angka, 'grafik': grafik})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===================================================================
# BAGIAN BARU: API untuk menyajikan data grafik interaktif
# ===================================================================
@app.route('/api/grafik-data')
def grafik_data():
    try:
        with open('static/grafik_data.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ===================================================================


# API untuk simulasi pribadi (KNN)
@app.route('/api/simulasi-pribadi', methods=['POST'])
def get_personal_simulation():
    # ... (logika fungsi ini tidak berubah) ...
    if not model_knn or not scaler:
        return jsonify({"error": "Model klasifikasi tidak siap di server."}), 500
    try:
        data = request.get_json()
        def gram_to_category(gram):
            if gram <= 150: return 1
            elif gram <= 400: return 2
            else: return 3
        organik_cat = gram_to_category(data['organik_gram'])
        daur_ulang_cat = gram_to_category(data['daur_ulang_gram'])
        residu_cat = gram_to_category(data['residu_gram'])
        input_data_kategori = [[organik_cat, daur_ulang_cat, residu_cat, data['kebiasaan_memilah']]]
        input_data_scaled = scaler.transform(input_data_kategori)
        hasil_profil = model_knn.predict(input_data_scaled)[0]
        
        if hasil_profil == 'Rendah':
            gelar = "Sahabat Lingkungan 🌱"
            deskripsi = "Keren! Kamu sudah sangat bijak dalam mengelola sampah. Teruskan kebiasaan baikmu!"
        elif hasil_profil == 'Sedang':
            gelar = "Pengguna Sehari-hari ♻️"
            deskripsi = "Sampah plastikmu masih di tingkat sedang. Yuk, kurangi pemakaian agar lebih ramah lingkungan!"
        else:
            gelar = "Pengguna Aktif Sampah ⚠"
            deskripsi = "Kebiasaanmu masih menghasilkan banyak sampah. Ayo mulai ubah langkah kecil untuk bantu bumi!"
            
        total_gram_harian = data['organik_gram'] + data['daur_ulang_gram'] + data['residu_gram']
        hari_ini = date.today()
        akhir_2026 = date(2026, 12, 31)
        sisa_hari = (akhir_2026 - hari_ini).days
        total_sampah_kg = round((total_gram_harian * sisa_hari) / 1000, 2)
        
        return jsonify({"gelar": gelar, "deskripsi": deskripsi, "total_kg": total_sampah_kg})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
