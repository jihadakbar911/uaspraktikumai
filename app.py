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
def prediksi(): return render_template('prediksi.html')

@app.route('/edukasi')
def edukasi(): return render_template('edukasi.html')

@app.route('/simulasi')
def simulasi(): return render_template('simulasi.html')

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
