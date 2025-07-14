# ===================================================================
# APLIKASI WEB FLASK - VERSI HYBRID (KUALITATIF + KUANTITATIF)
# ===================================================================

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import joblib
from datetime import date
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

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Fungsi untuk konversi gram ke kategori (1, 2, 3)
def gram_to_category(gram):
    if gram <= 150:
        return 1 # Sedikit
    elif gram <= 400:
        return 2 # Sedang
    else:
        return 3 # Banyak

# API untuk simulasi pribadi (KNN) - VERSI HYBRID
@app.route('/api/simulasi-pribadi', methods=['POST'])
def get_personal_simulation():
    if not model_knn or not scaler:
        return jsonify({"error": "Model klasifikasi tidak siap di server."}), 500
    
    try:
        data = request.get_json()
        
        # --- Bagian 1: Prediksi Kualitatif (AI) ---
        # Konversi input gram dari user ke kategori
        organik_cat = gram_to_category(data['organik_gram'])
        daur_ulang_cat = gram_to_category(data['daur_ulang_gram'])
        residu_cat = gram_to_category(data['residu_gram'])
        
        # Susun data kategori untuk dimasukkan ke model
        input_data_kategori = [[
            organik_cat,
            daur_ulang_cat,
            residu_cat,
            data['kebiasaan_memilah']
        ]]
        
        # Lakukan scaling dan prediksi profil
        input_data_scaled = scaler.transform(input_data_kategori)
        prediksi_profil = model_knn.predict(input_data_scaled)
        hasil_profil = prediksi_profil[0]
        
        # Terjemahkan hasil menjadi pesan menarik
        if hasil_profil == 'Rendah':
            gelar = "Sahabat Lingkungan 🌱"
            deskripsi = "Keren! Kamu sudah sangat bijak dalam mengelola sampah. Teruskan kebiasaan baikmu!"
        elif hasil_profil == 'Sedang':
            gelar = "Pengguna Plastik Sehari-hari 🚯"
            deskripsi = "Sampah plastikmu masih di tingkat sedang. Yuk, kurangi pemakaian agar lebih ramah lingkungan!"
        else: # 'Tinggi'
            gelar = "Pengguna Aktif Plastik ♻"
            deskripsi = "Kebiasaanmu masih menghasilkan banyak sampah. Ayo mulai ubah langkah kecil untuk bantu bumi!"
            
        # --- Bagian 2: Prediksi Kuantitatif (Kalkulasi) ---
        total_gram_harian = data['organik_gram'] + data['daur_ulang_gram'] + data['residu_gram']
        
        # Hitung sisa hari hingga akhir 2026
        hari_ini = date.today()
        akhir_2026 = date(2026, 12, 31)
        sisa_hari = (akhir_2026 - hari_ini).days
        
        total_sampah_gram = total_gram_harian * sisa_hari
        total_sampah_kg = round(total_sampah_gram / 1000, 2)
        
        # Kirim KEDUA hasil ke frontend
        return jsonify({
            "gelar": gelar,
            "deskripsi": deskripsi,
            "total_kg": total_sampah_kg
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=5000)
