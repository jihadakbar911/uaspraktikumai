# ===================================================================
# APLIKASI WEB FLASK - VERSI FINAL DENGAN GRAFIK GAMBAR STATIS
# ===================================================================

# 1. IMPORT LIBRARY
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import joblib
import os

import logging



logging.basicConfig(level=logging.INFO)

# 2. INISIALISASI APLIKASI
app = Flask(__name__)
CORS(app)

# 3. MEMUAT MODEL KNN & SCALER
try:
    model_knn = joblib.load('model_knn.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model KNN dan Scaler berhasil dimuat.")
except Exception as e:
    model_knn = None
    scaler = None
    print(f"⚠️ PERINGATAN: Gagal memuat model atau scaler: {e}")

# 4. ROUTE UNTUK HALAMAN UTAMA
@app.route('/')
def home():
    """Menyajikan halaman utama website (index.html)."""
    return render_template('index.html')

# 5. API UNTUK SIMULASI PRIBADI (KNN)
@app.route('/api/simulasi-pribadi', methods=['POST'])
def get_personal_simulation():
    """Menerima input dari form, melakukan prediksi dengan model KNN."""
    if not model_knn or not scaler:
        return jsonify({"error": "Model klasifikasi tidak siap di server."}), 500
    
    try:
        data = request.get_json()
        
        # Susun data sesuai urutan fitur
        input_data = [[
            data['jumlah_botol'],
            data['jumlah_kantong'],
            data['jumlah_bungkus'],
            data['daur_ulang']
        ]]
        
        # Lakukan scaling dan prediksi
        input_data_scaled = scaler.transform(input_data)
        prediksi_profil = model_knn.predict(input_data_scaled)
        hasil_profil = prediksi_profil[0]
        
        # Terjemahkan hasil menjadi pesan menarik
        if hasil_profil == 'Rendah':
            gelar = "Jawara Lingkungan ✨"
            deskripsi = "Luar biasa! Kamu adalah inspirasi dalam menjaga bumi. Pertahankan kebiasaan baikmu!"
        elif hasil_profil == 'Sedang':
            gelar = "Pejuang Diet Plastik ⭐"
            deskripsi = "Sudah bagus! Kamu sudah di jalur yang benar. Tingkatkan sedikit lagi untuk jadi jawara!"
        else: # 'Tinggi'
            gelar = "Pemula Hijau 👍"
            deskripsi = "Perjalananmu baru dimulai! Ayo kurangi sampah plastik sedikit demi sedikit setiap hari."
            
        return jsonify({
            "gelar": gelar,
            "deskripsi": deskripsi
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':

  port = int(os.environ.get('PORT', 5000))

  app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
