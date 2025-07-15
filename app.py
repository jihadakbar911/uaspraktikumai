# ===================================================================
# APLIKASI WEB FLASK - VERSI FINAL MULTI-HALAMAN
# ===================================================================

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import joblib
from datetime import date

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

# Route untuk halaman utama (Landing Page)
@app.route('/')
def home():
    return render_template('index.html')

# ===================================================================
# BAGIAN BARU: Route untuk halaman simulasi
# ===================================================================
@app.route('/simulasi')
def simulasi():
    """Menyajikan halaman form simulasi."""
    return render_template('simulasi.html')
# ===================================================================

# API untuk simulasi pribadi (KNN) - Tidak ada perubahan
@app.route('/api/simulasi-pribadi', methods=['POST'])
def get_personal_simulation():
    if not model_knn or not scaler:
        return jsonify({"error": "Model klasifikasi tidak siap di server."}), 500
    
    try:
        data = request.get_json()
        
        # Fungsi untuk konversi gram ke kategori (1, 2, 3)
        def gram_to_category(gram):
            if gram <= 150: return 1
            elif gram <= 400: return 2
            else: return 3

        organik_cat = gram_to_category(data['organik_gram'])
        daur_ulang_cat = gram_to_category(data['daur_ulang_gram'])
        residu_cat = gram_to_category(data['residu_gram'])
        
        input_data_kategori = [[organik_cat, daur_ulang_cat, residu_cat, data['kebiasaan_memilah']]]
        
        input_data_scaled = scaler.transform(input_data_kategori)
        prediksi_profil = model_knn.predict(input_data_scaled)
        hasil_profil = prediksi_profil[0]
        
        if hasil_profil == 'Rendah':
            gelar = "Jawara Lingkungan ✨"
            deskripsi = "Luar biasa! Pengelolaan sampahmu sudah sangat baik."
        elif hasil_profil == 'Sedang':
            gelar = "Pejuang Bumi ⭐"
            deskripsi = "Sudah bagus! Kamu sudah di jalur yang benar."
        else:
            gelar = "Pemula Hijau 👍"
            deskripsi = "Perjalananmu baru dimulai! Ayo fokus kurangi sampah residu."
            
        total_gram_harian = data['organik_gram'] + data['daur_ulang_gram'] + data['residu_gram']
        
        hari_ini = date.today()
        akhir_2026 = date(2026, 12, 31)
        sisa_hari = (akhir_2026 - hari_ini).days
        
        total_sampah_kg = round((total_gram_harian * sisa_hari) / 1000, 2)
        
        return jsonify({
            "gelar": gelar,
            "deskripsi": deskripsi,
            "total_kg": total_sampah_kg
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
