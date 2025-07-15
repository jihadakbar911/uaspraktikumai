document.addEventListener('DOMContentLoaded', function() {

    const formSimulasi = document.getElementById('simulasi-form');
    const hasilContainer = document.getElementById('hasil-simulasi');

    formSimulasi.addEventListener('submit', async function(event) {
        event.preventDefault();

        // Ambil data dari elemen form (dalam gram)
        const dataUntukDikirim = {
            organik_gram: parseInt(document.getElementById('organik').value),
            daur_ulang_gram: parseInt(document.getElementById('daur_ulang_item').value),
            residu_gram: parseInt(document.getElementById('residu').value),
            kebiasaan_memilah: parseInt(document.getElementById('kebiasaan_memilah').value)
        };

        try {
            // Panggil API backend kita untuk klasifikasi KNN
            const response = await fetch('/api/simulasi-pribadi', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(dataUntukDikirim)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const hasil = await response.json();

            // Tampilkan hasil gabungan: Penilaian AI + Prediksi Kuantitatif
            hasilContainer.innerHTML = `
                <h3>${hasil.gelar}</h3>
                <p>${hasil.deskripsi}</p>
                <hr>
                <p class="mt-3">
                    <strong>Proyeksi Akumulasi Sampahmu:</strong><br>
                    Dengan kebiasaan ini, kamu diproyeksikan akan menghasilkan sekitar
                    <strong style="font-size: 1.2em; color: #9b2226;">${hasil.total_kg} kg</strong>
                    sampah hingga akhir tahun 2026.
                </p>
            `;
            hasilContainer.style.display = 'block';

        } catch (error) {
            console.error('Gagal mengirim data simulasi:', error);
            hasilContainer.innerHTML = '<p style="color: red;">Gagal mendapatkan hasil. Pastikan server backend berjalan dengan benar.</p>';
            hasilContainer.style.display = 'block';
        }
    });
});
