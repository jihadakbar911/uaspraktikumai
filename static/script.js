// Menunggu semua konten HTML dimuat sebelum menjalankan JavaScript
document.addEventListener('DOMContentLoaded', function() {

    // ===============================================================
    // FUNGSI UNTUK MENGIRIM DATA FORM & MENAMPILKAN HASIL SIMULASI
    // ===============================================================
    const formSimulasi = document.getElementById('simulasi-form');
    const hasilContainer = document.getElementById('hasil-simulasi');

    formSimulasi.addEventListener('submit', async function(event) {
        // Mencegah form mengirim data dengan cara tradisional
        event.preventDefault();

        // Ambil data dari setiap input
        const dataUntukDikirim = {
            jumlah_botol: parseInt(document.getElementById('botol').value),
            jumlah_kantong: parseInt(document.getElementById('kantong').value),
            jumlah_bungkus: parseInt(document.getElementById('bungkus').value),
            daur_ulang: parseInt(document.getElementById('daur-ulang').value)
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

            // Ambil lagi data angka dari form untuk perhitungan sederhana
            const totalItemPerHari = dataUntukDikirim.jumlah_botol + dataUntukDikirim.jumlah_kantong + dataUntukDikirim.jumlah_bungkus;
            const estimasiTotalItem = totalItemPerHari * 500; // Asumsi 500 hari hingga akhir 2026

            // Tampilkan hasil gabungan
            hasilContainer.innerHTML = `
                <h3>${hasil.gelar}</h3>
                <p>${hasil.deskripsi}</p>
                <hr>
                <p class="mt-3"><strong>Sebagai gambaran,</strong> kebiasaan ini setara dengan <strong>${estimasiTotalItem.toLocaleString('id-ID')} item</strong> sampah plastik hingga akhir tahun 2026.</p>
            `;
            hasilContainer.style.display = 'block';

        } catch (error) {
            console.error('Gagal mengirim data simulasi:', error);
            hasilContainer.innerHTML = '<p style="color: red;">Gagal mendapatkan hasil. Pastikan server backend berjalan dengan benar.</p>';
            hasilContainer.style.display = 'block';
        }
    });
});
