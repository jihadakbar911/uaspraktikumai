document.addEventListener('DOMContentLoaded', async function() {
    const canvasElement = document.getElementById('grafikPrediksiInteraktif');
    if (!canvasElement) return;

    try {
        const response = await fetch('/api/grafik-data');
        const data = await response.json();

        const labels = data.map(item => item.tahun);
        const dataAsli = data.map(item => item.data_asli);
        const dataPrediksi = data.map(item => item.prediksi);

        const ctx = canvasElement.getContext('2d');

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Data Asli',
                        data: dataAsli,
                        backgroundColor: '#198754', // Hijau tua
                        borderColor: '#198754',
                        borderWidth: 1
                    },
                    {
                        label: 'Prediksi Model',
                        data: dataPrediksi,
                        backgroundColor: 'rgba(40, 167, 69, 0.5)', // Hijau lebih terang
                        borderColor: 'rgba(40, 167, 69, 0.5)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Jumlah Produksi Sampah per Tahun (Ton)',
                        font: { size: 16 },
                        color: '#212529' // Teks gelap
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    },
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#212529' // Teks gelap
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: { color: 'rgba(0, 0, 0, 0.1)' }, // Grid abu-abu
                        ticks: { color: '#6c757d' } // Teks abu-abu
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#6c757d' } // Teks abu-abu
                    }
                }
            }
        });

    } catch (error) {
        console.error('Gagal memuat atau menggambar grafik:', error);
        canvasElement.parentElement.innerHTML = '<p class="text-danger text-center">Gagal memuat grafik interaktif.</p>';
    }
});
