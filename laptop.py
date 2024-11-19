import numpy as np
import matplotlib.pyplot as plt

# Data penjualan laptop dalam 5 tahun terakhir berdasarkan kategori penggunaan
tahun = np.array([2019, 2020, 2021, 2022, 2023])
penjualan_ringan = np.array([500, 520, 540, 580, 620])  # Penjualan penggunaan ringan
penjualan_sedang = np.array([300, 320, 350, 380, 400])  # Penjualan penggunaan sedang
penjualan_berat = np.array([200, 230, 250, 270, 300])   # Penjualan penggunaan berat

# Total penjualan tahunan
total_penjualan = penjualan_ringan + penjualan_sedang + penjualan_berat

# Parameter smoothing untuk eksponensial smoothing (bisa di-tune)
alpha = 0.3

# Fungsi untuk eksponensial smoothing
def eksponensial_smoothing(data, alpha):
    hasil = np.zeros_like(data)
    hasil[0] = data[0]  # Inisialisasi nilai pertama
    for t in range(1, len(data)):
        hasil[t] = alpha * data[t] + (1 - alpha) * hasil[t-1]
    return hasil

# Menghitung prediksi penjualan menggunakan eksponensial smoothing
prediksi_ringan = eksponensial_smoothing(penjualan_ringan, alpha)
prediksi_sedang = eksponensial_smoothing(penjualan_sedang, alpha)
prediksi_berat = eksponensial_smoothing(penjualan_berat, alpha)

# Matriks Transisi Markov
# Keadaan: [Penggunaan Ringan, Penggunaan Sedang, Penggunaan Berat]
states = ["Ringan", "Sedang", "Berat"]
transition_matrix = np.array([[0.6, 0.3, 0.1],  # Probabilitas transisi dari "Ringan"
                              [0.2, 0.5, 0.3],  # Probabilitas transisi dari "Sedang"
                              [0.1, 0.3, 0.6]])  # Probabilitas transisi dari "Berat"

# Distribusi awal berdasarkan total penjualan pada tahun pertama
total_initial = total_penjualan[0]
state_distribution = np.array([penjualan_ringan[0] / total_initial,
                               penjualan_sedang[0] / total_initial,
                               penjualan_berat[0] / total_initial])

# Simulasi Markov Chain untuk beberapa tahun ke depan
def simulate_markov_chain(transition_matrix, state_distribution, steps):
    distributions = [state_distribution]
    for _ in range(steps):
        state_distribution = np.dot(state_distribution, transition_matrix)
        distributions.append(state_distribution)
    return distributions

# Simulasi untuk 4 langkah waktu (tahun berikutnya)
state_distributions = simulate_markov_chain(transition_matrix, state_distribution, 4)

# Menampilkan hasil eksponensial smoothing
plt.plot(tahun, penjualan_ringan, label='Penjualan Ringan (Sebenarnya)', marker='o')
plt.plot(tahun, prediksi_ringan, label='Prediksi Ringan (Eksponensial Smoothing)', linestyle='--')
plt.plot(tahun, penjualan_sedang, label='Penjualan Sedang (Sebenarnya)', marker='o')
plt.plot(tahun, prediksi_sedang, label='Prediksi Sedang (Eksponensial Smoothing)', linestyle='--')
plt.plot(tahun, penjualan_berat, label='Penjualan Berat (Sebenarnya)', marker='o')
plt.plot(tahun, prediksi_berat, label='Prediksi Berat (Eksponensial Smoothing)', linestyle='--')
plt.xlabel('Tahun')
plt.ylabel('Penjualan Laptop')
plt.title('Prediksi Penjualan Laptop Berdasarkan Kategori Penggunaan')
plt.legend()
plt.grid(True)
plt.show()

# Menampilkan hasil simulasi Markov Chain
print("\nDistribusi Prediksi Penjualan (Markov Chain):")
for step, distribution in enumerate(state_distributions):
    print(f"Langkah {step}: Ringan={distribution[0]:.2f}, Sedang={distribution[1]:.2f}, Berat={distribution[2]:.2f}")
