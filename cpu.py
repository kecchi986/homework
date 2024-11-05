import numpy as np
import matplotlib.pyplot as plt

# Data harga CPU dalam 5 tahun terakhir (contoh data, bisa diganti dengan data asli)
tahun = np.array([2019, 2020, 2021, 2022, 2023])
harga_cpu = np.array([300, 320, 350, 400, 450])  # Harga dalam satuan tertentu, misalnya USD

# Parameter smoothing untuk eksponensial smoothing (bisa di-tune)
alpha = 0.3

# Fungsi untuk eksponensial smoothing
def eksponensial_smoothing(data, alpha):
    hasil = np.zeros_like(data)
    hasil[0] = data[0]  # Inisialisasi nilai pertama
    for t in range(1, len(data)):
        hasil[t] = alpha * data[t] + (1 - alpha) * hasil[t-1]
    return hasil

# Menghitung prediksi harga menggunakan eksponensial smoothing
prediksi_harga = eksponensial_smoothing(harga_cpu, alpha)

# Menampilkan hasil
plt.plot(tahun, harga_cpu, label='Harga Sebenarnya', marker='o')
plt.plot(tahun, prediksi_harga, label='Prediksi Harga', linestyle='--')
plt.xlabel('Tahun')
plt.ylabel('Harga CPU (USD)')
plt.title('Prediksi Kenaikan Harga CPU Menggunakan Eksponensial Smoothing')
plt.legend()
plt.grid(True)
plt.show()