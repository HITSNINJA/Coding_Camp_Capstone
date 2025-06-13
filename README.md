# Stress Detection using Wearable Devices to Enhance Well-Being

ID Tim: CC25-CR439
Anggota:
- (ML) MC299D5Y2471 - Muhamad Furqon Al-Haqqi
- (ML) MC262D5Y2274 - Muhammad Zawawi Basri
---

## 📝 Ringkasan Proyek

Stres kronis merupakan salah satu tantangan kesehatan mental dan fisik yang signifikan di era modern. Deteksi dini tingkat stres dapat membantu individu untuk mengelola kesehatan dan kesejahteraan (well-being) mereka dengan lebih baik. Proyek ini memanfaatkan dataset **WESAD** untuk membangun sebuah model deep learning yang mampu mengklasifikasikan kondisi fisiologis pengguna menjadi keadaan **"Stres"** atau **"Non-Stres"**.

---
## ⚙️ Alur Kerja Proyek
Proyek ini dibagi menjadi dua tahap utama: analisis dan ekstraksi fitur, diikuti oleh pelatihan model.

1. Data Analisis & Ekstraksi Fitur

Tahap ini bertujuan untuk mengubah data sinyal mentah dari dataset WESAD menjadi serangkaian fitur yang bermakna. Proses ini difokuskan hanya pada data yang dapat ditangkap oleh smartwatch konvensional untuk mensimulasikan penggunaan di dunia nyata.
- Sumber Data: Proyek ini menggunakan dataset WESAD, dengan fokus eksklusif pada data dari perangkat pergelangan tangan (wrist).
- Sinyal yang Digunakan: Hanya tiga sinyal utama yang diambil dari data pergelangan tangan: 
    - ACC (Akselerometer 3-sumbu) dengan sampling rate 32 Hz.
    - BVP (Blood Volume Pulse) dengan sampling rate 64 Hz.
    - TEMP (Temperatur kulit) dengan sampling rate 4 Hz.
- Ekstraksi Fitur: Proses ini dilakukan menggunakan metode sliding window: 
    - Window 5 detik untuk data ACC.
    - Window 60 detik untuk data BVP dan TEMP.
    - Pustaka neurokit2 digunakan secara ekstensif untuk menghitung fitur-fitur relevan, terutama metrik HRV (Heart Rate Variability) dari sinyal BVP, dan fitur statistik (mean, std, min, max, slope) dari sinyal ACC dan TEMP.
- Output: Hasil dari tahap ini adalah sebuah file CSV (combined.csv) yang berisi matriks fitur yang siap digunakan untuk tahap pelatihan model.

2. Pelatihan Model Machine Learning
Tahap ini didokumentasikan dalam notebook coding-camp-capstone-training-model.ipynb.

- Tugas Klasifikasi: Label asli disederhanakan menjadi masalah klasifikasi biner: kelas 1 (Stres) dan kelas 0 (Non-Stres).
- Pembagian Data: Data dibagi menjadi 70% set pelatihan dan 30% set pengujian menggunakan metode pemisahan berbasis subjek (subject-independent split) untuk memastikan model dievaluasi pada data dari individu yang benar-benar baru.
- Arsitektur Model: Model yang digunakan adalah Multi-Layer Perceptron (MLP) yang dibangun dengan TensorFlow/Keras. Arsitekturnya terdiri dari beberapa hidden layer Dense dengan aktivasi ReLU dan Dropout (0.5 dan 0.3) untuk mencegah overfitting.
- Proses Pelatihan:
    - Preprocessing: Fitur diskalakan menggunakan StandardScaler.
    - Konfigurasi: Model dilatih dengan optimizer adam dan loss function binary_crossentropy.
    - Regularisasi: Callback EarlyStopping digunakan untuk menghentikan pelatihan jika val_loss tidak membaik selama 10 epoch, dan mengembalikan bobot model terbaik.
- Hasil: Model mencapai performa yang baik pada test set: 
    - Akurasi: ~85.3%
    - F1-Score (Stress): 76%
    - F1-Score (Non-Stress): 89%
- Artefak: Model yang telah dilatih (wesad_model.h5) dan objek scaler (scaler.pkl) disimpan untuk digunakan dalam aplikasi inferensi.

---

# 🚀 Inferensi Menggunakan Streamlit
Untuk menjalankan aplikasi deteksi stres secara lokal, ikuti langkah-langkah berikut.

1. Prasyarat

Pastikan Anda memiliki Python 3.9+ dan pip terinstal. Buat sebuah virtual environment (opsional tapi direkomendasikan).

2. Instalasi

Clone repositori ini dan instal semua pustaka yang dibutuhkan menggunakan file requirements.txt.

```bash
# Kloning repositori (jika belum)
git clone https://github.com/username/repository-name.git
cd repository-name

# Buat dan aktifkan virtual environment
python -m venv venv
source venv/bin/activate  # Pada Windows, gunakan: venv\Scripts\activate

# Instal pustaka yang dibutuhkan
pip install -r requirements.txt
```

3. Menjalankan Aplikasi

Setelah instalasi selesai, jalankan aplikasi menggunakan perintah berikut dari terminal:

```bash
streamlit run app.py
```

Aplikasi akan terbuka secara otomatis di browser web Anda. Anda dapat mulai dengan mengunggah file data sensor mentah dalam format .csv pada halaman UPLOAD & PREDIKSI.

---

Dataset: [WESAD](https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset/data)

Pengerjaan dilakukan menggunakan platform kaggle:
- Data analysis ([kaggle](https://www.kaggle.com/code/furqonalhaqqi/coding-camp-capstone-data-analysis))
- Training Model ([Kaggle](https://www.kaggle.com/code/furqonalhaqqi/coding-camp-capstone-training-model))

Aplikasi Deploy
- Application ([Streamlit](https://codingcampcapstone-miso3xszrjgxaegtdvqize.streamlit.app/))