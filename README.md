# 🎮 Analisis Sentimen Review Steam (Sistem Cerdas)

Aplikasi web berbasis Flask untuk menganalisis dan mengklasifikasikan sentimen review game Steam (*Recommended* atau *Not Recommended*) menggunakan algoritma **Multinomial Naive Bayes**.

---

## 🚀 Fitur Utama

- **Pembersihan & Pra-pemrosesan Teks**: Tokenisasi, normalisasi, dan penyaringan kata tanpa ketergantungan library berat.
- **Klasifikasi Multinomial Naive Bayes**: Dilengkapi dengan *Laplace Smoothing* ($\alpha = 1$) dan bobot probabilitas logaritmik.
- **Dukungan Multi-Bahasa**: Menerjemahkan review bahasa lain secara otomatis ke Bahasa Inggris sebelum klasifikasi.
- **Mekanisme Umpan Balik (Active Learning)**: Pengguna dapat memberikan koreksi hasil prediksi, dan sistem akan melakukan pelatihan ulang (*retraining*) secara background.
- **Antarmuka Modern**: UI web responsif dan interaktif.

---

## 📁 Struktur Direktori

```text
├── core/
│   ├── naive_bayes.py       # Implementasi kelas Naive Bayes Classifier
│   └── preprocessing.py     # Modul pembersihan dan tokenisasi teks
├── data/
│   └── .gitkeep             # Folder penampung dataset (steam_reviews.csv)
├── model/
│   └── nb_weights.json      # Bobot model Naive Bayes yang sudah dilatih
├── templates/
│   └── index.html           # Tampilan antarmuka web
├── controller.py            # Aplikasi web Flask & API endpoint
├── train.py                 # Skrip pelatihan model Naive Bayes
├── requirements.txt         # Daftar pustaka yang dibutuhkan
└── README.md
```

---

## 🛠️ Instalasi & Menjalankan

### 1. Klon Repositori
```bash
git clone https://github.com/julianputra-dev/tugas-sitem-cerdas.git
cd tugas-sitem-cerdas
```

### 2. Buat & Aktifkan Virtual Environment (Opsional tapi disarankan)
```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 3. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
python controller.py
```
Buka browser dan akses [http://localhost:5000](http://localhost:5000).

---

## 📊 Dataset
Dataset asli (`data/steam_reviews.csv`) berukuran besar (>100 MB) dan diabaikan dari repositori Git. Model sudah dilengkapi dengan bobot terlatih di `model/nb_weights.json` sehingga aplikasi dapat langsung digunakan tanpa perlu melatih ulang dataset dari awal.
