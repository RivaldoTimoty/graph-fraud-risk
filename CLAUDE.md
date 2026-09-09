# Project Context

Graph-based fraud detection pada dataset IEEE-CIS (Vesta). Tujuan: mengukur
incremental lift dari network/graph features di atas baseline LightGBM tabular,
dengan validasi out-of-time yang jujur.

Ini project portofolio untuk role Data Scientist credit scoring/risk di fintech.
Kualitas engineering dan kejujuran metodologis lebih penting dari skor tinggi.

## Aturan Wajib

### Data leakage
- SEMUA split adalah temporal, berdasarkan `TransactionDT`. JANGAN PERNAH pakai
  random split atau StratifiedKFold biasa.
- Fitur berbasis target (neighbor fraud rate, target encoding, WoE) HARUS dihitung
  hanya dari data periode training, lalu di-apply ke validation/test. Kalau dihitung
  dari seluruh data, itu leakage dan hasilnya tidak sah.
- Graph boleh dibangun dari seluruh data (struktur), TAPI statistik label di atas
  graph hanya dari periode training.
- Setiap kali menambah fitur baru, tanyakan: "apakah fitur ini bisa dihitung pada
  saat transaksi terjadi, tanpa tahu masa depan?" Kalau tidak, tolak fitur itu.

### Memori
- Dataset mentah ~1.5GB. Selalu jalankan dtype reduction setelah load.
- Simpan intermediate dalam parquet, bukan CSV.
- Untuk operasi graph, pakai scipy.sparse. Jangan pakai NetworkX untuk komputasi
  di full graph — hanya untuk visualisasi subgraph kecil.

### Kode
- Semua path dan hyperparameter dari file YAML di `configs/`. Tidak ada hardcode.
- Setiap modul di `src/` harus bisa diimport dan ditest secara independen.
- Notebook tidak boleh jadi sumber kebenaran. Logic apa pun yang dipakai lebih dari
  sekali harus pindah ke `src/`.
- Setiap fungsi feature engineering punya docstring: apa yang dihitung, kenapa
  masuk akal secara bisnis, dan risiko leakage-nya.
- Set random seed di semua tempat. Simpan seed di config.

### Evaluasi
- Metrik utama: AUC-ROC, KS statistic, PR-AUC. KS wajib ada karena ini standar
  industri credit scoring.
- Selalu laporkan metrik pada out-of-time test set, bukan validation.
- Setiap eksperimen dicatat ke `artifacts/experiments.csv` dengan: nama, feature set,
  hyperparameter hash, metrik val, metrik test, timestamp.
- Kalau graph features tidak memberikan lift, laporkan apa adanya. Jangan tuning
  sampai kelihatan bagus.

## Perintah
- `make data` — download + join + reduce memory
- `make graph` — bangun graph dan hitung fitur graph
- `make features` — bangun feature matrix
- `make train MODEL=lgbm_baseline` — training
- `make eval` — evaluasi semua model dan tulis laporan perbandingan
- `make test` — pytest

## Gaya Interaksi
- Sebelum menulis kode untuk tahap baru, jelaskan rencananya dulu dalam 3-5 poin
  dan tunggu konfirmasi.
- Jangan menulis fungsi lebih dari 50 baris tanpa memecahnya.
- Kalau ada keputusan metodologis yang ambigu (misal: bagaimana mendefinisikan UID),
  sebutkan trade-off-nya, jangan langsung pilih diam-diam.
