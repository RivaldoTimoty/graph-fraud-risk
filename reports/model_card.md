# Model Card — Fraud Risk Scorecard (M3)

**Versi:** 1.0 | **Status:** portofolio / riset, BUKAN produksi
**Terakhir diperbarui:** setelah Fase 5

---

## Ringkasan

Model klasifikasi biner yang memberi skor risiko fraud pada transaksi e-commerce,
dilatih pada dataset IEEE-CIS (Vesta Corporation). Output berupa probabilitas
fraud yang dikonversi ke skala skor 300–850 mengikuti konvensi credit scoring.

| | |
|---|---|
| Algoritma | XGBoost (gradient boosted trees), 447 fitur |
| Target | `isFraud` — biner, prevalensi 3,50% |
| Output | Probabilitas [0,1] → skor 300–850 (tinggi = risiko rendah) |
| Hyperparameter kunci | `max_depth=7`, `learning_rate=0,05`, `scale_pos_weight=1,0` |
| Seed | 42 (semua tahap) |

**Fitur graph TIDAK termasuk dalam model final.** Alasannya di bagian
[Keputusan desain](#keputusan-desain-yang-perlu-diketahui).

---

## Data

| | |
|---|---|
| Sumber | IEEE-CIS Fraud Detection (Kaggle), data transaksi Vesta |
| Ukuran | 590.540 transaksi, 434 kolom mentah |
| Periode | 182 hari (26 minggu penuh) |
| Prevalensi fraud | 3,50% keseluruhan |
| Identity coverage | 24,4% transaksi punya record identity |

**Pembagian temporal** (dipotong pada batas hari, bukan persentil baris):

| Split | Hari | Baris | Fraud rate |
|---|---|---|---|
| Training | 0–118 | 410.601 (69,5%) | 3,51% |
| Gap (dibuang) | 119–125 | 23.575 (4,0%) | 3,53% |
| Validation | 126–150 | 67.038 (11,4%) | 3,43% |
| Test (OOT) | 151–181 | 89.326 (15,1%) | 3,49% |

**Kelompok fitur:** kolom numerik Vesta (V1–V339, 339 kolom), counting features
(C1–C14), timedelta (D1–D15), identity (id_01–id_38), kategorikal ter-encode,
frequency encoding, dan agregasi `TransactionAmt` per entitas.

---

## Performa (test out-of-time, 89.326 transaksi)

| Metrik | Nilai |
|---|---|
| AUC-ROC | 0,9007 |
| KS statistic | 0,6471 |
| PR-AUC | 0,5196 |
| recall@1% | 25,3% |
| recall@5% | 56,3% |
| recall@10% | 69,4% |
| PSI skor (train→test) | 0,0032 (stabil) |
| ECE | 0,00348 (terkalibrasi tanpa post-processing) |

Skor rata-rata: fraud **514**, non-fraud **626**.

Distribusi per band (10 band, masing-masing 10% populasi):

| Band | Rentang skor | Fraud rate | Cum. fraud tertangkap |
|---|---|---|---|
| 0 (terburuk) | 300–569 | 24,19% | 69,40% |
| 1 | 569–593 | 4,30% | 81,73% |
| 2 | 593–607 | 2,17% | 87,96% |
| 3–8 | 607–672 | 1,24% → 0,17% | 99,49% |
| 9 (terbaik) | 672–762 | 0,18% | 100% |

---

## Asumsi

**Asumsi data**

1. Distribusi transaksi masa depan menyerupai periode training. Fraud rate
   mingguan pada data ini berkisar 2,07%–5,07% (rasio 2,45×), jadi asumsi ini
   sudah tegang bahkan dalam periode pengamatan.
2. `TransactionDT` monoton dan mencerminkan urutan kejadian sebenarnya.
3. Label `isFraud` lengkap dan benar pada periode training. Dalam praktik, fraud
   yang tidak pernah dilaporkan akan terlabel sebagai non-fraud.
4. `hour` **bukan jam lokal pengguna** — profil volume menunjukkan offset ~5–7 jam
   dari tengah malam lokal. Fitur ini valid sebagai sinyal siklikal, tapi tidak
   boleh diberi interpretasi naratif seperti "transaksi dini hari".

**Asumsi bisnis** (dipakai untuk analisis biaya)

5. Biaya false negative = nilai transaksi (`TransactionAmt`). Mengabaikan biaya
   penanganan sengketa dan kerugian reputasi.
6. Biaya false positive = konstanta per transaksi. **Ini asumsi, bukan angka
   terukur** — karena itu hasil disajikan sebagai sensitivitas terhadap $2/$5/$10/$25.
7. Setiap transaksi ter-flag menerima review manusia dengan biaya seragam.
8. Kapasitas review tidak terbatas. Model tidak memodelkan antrean atau SLA.

---

## Keputusan desain yang perlu diketahui

**Fitur graph tidak dipakai, meski sudah dibangun.**
Ablation menunjukkan graph features menurunkan AUC test 1,91 pp. Dekomposisi
memisahkan dua perilaku berlawanan: fitur struktural (degree, PageRank, component
size) memberi **+2,30 pp di validation**, sementara fitur berbasis label
(neighbor fraud rate, UID fraud rate) merusak generalisasi. Karena keduanya diuji
bersama di test, kombinasi terbaik (baseline + struktural saja) **belum
terverifikasi out-of-time**.

**`scale_pos_weight=1,0`, bukan 5.**
`spw=5` memberi AUC +1,28 pp di validation, dan isotonic memulihkan kalibrasinya
(ECE 0,0371 → 0,0058). Tidak dipakai karena memilihnya berdasarkan validation lalu
mengevaluasinya di test akan menjadi pembukaan test set kedua.

**Tanpa kalibrasi post-processing.**
Isotonic justru **memperburuk** ECE di test (0,00348 → 0,00363): kalibrator di-fit
pada validation (fraud rate 0,03426) lalu diterapkan ke test (0,03486), sehingga
mengimpor drift. Model raw sudah terkalibrasi baik (bias −0,00037).

**Tanpa SMOTE.** Fraud sintetis bukan fraud; reweighting lewat `scale_pos_weight`
dinilai lebih tepat.

---

## Kapan model ini TIDAK boleh dipakai

**Jangan pakai sebagai satu-satunya dasar penolakan otomatis.**
Pada review rate 10%, model melewatkan 30,6% fraud. Band terburuk punya fraud rate
24,2% — artinya **75,8% transaksi di band itu sah**. Penolakan otomatis akan
menolak tiga pelanggan jujur untuk setiap satu penipu.

**Jangan pakai di luar populasi e-commerce serupa.**
Model dilatih pada transaksi e-commerce Vesta. Tidak ada bukti generalisasi ke
pinjaman, transfer P2P, transaksi kartu present, atau pasar dengan komposisi
pembayaran berbeda.

**Jangan pakai lebih dari ~3 bulan tanpa retrain.**
Fraud rate bergeser 2,45× dalam 26 minggu data. Selain itu, empat fitur (`M3`,
`M7`, `M8`, `M9`) punya CSI 0,28–0,34 karena **missingness-nya berubah** dari 67%
ke 39% antar periode — makna fitur berubah seiring waktu.

**Jangan pakai probabilitasnya untuk keputusan di luar rentang teramati.**
Kalibrasi diverifikasi pada rentang probabilitas yang muncul di test. Ekstrapolasi
ke keputusan pricing atau limit kredit di luar rentang itu tidak didukung bukti.

**Jangan pakai untuk mengambil keputusan tentang individu tanpa jalur banding.**
Model tidak diaudit untuk bias demografis — dataset tidak menyediakan atribut yang
diperlukan untuk itu, sehingga ketiadaan bias **tidak dapat diklaim**.

**Jangan pakai fitur graph berbasis label dari repo ini di produksi**, kecuali
setelah diverifikasi ulang pada periode baru. Terbukti merusak generalisasi
temporal meski implementasinya lolos setiap pengecekan leakage.

---

## Monitoring yang wajib

| Yang dipantau | Ambang | Alasan |
|---|---|---|
| PSI skor | > 0,10 perhatian, > 0,25 retrain | Standar industri |
| CSI per fitur | > 0,25 | PSI skor sehat tidak menjamin fitur sehat |
| **Missingness per fitur** | perubahan > 10 pp | M3/M7/M8/M9 bergeser 67% → 39% |
| Fraud rate aktual mingguan | di luar 2–5% | Rentang historis teramati |
| Kalibrasi (mean prediksi vs aktual) | bias > 0,005 | Keputusan biaya butuh probabilitas benar |

**Jika fitur berbasis entitas dipakai kelak:** pantau **jumlah observasi
pendukung**, bukan hanya nilai fiturnya. Pelajaran dari Fase 4 —
`uid_fraud_rate` punya nilai stabil (drift −2,8%) sementara kekuatan buktinya
runtuh 52%. PSI tidak menangkap kegagalan seperti itu.

---

## Reproduksibilitas

Seed 42 di semua tahap. Pipeline deterministik: M3 dilatih ulang di Fase 5
mereproduksi AUC test 0,9007 persis.

Semua path dan hyperparameter berada di `configs/*.yaml`. Riwayat eksperimen di
`artifacts/experiments.csv` — 9 baris, 4 di antaranya punya `test_auc` terisi,
yang merupakan audit trail bahwa test set dibuka satu kali untuk empat model.

79 unit test, sebagian besar menguji pencegahan leakage.

---

## Kontak & atribusi

Dibangun sebagai portofolio untuk peran Data Scientist credit scoring/risk.
Dataset: IEEE-CIS Fraud Detection, Vesta Corporation, via Kaggle.
Detail keputusan metodologis: [`reports/decisions.md`](decisions.md).
