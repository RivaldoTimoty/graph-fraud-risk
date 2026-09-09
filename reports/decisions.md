# Catatan Keputusan

Keputusan metodologis beserta alasannya, dicatat saat diambil. Bahan mentah untuk
technical write-up.

---

## Fase 1 — Temporal Split & EDA

### D1. Split dipotong pada batas hari, bukan persentil baris

**Keputusan.** Batas split ditentukan oleh indeks hari, bukan `quantile(0.70)` /
`quantile(0.85)` atas baris.

```
train : hari   0–118  (119 hari)   410.601 baris   69,5%
gap   : hari 119–125  (7 hari)      23.575 baris    4,0%   ← dibuang
val   : hari 126–150  (25 hari)     67.038 baris   11,4%
test  : hari 151–181  (31 hari)     89.326 baris   15,1%   ← OOT, sekali pakai
```

**Alasan.** Persentil baris jatuh di tengah hari (kuantil 0,70 → hari 119; 0,85 →
hari 151), sehingga transaksi dari hari yang sama terbelah antara dua split. Fitur
agregasi harian di Fase 3 akan membocorkan informasi lintas batas kalau itu terjadi.
Memotong di batas hari menghilangkan risiko tersebut sepenuhnya.

**Trade-off.** Proporsi tidak persis 70/15/15. Hasil aktual 69,5 / 11,4 / 15,1
(sisanya gap) karena volume harian tidak seragam — paruh kedua data lebih tipis.
Validation lebih kecil dari nominal; dapat diterima karena val hanya dipakai untuk
early stopping dan tuning, bukan pelaporan.

### D2. Gap 7 hari antara train dan validation

**Keputusan.** Hari 119–125 dibuang seluruhnya.

**Alasan.** Meniru delay pelaporan fraud di dunia nyata: pada saat model dilatih
ulang, label untuk transaksi paling baru belum tersedia karena chargeback butuh
waktu. Tanpa gap, model dievaluasi pada kondisi yang lebih mudah daripada produksi.

**Trade-off.** MASTER_PLAN menyebut 1–2 minggu. Dipilih 7 hari, bukan 14, karena
total data hanya 182 hari — gap 14 hari memakan 7,7% data dan memperkecil train
tanpa manfaat metodologis tambahan yang jelas. Fraud rate di periode gap (3,53%)
tidak berbeda dari periode sekitarnya, jadi tidak ada bias sistematis dari membuang
jendela ini.

### D3. Test set adalah out-of-time dan sekali pakai

Hari 151–181 tidak boleh disentuh sampai Fase 5. Semua tuning, seleksi fitur, dan
early stopping memakai validation. Setiap evaluasi test dicatat di
`artifacts/experiments.csv` supaya jumlah pembukaan test set dapat diaudit.

### D4. Mask disimpan ke parquet

Split ditulis ke `data/interim/split_masks.parquet` (`TransactionID`, `is_train`,
`is_val`, `is_test`, `is_gap`) alih-alih dihitung ulang tiap kali. Alasan:
reproducibility — semua fase memakai definisi split yang identik, dan perubahan
tidak sengaja pada config akan terdeteksi oleh test yang membandingkan mask
tersimpan dengan hasil hitung ulang. Kolom `is_train` akan dipakai langsung sebagai
`label_mask` untuk fitur graph berbasis label di Fase 3.

---

### Temuan EDA yang memengaruhi fase berikutnya

#### T1. Fraud rate tidak stabil — drift 2,45× antar minggu

Rentang data 182 hari (26 minggu penuh), fraud rate keseluruhan **3,50%**.

| Ukuran | Nilai |
|---|---|
| Minggu dengan fraud rate terendah | **minggu 3 — 2,07%** |
| Minggu dengan fraud rate tertinggi | **minggu 16 — 5,07%** |
| Rasio max/min | **2,45×** |
| Standar deviasi antar minggu | 0,69 pp |
| Rata-rata minggu 0–3 | **2,49%** |
| Rata-rata minggu 5–10 | **4,19%** |

Polanya bukan noise acak: ada kenaikan level yang jelas dari minggu 0–3 (~2,5%) ke
minggu 5–10 (~4,2%), lalu berfluktuasi di kisaran 3–4% sampai akhir. Kenaikan
sebesar 1,7 pp dalam sebulan adalah perubahan rezim, bukan variasi musiman biasa.

**Implikasi.** (a) Model yang dilatih pada periode awal akan under-predict di
periode berikutnya — kalibrasi wajib dicek ulang pada test OOT, bukan hanya
diasumsikan. (b) Ini alasan tambahan mengapa metrik berbasis ranking (AUC, KS)
harus dilaporkan berdampingan dengan kalibrasi. (c) Menjadi bahan bagian limitasi
di artikel: lift graph features harus dinilai relatif terhadap baseline yang sama-sama
terkena drift ini.

Untungnya fraud rate antar split relatif seimbang (train 3,51%, val 3,43%, test
3,49%), jadi drift ini tidak menciptakan ketimpangan prior antar partisi.

#### T2. `hour` bukan jam lokal pengguna

Volume terendah di hour 7–10 dan tertinggi di hour 18–21. Kalau acuan `TransactionDT`
adalah tengah malam waktu lokal, palung seharusnya jatuh di jam 2–5. Selisih ini
menunjukkan offset ~5–7 jam, konsisten dengan DT dalam UTC sementara populasi
pengguna berada di zona waktu Amerika.

**Implikasi.** `hour` tetap dipakai sebagai fitur siklikal, tapi di laporan tidak
boleh diberi label interpretatif seperti "transaksi dini hari". Fraud rate pada
hour 7 mencapai 10,6% versus 2,3% pada hour 13 — sinyalnya kuat, tapi penjelasan
naratifnya harus hati-hati.

#### T3. Kandidat node graph — kardinalitas

| Kolom | Unik | % missing | % nilai singleton | Median ukuran grup |
|---|---|---|---|---|
| `card1` | 13.553 | 0,0% | 25,4% | 4 |
| `DeviceInfo` | 1.786 | 79,9% | 24,6% | 4 |
| `card2` | 500 | 1,5% | 0,0% | 176 |
| `addr1` | 332 | 11,1% | 37,1% | 3 |
| `id_31` (browser) | 130 | 76,3% | 11,5% | 80 |
| `card5` | 119 | 0,7% | 14,3% | 13 |
| `P_emaildomain` | 59 | 16,0% | 0,0% | 305 |
| `card4`, `card6`, `DeviceType` | 2–4 | 0,3–76% | 0,0% | sangat besar |

**Implikasi untuk desain graph (Fase 3).**
- `card1` adalah kandidat node terkuat: kardinalitas tinggi, tanpa missing, ukuran
  grup wajar.
- `DeviceInfo` informatif tapi 79,9% missing — node hanya terbentuk untuk seperlima
  transaksi. Perlu diputuskan apakah missing diperlakukan sebagai node tersendiri
  (tidak disarankan: akan menciptakan hub raksasa palsu) atau transaksi tersebut
  dibiarkan tanpa edge device.
- `card3`, `card4`, `card6`, `addr2`, `DeviceType` punya konsentrasi ekstrem (satu
  nilai menampung 65–88% baris). Menjadikannya node akan menghasilkan hub yang
  menghubungkan hampir semua transaksi — tidak informatif dan mahal secara komputasi.
  **Kolom-kolom ini tidak dipakai sebagai node.**
- 25–37% nilai pada `card1`/`addr1`/`DeviceInfo` hanya muncul sekali. Node berderajat
  1 tidak menghubungkan transaksi mana pun; fitur graph untuk baris tersebut akan
  bernilai default. Perlu dilaporkan berapa persen transaksi yang benar-benar
  mendapat sinyal graph.

#### T4. Coverage entitas lintas periode

Berapa persen entitas di val/test yang sudah pernah terlihat di periode training:

| Kolom | Periode | % baris ter-cover | % nilai unik ter-cover |
|---|---|---|---|
| `card1` | val | 98,9% | 90,5% |
| `card1` | test | 98,6% | 87,7% |
| `addr1` | val | 100,0% | 91,7% |
| `addr1` | test | 100,0% | 97,6% |
| `DeviceInfo` | val | 97,1% | 82,1% |
| `DeviceInfo` | test | 96,6% | 78,9% |

**Implikasi.** Coverage berbobot baris tinggi (96,6–100%), artinya secara volume
hampir semua transaksi OOT terhubung ke entitas yang statistiknya sudah diketahui
dari training — kabar baik untuk fitur graph berbasis label. Namun coverage berbobot
nilai unik jauh lebih rendah (78,9–97,6%): entitas baru terus bermunculan, hanya saja
masing-masing bervolume kecil. Untuk `DeviceInfo` di test, 21,1% nilai unik belum
pernah terlihat.

Konsekuensinya, fitur berbasis label pada entitas baru harus punya perlakuan default
yang eksplisit (smoothing ke prior global, bukan nilai 0 yang akan disalahartikan
model sebagai "tidak pernah fraud").

#### T5. Kolom V terbagi menjadi 15 blok dengan pola missing identik

Seluruh 339 kolom `Vxxx` jatuh ke dalam **15 grup** yang pola missing-nya persis
sama (grup terbesar 46 kolom, disusul 43, 32, 31, 23...). Ini menguatkan dugaan
bahwa kolom V berasal dari 15 sumber/blok fitur Vesta yang berbeda, bukan 339 fitur
independen.

**Implikasi.** Seleksi fitur sebaiknya dilakukan per blok, bukan per kolom
individual. Korelasi dalam blok kemungkinan sangat tinggi, sehingga reduksi
dimensi per blok layak dipertimbangkan di Fase 2.

---

## Fase 2 — Baseline Tabular

### D5. Backend adalah XGBoost, bukan LightGBM

**Keputusan.** Semua model gradient boosting memakai XGBoost 3.4.1.

**Alasan.** Binary LightGBM 4.7.0 crash di mesin ini dengan access violation pada
`LGBM_DatasetSetField` — kegagalan di level C library, bukan di kode project.
Dikonfirmasi terjadi pada data acak 1000x5 di venv bersih, dan bahkan saat C API
dipanggil langsung tanpa perantara pandas. Empat perbaikan dicoba dan semuanya
gagal: downgrade numpy (2.5.2 -> 2.2.6), force-reinstall lightgbm, kombinasi
keduanya, dan venv baru dengan numpy 1.26.4 + pandas 2.2.3. XGBoost berjalan
normal di venv yang sama, sehingga masalahnya terisolasi pada binary LightGBM.

**Dampak metodologis: tidak ada.** MASTER_PLAN sudah mencantumkan XGBoost sebagai
pembanding; yang tertukar hanya peran utama dan pembanding. XGBoost menangani NaN
secara native (arah default dipelajari per split), sehingga keputusan "unseen
entity -> NaN" tetap berlaku persis sama. Blok `params` LightGBM dipertahankan di
kedua file config supaya bisa langsung dipakai lagi kalau binary-nya diperbaiki.

Padanan hyperparameter yang dipakai: `num_leaves: 128` -> `max_depth: 7`,
`feature_fraction` -> `colsample_bytree`, `bagging_fraction` -> `subsample`,
`min_data_in_leaf` -> `min_child_weight`.

### D6. Dua varian baseline, bukan satu

**Keputusan.** Dua baseline dilatih dengan pipeline identik, beda hanya pada
keikutsertaan C1-C14.

| Model | Fitur | AUC | KS | PR-AUC | recall@1% | recall@5% | recall@10% |
|---|---|---|---|---|---|---|---|
| B1 `baseline_noC` | 433 | 0,8882 | 0,6109 | 0,4497 | 23,1% | 51,2% | 65,1% |
| B2 `baseline_full` | 447 | **0,9031** | **0,6435** | 0,5067 | 24,3% | 56,5% | 70,1% |
| **Selisih (C1-C14)** | 14 | **+1,49 pp** | **+3,27 pp** | +5,70 pp | +1,2 pp | +5,3 pp | +5,0 pp |

**Alasan.** Klaim MASTER_PLAN bahwa C1-C14 "sudah semi-graph" diuji, dan hasilnya
lebih spesifik dari dugaan awal. Korelasi Spearman pada periode training:

| | C13 | C9 | C2 | C1 |
|---|---|---|---|---|
| vs degree `card1` | -0,01 | -0,00 | 0,01 | -0,00 |
| vs degree UID (`card1`+`addr1`+`D1n`) | **0,46** | **0,35** | **0,30** | 0,25 |

C-features hampir tidak berkorelasi dengan degree kartu, tapi berkorelasi kuat
dengan degree **klien**. Jadi mereka bukan "semi-graph" secara umum — mereka
counting features di level entitas klien yang sudah di-resolve Vesta. Itu persis
lapisan yang akan dibangun ulang di Fase 3.

**Implikasi untuk ablation Fase 4 — ini alasan utama dua varian dipertahankan.**
Graph features harus dibandingkan terhadap KEDUANYA:
- Diuji hanya terhadap B2, lift graph akan tampak kecil secara artifisial, karena
  sebagian sinyal entity-level sudah disediakan C. Kesimpulan "graph tidak
  berguna" akan menyesatkan.
- Diuji hanya terhadap B1, lift akan tampak besar secara artifisial, karena graph
  sebagian hanya menemukan kembali apa yang sudah ada di C.

Selisih B1 vs B2 (**+1,49 pp AUC**) adalah tolok ukur yang bermakna: itulah nilai
sinyal entity-counting yang sudah tersedia secara gratis di dataset. Kalau graph
features menambah jauh di bawah angka itu di atas B2, graph tidak memberi
informasi baru. Kalau menambah jauh di atasnya, graph menangkap struktur yang
tidak bisa diwakili counting per-klien sederhana.

### D7. `scale_pos_weight` = 1,0 — dipilih meski bukan yang ber-AUC tertinggi

**Hasil eksperimen** (B2, validation, fraud rate aktual 3,43%):

| spw | AUC | KS | PR-AUC | ECE | mean_pred | best_iter |
|---|---|---|---|---|---|---|
| **1,0** | 0,9031 | 0,6435 | 0,5067 | **0,0049** | **0,0307** | 863 |
| 5,0 | **0,9159** | **0,6848** | **0,5335** | 0,0360 | 0,0703 | 755 |
| 27,6 | 0,9110 | 0,6643 | 0,5114 | 0,1260 | 0,1603 | 621 |

**Prediksi awal saya salah dan itu dicatat di sini dengan sengaja.** Rencana Fase 2
menyatakan reweighting "tidak akan berpengaruh pada metrik ranking karena AUC/KS
invarian terhadap transformasi monoton". Itu keliru: reweighting bukan transformasi
monoton pada skor akhir, melainkan mengubah *fungsi loss*, sehingga pohon yang
tumbuh pun berbeda. Hasilnya `spw=5` menaikkan AUC +1,28 pp dan KS +4,13 pp.

**Keputusan tetap `spw=1,0`,** dengan alasan berikut:

1. **Kalibrasi.** ECE naik 7x pada `spw=5` (0,0049 -> 0,0360) dan 26x pada
   `spw=27,6` (0,1260). Rata-rata prediksi `spw=27,6` adalah 0,160 versus fraud
   rate aktual 0,034 — model over-predict hampir 5x. Fase 5 membutuhkan
   probabilitas bermakna untuk cost-based threshold; skor yang hanya benar secara
   ranking tidak cukup untuk menghitung ekspektasi kerugian dalam rupiah.
2. **Konsistensi lintas fase.** Baseline resmi harus sama di semua eksperimen agar
   lift graph features di Fase 4 tidak tercampur dengan efek reweighting.
3. **Kejujuran ablation.** Menaikkan baseline lewat tuning yang tidak akan
   diterapkan pada model graph akan membuat perbandingan tidak apple-to-apple.

**Catatan untuk Fase 5.** Kenaikan AUC dari `spw=5` cukup besar untuk ditinjau
ulang setelah kalibrasi isotonic diimplementasikan. Kalau isotonic memulihkan ECE
tanpa menurunkan ranking, kombinasi `spw=5` + isotonic layak dipertimbangkan
sebagai model produksi — tapi keputusan itu harus diambil SETELAH ablation graph
selesai, bukan sebelumnya, dan diterapkan seragam ke semua varian model.

### D8. Anti-leakage pada feature engineering

Semua statistik turunan di-fit hanya pada `is_train`, lalu di-transform ke seluruh
baris. Perlakuan kategori tak terlihat:

- frequency encoding -> **0** (benar: entitas ini memang tidak pernah muncul saat
  training)
- agregasi per entitas -> **NaN**, bukan 0. Nilai 0 akan dibaca model sebagai
  "nominal jauh di bawah rata-rata", padahal artinya "tidak diketahui". Ini bukan
  detail kosmetik: 12,3% nilai `card1` di test belum pernah terlihat saat training
  (T4).
- label encoding -> **-1**, terpisah dari kategori valid `0..n-1`.

Tiga test di `tests/test_tabular.py` menjaga ini, yang terpenting adalah
`test_*_ignores_non_training_rows`: mengubah data periode val/test tidak boleh
mengubah nilai fitur pada baris training. Kalau berubah, berarti statistik ikut
dihitung dari periode yang seharusnya belum terlihat.

**Temuan implementasi.** 15 kolom `id_*` (id_12..id_38) ternyata bertipe string dan
awalnya lolos dari daftar `label_encode`, menyebabkan training gagal. Sekarang
kolom-kolom itu ikut di-encode dan versi mentahnya dikecualikan dari blok fitur.

### D9. Test set masih belum disentuh

Seluruh angka Fase 2 berasal dari validation. Kolom `test_*` di
`artifacts/experiments.csv` masih kosong untuk semua baris — audit trail bahwa
test set out-of-time belum pernah dibuka.

---

## Fase 3 — Graph Construction & Features

### D10. Node graph: 7 kolom, lima kolom sengaja dibuang

Node atribut: `card1`, `addr1`, `DeviceInfo`, `P_emaildomain`, `R_emaildomain`,
`id_30`, `id_31`. Graph akhir: 590.540 x 15.995, nnz 2.085.262 (density 2,2e-04).

`card3`, `card4`, `card6`, `addr2`, `DeviceType` dikecualikan karena satu nilai
menampung 65-88% baris (T3) — sebagai node mereka akan menghubungkan hampir semua
transaksi tanpa membawa informasi.

Missing value TIDAK dijadikan node. Menyatukan 79,9% transaksi tanpa `DeviceInfo`
ke satu node akan menciptakan hub palsu raksasa yang tidak punya makna bisnis.

### D11. Adjacency 590K x 590K tidak pernah dimaterialisasi

Estimasi nnz `A @ A.T` adalah **89,5 miliar**, dengan `P_emaildomain` menyumbang
66,8 miliar sendirian. Semua agregasi memakai `A @ (A.T @ v)` — biayanya linier
terhadap nnz(A) = 2,08 juta. Urutan kurung ini wajib; `(A @ A.T) @ v` akan
mencoba membentuk matriks penuh.

`test_neighbor_sum_equals_dense_adjacency_result` membuktikan trik ini menghasilkan
angka yang identik dengan perhitungan dense pada toy graph — bukan sekadar lebih
cepat, tapi benar.

### D12. Hub cap 5.000 untuk propagasi tetangga

Atribut dengan degree > 5.000 dikeluarkan dari agregasi tetangga, tapi **tetap
dipakai sebagai fitur degree**. 69 node terkena, termasuk `gmail.com` dengan
228.355 transaksi (38,7% data).

Alasan: "berbagi gmail" bukan sinyal fraud. Kalau ikut dipropagasikan, setiap
transaksi gmail menjadi tetangga setiap transaksi gmail lain — noise yang
menenggelamkan sinyal dari atribut spesifik seperti device atau kartu.

### D13. `graph_component_id` dibuang dari output

ID komponen bersifat arbitrer (bergantung urutan penelusuran), sehingga model
akan memperlakukan jarak antar-ID sebagai bermakna padahal tidak. Hanya
`graph_component_size` yang informatif. Ditemukan saat inspeksi distribusi, dan
sekarang dijaga oleh `test_component_id_is_not_exposed_as_feature`.

### D14. Level 3 — leave-one-out secara sparse

**Formula.** Agregasi `A @ (A.T @ v)` menyertakan transaksi itu sendiri sebagai
tetangganya. Bobot kontribusi diri adalah diagonal `(A @ A.T)[i,i] = sum_j A[i,j]^2`,
yang untuk A biner sama dengan jumlah atribut yang dimiliki i — dihitung tanpa
membentuk matriksnya:

```
loo_positive = A @ (A.T @ y_masked) - self_weight * y_masked
loo_labeled  = A @ (A.T @ mask)     - self_weight * mask
```

Untuk baris di dalam mask, kontribusi dirinya hilang tepat. Untuk baris di luar
mask, pengurangannya nol — benar, karena labelnya tidak pernah ikut sejak awal.
Diverifikasi terhadap perhitungan dense dengan diagonal di-nol-kan.

**Smoothing.** `rate = (pos + alpha*prior) / (labeled + alpha)` dengan alpha = 20
dan prior = fraud rate periode training = **0,03512**. Ketika `labeled = 0`, hasil
jatuh tepat ke prior tanpa percabangan khusus. **78.939 baris** memang jatuh ke
prior karena tidak punya tetangga berlabel.

**label_mask = is_train untuk SEMUA baris,** termasuk saat menghitung fitur untuk
test. Ini pilihan konservatif yang disengaja: di produksi, label validation
sebenarnya sudah diketahui saat scoring test, sehingga fitur test di sini lebih
lemah dari yang bisa dicapai. Dicatat sebagai trade-off sadar, bukan kelalaian.

**Verifikasi pada data nyata.** Membalik SELURUH label val+test pada 590.540 baris
menghasilkan fitur Level 3 yang **identik bit-per-bit**. Ini bukan hanya test unit
pada toy graph — dijalankan pada dataset penuh.

**Limitasi 2-hop yang harus dinyatakan.** Koreksi LOO pada 2-hop tidak seeksak
1-hop: jalur bolak-balik (i -> j -> i) tetap tersisa dalam bentuk tak-langsung
setelah kontribusi diri dikurangi di kedua tingkat. Fitur `graph_nb_fraud_rate_2hop`
karena itu punya jaminan lebih lemah daripada versi 1-hop. Kalau di Fase 4
kontribusinya kecil, sebaiknya digugurkan saja daripada dipertahankan dengan
jaminan yang tidak penuh.

### T6. Kekuatan sinyal Level 3 dan diagnostik leakage

AUC univariat per split:

| Fitur | train | val | test | gap train-val |
|---|---|---|---|---|
| `uid_fraud_rate` | 0,8957 | 0,7802 | 0,6977 | 0,1155 |
| `graph_nb_fraud_rate_1hop` | 0,8066 | 0,7471 | 0,7269 | 0,0595 |
| `graph_nb_fraud_rate_2hop` | 0,7828 | 0,7258 | 0,7346 | 0,0570 |
| `graph_community_fraud_rate` | 0,5823 | 0,6487 | 0,6753 | **-0,0664** |

Gap besar pada `uid_fraud_rate` awalnya tampak seperti gejala menghafal label.
Investigasi menunjukkan sebaliknya — penyebabnya **coverage**, bukan leakage:

| Split | Baris tanpa tetangga UID berlabel | AUC pada baris yang PUNYA tetangga |
|---|---|---|
| train | 33,2% | 0,9725 |
| val | 57,7% | 0,9957 |
| test | **67,3%** | **0,9751** |

Pada baris yang benar-benar punya tetangga UID berlabel, AUC test (0,975) bahkan
sedikit lebih tinggi dari train (0,973) — tidak ada tanda menghafal sama sekali.
Penurunan AUC agregat murni karena dua pertiga baris test jatuh ke prior yang
konstan, sehingga tidak membawa informasi apa pun.

Ini konsisten dengan temuan T4: coverage UID train->test hanya 33,9% baris. Karena
itu `uid_labeled_count` disertakan sebagai fitur — model perlu tahu kapan
`uid_fraud_rate` layak dipercaya dan kapan ia hanya prior.

`graph_community_fraud_rate` justru punya gap NEGATIF (performa val/test lebih
baik dari train), yang menyingkirkan kecurigaan leakage untuk fitur itu.

### T7. Fitur Level 3 vs C-features — bukti untuk artikel

Korelasi Spearman pada periode training:

| C-feature | `nb_fraud_rate_1hop` | `nb_fraud_rate_2hop` | `community_fraud_rate` | `uid_fraud_rate` |
|---|---|---|---|---|
| C4 | 0,431 | **0,492** | 0,366 | 0,381 |
| C7 | 0,409 | 0,481 | **0,485** | 0,385 |
| C8 | 0,424 | 0,480 | 0,340 | 0,379 |
| C10 | 0,406 | 0,460 | 0,321 | 0,363 |
| C12 | 0,357 | 0,419 | 0,426 | 0,286 |
| C9 | -0,350 | -0,399 | -0,277 | **-0,444** |
| C13 | -0,220 | -0,249 | -0,141 | **-0,461** |

**Temuan utama, dan lebih bernuansa dari hipotesis awal.** Hipotesis di Fase 2
adalah C13 akan berkorelasi tinggi dengan `nb_fraud_rate_1hop`, karena C13 sudah
berkorelasi 0,46 dengan degree UID. Yang terjadi:

1. **C13 memang berkorelasi kuat, tapi dengan `uid_fraud_rate` (-0,461), bukan
   dengan fitur bipartite (-0,220).** Ini justru mempertajam kesimpulan Fase 2:
   C13 meng-encode sesuatu di level KLIEN, bukan level atribut bersama. Korelasi
   negatif berarti C13 tinggi menyertai fraud rate klien yang rendah — arah yang
   berlawanan, tapi kekuatan hubungannya nyata.

2. **Blok C4/C7/C8/C10/C12 berkorelasi 0,36-0,49 dengan fitur graph berbasis
   label.** Ini kelompok C yang perilakunya paling mirip neighbor fraud rate.

3. **Tidak ada korelasi yang mendekati 0,8+.** Artinya C-features TIDAK menduplikasi
   fitur graph — mereka menangkap sinyal yang beririsan tapi berbeda. Ini kabar
   baik untuk ablation: masih ada ruang bagi graph features untuk menambah
   informasi di atas B2.

**Implikasi untuk artikel.** Narasinya bukan "Vesta sudah punya fitur graph
sehingga graph tidak berguna", melainkan lebih menarik: *Vesta meng-encode sinyal
counting di level klien (C13, C9) dan sesuatu yang menyerupai neighbor fraud rate
(C4, C7, C8, C10), tapi korelasi 0,36-0,49 menunjukkan keduanya tidak sama.*
Ablation Fase 4 akan mengukur berapa banyak sisa informasi yang benar-benar baru.
