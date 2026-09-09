# Catatan Keputusan

Keputusan metodologis beserta alasannya, dicatat saat diambil. Bahan mentah untuk
technical write-up.

---

## Fase 1 - Temporal Split & EDA

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
(sisanya gap) karena volume harian tidak seragam - paruh kedua data lebih tipis.
Validation lebih kecil dari nominal; dapat diterima karena val hanya dipakai untuk
early stopping dan tuning, bukan pelaporan.

### D2. Gap 7 hari antara train dan validation

**Keputusan.** Hari 119–125 dibuang seluruhnya.

**Alasan.** Meniru delay pelaporan fraud di dunia nyata: pada saat model dilatih
ulang, label untuk transaksi paling baru belum tersedia karena chargeback butuh
waktu. Tanpa gap, model dievaluasi pada kondisi yang lebih mudah daripada produksi.

**Trade-off.** MASTER_PLAN menyebut 1–2 minggu. Dipilih 7 hari, bukan 14, karena
total data hanya 182 hari - gap 14 hari memakan 7,7% data dan memperkecil train
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
reproducibility - semua fase memakai definisi split yang identik, dan perubahan
tidak sengaja pada config akan terdeteksi oleh test yang membandingkan mask
tersimpan dengan hasil hitung ulang. Kolom `is_train` akan dipakai langsung sebagai
`label_mask` untuk fitur graph berbasis label di Fase 3.

---

### Temuan EDA yang memengaruhi fase berikutnya

#### T1. Fraud rate tidak stabil - drift 2,45× antar minggu

Rentang data 182 hari (26 minggu penuh), fraud rate keseluruhan **3,50%**.

| Ukuran | Nilai |
|---|---|
| Minggu dengan fraud rate terendah | **minggu 3 - 2,07%** |
| Minggu dengan fraud rate tertinggi | **minggu 16 - 5,07%** |
| Rasio max/min | **2,45×** |
| Standar deviasi antar minggu | 0,69 pp |
| Rata-rata minggu 0–3 | **2,49%** |
| Rata-rata minggu 5–10 | **4,19%** |

Polanya bukan noise acak: ada kenaikan level yang jelas dari minggu 0–3 (~2,5%) ke
minggu 5–10 (~4,2%), lalu berfluktuasi di kisaran 3–4% sampai akhir. Kenaikan
sebesar 1,7 pp dalam sebulan adalah perubahan rezim, bukan variasi musiman biasa.

**Implikasi.** (a) Model yang dilatih pada periode awal akan under-predict di
periode berikutnya - kalibrasi wajib dicek ulang pada test OOT, bukan hanya
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
hour 7 mencapai 10,6% versus 2,3% pada hour 13 - sinyalnya kuat, tapi penjelasan
naratifnya harus hati-hati.

#### T3. Kandidat node graph - kardinalitas

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
- `DeviceInfo` informatif tapi 79,9% missing - node hanya terbentuk untuk seperlima
  transaksi. Perlu diputuskan apakah missing diperlakukan sebagai node tersendiri
  (tidak disarankan: akan menciptakan hub raksasa palsu) atau transaksi tersebut
  dibiarkan tanpa edge device.
- `card3`, `card4`, `card6`, `addr2`, `DeviceType` punya konsentrasi ekstrem (satu
  nilai menampung 65–88% baris). Menjadikannya node akan menghasilkan hub yang
  menghubungkan hampir semua transaksi - tidak informatif dan mahal secara komputasi.
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
dari training - kabar baik untuk fitur graph berbasis label. Namun coverage berbobot
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

## Fase 2 - Baseline Tabular

### D5. Backend adalah XGBoost, bukan LightGBM

**Keputusan.** Semua model gradient boosting memakai XGBoost 3.4.1.

**Alasan.** Binary LightGBM 4.7.0 crash di mesin ini dengan access violation pada
`LGBM_DatasetSetField` - kegagalan di level C library, bukan di kode project.
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
dengan degree **klien**. Jadi mereka bukan "semi-graph" secara umum - mereka
counting features di level entitas klien yang sudah di-resolve Vesta. Itu persis
lapisan yang akan dibangun ulang di Fase 3.

**Implikasi untuk ablation Fase 4 - ini alasan utama dua varian dipertahankan.**
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

### D7. `scale_pos_weight` = 1,0 - dipilih meski bukan yang ber-AUC tertinggi

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
   rate aktual 0,034 - model over-predict hampir 5x. Fase 5 membutuhkan
   probabilitas bermakna untuk cost-based threshold; skor yang hanya benar secara
   ranking tidak cukup untuk menghitung ekspektasi kerugian dalam rupiah.
2. **Konsistensi lintas fase.** Baseline resmi harus sama di semua eksperimen agar
   lift graph features di Fase 4 tidak tercampur dengan efek reweighting.
3. **Kejujuran ablation.** Menaikkan baseline lewat tuning yang tidak akan
   diterapkan pada model graph akan membuat perbandingan tidak apple-to-apple.

**Catatan untuk Fase 5.** Kenaikan AUC dari `spw=5` cukup besar untuk ditinjau
ulang setelah kalibrasi isotonic diimplementasikan. Kalau isotonic memulihkan ECE
tanpa menurunkan ranking, kombinasi `spw=5` + isotonic layak dipertimbangkan
sebagai model produksi - tapi keputusan itu harus diambil SETELAH ablation graph
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
`artifacts/experiments.csv` masih kosong untuk semua baris - audit trail bahwa
test set out-of-time belum pernah dibuka.

---

## Fase 3 - Graph Construction & Features

### D10. Node graph: 7 kolom, lima kolom sengaja dibuang

Node atribut: `card1`, `addr1`, `DeviceInfo`, `P_emaildomain`, `R_emaildomain`,
`id_30`, `id_31`. Graph akhir: 590.540 x 15.995, nnz 2.085.262 (density 2,2e-04).

`card3`, `card4`, `card6`, `addr2`, `DeviceType` dikecualikan karena satu nilai
menampung 65-88% baris (T3) - sebagai node mereka akan menghubungkan hampir semua
transaksi tanpa membawa informasi.

Missing value TIDAK dijadikan node. Menyatukan 79,9% transaksi tanpa `DeviceInfo`
ke satu node akan menciptakan hub palsu raksasa yang tidak punya makna bisnis.

### D11. Adjacency 590K x 590K tidak pernah dimaterialisasi

Estimasi nnz `A @ A.T` adalah **89,5 miliar**, dengan `P_emaildomain` menyumbang
66,8 miliar sendirian. Semua agregasi memakai `A @ (A.T @ v)` - biayanya linier
terhadap nnz(A) = 2,08 juta. Urutan kurung ini wajib; `(A @ A.T) @ v` akan
mencoba membentuk matriks penuh.

`test_neighbor_sum_equals_dense_adjacency_result` membuktikan trik ini menghasilkan
angka yang identik dengan perhitungan dense pada toy graph - bukan sekadar lebih
cepat, tapi benar.

### D12. Hub cap 5.000 untuk propagasi tetangga

Atribut dengan degree > 5.000 dikeluarkan dari agregasi tetangga, tapi **tetap
dipakai sebagai fitur degree**. 69 node terkena, termasuk `gmail.com` dengan
228.355 transaksi (38,7% data).

Alasan: "berbagi gmail" bukan sinyal fraud. Kalau ikut dipropagasikan, setiap
transaksi gmail menjadi tetangga setiap transaksi gmail lain - noise yang
menenggelamkan sinyal dari atribut spesifik seperti device atau kartu.

### D13. `graph_component_id` dibuang dari output

ID komponen bersifat arbitrer (bergantung urutan penelusuran), sehingga model
akan memperlakukan jarak antar-ID sebagai bermakna padahal tidak. Hanya
`graph_component_size` yang informatif. Ditemukan saat inspeksi distribusi, dan
sekarang dijaga oleh `test_component_id_is_not_exposed_as_feature`.

### D14. Level 3 - leave-one-out secara sparse

**Formula.** Agregasi `A @ (A.T @ v)` menyertakan transaksi itu sendiri sebagai
tetangganya. Bobot kontribusi diri adalah diagonal `(A @ A.T)[i,i] = sum_j A[i,j]^2`,
yang untuk A biner sama dengan jumlah atribut yang dimiliki i - dihitung tanpa
membentuk matriksnya:

```
loo_positive = A @ (A.T @ y_masked) - self_weight * y_masked
loo_labeled  = A @ (A.T @ mask)     - self_weight * mask
```

Untuk baris di dalam mask, kontribusi dirinya hilang tepat. Untuk baris di luar
mask, pengurangannya nol - benar, karena labelnya tidak pernah ikut sejak awal.
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
pada toy graph - dijalankan pada dataset penuh.

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
Investigasi menunjukkan sebaliknya - penyebabnya **coverage**, bukan leakage:

| Split | Baris tanpa tetangga UID berlabel | AUC pada baris yang PUNYA tetangga |
|---|---|---|
| train | 33,2% | 0,9725 |
| val | 57,7% | 0,9957 |
| test | **67,3%** | **0,9751** |

Pada baris yang benar-benar punya tetangga UID berlabel, AUC test (0,975) bahkan
sedikit lebih tinggi dari train (0,973) - tidak ada tanda menghafal sama sekali.
Penurunan AUC agregat murni karena dua pertiga baris test jatuh ke prior yang
konstan, sehingga tidak membawa informasi apa pun.

Ini konsisten dengan temuan T4: coverage UID train->test hanya 33,9% baris. Karena
itu `uid_labeled_count` disertakan sebagai fitur - model perlu tahu kapan
`uid_fraud_rate` layak dipercaya dan kapan ia hanya prior.

`graph_community_fraud_rate` justru punya gap NEGATIF (performa val/test lebih
baik dari train), yang menyingkirkan kecurigaan leakage untuk fitur itu.

### T7. Fitur Level 3 vs C-features - bukti untuk artikel

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
   negatif berarti C13 tinggi menyertai fraud rate klien yang rendah - arah yang
   berlawanan, tapi kekuatan hubungannya nyata.

2. **Blok C4/C7/C8/C10/C12 berkorelasi 0,36-0,49 dengan fitur graph berbasis
   label.** Ini kelompok C yang perilakunya paling mirip neighbor fraud rate.

3. **Tidak ada korelasi yang mendekati 0,8+.** Artinya C-features TIDAK menduplikasi
   fitur graph - mereka menangkap sinyal yang beririsan tapi berbeda. Ini kabar
   baik untuk ablation: masih ada ruang bagi graph features untuk menambah
   informasi di atas B2.

**Implikasi untuk artikel.** Narasinya bukan "Vesta sudah punya fitur graph
sehingga graph tidak berguna", melainkan lebih menarik: *Vesta meng-encode sinyal
counting di level klien (C13, C9) dan sesuatu yang menyerupai neighbor fraud rate
(C4, C7, C8, C10), tapi korelasi 0,36-0,49 menunjukkan keduanya tidak sama.*
Ablation Fase 4 akan mengukur berapa banyak sisa informasi yang benar-benar baru.

---

## Fase 4a - Ablation Study: Graph Features Menurunkan Performa

**Ringkasan: fitur graph BERBASIS LABEL (Level 3) memperburuk performa
out-of-time secara serius, sementara fitur graph STRUKTURAL (Level 1-2) justru
memberi lift substansial di validation (+2,30 pp AUC). Karena keduanya diuji
bersama di test, model bergraf terbaik yang terukur out-of-time tetap kalah dari
baseline murni.**

Dari empat model yang dievaluasi di test, yang terbaik adalah M3 - baseline tanpa
graph. Tetapi dekomposisi di bagian 2 menunjukkan itu bukan karena graph tidak
berguna, melainkan karena Level 3 menenggelamkan kontribusi Level 1-2.

Hasil negatif ini dilaporkan apa adanya sesuai aturan proyek, termasuk fakta bahwa
kombinasi terbaik (B2 + Level 1-2 saja) tidak sempat terverifikasi di test.

### 1. Tabel hasil - test set out-of-time (hari 151-181)

Test set dibuka SEKALI untuk keempat model, setelah konfigurasi dikunci.
Audit trail: `artifacts/experiments.csv` punya tepat 4 baris dengan `test_auc`
terisi dari total 9 baris eksperimen.

| Metrik | M1 (B1) | M2 (B1+graph) | M3 (B2) | M4 (B2+graph) |
|---|---|---|---|---|
| AUC | 0,8922 | 0,8099 | **0,9007** | 0,8816 |
| KS | 0,6281 | 0,4177 | **0,6471** | 0,5942 |
| PR-AUC | 0,4806 | 0,2942 | **0,5196** | 0,3901 |
| recall@1% | 0,2408 | 0,2039 | **0,2531** | 0,2200 |
| recall@5% | 0,5337 | 0,4252 | **0,5633** | 0,4730 |
| recall@10% | 0,6731 | 0,4974 | **0,6940** | 0,6008 |

Lift dan tumpang-tindih:

| Metrik | Lift_B1 (M2-M1) | Lift_B2 (M4-M3) | Nilai C (M3-M1) | **Tumpang-tindih C<->graph** |
|---|---|---|---|---|
| AUC | -0,0823 | -0,0191 | +0,0085 | **-0,0633** |
| KS | -0,2103 | -0,0529 | +0,0191 | **-0,1574** |
| PR-AUC | -0,1864 | -0,1295 | +0,0390 | **-0,0569** |
| recall@1% | -0,0369 | -0,0331 | +0,0122 | -0,0039 |
| recall@5% | -0,1085 | -0,0902 | +0,0295 | -0,0183 |
| recall@10% | -0,1757 | -0,0931 | +0,0209 | -0,0825 |

**Cara membaca tumpang-tindih.** Angka ini dirancang untuk mengukur berapa sinyal
graph yang sudah ada di C-features. Karena kedua lift negatif, tandanya berbalik
makna: tumpang-tindih -0,0633 AUC berarti **kerusakan yang ditimbulkan graph jauh
lebih kecil ketika C-features hadir** (-1,91 pp) dibanding ketika tidak ada
(-8,23 pp). C-features berfungsi sebagai peredam: model yang sudah punya sinyal
entity-level yang stabil lebih sedikit bergantung pada fitur graph yang rapuh.

Catatan: "Nilai C" di test (+0,85 pp AUC) lebih kecil daripada di validation
(+1,49 pp, Fase 2) - konsisten dengan drift yang sudah terdokumentasi.

### 2. Dekomposisi: Level 1-2 vs Level 3

**Temuan yang mengubah kesimpulan.** Diagnostik terpisah (validation, konfigurasi
identik) memisahkan kontribusi kedua level:

| Feature set | n fitur | VAL AUC | VAL KS | vs B2 |
|---|---|---|---|---|
| B2 (baseline) | 447 | 0,9031 | 0,6435 | - |
| **B2 + Level 1-2 saja** | 473 | **0,9261** | **0,7096** | **+2,30 pp / +6,61 pp** |
| B2 + Level 3 saja | 453 | 0,9067 | 0,6557 | +0,36 pp / +1,22 pp |
| B2 + semua (M4) | 479 | 0,9081 | 0,6630 | +0,50 pp / +1,95 pp |

**Level 1-2 sendirian adalah feature set terbaik di validation** - lebih baik
daripada menambahkan seluruh fitur graph. Menambahkan Level 3 ke B2+L12 justru
MENURUNKAN AUC dari 0,9261 ke 0,9081 (-1,80 pp).

Jadi pernyataan "graph features merusak" TIDAK tepat. Yang tepat: **fitur graph
struktural (Level 1-2) memberi lift substansial, sementara fitur berbasis label
(Level 3) merusak dan menyeret Level 1-2 turun bersamanya.**

**Keterbatasan yang harus dinyatakan.** Angka B2+L12 di atas adalah VALIDATION.
Test set sudah dibuka untuk empat model terkunci dan tidak boleh dibuka lagi untuk
kombinasi baru - itu akan mengubah test menjadi alat seleksi. Karena itu:

- Lift Level 1-2 sebesar +2,30 pp AUC **belum terverifikasi out-of-time**.
- Diagnostik drift (bagian 3b) menunjukkan fitur Level 1-2 sangat stabil kecuali
  `uid_size` (+250%), sehingga lift ini kemungkinan bertahan - tapi itu dugaan,
  bukan hasil terukur.
- Verifikasi yang benar memerlukan test set periode baru, atau dinyatakan sebagai
  hipotesis untuk pekerjaan lanjutan.

Ini konsekuensi sah dari disiplin membuka test sekali. Mencatatnya sebagai
"belum terukur" lebih jujur daripada membuka test lagi demi angka yang lebih baik.

#### Bukti pendukung dari SHAP

SHAP pada M4 (validation, 50K subsample) menunjukkan model mengandalkan graph
secara ekstrem:

| Kelompok | n fitur | total mean\|SHAP\| | % dari total |
|---|---|---|---|
| `graph_*` | 23 | 0,10509 | **46,2%** |
| `uid_*` | 9 | 0,04882 | **21,5%** |
| V (Vesta) | 339 | 0,04230 | 18,6% |
| C (Vesta counting) | 14 | 0,02846 | 12,5% |
| D (timedelta) | 15 | 0,00293 | 1,3% |

Empat peringkat teratas seluruhnya fitur Level 3:

| Rank | Fitur | mean\|SHAP\| |
|---|---|---|
| 1 | `graph_nb_fraud_rate_1hop` | 0,03606 |
| 2 | `graph_nb_fraud_rate_2hop` | 0,03553 |
| 3 | `uid_fraud_rate` | 0,03441 |
| 4 | `graph_community_fraud_rate` | 0,02870 |
| 5 | V244 (fitur non-graph pertama) | 0,01467 |

Fitur graph/uid di top 20: **7 dari 20**.

**Keputusan 2-hop: DIPERTAHANKAN.** Kriteria gugur ditetapkan sebelum angka
dilihat (mean|SHAP| < 20% dari 1-hop, atau rank > 50). Hasilnya rank 2 dengan
98,5% dari 1-hop - jauh melewati ambang. Menggugurkan 2-hop bukan solusi atas
masalah ini, karena 1-hop pun sama bermasalahnya.

**Inti dekomposisinya:** jarak SHAP antara fitur Level 3 (0,029-0,036) dan fitur
non-graph terbaik (V244, 0,015) adalah 2-2,5 kali. Model tidak sekadar memakai
fitur Level 3 - ia menggantungkan sebagian besar keputusannya pada empat fitur
tersebut, lalu performanya runtuh di periode berikutnya.

### 3. Root cause analysis

Ada DUA mekanisme berbeda, keduanya lolos dari test anti-leakage karena keduanya
bukan leakage label.

#### 3a. Coverage collapse - bukti melemah, model tidak tahu

`uid_labeled_count` = jumlah tetangga UID berlabel setelah leave-one-out:

| Split | mean | median | % bernilai nol |
|---|---|---|---|
| train | 6,00 | 2 | 33,2% |
| val | 3,75 | 0 | 57,7% |
| test | **2,87** | **0** | **67,3%** |

Turun 52% dari train ke test. Konsekuensinya berlapis:

- Di training, model belajar bahwa `uid_fraud_rate` layak dipercaya karena
  rata-rata didukung 6 tetangga berlabel.
- Di test, dua pertiga baris tidak punya tetangga berlabel sama sekali, sehingga
  `uid_fraud_rate` = prior konstan 0,03512 - **tidak membawa informasi apa pun**.
- Model tetap memberi bobot besar pada fitur itu, karena selama training fitur
  tersebut memang sangat prediktif.

Yang penting: distribusi NILAI rate-nya stabil (drift hanya -2,8%). Yang runtuh
adalah kekuatan buktinya, bukan nilainya. Inilah sebabnya pengecekan drift biasa
(membandingkan mean/PSI fitur) TIDAK akan menangkap masalah ini.

Bukti bahwa fiturnya sendiri tidak bocor: pada baris yang benar-benar punya
tetangga UID berlabel, AUC test 0,9751 versus train 0,9725 (T6) - sedikit lebih
BAIK di test. Fitur ini valid; yang gagal adalah ketersediaannya.

#### 3b. `uid_size` dihitung atas seluruh periode

Ini kesalahan desain yang saya temukan saat investigasi, bukan saat implementasi.

`uid_size` (Level 1-2, dianggap "struktural dan aman") diagregasi atas SELURUH
data. Akibatnya transaksi test mewarisi ukuran klien yang sebagian besar terbentuk
SETELAH periode training:

| Perhitungan | uid_size rata-rata di test |
|---|---|
| Seluruh periode (yang dipakai) | **29,79** |
| Hanya periode training | 2,87 |

Drift +250,8% dari train (8,48) ke test (29,79). Contoh paling ekstrem: UID 158745
punya 1.414 transaksi, **1.393 di antaranya di test dan 0 di train** - entitas yang
praktis tidak eksis saat model dilatih, tapi muncul sebagai klien raksasa di test.

Ini bukan leakage label (tidak ada `isFraud` yang tersentuh), dan CLAUDE.md memang
mengizinkan graph dibangun dari seluruh data. Tapi untuk fitur AGREGASI, "struktur
dari seluruh data" ternyata tetap menciptakan pergeseran distribusi yang parah.
Aturan yang benar seharusnya: **statistik agregasi apa pun - termasuk yang tidak
menyentuh label - harus dihitung dari periode training saja.**

Fitur Level 1-2 lain yang murni struktural justru sangat stabil:

| Fitur | train | test | drift |
|---|---|---|---|
| `graph_deg_card1` | 2511,5 | 2572,9 | +2,4% |
| `graph_two_hop_count` | 1707,7 | 1841,1 | +7,8% |
| `graph_component_size` | 421400 | 420077 | -0,3% |
| `graph_pagerank` | - | - | +0,6% |
| **`uid_size`** | **8,48** | **29,79** | **+250,8%** |

Jadi masalahnya bukan "fitur graph" secara umum, melainkan spesifik pada fitur
yang bergantung pada agregasi lintas waktu.

#### 3c. Kenapa model over-relies pada fitur yang justru degradasi

Gradient boosting memilih split yang paling menurunkan loss DI DATA TRAINING.
Fitur Level 3 di periode training punya AUC univariat 0,81-0,90 - jauh mengalahkan
fitur terbaik lainnya. Model rasional memberinya bobot besar.

Early stopping tidak menyelamatkan, karena validation (hari 126-150) hanya
mengalami degradasi separuh jalan: `uid_labeled_count` di val masih 3,75 versus
2,87 di test. Model berhenti pada titik yang optimal untuk kondisi val, lalu
menghadapi kondisi yang lebih buruk lagi di test.

Ini pola yang secara struktural sama dengan overfitting, tapi mekanismenya
berbeda: bukan menghafal noise, melainkan **mengandalkan fitur yang kualitasnya
meluruh seiring waktu**.

### 4. Implikasi metodologis - apa yang terjadi kalau split-nya random

Prediksi eksplisit, dicatat sebagai klaim yang bisa diuji siapa pun:

Dengan random split (atau StratifiedKFold biasa), lift fitur Level 3 akan
**positif dan besar** - perkiraan +3 sampai +8 pp AUC, dengan Level 3 mendominasi
feature importance persis seperti yang terlihat di SHAP. Dengan temporal split,
lift yang sama menjadi negatif.

(Level 1-2 tidak termasuk dalam prediksi ini: kontribusinya positif di validation
temporal, jadi tidak bergantung pada jenis split.)

Alasannya langsung mengikuti root cause di atas:

1. **Coverage tidak akan runtuh.** Dengan pembagian acak, transaksi dari satu UID
   tersebar merata antara train dan test, sehingga `uid_labeled_count` di test
   akan setara dengan di train (~6, bukan 2,87). Fitur Level 3 tetap informatif.
2. **`uid_size` tidak akan bergeser.** Agregasi seluruh periode menjadi tidak
   bermasalah ketika train dan test berasal dari periode yang sama.
3. **Leave-one-out tetap lolos semua test anti-leakage.** Tidak ada label val/test
   yang dipakai. Secara teknis benar, tapi hasilnya tetap menyesatkan.

Inilah bagian yang paling layak ditulis: **implementasi Level 3 di project ini
lolos setiap pengecekan leakage yang biasa dilakukan** - label_mask wajib,
leave-one-out terverifikasi terhadap perhitungan dense, membalik seluruh label
val+test pada 590K baris menghasilkan output identik bit-per-bit. Semua benar.
Yang membuat hasilnya jujur bukan test-test itu, melainkan **temporal split**.

Sebagian besar repo fraud detection yang melaporkan "graph features memberi lift
besar" kemungkinan berada dalam kondisi ini: kodenya benar, validasinya yang salah.

### 5. Rekomendasi produksi

Kalau sistem ini di-deploy, berikut klasifikasi fiturnya.

**Aman dipakai - struktural, drift < 10%:**

| Fitur | Drift | Catatan |
|---|---|---|
| `graph_deg_*` | +2,4% | Degree atribut, stabil |
| `graph_two_hop_count` | +7,8% | Ukuran lingkungan |
| `graph_component_size` | -0,3% | Sangat stabil |
| `graph_pagerank` | +0,6% | Sangat stabil |
| `graph_attr_entropy`, `graph_max_kcore` | - | Murni struktural |

Fitur-fitur ini boleh dipakai tanpa perlakuan khusus, dan diagnostik dekomposisi
menunjukkan mereka MEMBERI lift (+2,30 pp AUC di validation). Limitasi yang harus
disertakan: angka itu belum terverifikasi out-of-time karena test set sudah
dikunci. Rekomendasi konkret untuk deployment: latih ulang dengan feature set
B2 + Level 1-2, lalu validasi pada periode baru sebelum produksi.

**Butuh monitoring ketat - jangan dipakai tanpa pengaman:**

| Fitur | Risiko | Pengaman wajib |
|---|---|---|
| `uid_fraud_rate` | Coverage runtuh 52% | Monitor `uid_labeled_count`, bukan hanya nilai rate |
| `graph_nb_fraud_rate_1hop` / `2hop` | Idem | Idem, plus alert bila % baris di prior naik |
| `graph_community_fraud_rate` | Komunitas berubah antar periode | Re-run Louvain tiap retrain |
| `uid_size` | **Drift +250%** | HARUS dihitung ulang dari jendela training saja |

**Aturan operasional yang dihasilkan analisis ini:**

1. Monitor **kekuatan bukti**, bukan hanya nilai fitur. PSI pada `uid_fraud_rate`
   akan terlihat sehat (drift -2,8%) sementara fitur itu sudah lumpuh. Yang harus
   dipantau adalah `*_labeled_count` dan persentase baris yang jatuh ke prior.
2. Semua agregasi entitas dihitung dari jendela training bergerak, tidak pernah
   dari seluruh data yang tersedia.
3. Retrain lebih sering daripada model tabular biasa, karena fitur berbasis entitas
   meluruh lebih cepat.

### 6. Framing untuk artikel

**Ini bukan kegagalan graph features. Ini demonstrasi bahwa temporal validation
memisahkan fitur graph yang benar-benar berguna dari yang hanya tampak berguna -
pemisahan yang tidak bisa dilakukan oleh pengecekan leakage mana pun.**

Alur artikel yang paling kuat:

1. **Setup**: pertanyaan yang wajar - berapa nilai tambah graph features di atas
   baseline tabular yang kuat?
2. **Implementasi yang benar**: bipartite sparse tanpa materialisasi adjacency
   89,5 miliar nnz, leave-one-out terverifikasi terhadap perhitungan dense,
   label_mask wajib, membalik seluruh label val+test menghasilkan output identik.
3. **Twist**: SHAP menunjukkan model mencintai fitur berbasis label - 68% total
   kontribusi, empat peringkat teratas. Semua tanda menunjukkan keberhasilan besar.
4. **Kenyataan**: di test out-of-time, AUC turun 1,91 pp (dengan C) dan 8,23 pp
   (tanpa C). Model terbaik dari empat yang diuji adalah yang tidak memakai graph.
5. **Pembalikan kedua**: dekomposisi menunjukkan fitur graph STRUKTURAL sebenarnya
   memberi +2,30 pp AUC. Yang merusak hanya fitur berbasis label - dan ia menyeret
   yang baik turun bersamanya.
6. **Diagnosis**: dua mekanisme - coverage collapse (`uid_labeled_count` -52%) dan
   agregasi lintas periode (`uid_size` +250%). Keduanya bukan leakage label, dan
   keduanya tak terlihat oleh drift check standar yang hanya memantau nilai fitur.
7. **Pelajaran**: dengan random split, Level 3 akan tampak sebagai kemenangan
   +3 sampai +8 pp dan kemungkinan besar dipilih sebagai fitur andalan. Kode yang
   sama, kesimpulan yang berlawanan.

Judul yang diusulkan: *"Fitur graph terbaik saya ternyata yang paling
membosankan"* - degree dan component size menang, neighbor fraud rate kalah.

Nilai portofolio dari hasil ini lebih tinggi daripada lift positif sederhana:
hampir semua orang bisa menghasilkan angka bagus dengan random split. Yang
membedakan adalah kemampuan mendeteksi kapan angka bagus itu palsu, memisahkan
komponen yang benar-benar bekerja, dan bersedia melaporkan bahwa kombinasi
terbaik belum sempat diverifikasi karena disiplin test-sekali-pakai.

### 7. Keputusan lanjutan

**Model utama untuk Fase 5 adalah M3 (`lgbm_baseline_full`)** - baseline tanpa
graph, AUC test 0,9007 / KS 0,6471. Ini satu-satunya model dengan performa
out-of-time terukur yang terbaik, dan Fase 5 (scorecard, PSI, kalibrasi,
cost-based threshold) dibangun di atasnya.

Analisis graph disimpan sebagai temuan riset. Dua hal yang dibawa ke laporan akhir:

1. Fitur graph struktural (Level 1-2) menunjukkan lift +2,30 pp AUC di validation,
   **belum terverifikasi out-of-time** - kandidat terkuat untuk pekerjaan lanjutan.
2. Fitur graph berbasis label (Level 3) terbukti merusak generalisasi temporal
   meski implementasinya lolos setiap pengecekan leakage.

Tidak ada model atau fitur yang diubah setelah test set dibuka. Angka test di atas
final. Diagnostik dekomposisi di bagian 2 dijalankan pada validation saja, setelah
test dibuka, dan tidak dipakai untuk mengubah model mana pun.

---

## Fase 5 - Evaluasi Risk-Style & Framing Bisnis

Model: M3 (`lgbm_baseline_full`, backend XGBoost, spw=1,0). Dilatih ulang dengan
konfigurasi identik dan **mereproduksi AUC test 0,9007 persis** - konfirmasi bahwa
pipeline deterministik dan angka Fase 4 dapat direplikasi.

### D15. Scorecard 300-850, PDO 20

Anchor: skor 600 pada odds non-fraud 1:50. Faktor skala = 20/ln(2) = 28,85 poin
per satuan log-odds. Diverifikasi lewat test: odds berlipat dua menghasilkan
selisih tepat 20,00 poin.

Pemisahan pada test out-of-time: **skor rata-rata fraud 514 versus non-fraud 626**
- selisih 112 poin, setara lebih dari 5 kali PDO.

Tabel band (test, 89.326 transaksi):

| Band | Rentang skor | Populasi | Fraud rate | Cum. recall |
|---|---|---|---|---|
| 0 (terburuk) | 300-569 | 10% | **24,19%** | **69,40%** |
| 1 | 569-593 | 10% | 4,30% | 81,73% |
| 2 | 593-607 | 10% | 2,17% | 87,96% |
| 3 | 607-618 | 10% | 1,24% | 91,52% |
| 4 | 618-628 | 10% | 1,05% | 94,54% |
| 5-8 | 628-672 | 40% | 0,74% -> 0,17% | 99,49% |
| 9 (terbaik) | 672-762 | 10% | 0,18% | 100% |

Band terburuk (10% populasi) menampung 69,4% seluruh fraud dengan fraud rate 24,2%
- tujuh kali fraud rate keseluruhan. Band terbaik hanya 0,18%, yaitu 134 kali lebih
aman. Ini bentuk yang langsung bisa dipakai tim risk untuk menetapkan kebijakan
per-band.

### D16. Cost-based threshold - alat, bukan satu angka

Asumsi yang dinyatakan:
- **Biaya false negative** = `TransactionAmt` transaksi itu sendiri (kerugian
  aktual per transaksi, bukan rata-rata).
- **Biaya false positive** = konstanta per transaksi (review manual + friksi
  pelanggan). Ini ASUMSI, bukan angka terukur.

Biaya tanpa model sama sekali (seluruh fraud lolos): **$477.356** pada periode test
31 hari.

| Biaya FP | Threshold optimal | Review rate | Recall | Total biaya | Penghematan | Penghematan/bulan | Reduksi biaya |
|---|---|---|---|---|---|---|---|
| $2 | 0,0121 | **35,7%** | 90,2% | $105.054 | $372.302 | $360.292 | **78,0%** |
| $5 | 0,0270 | 18,6% | 81,0% | $166.549 | $310.806 | $300.780 | 65,1% |
| $10 | 0,0447 | 12,1% | 73,7% | $213.513 | $263.843 | $255.332 | 55,3% |
| $25 | 0,0822 | **7,0%** | 63,1% | $299.375 | $177.981 | $172.239 | 37,3% |

**Ini deliverable yang sebenarnya.** Threshold optimal bergeser lima kali lipat
(35,7% -> 7,0%) hanya karena asumsi biaya FP berubah dari $2 ke $25. Menyodorkan
satu angka "threshold optimal 0,027" akan menyembunyikan bahwa angka itu sepenuhnya
bergantung pada asumsi yang tidak pernah benar-benar diketahui.

Yang bisa dikatakan dengan yakin ke stakeholder: **model ini menghemat 37-78% biaya
fraud tergantung berapa mahal review manual di organisasi Anda.** Bahkan pada
asumsi paling pesimis ($25 per review), penghematannya $172 ribu per bulan.

Yang TIDAK dimodelkan, dan harus disebut: biaya reputasi jangka panjang, churn
pelanggan yang salah ditolak, dan biaya tetap operasional tim review.

### D17. PSI skor sangat stabil, tapi CSI menemukan masalah lain

**PSI skor train -> test = 0,0032** - jauh di bawah ambang 0,10, kategori "stabil".
Distribusi skor model praktis tidak bergeser antar periode.

Tapi CSI per fitur menemukan sesuatu yang PSI skor tidak tunjukkan:

| Fitur | CSI | Verdict | % missing train | % missing test |
|---|---|---|---|---|
| `le_M8` | 0,3360 | bermasalah | 67,1% | **38,6%** |
| `le_M9` | 0,3360 | bermasalah | 67,1% | 38,6% |
| `le_M7` | 0,3360 | bermasalah | 67,1% | 38,6% |
| `le_M3` | 0,2833 | bermasalah | 54,0% | 28,1% |
| sisanya | < 0,023 | stabil | - | - |

Nilai CSI M7/M8/M9 identik persis karena ketiganya berbagi pola missing yang sama
(M8 dan M9 identik; M7 hampir identik). **Pergeserannya adalah pergeseran
MISSINGNESS, bukan pergeseran nilai** - Vesta mulai mengisi kolom M di paruh kedua
periode, dari 67% kosong menjadi 39% kosong.

Implikasi: keempat fitur ini berubah makna antar periode. Nilai "-1" (missing)
yang di training berarti "mayoritas kasus" berubah menjadi "minoritas kasus" di
test. Model tetap berjalan baik (PSI skor 0,0032) karena bobotnya tersebar ke 447
fitur, tapi dalam produksi keempat fitur ini layak dipantau atau di-drop.

**Pelajaran metodologis:** PSI skor yang sehat tidak menjamin fitur-fiturnya sehat.
Keduanya harus dipantau. Ini melengkapi temuan Fase 4 bahwa PSI juga tidak
menangkap runtuhnya kekuatan bukti pada fitur berbasis entitas.

### D18. Model sudah terkalibrasi; isotonic justru memperburuk

Hasil pada test out-of-time:

| | ECE | Brier | Mean predicted | Actual rate | Bias |
|---|---|---|---|---|---|
| **Sebelum (raw)** | **0,00348** | **0,02268** | 0,03450 | 0,03486 | **-0,00037** |
| Isotonic | 0,00363 | 0,02289 | 0,03839 | 0,03486 | +0,00353 |

AUC 0,9007 -> 0,9000 (isotonic monoton, ranking praktis tak berubah).

**Isotonic memperburuk ECE.** Penyebabnya bukan bug melainkan drift:

- Model spw=1 sudah nyaris terkalibrasi sempurna di test - bias hanya -0,00037,
  yaitu 1% dari fraud rate.
- Isotonic di-fit pada **validation** yang fraud rate-nya 0,03426, lalu diterapkan
  ke **test** yang fraud rate-nya 0,03486.
- Kalibrator mewarisi level validation, menaikkan mean prediksi ke 0,03839 dan
  membuat bias menjadi +0,00353 - sepuluh kali lebih buruk.

Jadi kalibrator MENGIMPOR drift val->test. Ini konsekuensi langsung dari drift
fraud rate yang terdokumentasi di T1 (rentang mingguan 2,07%-5,07%).

**Keputusan: tidak memakai kalibrasi isotonic pada model final.** Model raw sudah
lebih baik. Kalibrasi baru berguna kalau model memang tidak terkalibrasi - dan
`scale_pos_weight=1,0` yang dipilih di Fase 2 justru dipilih karena alasan itu.
Keputusan D7 terbukti benar dua fase kemudian.

### D19. Peninjauan `scale_pos_weight` - spw=5 + isotonic menang di validation

Dievaluasi pada paruh kedua validation; isotonic di-fit pada paruh pertama supaya
tidak menilai dirinya sendiri.

| spw | Kalibrasi | AUC | KS | PR-AUC | ECE | Brier | Bias |
|---|---|---|---|---|---|---|---|
| 1,0 | raw | 0,8943 | 0,6383 | 0,4561 | 0,0052 | 0,0232 | -0,0026 |
| 1,0 | isotonic | 0,8932 | 0,6300 | 0,4312 | 0,0057 | 0,0236 | +0,0022 |
| 5,0 | raw | **0,9077** | **0,6651** | **0,4868** | 0,0371 | 0,0262 | +0,0371 |
| **5,0** | **isotonic** | **0,9071** | **0,6635** | 0,4629 | **0,0058** | **0,0227** | +0,0020 |

**Jawaban atas pertanyaan Fase 2: ya, isotonic memulihkan kalibrasi spw=5 hampir
sepenuhnya.** ECE turun dari 0,0371 ke 0,0058 (setara spw=1), Brier bahkan menjadi
yang terbaik (0,0227), sementara AUC hanya turun 0,0006. Hasilnya: AUC +1,28 pp dan
KS +2,52 pp di atas spw=1, dengan kalibrasi yang setara.

**Tetapi spw=5 TIDAK dipakai sebagai model final.** Alasannya metodologis, bukan
teknis:

1. Angka di atas adalah **validation**. Memilih spw=5 berdasarkan validation lalu
   mengevaluasinya di test akan menjadi pembukaan test set kedua untuk model yang
   dipilih berdasarkan hasil - persis yang dilarang.
2. Satu-satunya model dengan angka out-of-time yang sah adalah spw=1 (M3).
3. Ironi yang perlu dicatat: keunggulan spw=5 justru pada kalibrasi setelah
   isotonic - padahal D18 menunjukkan isotonic mengimpor drift val->test. Belum
   tentu keunggulan itu bertahan di OOT.

**Status: temuan terverifikasi di validation, belum terukur out-of-time.** Sama
seperti fitur graph Level 1-2 (+2,30 pp AUC). Kedua kandidat ini adalah rekomendasi
utama untuk pekerjaan lanjutan dengan periode data baru.

### D20. Ringkasan model final

| Aspek | Nilai |
|---|---|
| Model | M3 - XGBoost, 447 fitur, tanpa graph, spw=1,0 |
| AUC test OOT | 0,9007 |
| KS test OOT | 0,6471 |
| PR-AUC test OOT | 0,5196 |
| recall@1% / 5% / 10% | 25,3% / 56,3% / 69,4% |
| PSI skor train->test | 0,0032 (stabil) |
| ECE | 0,00348 (terkalibrasi, tanpa post-processing) |
| Skor fraud vs non-fraud | 514 vs 626 |
| Penghematan biaya | 37-78% tergantung asumsi biaya FP |

Target MASTER_PLAN Fase 2 (AUC 0,90-0,93, KS 0,60-0,70) tercapai pada test
out-of-time, bukan hanya validation.
