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
