# Deteksi Fraud Transaksi — Ringkasan Eksekutif

---

## Masalah

Dari setiap 100 transaksi yang masuk, sekitar 3–4 adalah penipuan. Pada volume
yang dianalisis — 89.000 transaksi dalam satu bulan — kerugian yang lolos tanpa
sistem penyaringan mencapai **$477.000 per bulan**.

Tim risk tidak mungkin meninjau semua transaksi secara manual. Pertanyaannya
bukan "bagaimana menangkap semua penipuan", melainkan **"transaksi mana yang
layak ditinjau lebih dulu, dengan kapasitas tim yang terbatas"**.

---

## Pendekatan

Sistem memberi setiap transaksi sebuah skor risiko, mirip skor kredit: makin
rendah skornya, makin besar kemungkinan penipuan. Tim review lalu bekerja dari
skor terendah ke atas, sesuai kapasitas yang tersedia.

Sistem dibangun dan diuji dengan satu aturan ketat: **model hanya boleh belajar
dari masa lalu, lalu diuji pada periode yang belum pernah dilihatnya.** Ini
mencerminkan kondisi nyata — saat menilai transaksi hari ini, kita tidak tahu apa
yang akan terjadi bulan depan.

---

## Hasil

**Jika tim meninjau 10% transaksi berskor terendah, 69 dari setiap 100 penipuan
akan tertangkap.**

Kelompok 10% terburuk itu punya tingkat penipuan **24%** — tujuh kali lebih tinggi
daripada rata-rata. Sebaliknya, kelompok 10% teraman hanya 0,18%, yaitu **134 kali
lebih aman**. Tim tidak perlu lagi meninjau transaksi secara acak.

| Kapasitas review | Penipuan tertangkap |
|---|---|
| 1% transaksi | 25% |
| 5% transaksi | 56% |
| 10% transaksi | **69%** |
| 19% transaksi | 81% |

---

## Estimasi dampak finansial

Dampaknya bergantung pada satu angka yang hanya bisa ditentukan oleh bisnis:
**berapa biaya meninjau satu transaksi** — mencakup waktu staf dan ketidaknyamanan
pelanggan yang transaksinya tertunda.

| Biaya per review | Kapasitas optimal | Penipuan tertangkap | Penghematan/bulan |
|---|---|---|---|
| $2 | 36% transaksi | 90% | $360.000 |
| **$5** | **19% transaksi** | **81%** | **$300.000** |
| $10 | 12% transaksi | 74% | $255.000 |
| $25 | 7% transaksi | 63% | $172.000 |

**Pada asumsi $5 per review, sistem menghemat sekitar $300.000 per bulan** —
setara 65% dari kerugian yang sebelumnya lolos begitu saja.

Yang perlu diperhatikan: kapasitas review optimal bergeser **lima kali lipat**
(dari 36% menjadi 7%) hanya karena asumsi biaya berubah. Karena itu yang
diserahkan bukan satu rekomendasi tunggal, melainkan alat untuk memilih titik
operasi sesuai kondisi biaya yang sebenarnya.

Bahkan pada asumsi paling konservatif, penghematannya **$172.000 per bulan**.

---

## Risiko

**Sistem tidak boleh menolak transaksi secara otomatis.**
Di kelompok paling berisiko sekalipun, 76 dari 100 transaksi tetap sah. Menolak
otomatis berarti menolak tiga pelanggan jujur untuk setiap satu penipu. Sistem ini
alat penentu prioritas, bukan pengambil keputusan.

**Pola penipuan berubah seiring waktu.**
Dalam enam bulan data, tingkat penipuan bergerak antara 2% dan 5% — selisih dua
setengah kali lipat. Sistem perlu dilatih ulang secara berkala, dan performanya
harus dipantau bulanan.

**Estimasi tidak mencakup biaya reputasi.**
Pelanggan yang berulang kali tertahan bisa berhenti memakai layanan. Angka di atas
hanya menghitung biaya operasional langsung.

**Sistem belum diaudit untuk keadilan antar kelompok pelanggan.**
Data yang tersedia tidak memuat informasi yang diperlukan untuk audit tersebut.
Sebelum digunakan pada keputusan yang berdampak langsung ke pelanggan, audit ini
harus dilakukan.

---

## Rekomendasi

1. **Mulai sebagai alat prioritas, bukan penolakan otomatis.** Arahkan tim review
   ke 10% transaksi paling berisiko lebih dulu, dan ukur berapa penipuan yang
   benar-benar tertangkap selama tiga bulan pertama.

2. **Tetapkan biaya review yang sebenarnya.** Angka itu menentukan kapasitas
   optimal, dan hanya bisa ditentukan dari data operasional internal — bukan dari
   analisis ini.

3. **Siapkan pemantauan bulanan** atas tingkat penipuan aktual dan akurasi
   sistem. Jika keduanya bergeser di luar rentang historis, jadwalkan pelatihan
   ulang.

4. **Sediakan jalur banding** bagi pelanggan yang transaksinya tertahan keliru.
   Ini kebutuhan operasional sekaligus sumber data untuk perbaikan sistem.

5. **Lanjutkan pengembangan yang tertunda.** Dua peningkatan sudah teridentifikasi
   dan menunjukkan hasil menjanjikan pada pengujian awal, tetapi belum
   diverifikasi pada periode baru. Keduanya layak diuji sebelum sistem diperluas.

---

*Seluruh angka berasal dari pengujian pada periode satu bulan yang tidak pernah
digunakan saat membangun sistem. Detail teknis: `README.md` dan
`reports/model_card.md`.*
