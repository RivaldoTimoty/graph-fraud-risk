# Master Plan — Graph-Based Fraud Risk Model (IEEE-CIS)

Project portofolio untuk memperkuat klaim *"network risk model leveraging graph-based features"* di CV.
Dirancang untuk dikerjakan dengan Claude Code, dalam 6 fase / ~4 minggu part-time.

---

## 1. Tujuan & Positioning

### Pertanyaan riset yang dijawab project ini

> Seberapa besar tambahan predictive power yang diberikan **graph/network features** di atas model tabular gradient boosting yang sudah kuat, pada data transaksi fraud dunia nyata — dan apakah tambahan itu bertahan pada validasi out-of-time?

Ini pertanyaan yang tepat karena:
- Banyak orang punya project "XGBoost di IEEE-CIS". Hampir tidak ada yang punya **ablation study graph vs non-graph yang jujur**.
- Framing "berapa lift-nya" adalah bahasa yang dipakai risk manager dan C-level, bukan bahasa Kaggle.
- Kalau ternyata lift-nya kecil, itu tetap hasil yang menarik dan menunjukkan kematangan. Jangan paksakan cerita sukses.

### Yang dibuktikan ke recruiter/hiring manager

| Klaim di CV | Dibuktikan lewat |
|---|---|
| Network risk model dengan graph-based features | Fase 3–4 |
| Feature store engineering | Fase 3 (feature registry + versioning) |
| Credit/risk scorecard discipline | Fase 5 (KS, gain chart, score binning, PSI) |
| Presentasi ke C-level | Fase 6 (executive summary + dashboard) |
| Produksi, bukan notebook-only | Struktur repo, config-driven, reproducible |

### Non-goals (penting, supaya scope tidak meledak)

- Bukan mengejar leaderboard Kaggle. Target bukan AUC 0.96.
- Bukan riset arsitektur GNN baru. GraphSAGE/GAT standar sudah cukup.
- Bukan real-time serving infrastructure. Batch scoring cukup.

---

## 2. Deliverable Akhir

1. **GitHub repo** — kode reproducible, README kuat, config-driven, ada test.
2. **Executive one-pager** — 1 halaman: masalah, pendekatan, hasil, rekomendasi bisnis. Format yang biasa dilihat C-level.
3. **Dashboard Streamlit** — visualisasi fraud ring, perbandingan model, score distribution.
4. **Technical write-up** — artikel Medium/blog, 1500–2500 kata.
5. **Model card** — dokumen singkat: data, asumsi, limitasi, kapan model tidak boleh dipakai.

---

## 3. Tech Stack

```
Data          : pandas, pyarrow, duckdb
Graph build   : scipy.sparse (utama), networkx (visualisasi subgraph saja), igraph (opsional, community detection cepat)
GNN           : PyTorch Geometric (torch-geometric)
Model tabular : LightGBM (utama), XGBoost (pembanding)
Explainability: SHAP
Tracking      : MLflow (lokal, file-based) atau CSV sederhana kalau mau ringan
Dashboard     : Streamlit + Plotly
Config        : hydra atau pydantic-settings + YAML
Testing       : pytest
Lint          : ruff + black
```

**Catatan penting soal library graph:** jangan pakai NetworkX untuk menghitung fitur di graph 590K node — akan sangat lambat dan boros memori. Pakai `scipy.sparse` untuk operasi matriks (degree, neighbor aggregation, PageRank via power iteration) dan `igraph` untuk community detection. NetworkX hanya untuk menggambar subgraph kecil di dashboard.

---

## 4. Struktur Repo

```
graph-fraud-risk/
├── CLAUDE.md                      # instruksi untuk Claude Code (lihat Bagian 5)
├── README.md
├── pyproject.toml
├── Makefile                       # make data / make features / make train / make report
├── configs/
│   ├── data.yaml                  # path, dtype, sampling
│   ├── graph.yaml                 # definisi node/edge, fitur graph mana yang dihitung
│   ├── features.yaml              # feature registry
│   └── model/
│       ├── lgbm_baseline.yaml
│       ├── lgbm_graph.yaml
│       └── graphsage.yaml
├── data/
│   ├── raw/                       # .gitignore — hasil download Kaggle
│   ├── interim/                   # parquet hasil join + dtype reduction
│   ├── graph/                     # sparse matrices, edge lists, node mapping
│   └── features/                  # feature matrices per versi
├── src/
│   ├── data/
│   │   ├── load.py                # load + join transaction & identity
│   │   ├── reduce_memory.py       # downcast dtype
│   │   └── split.py               # temporal split (KRITIS)
│   ├── graph/
│   │   ├── build.py               # konstruksi bipartite graph
│   │   ├── entity_resolution.py   # UID / client identification
│   │   └── features.py            # degree, pagerank, community, neighbor stats
│   ├── features/
│   │   ├── tabular.py             # aggregation, frequency encoding, time features
│   │   ├── woe.py                 # WoE / IV binning
│   │   └── registry.py            # feature store sederhana
│   ├── models/
│   │   ├── gbdt.py
│   │   ├── gnn.py
│   │   └── calibration.py         # isotonic / Platt scaling
│   ├── evaluation/
│   │   ├── metrics.py             # AUC, KS, PR-AUC, lift, gain
│   │   ├── scorecard.py           # scaling ke skala 300-850
│   │   ├── stability.py           # PSI, CSI
│   │   └── business.py            # cost-based threshold optimization
│   └── viz/
├── notebooks/                     # HANYA untuk eksplorasi, bukan sumber kebenaran
│   ├── 01_eda.ipynb
│   └── 02_graph_exploration.ipynb
├── dashboard/
│   └── app.py
├── tests/
├── reports/
│   ├── executive_summary.md
│   ├── model_card.md
│   └── figures/
└── artifacts/                     # model tersimpan, metrics.json
```

---

## 5. CLAUDE.md (copy-paste ke root repo)

Ini file yang membuat Claude Code konsisten sepanjang project. Isi persis seperti ini:

````markdown
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
````

---

## 6. Rencana Per Fase

### Fase 0 — Setup (½ hari)

**Tujuan:** repo jalan, data terunduh, `make test` hijau.

Langkah:
1. Inisialisasi repo, `pyproject.toml`, ruff + black, pre-commit.
2. Buat `CLAUDE.md` dari Bagian 5.
3. Download data. Perlu join kompetisi IEEE-CIS dulu dari akun Kaggle:
   `kaggle competitions download -c ieee-fraud-detection`
4. Buat `src/data/load.py` + `reduce_memory.py`, simpan ke parquet.
5. Satu test sanity: row count 590540, fraud rate ≈ 3.5%.

**Prompt awal untuk Claude Code:**
> Baca CLAUDE.md. Buatkan scaffolding repo sesuai struktur di README, lalu implementasi `src/data/load.py` yang join transaction dan identity via TransactionID, dan `src/data/reduce_memory.py` yang downcast dtype numerik. Simpan hasil ke `data/interim/merged.parquet`. Tambahkan pytest yang memverifikasi shape dan fraud rate. Jelaskan rencanamu dulu sebelum menulis kode.

**Definition of done:** `make data` menghasilkan parquet < 500MB, `make test` hijau.

---

### Fase 1 — EDA & Temporal Split (1 hari)

**Tujuan:** paham struktur waktu data, dan mengunci strategi validasi sebelum menyentuh model.

Ini fase yang paling sering dilewati orang dan paling sering jadi sumber hasil palsu.

Yang harus dipahami:
- `TransactionDT` adalah detik relatif dari suatu titik acuan, bukan timestamp absolut. Konversi ke hari/jam/hari-dalam-minggu.
- Rentang data ~6 bulan untuk train. Train dan test asli Kaggle tidak overlap secara waktu, dengan gap ~1 bulan di antaranya.
- Karena label test set Kaggle tidak tersedia, kamu akan **memecah train set sendiri** secara temporal.

Rekomendasi split:
```
Periode 0-70%   → training
Periode 70-85%  → validation (untuk early stopping & tuning)
Periode 85-100% → out-of-time test (SEKALI PAKAI, di akhir saja)
```
Tambahkan gap 1-2 minggu antara train dan validation untuk meniru delay label di dunia nyata.

Yang di-EDA:
- Distribusi fraud rate per minggu — apakah stabil? Kalau tidak, itu tantangan drift yang perlu dibahas di laporan.
- Missing value pattern. Banyak kolom Vxxx punya pola missing yang identik — itu petunjuk mereka berasal dari satu sumber.
- Kardinalitas `card1`, `addr1`, `P_emaildomain`, `DeviceInfo` — ini calon node graph.

**Prompt:**
> Buat `src/data/split.py` dengan fungsi `temporal_split(df, train_frac, val_frac, gap_days)`. Lalu buat notebook `01_eda.ipynb` yang menganalisis: (1) distribusi TransactionDT dan fraud rate per minggu, (2) pola missing per grup kolom, (3) kardinalitas kandidat node graph: card1-card6, addr1, addr2, P_emaildomain, R_emaildomain, DeviceInfo, id_30, id_31. Simpan figur ke reports/figures/.

**Definition of done:** kamu bisa menjawab "berapa hari data ini?" dan "apakah fraud rate stabil sepanjang waktu?" tanpa membuka kode.

---

### Fase 2 — Baseline Tabular (1–2 hari)

**Tujuan:** baseline yang kuat dan jujur. Kalau baselinenya lemah, lift graph jadi tidak bermakna.

Fitur baseline:
- Kolom asli numerik apa adanya (LightGBM tahan missing value).
- Label encoding untuk kategorikal.
- Frequency encoding untuk kolom kardinalitas tinggi (`card1`, `addr1`, `P_emaildomain`).
- Fitur waktu: jam transaksi, hari dalam minggu.
- Aggregation: `TransactionAmt` dibagi mean/std per `card1`, per `card4`.
- Log transform `TransactionAmt` — distribusinya sangat skew, setelah log jadi mendekati normal.

Yang **belum** dimasukkan di fase ini: apa pun yang berbau graph, termasuk fitur C1–C14 (fitur counting Vesta sebenarnya sudah semi-graph — pertimbangkan membuat dua varian baseline, dengan dan tanpa C-features, dan bahas ini di laporan sebagai nuance yang menarik).

Handling imbalance: mulai dengan `scale_pos_weight` di LightGBM, bukan SMOTE. SMOTE pada data tabular berdimensi tinggi sering merusak dan tidak masuk akal untuk fraud (fraud sintetis bukan fraud). Kalau mau bahas SMOTE, taruh sebagai eksperimen yang gagal di laporan — itu justru menunjukkan judgment.

**Prompt:**
> Implementasi `src/features/tabular.py` dan `src/models/gbdt.py`. Fitur sesuai daftar di master plan Fase 2. Training LightGBM dengan early stopping pada validation set temporal. Catat AUC, KS, PR-AUC ke `artifacts/experiments.csv`. Pastikan frequency encoding dihitung hanya dari periode training.

**Target realistis:** AUC validation 0.90–0.93, KS 0.60–0.70.

**Definition of done:** ada satu angka baseline yang kamu percaya, tercatat rapi.

---

### Fase 3 — Graph Construction & Graph Features (3–5 hari) ⭐

Ini inti project. Kerjakan pelan-pelan.

#### 3a. Entity resolution

Sebelum membangun graph, kamu perlu memutuskan **apa yang jadi node**. Ini keputusan desain terpenting.

Dua pendekatan, kerjakan keduanya:

**Pendekatan A — Bipartite graph (lebih aman, mulai dari sini):**
```
Node tipe 1: transaksi (590K)
Node tipe 2: atribut — nilai unik dari card1, addr1, P_emaildomain, DeviceInfo, dst
Edge: transaksi —— atribut yang dimilikinya
```
Ini tidak butuh asumsi apa pun. Dua transaksi "terhubung" kalau berbagi atribut, lewat jalur 2-hop.

**Pendekatan B — Client/UID graph (lebih powerful, lebih berisiko):**
Komunitas Kaggle menemukan bahwa mengidentifikasi klien individual sangat meningkatkan performa. Pendekatan yang dikenal: kombinasi `card1`, `addr1`, dan `D1` yang dinormalisasi terhadap waktu (`D1n = hari_transaksi - D1`) menghasilkan identifier klien yang cukup stabil.

Kenapa berisiko: UID adalah bentuk agregasi yang bisa membocorkan informasi lintas waktu kalau tidak hati-hati. Kalau satu klien punya transaksi di train dan test, statistik klien dari train akan "membocorkan" ke test. Ini sah kalau meniru produksi (kamu memang tahu histori klien), tapi harus dinyatakan eksplisit di laporan.

Bahas trade-off ini di write-up — ini jenis diskusi yang membedakan kandidat senior.

#### 3b. Fitur graph yang dihitung

Kelompokkan dalam tiga tingkat, dari murah ke mahal:

**Level 1 — Struktural lokal (murah, hitung dulu):**
| Fitur | Arti bisnis |
|---|---|
| `degree_card`, `degree_email`, `degree_device` | Berapa transaksi berbagi kartu/email/device ini |
| `n_unique_cards_per_device` | Satu device banyak kartu → sinyal fraud farm |
| `n_unique_devices_per_card` | Satu kartu banyak device → kartu dicuri |
| `n_unique_emails_per_card` | Rasio penyalahgunaan identitas |
| `attr_entropy` | Keberagaman atribut dalam satu cluster |

**Level 2 — Struktural global (sedang):**
| Fitur | Arti bisnis |
|---|---|
| `pagerank` | Sentralitas node dalam jaringan transaksi |
| `component_size` | Ukuran connected component — komponen besar sering fraud ring |
| `community_id` + `community_size` | Louvain/Leiden clustering |
| `clustering_coefficient` (2-hop) | Seberapa rapat lingkungan node |
| `k_core_number` | Kedalaman node dalam struktur padat |

**Level 3 — Berbasis label (paling powerful, PALING BERBAHAYA):**
| Fitur | Aturan wajib |
|---|---|
| `neighbor_fraud_rate_1hop` | Hitung HANYA dari label periode training |
| `neighbor_fraud_rate_2hop` | Sama, plus smoothing untuk node degree rendah |
| `community_fraud_rate` | Sama |
| `days_since_neighbor_fraud` | Hanya dari fraud yang sudah "diketahui" pada saat itu |

**Aturan emas untuk Level 3:** implementasikan sebagai fungsi yang menerima `label_mask` — boolean array menandai baris mana yang labelnya boleh dipakai. Untuk baris training sendiri, gunakan leave-one-out atau out-of-fold agar tidak membocorkan label sendiri. Ini satu-satunya bagian project yang benar-benar mudah salah, dan reviewer teknis akan langsung mengeceknya.

#### 3c. Implementasi teknis

Bangun matriks insiden sparse:
```python
# Konseptual
# A: matriks sparse (n_transaksi x n_nilai_atribut), A[i,j]=1 jika transaksi i punya atribut j
# Adjacency transaksi-ke-transaksi lewat atribut = A @ A.T
# JANGAN materialize A @ A.T secara penuh — 590K x 590K akan meledak.
# Sebagai gantinya, hitung fitur langsung:
#   degree per atribut      = A.sum(axis=0)
#   neighbor fraud count    = A @ (A.T @ y_masked)
```
Trik ini membuat semua fitur 1-hop dan 2-hop bisa dihitung tanpa pernah membentuk matriks padat. Sampaikan ke Claude Code secara eksplisit.

**Prompt:**
> Implementasi `src/graph/build.py` yang membangun matriks insiden sparse transaksi-atribut untuk daftar kolom di `configs/graph.yaml`. Lalu `src/graph/features.py` dengan fitur Level 1 dan Level 2 dari master plan. Gunakan scipy.sparse; jangan pernah materialize adjacency transaksi-ke-transaksi secara penuh — hitung neighbor aggregation sebagai `A @ (A.T @ v)`. Tambahkan test yang memverifikasi degree dihitung benar pada graph mainan berukuran 10 node.

Setelah Level 1–2 selesai dan tertest, baru kerjakan Level 3 dengan prompt terpisah yang menekankan `label_mask`.

**Definition of done:** feature matrix graph tersimpan, ada test unit untuk minimal 3 fungsi fitur, dan kamu bisa menjelaskan setiap fitur dalam satu kalimat bisnis.

---

### Fase 4 — Model dengan Graph Features & GNN (2–3 hari)

#### 4a. GBDT + graph features

Latih ulang LightGBM dengan feature set = baseline + graph features. Bandingkan dengan baseline pada test set out-of-time yang sama.

Ini **model utama** project kamu. GNN adalah bonus.

#### 4b. GraphSAGE (opsional tapi bernilai tinggi)

Pakai PyTorch Geometric. Setup:
- Node = transaksi, fitur node = fitur tabular yang sudah dinormalisasi.
- Edge = transaksi berbagi atribut (sample edge kalau terlalu banyak — batasi degree maksimum, misal 50 tetangga per node, dengan sampling).
- Task = node classification, binary.
- Loss = weighted BCE atau focal loss.
- **Splitting temporal tetap berlaku:** node validation/test tidak boleh dipakai saat menghitung loss training, meskipun mereka ada di graph (ini transductive setting — nyatakan di laporan).

Kalau GNN sulit atau lambat, tidak masalah — gugurkan dan tulis di laporan kenapa GBDT + graph features lebih praktis untuk kasus ini. Itu kesimpulan yang realistis dan sering benar di industri.

**Prompt:**
> Latih ulang LightGBM dengan feature set baseline + graph, config `configs/model/lgbm_graph.yaml`. Bandingkan head-to-head dengan baseline pada test out-of-time. Hasilkan tabel perbandingan AUC, KS, PR-AUC, dan precision@top1%. Lalu buat SHAP summary plot yang khusus menyoroti kontribusi graph features.

---

### Fase 5 — Evaluasi Risk-Style & Business Framing (2 hari)

Fase ini yang membuat project terasa seperti kerja fintech, bukan latihan Kaggle.

**Yang dibuat:**

1. **Scorecard scaling** — konversi probabilitas ke skala skor (misal 300–850) dengan PDO (points to double the odds). Tunjukkan distribusi skor untuk fraud vs non-fraud.

2. **Gain & lift chart** — kalau kita review 1% transaksi teratas berdasarkan skor, berapa persen fraud yang tertangkap? Ini pertanyaan yang sebenarnya ditanyakan tim risk.

3. **Cost-based threshold** — buat asumsi eksplisit:
   ```
   Biaya false negative = rata-rata kerugian fraud (pakai TransactionAmt)
   Biaya false positive = biaya review manual + friksi pelanggan (asumsikan angka, nyatakan)
   ```
   Cari threshold yang meminimalkan total biaya. Ini mengubah "AUC 0.94" jadi "menghemat estimasi $X per bulan" — bahasa C-level.

4. **PSI (Population Stability Index)** — hitung stabilitas skor antara periode train dan test. Ini metrik monitoring wajib di credit scoring.

5. **Kalibrasi** — plot reliability curve. Model boosting biasanya overconfident; terapkan isotonic regression dan tunjukkan perbaikannya. Penting karena keputusan limit/pricing butuh probabilitas yang bermakna, bukan sekadar ranking.

**Prompt:**
> Implementasi `src/evaluation/` sesuai Fase 5 master plan: metrics.py (AUC, KS, PR-AUC, lift@k), scorecard.py (scaling PDO), stability.py (PSI), business.py (cost-based threshold dengan biaya FN dari TransactionAmt dan biaya FP sebagai parameter config), calibration.py (isotonic + reliability curve). Semua menghasilkan figur ke reports/figures/.

---

### Fase 6 — Packaging & Storytelling (2–3 hari)

Bagian yang paling sering diabaikan, padahal paling menentukan apakah project ini dilihat orang.

#### Dashboard Streamlit

Tiga tab saja, jangan lebih:
1. **Model comparison** — tabel metrik, ROC/PR curve, gain chart.
2. **Fraud network explorer** — pilih satu `card1` atau `DeviceInfo`, tampilkan subgraph-nya dengan pyvis/plotly, node diwarnai berdasarkan status fraud. Ini yang membuat orang berhenti scroll.
3. **Score explorer** — distribusi skor, simulasi threshold dengan slider yang langsung menunjukkan trade-off biaya.

#### README

Struktur yang efektif:
```
1. Satu kalimat: apa ini dan apa temuannya (dengan angka)
2. GIF/screenshot dashboard — sebelum teks apa pun
3. Temuan utama dalam tabel
4. Metodologi singkat + diagram arsitektur
5. Bagian "Bagaimana saya mencegah leakage" ← pembeda utama
6. Limitasi & apa yang akan saya lakukan berikutnya
7. Cara menjalankan
```

Bagian nomor 5 adalah yang membedakan. Sebagian besar repo fraud detection punya leakage dan tidak menyadarinya. Menunjukkan kamu memikirkannya secara eksplisit adalah sinyal kuat.

#### Executive one-pager

Satu halaman, tanpa istilah teknis:
- Masalah bisnis dan ukurannya
- Pendekatan dalam 3 kalimat
- Hasil: berapa fraud tertangkap pada berapa persen review rate
- Estimasi dampak finansial dengan asumsi yang dinyatakan
- Risiko dan rekomendasi implementasi

#### Artikel teknis

Angle yang paling menarik untuk ditulis: **"Berapa nilai sebenarnya dari graph features dalam fraud detection — sebuah ablation yang jujur."** Bukan tutorial. Tutorial IEEE-CIS sudah ratusan. Analisis dengan sudut pandang dan angka adalah yang dibaca dan dibagikan.

---

## 7. Jebakan yang Harus Dihindari

| Jebakan | Kenapa berbahaya | Pencegahan |
|---|---|---|
| Random split | Data punya struktur waktu kuat; hasilnya akan terlalu optimistis | Temporal split, dikunci di Fase 1 |
| Neighbor fraud rate dari semua data | Leakage paling parah dan paling umum | `label_mask` + out-of-fold |
| Tuning di test set | Test set jadi tidak valid | Test set dibuka sekali, di akhir |
| Materialize adjacency 590K² | Kehabisan memori | Sparse ops, `A @ (A.T @ v)` |
| SMOTE pada fraud | Fraud sintetis bukan fraud; sering memperburuk | `scale_pos_weight` |
| Membandingkan model dengan preprocessing berbeda | Perbandingan jadi tidak apple-to-apple | Satu pipeline, beda feature set saja |
| Notebook sebagai deliverable | Terlihat seperti latihan, bukan engineering | Logic di `src/`, notebook hanya eksplorasi |
| Memaksakan cerita "graph menang besar" | Reviewer berpengalaman akan curiga | Laporkan apa adanya, termasuk kalau liftnya kecil |

---

## 8. Timeline

| Minggu | Fase | Output |
|---|---|---|
| 1 | Fase 0–1 | Repo jalan, EDA selesai, split terkunci |
| 2 | Fase 2–3a | Baseline tercatat, graph terbangun |
| 3 | Fase 3b–4 | Graph features lengkap, model perbandingan selesai |
| 4 | Fase 5–6 | Evaluasi bisnis, dashboard, README, artikel |

Kalau waktu terbatas, urutan prioritas: Fase 3 > Fase 5 > Fase 6 > Fase 4b (GNN).
Project tanpa GNN tapi dengan graph features yang bersih dan evaluasi bisnis yang kuat jauh lebih baik daripada GNN yang bocor tanpa framing.

---

## 9. Checklist Definition of Done

Sebelum menyebut project ini selesai:

- [ ] `git clone` + `make data` + `make train` berjalan di mesin bersih
- [ ] Semua eksperimen tercatat di `artifacts/experiments.csv`
- [ ] Ada minimal 10 unit test, semuanya hijau
- [ ] Test set out-of-time hanya dievaluasi sekali
- [ ] Setiap fitur berbasis label bisa ditunjuk barisnya di mana `label_mask` diterapkan
- [ ] README punya screenshot dan angka temuan di paragraf pertama
- [ ] Ada bagian limitasi yang jujur
- [ ] Model card selesai
- [ ] Executive one-pager bisa dipahami orang non-teknis
- [ ] Kamu bisa menjelaskan seluruh project dalam 3 menit tanpa slide

---

## 10. Prompt Pembuka untuk Claude Code

Simpan ini, pakai di sesi pertama:

> Saya membangun project graph-based fraud detection pada dataset IEEE-CIS untuk portofolio Data Scientist credit scoring. Master plan lengkap ada di `MASTER_PLAN.md` dan aturan project ada di `CLAUDE.md` — baca keduanya dulu.
>
> Kita mulai dari Fase 0. Sebelum menulis kode apa pun, ringkas pemahamanmu tentang project ini dalam 5 poin, sebutkan bagian mana yang menurutmu paling berisiko secara metodologis, lalu ajukan rencana implementasi Fase 0. Tunggu konfirmasi saya sebelum mulai.

Pola yang efektif sepanjang project: satu fase satu sesi, selalu minta rencana dulu, dan di akhir setiap fase minta Claude Code menulis ringkasan keputusan yang diambil ke `reports/decisions.md`. File itu nanti jadi bahan mentah artikel kamu.
