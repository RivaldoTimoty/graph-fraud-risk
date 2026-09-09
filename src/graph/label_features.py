"""Fitur graph Level 3: berbasis label. BAGIAN PALING BERISIKO DI PROJECT INI.

Setiap fungsi di modul ini WAJIB menerima `label_mask` - boolean array yang
menandai baris mana yang labelnya boleh dipakai. Tidak ada nilai default, supaya
tidak mungkin lupa memberikannya.

TIGA ATURAN YANG MENGIKAT:

1. `label_mask` selalu = periode training saja, untuk SEMUA baris. Fitur pada
   baris test dihitung dari label training, bukan train+val. Ini pilihan
   konservatif yang disengaja: di produksi, label validation sebenarnya sudah
   diketahui saat scoring test, sehingga fitur test di sini lebih lemah dari yang
   bisa dicapai. Dicatat sebagai trade-off sadar di decisions.md.

2. Leave-one-out: transaksi tidak boleh melihat labelnya sendiri lewat rata-rata
   tetangganya. Tanpa LOO, `neighbor_fraud_rate` untuk baris training akan
   mengandung labelnya sendiri - model lalu "menghafal" dan performa validation
   terlihat jauh lebih baik daripada kenyataan.

3. Node tanpa tetangga berlabel mendapat prior global dari periode training,
   bukan 0 dan bukan NaN. Nilai 0 akan dibaca model sebagai "terbukti tidak
   pernah fraud", padahal artinya "tidak ada informasi".

MATEMATIKA LEAVE-ONE-OUT SECARA SPARSE
--------------------------------------
Agregasi 1-hop memakai pola `A @ (A.T @ v)`, yang menyertakan transaksi itu
sendiri sebagai tetangganya sendiri. Bobot kontribusi diri untuk transaksi i
adalah elemen diagonal (A @ A.T)[i,i] = sum_j A[i,j]^2. Untuk A biner ini sama
dengan jumlah atribut yang dimiliki i, sehingga dapat dihitung tanpa pernah
membentuk A @ A.T:

    self_weight = A.sum(axis=1)

Koreksi LOO menjadi pengurangan langsung:

    loo_positive = A @ (A.T @ y_masked) - self_weight * y_masked
    loo_labeled  = A @ (A.T @ mask)     - self_weight * mask

Untuk baris di dalam mask, kontribusi dirinya hilang tepat. Untuk baris di luar
mask, pengurangannya nol - memang benar, karena labelnya tidak pernah ikut
dijumlahkan sejak awal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _validate_mask(label_mask: np.ndarray, n_rows: int) -> np.ndarray:
    if label_mask is None:
        raise ValueError("label_mask wajib diberikan - fitur Level 3 tidak boleh tanpa mask")
    mask = np.asarray(label_mask)
    if mask.shape != (n_rows,):
        raise ValueError(f"label_mask harus berbentuk ({n_rows},), bukan {mask.shape}")
    if mask.dtype != bool:
        raise TypeError(f"label_mask harus boolean, bukan {mask.dtype}")
    return mask


def training_prior(y: np.ndarray, label_mask: np.ndarray) -> float:
    """Fraud rate global dari periode training saja - target smoothing."""
    mask = _validate_mask(label_mask, len(y))
    if not mask.any():
        raise ValueError("label_mask kosong: tidak ada baris yang labelnya boleh dipakai")
    return float(np.asarray(y)[mask].mean())


def smooth_rate(positive: np.ndarray, labeled: np.ndarray, prior: float, alpha: float):
    """Smoothing Bayesian: (pos + alpha*prior) / (labeled + alpha).

    Ketika `labeled` nol, hasilnya jatuh tepat ke `prior` tanpa percabangan
    khusus - inilah yang memenuhi aturan "node baru dapat prior global".
    """
    return ((positive + alpha * prior) / (labeled + alpha)).astype("float32")


def _self_weight(incidence: sp.csr_matrix) -> np.ndarray:
    """Diagonal (A @ A.T) tanpa membentuk matriksnya.

    Untuk A biner, sum_j A[i,j]^2 == sum_j A[i,j]. `multiply` dipakai eksplisit
    agar tetap benar seandainya bobot non-biner dipakai di kemudian hari.
    """
    return np.asarray(incidence.multiply(incidence).sum(axis=1)).ravel()


def neighbor_label_stats(
    incidence: sp.csr_matrix,
    y: np.ndarray,
    label_mask: np.ndarray,
    leave_one_out: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Jumlah tetangga fraud dan tetangga berlabel, 1-hop, dengan koreksi LOO.

    Mengembalikan (positive, labeled) - belum di-smooth, supaya bisa dipakai
    ulang untuk propagasi 2-hop.
    """
    mask = _validate_mask(label_mask, incidence.shape[0])
    mask_f = mask.astype(np.float64)
    y_masked = np.asarray(y, dtype=np.float64) * mask_f

    positive = incidence @ (incidence.T @ y_masked)
    labeled = incidence @ (incidence.T @ mask_f)

    if leave_one_out:
        weight = _self_weight(incidence)
        positive = positive - weight * y_masked
        labeled = labeled - weight * mask_f

    # Galat pembulatan float bisa menghasilkan -1e-15; jepit ke nol.
    return np.maximum(positive, 0.0), np.maximum(labeled, 0.0)


def two_hop_label_stats(
    incidence: sp.csr_matrix, y: np.ndarray, label_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Statistik label 2-hop: tetangga dari tetangga.

    Propagasi dijalankan pada vektor yang SUDAH dikoreksi LOO di tingkat 1-hop,
    lalu kontribusi diri dikurangi lagi di tingkat kedua. Koreksi ini tidak eksak
    seperti pada 1-hop karena jalur bolak-balik (i -> j -> i) tetap ada dalam
    bentuk lain, sehingga fitur ini lebih lemah secara jaminan.

    `test_two_hop_loo_removes_direct_self_contribution` memverifikasi bahwa
    kontribusi langsung sudah hilang; sisa kebocoran tak-langsung dinyatakan
    sebagai limitasi di decisions.md.
    """
    mask = _validate_mask(label_mask, incidence.shape[0])
    mask_f = mask.astype(np.float64)
    y_masked = np.asarray(y, dtype=np.float64) * mask_f
    weight = _self_weight(incidence)

    def propagate(vector: np.ndarray, own: np.ndarray) -> np.ndarray:
        first = incidence @ (incidence.T @ vector) - weight * own
        second = incidence @ (incidence.T @ first) - weight * first
        return np.maximum(second, 0.0)

    return propagate(y_masked, y_masked), propagate(mask_f, mask_f)


def group_label_stats(
    group_ids: np.ndarray, y: np.ndarray, label_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Statistik label per grup (komunitas / UID) dengan leave-one-out.

    Lebih sederhana dari kasus graph: keanggotaan tunggal, sehingga kontribusi
    diri selalu tepat 1. Grup dengan id negatif (UID tak lengkap) diperlakukan
    sebagai grup beranggota satu, sehingga LOO menyisakan nol tetangga berlabel
    dan hasilnya jatuh ke prior.
    """
    mask = _validate_mask(label_mask, len(y))
    frame = pd.DataFrame(
        {
            "group": np.asarray(group_ids),
            "y_masked": np.asarray(y, dtype=np.float64) * mask,
            "labeled": mask.astype(np.float64),
        }
    )
    totals = frame.groupby("group")[["y_masked", "labeled"]].transform("sum")

    positive = totals["y_masked"].to_numpy() - frame["y_masked"].to_numpy()
    labeled = totals["labeled"].to_numpy() - frame["labeled"].to_numpy()

    singleton = np.asarray(group_ids) < 0
    positive[singleton] = 0.0
    labeled[singleton] = 0.0
    return np.maximum(positive, 0.0), np.maximum(labeled, 0.0)


def build_level3_features(
    incidence: sp.csr_matrix,
    y: np.ndarray,
    label_mask: np.ndarray,
    community_ids: np.ndarray,
    uid: np.ndarray,
    alpha: float,
    include_2hop: bool = True,
) -> pd.DataFrame:
    """Rangkai seluruh fitur Level 3. `label_mask` wajib, tanpa default."""
    mask = _validate_mask(label_mask, incidence.shape[0])
    prior = training_prior(y, mask)

    pos1, lab1 = neighbor_label_stats(incidence, y, mask)
    out = {
        "graph_nb_fraud_rate_1hop": smooth_rate(pos1, lab1, prior, alpha),
        # Bukan fitur berbasis label: hanya menghitung BERAPA tetangga berlabel,
        # tidak menyentuh nilai y. Penting untuk interpretasi - rate dengan count
        # rendah tidak bisa dipercaya, dan model perlu tahu itu.
        "graph_nb_labeled_count": lab1.astype("float32"),
    }

    if include_2hop:
        pos2, lab2 = two_hop_label_stats(incidence, y, mask)
        out["graph_nb_fraud_rate_2hop"] = smooth_rate(pos2, lab2, prior, alpha)

    pos_c, lab_c = group_label_stats(community_ids, y, mask)
    out["graph_community_fraud_rate"] = smooth_rate(pos_c, lab_c, prior, alpha)

    pos_u, lab_u = group_label_stats(uid, y, mask)
    out["uid_fraud_rate"] = smooth_rate(pos_u, lab_u, prior, alpha)
    out["uid_labeled_count"] = lab_u.astype("float32")

    return pd.DataFrame(out)
