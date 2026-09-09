"""Fitur graph Level 1-2: struktural, tanpa menyentuh label.

Semua fitur di sini bisa dihitung pada saat transaksi terjadi tanpa mengetahui
masa depan maupun label, sehingga bebas dari risiko leakage target. Fitur berbasis
label (Level 3) berada di modul terpisah dan wajib memakai label_mask.

Prefiks nama fitur:
  graph_*  -> berasal dari bipartite graph transaksi-atribut
  uid_*    -> berasal dari entity resolution UID (Pendekatan B)

POLA KOMPUTASI. Agregasi tetangga selalu berbentuk A @ (A.T @ v) sehingga
adjacency transaksi-ke-transaksi tidak pernah dimaterialisasi. Urutan kurung
ini WAJIB: (A @ A.T) @ v akan mencoba membentuk matriks 590K x 590K.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from src.graph.build import BipartiteGraph


def neighbor_sum(incidence: sp.csr_matrix, values: np.ndarray) -> np.ndarray:
    """Jumlahkan `values` atas semua transaksi yang berbagi atribut.

    Menghitung (A @ A.T) @ v tanpa membentuk A @ A.T. Hasil untuk transaksi i
    adalah jumlah v atas seluruh tetangga i, dengan bobot = banyaknya atribut
    yang dibagi bersama. Termasuk kontribusi diri sendiri (i tetangga dirinya).
    """
    return incidence @ (incidence.T @ values)


def cap_high_degree_columns(incidence: sp.csr_matrix, max_degree: int) -> tuple[sp.csr_matrix, int]:
    """Nolkan kolom atribut yang degree-nya melampaui batas.

    Alasan bisnis: P_emaildomain=gmail.com dimiliki 228.355 transaksi (38,7%).
    "Berbagi gmail" bukan sinyal fraud, tapi kalau ikut dipropagasikan, setiap
    transaksi gmail menjadi tetangga setiap transaksi gmail lain - noise yang
    menenggelamkan sinyal dari atribut spesifik, sekaligus menyumbang 66,8 miliar
    dari 89,5 miliar nnz.

    Degree atribut itu sendiri TETAP dipakai sebagai fitur; yang dibuang hanya
    perannya dalam propagasi tetangga.
    """
    degree = np.asarray(incidence.sum(axis=0)).ravel()
    keep = degree <= max_degree
    mask = sp.diags(keep.astype(np.float32))
    return (incidence @ mask).tocsr(), int((~keep).sum())


def degree_features(graph: BipartiteGraph) -> pd.DataFrame:
    """Level 1: berapa transaksi lain berbagi tiap atribut dengan transaksi ini.

    Arti bisnis: degree tinggi pada card1 berarti kartu dipakai sangat sering -
    bisa merchant sah, bisa kartu yang dieksploitasi. Model yang menentukan.
    """
    out = {}
    for col, sl in graph.column_slices.items():
        block = graph.incidence[:, sl]
        attr_degree = np.asarray(block.sum(axis=0)).ravel()
        out[f"graph_deg_{col}"] = (block @ attr_degree).astype("float32")
    return pd.DataFrame(out)


def cross_cardinality_features(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Level 1: berapa nilai unik kolom B yang terkait dengan nilai kolom A.

    Arti bisnis per pasangan:
      DeviceInfo->card1        : satu device banyak kartu -> fraud farm
      card1->DeviceInfo        : satu kartu banyak device -> kartu dicuri
      card1->P_emaildomain     : satu kartu banyak email -> penyalahgunaan identitas
      addr1->card1             : satu alamat banyak kartu
    """
    out = {}
    for group_col, target_col in pairs:
        counts = df.groupby(group_col)[target_col].nunique()
        name = f"graph_n_unique_{target_col}_per_{group_col}"
        out[name] = df[group_col].map(counts).astype("float32")
    return pd.DataFrame(out, index=df.index)


def neighbor_count_features(incidence: sp.csr_matrix) -> pd.DataFrame:
    """Level 1: ukuran lingkungan 2-hop tiap transaksi.

    `graph_two_hop_count` menghitung berapa banyak pasangan (tetangga, atribut
    bersama) yang dimiliki transaksi - proksi seberapa terhubung transaksi ini
    dalam jaringan, tanpa membentuk adjacency.
    """
    ones = np.ones(incidence.shape[0], dtype=np.float32)
    total = neighbor_sum(incidence, ones)
    n_attrs = np.asarray(incidence.sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(n_attrs > 0, total / n_attrs, 0.0)
    return pd.DataFrame(
        {
            "graph_two_hop_count": total.astype("float32"),
            "graph_n_attributes": n_attrs.astype("float32"),
            "graph_avg_neighbors_per_attr": avg.astype("float32"),
        }
    )


def attribute_entropy(graph: BipartiteGraph) -> pd.Series:
    """Level 1: keberagaman ukuran atribut yang dimiliki satu transaksi.

    Entropi rendah berarti semua atribut transaksi ini berukuran serupa; tinggi
    berarti campuran atribut langka dan umum. Transaksi fraud kerap memadukan
    atribut sangat langka (device baru) dengan atribut sangat umum (email publik).
    """
    degree = graph.attribute_degree()
    weighted = graph.incidence.multiply(degree[np.newaxis, :]).tocsr()
    row_total = np.asarray(weighted.sum(axis=1)).ravel()

    entropy = np.zeros(weighted.shape[0], dtype=np.float64)
    indptr, data = weighted.indptr, weighted.data
    for i in range(weighted.shape[0]):
        total = row_total[i]
        if total <= 0:
            continue
        p = data[indptr[i] : indptr[i + 1]] / total
        entropy[i] = -np.sum(p * np.log(p, where=p > 0, out=np.zeros_like(p)))
    return pd.Series(entropy.astype("float32"), name="graph_attr_entropy")


def pagerank(incidence: sp.csr_matrix, damping: float, max_iter: int, tol: float) -> np.ndarray:
    """Level 2: sentralitas transaksi lewat power iteration pada bipartite graph.

    Random surfer berjalan transaksi -> atribut -> transaksi. Skor tinggi berarti
    transaksi berada di pusat klaster yang saling terhubung rapat. Dihitung tanpa
    membentuk adjacency: tiap iterasi hanya dua perkalian sparse-vektor.
    """
    n = incidence.shape[0]
    row_norm = np.asarray(incidence.sum(axis=1)).ravel()
    col_norm = np.asarray(incidence.sum(axis=0)).ravel()
    row_inv = np.divide(1.0, row_norm, out=np.zeros_like(row_norm), where=row_norm > 0)
    col_inv = np.divide(1.0, col_norm, out=np.zeros_like(col_norm), where=col_norm > 0)

    rank = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        to_attr = incidence.T @ (rank * row_inv)
        updated = teleport + damping * (incidence @ (to_attr * col_inv))
        # Dangling mass: transaksi tanpa atribut tidak mendistribusikan skornya.
        updated += (1.0 - updated.sum()) / n
        if np.abs(updated - rank).sum() < tol:
            rank = updated
            break
        rank = updated
    return rank


def component_features(incidence: sp.csr_matrix) -> pd.DataFrame:
    """Level 2: ukuran connected component tempat transaksi berada.

    Arti bisnis: komponen besar berarti sekumpulan transaksi yang saling terkait
    lewat rantai atribut bersama - kandidat fraud ring bila juga padat. Dihitung
    pada graph bipartit gabungan (transaksi + atribut) sebagai satu graph.
    """
    n_tx = incidence.shape[0]
    top_right = sp.csr_matrix(incidence)
    adjacency = sp.bmat([[None, top_right], [top_right.T, None]], format="csr", dtype=np.float32)
    n_components, labels = connected_components(adjacency, directed=False)

    tx_labels = labels[:n_tx]
    sizes = np.bincount(labels, minlength=n_components)
    # component_id sengaja TIDAK dikembalikan sebagai fitur: nilainya arbitrer
    # (bergantung urutan penelusuran), sehingga model akan memperlakukan jarak
    # antar-ID sebagai bermakna padahal tidak. Hanya ukurannya yang informatif.
    return pd.DataFrame({"graph_component_size": sizes[tx_labels].astype("float32")})
