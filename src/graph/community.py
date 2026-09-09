"""Level 2: community detection dan k-core lewat igraph.

Louvain dijalankan pada PROJECTED ATTRIBUTE GRAPH (atribut-ke-atribut, ~16K node),
bukan pada 590K node transaksi. Alasannya: projeksi atribut jauh lebih kecil,
dan komunitas atribut adalah yang punya makna bisnis — sekumpulan kartu, alamat,
dan device yang saling terkait adalah kandidat fraud ring. Transaksi kemudian
mewarisi komunitas dari atribut yang dimilikinya.

Tetap bebas label: struktur murni, tidak menyentuh isFraud.
"""

from __future__ import annotations

import random

import igraph as ig
import numpy as np
import pandas as pd
import scipy.sparse as sp


def project_attribute_graph(incidence: sp.csr_matrix) -> sp.csr_matrix:
    """Graph atribut-ke-atribut: dua atribut terhubung bila berbagi transaksi.

    Aman dimaterialisasi — dimensinya n_atribut (~16K), bukan n_transaksi (590K).
    Bobot edge = jumlah transaksi yang memiliki kedua atribut.
    """
    projected = (incidence.T @ incidence).tocsr()
    projected.setdiag(0)
    projected.eliminate_zeros()
    return projected


def _to_igraph(adjacency: sp.csr_matrix) -> ig.Graph:
    coo = sp.triu(adjacency, k=1).tocoo()
    graph = ig.Graph(n=adjacency.shape[0], edges=list(zip(coo.row, coo.col, strict=True)))
    graph.es["weight"] = coo.data.tolist()
    return graph


def community_and_core(incidence: sp.csr_matrix, resolution: float, seed: int) -> pd.DataFrame:
    """Komunitas Louvain dan k-core per transaksi, diwarisi dari atributnya.

    `graph_community_size` : ukuran komunitas atribut terbesar yang dimiliki
        transaksi — proksi skala klaster tempat transaksi berada.
    `graph_max_kcore`      : k-core tertinggi di antara atribut transaksi. Nilai
        tinggi berarti atribut berada dalam struktur yang sangat padat, ciri khas
        ring yang saling terhubung rapat.
    """
    projected = project_attribute_graph(incidence)
    graph = _to_igraph(projected)

    # igraph memakai antarmuka bergaya `random.Random`, bukan numpy Generator.
    ig.set_random_number_generator(random.Random(seed))
    membership = np.asarray(
        graph.community_multilevel(weights="weight", resolution=resolution).membership
    )
    coreness = np.asarray(graph.coreness())

    community_sizes = np.bincount(membership)
    attr_community_size = community_sizes[membership].astype(np.float32)

    # Transaksi mewarisi nilai maksimum dari atribut yang dimilikinya.
    binary = incidence.copy()
    binary.data = np.ones_like(binary.data)
    max_community = _rowwise_max(binary, attr_community_size)
    max_core = _rowwise_max(binary, coreness.astype(np.float32))

    return pd.DataFrame(
        {
            "graph_community_size": max_community.astype("float32"),
            "graph_max_kcore": max_core.astype("float32"),
        }
    )


def _rowwise_max(incidence: sp.csr_matrix, values: np.ndarray) -> np.ndarray:
    """Maksimum `values` atas atribut yang dimiliki tiap transaksi."""
    weighted = incidence.multiply(values[np.newaxis, :]).tocsr()
    result = np.zeros(weighted.shape[0], dtype=np.float32)
    indptr, data = weighted.indptr, weighted.data
    for i in range(weighted.shape[0]):
        chunk = data[indptr[i] : indptr[i + 1]]
        if chunk.size:
            result[i] = chunk.max()
    return result
