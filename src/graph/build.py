"""Konstruksi bipartite graph transaksi-atribut sebagai matriks insiden sparse.

STRUKTUR. A adalah matriks (n_transaksi x n_nilai_atribut) dengan A[i,j]=1 bila
transaksi i memiliki nilai atribut j. Dua transaksi "bertetangga" bila berbagi
minimal satu nilai atribut, yaitu lewat jalur 2-hop di A.

ATURAN YANG TIDAK BOLEH DILANGGAR: adjacency transaksi-ke-transaksi (A @ A.T)
TIDAK PERNAH dimaterialisasi. Untuk data ini ukurannya ~89,5 miliar nnz - akan
kehabisan memori. Semua agregasi tetangga dihitung sebagai A @ (A.T @ v), yang
biayanya linier terhadap nnz(A) = 2,08 juta.

Level 1-2 murni struktural: tidak ada fungsi di modul ini yang menyentuh isFraud.
Graph dibangun dari SELURUH data (struktur), sesuai CLAUDE.md; statistik label
baru masuk di Level 3 dengan label_mask.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.config import load_config, resolve_path
from src.data.split import add_time_features

UID_MISSING = -1


@dataclass
class BipartiteGraph:
    """Matriks insiden beserta metadata kolom asalnya.

    `column_slices` memetakan nama kolom sumber ke rentang indeks di A, sehingga
    fitur per-tipe-atribut (degree card vs degree device) bisa dihitung tanpa
    membangun ulang matriks terpisah.
    """

    incidence: sp.csr_matrix
    column_slices: dict[str, slice]
    node_values: pd.DataFrame  # kolom: source_column, value, node_index

    @property
    def n_transactions(self) -> int:
        return self.incidence.shape[0]

    @property
    def n_attribute_nodes(self) -> int:
        return self.incidence.shape[1]

    def slice_for(self, column: str) -> sp.csr_matrix:
        """Sub-matriks insiden untuk satu kolom atribut saja."""
        return self.incidence[:, self.column_slices[column]]

    def attribute_degree(self) -> np.ndarray:
        """Degree tiap node atribut = berapa transaksi memilikinya."""
        return np.asarray(self.incidence.sum(axis=0)).ravel()


def _encode_column(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Petakan nilai kategori ke indeks 0..k-1. NaN -> -1 (tidak menjadi node).

    Missing sengaja TIDAK dijadikan node tersendiri: menyatukan semua transaksi
    tanpa DeviceInfo (79,9% data) ke satu node akan menciptakan hub palsu raksasa
    yang tidak punya makna bisnis.
    """
    codes, uniques = pd.factorize(values, use_na_sentinel=True)
    return codes, uniques


def build_incidence(df: pd.DataFrame, node_columns: list[str]) -> BipartiteGraph:
    """Bangun matriks insiden gabungan untuk semua kolom node."""
    n = len(df)
    blocks, slices, records, offset = [], {}, [], 0

    for col in node_columns:
        codes, uniques = _encode_column(df[col])
        valid = codes >= 0
        rows = np.flatnonzero(valid)
        cols = codes[valid]

        block = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n, len(uniques)),
        )
        blocks.append(block)
        slices[col] = slice(offset, offset + len(uniques))
        records.append(
            pd.DataFrame(
                {
                    "source_column": col,
                    "value": uniques.astype(str),
                    "node_index": np.arange(offset, offset + len(uniques)),
                }
            )
        )
        offset += len(uniques)

    return BipartiteGraph(
        incidence=sp.hstack(blocks, format="csr"),
        column_slices=slices,
        node_values=pd.concat(records, ignore_index=True),
    )


def build_uid(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Identifier klien: kombinasi atribut + D1 yang dinormalkan terhadap waktu.

    D1n = day - D1 membuat identifier stabil sepanjang periode: D1 adalah hari
    sejak kejadian pertama klien, sehingga selisihnya konstan untuk klien yang sama.

    RISIKO LEAKAGE (dinyatakan eksplisit, dibahas di decisions.md): UID adalah
    agregasi lintas waktu. Satu klien yang muncul di train dan test membuat
    statistik train "diketahui" saat scoring test. Ini SAH karena meniru produksi
    (histori klien memang tersedia), tapi harus dinyatakan, bukan disembunyikan.

    Baris dengan D1 atau komponen lain yang missing mendapat UID unik tersendiri
    (bukan digabung), supaya tidak menciptakan klaster palsu.
    """
    uid_cfg = cfg["uid"]
    frame = add_time_features(df) if "day" not in df.columns else df
    d1n = frame["day"] - frame[uid_cfg["time_normalized"]]

    parts = [frame[c].astype("string") for c in uid_cfg["components"]]
    parts.append(d1n.astype("string"))
    key = parts[0].str.cat(parts[1:], sep="_", na_rep="?")

    incomplete = d1n.isna()
    for col in uid_cfg["components"]:
        incomplete |= frame[col].isna()

    codes, _ = pd.factorize(key.where(~incomplete), use_na_sentinel=True)
    codes = codes.astype("int64")
    # Baris tak lengkap: beri kode unik negatif agar tidak saling bergabung.
    codes[incomplete] = UID_MISSING - np.arange(incomplete.sum())
    return pd.Series(codes, index=frame.index, name="uid")


def load_source_frame(cfg: dict) -> pd.DataFrame:
    """Baca kolom yang dibutuhkan graph dari parquet merged."""
    data_cfg = load_config("data")
    needed = sorted(
        {
            "TransactionID",
            "TransactionDT",
            cfg["uid"]["aggregate_col"],
            cfg["uid"]["time_normalized"],
            *cfg["node_columns"],
            *cfg["uid"]["components"],
        }
    )
    return pd.read_parquet(resolve_path(data_cfg["paths"]["merged_train"]), columns=needed)


def main(config_name: str = "graph"):
    cfg = load_config(config_name)
    df = load_source_frame(cfg)
    graph = build_incidence(df, cfg["node_columns"])
    uid = build_uid(df, cfg)

    out = resolve_path(cfg["paths"]["incidence"])
    out.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out, graph.incidence)
    graph.node_values.to_parquet(resolve_path(cfg["paths"]["node_mapping"]), index=False)

    nnz = graph.incidence.nnz
    print(f"incidence: {graph.n_transactions:,} x {graph.n_attribute_nodes:,}, nnz={nnz:,}")
    print(f"density: {nnz / (graph.n_transactions * graph.n_attribute_nodes):.2e}")
    print(f"uid: {uid[uid >= 0].nunique():,} unik, {(uid < 0).sum():,} baris tak lengkap")
    print(f"written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build bipartite incidence matrix")
    parser.add_argument("--config", default="graph")
    main(parser.parse_args().config)
