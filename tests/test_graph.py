"""Test fitur graph Level 1-2 pada graph mainan yang dihitung manual.

Test terpenting adalah `test_neighbor_sum_equals_dense_adjacency_result`: ia
membuktikan trik A @ (A.T @ v) menghasilkan angka yang PERSIS SAMA dengan
(A @ A.T) @ v yang dihitung secara dense. Kalau trik itu salah, seluruh fitur
graph salah tanpa ada yang error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.graph.build import build_incidence, build_uid
from src.graph.community import project_attribute_graph
from src.graph.features import (
    attribute_entropy,
    cap_high_degree_columns,
    component_features,
    cross_cardinality_features,
    degree_features,
    neighbor_count_features,
    neighbor_sum,
    pagerank,
)
from src.graph.uid_features import uid_activity_features


@pytest.fixture
def toy():
    """Graph mainan 10 transaksi, dua kolom atribut.

    card: A A A B B | C C D D D
    dev : X X Y Y Z | Z W W nan nan

    Struktur yang sengaja dibuat: transaksi 0-4 dan 5-9 terhubung hanya lewat
    dev=Z (transaksi 4 dan 5), sehingga membentuk satu komponen. Dua baris
    terakhir punya device missing untuk menguji bahwa NaN tidak menjadi node.
    """
    return pd.DataFrame(
        {
            "card": ["A", "A", "A", "B", "B", "C", "C", "D", "D", "D"],
            "dev": ["X", "X", "Y", "Y", "Z", "Z", "W", "W", None, None],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        }
    )


def test_incidence_shape_and_nnz(toy):
    graph = build_incidence(toy, ["card", "dev"])
    # 4 nilai card (A,B,C,D) + 4 nilai dev (X,Y,Z,W) = 8 node atribut
    assert graph.incidence.shape == (10, 8)
    # 10 edge dari card + 8 edge dari dev (2 baris dev-nya NaN) = 18
    assert graph.incidence.nnz == 18


def test_missing_value_does_not_become_node(toy):
    graph = build_incidence(toy, ["dev"])
    assert graph.incidence.shape[1] == 4  # X, Y, Z, W — bukan 5
    # Dua baris terakhir tidak punya edge sama sekali
    assert graph.incidence[8:].nnz == 0


def test_attribute_degree_matches_hand_count(toy):
    graph = build_incidence(toy, ["card"])
    degree = graph.attribute_degree()
    # A muncul 3x, B 2x, C 2x, D 3x
    np.testing.assert_array_equal(degree, [3.0, 2.0, 2.0, 3.0])


def test_degree_features_match_hand_count(toy):
    out = degree_features(build_incidence(toy, ["card"]))
    # Tiap transaksi mewarisi degree kartunya
    assert out["graph_deg_card"].tolist() == [3, 3, 3, 2, 2, 2, 2, 3, 3, 3]


def test_neighbor_sum_equals_dense_adjacency_result(toy):
    """Trik sparse HARUS identik dengan perhitungan dense. Ini test paling kritis."""
    graph = build_incidence(toy, ["card", "dev"])
    A = graph.incidence
    values = np.arange(1.0, 11.0, dtype=np.float32)

    sparse_result = neighbor_sum(A, values)
    dense_result = (A @ A.T).toarray() @ values

    np.testing.assert_allclose(sparse_result, dense_result, rtol=1e-6)


def test_neighbor_sum_hand_computed(toy):
    """Verifikasi satu nilai secara manual, bukan hanya konsisten dengan dense."""
    graph = build_incidence(toy, ["card"])
    ones = np.ones(10, dtype=np.float32)
    result = neighbor_sum(graph.incidence, ones)
    # Transaksi 0 berbagi card A dengan transaksi 0,1,2 -> 3
    assert result[0] == pytest.approx(3.0)
    # Transaksi 3 berbagi card B dengan transaksi 3,4 -> 2
    assert result[3] == pytest.approx(2.0)


def test_cap_high_degree_removes_only_hubs(toy):
    graph = build_incidence(toy, ["card"])
    capped, n_removed = cap_high_degree_columns(graph.incidence, max_degree=2)

    assert n_removed == 2  # A dan D punya degree 3
    # Transaksi 0 (card A) kehilangan seluruh edge-nya
    assert capped[0].nnz == 0
    # Transaksi 3 (card B, degree 2) tetap punya edge
    assert capped[3].nnz == 1


def test_cross_cardinality_counts_distinct(toy):
    out = cross_cardinality_features(toy, [("card", "dev")])
    col = "graph_n_unique_dev_per_card"
    # card A punya dev {X, Y} -> 2; card D punya dev {W} + 2 NaN -> 1
    assert out.loc[0, col] == pytest.approx(2.0)
    assert out.loc[7, col] == pytest.approx(1.0)


def test_neighbor_count_features_are_consistent(toy):
    graph = build_incidence(toy, ["card", "dev"])
    out = neighbor_count_features(graph.incidence)

    assert out["graph_n_attributes"].tolist()[:3] == [2, 2, 2]
    assert out["graph_n_attributes"].tolist()[8:] == [1, 1]  # dev NaN
    assert (out["graph_two_hop_count"] > 0).all()


def test_attribute_entropy_zero_for_equal_degrees():
    """Entropi nol bila semua atribut transaksi berukuran sama persis."""
    df = pd.DataFrame({"a": ["p", "p"], "b": ["q", "q"]})
    graph = build_incidence(df, ["a", "b"])
    # Kedua atribut punya degree 2 -> proporsi 0.5/0.5 -> entropi = log(2)
    entropy = attribute_entropy(graph)
    assert entropy.iloc[0] == pytest.approx(np.log(2), rel=1e-5)


def test_pagerank_sums_to_one_and_is_positive(toy):
    graph = build_incidence(toy, ["card", "dev"])
    rank = pagerank(graph.incidence, damping=0.85, max_iter=100, tol=1e-12)

    assert rank.sum() == pytest.approx(1.0, abs=1e-6)
    assert (rank > 0).all()


def test_pagerank_higher_for_well_connected_node():
    """Transaksi di klaster padat harus dapat skor lebih tinggi."""
    df = pd.DataFrame({"a": ["hub", "hub", "hub", "lonely"]})
    graph = build_incidence(df, ["a"])
    rank = pagerank(graph.incidence, damping=0.85, max_iter=100, tol=1e-12)
    assert rank[0] > rank[3]


def test_component_features_detect_disconnected_groups():
    """Dua kelompok yang tidak berbagi atribut apa pun harus terpisah."""
    df = pd.DataFrame({"a": ["p", "p", "q", "q", "q"]})
    out = component_features(build_incidence(df, ["a"]).incidence)

    # Komponen 'p': 2 transaksi + 1 node atribut = 3. Komponen 'q': 3 + 1 = 4.
    assert out["graph_component_size"].tolist() == [3.0, 3.0, 4.0, 4.0, 4.0]


def test_component_id_is_not_exposed_as_feature():
    """ID komponen arbitrer — kalau bocor sebagai fitur, model salah menafsirkannya."""
    df = pd.DataFrame({"a": ["p", "p", "q", "q"]})
    out = component_features(build_incidence(df, ["a"]).incidence)
    assert "graph_component_id" not in out.columns


def test_projected_attribute_graph_has_no_self_loops(toy):
    projected = project_attribute_graph(build_incidence(toy, ["card", "dev"]).incidence)
    assert projected.diagonal().sum() == 0


def test_uid_groups_same_client(toy):
    df = toy.assign(
        card1=["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
        addr1=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        day=[10, 11, 10, 11, 10, 11, 10, 11, 10, 11],
        D1=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    )
    cfg = {"uid": {"components": ["card1", "addr1"], "time_normalized": "D1"}}
    uid = build_uid(df, cfg)

    # D1n = day - D1 = 10 untuk semua; pasangan berurutan berbagi card1+addr1
    assert uid[0] == uid[1]
    assert uid[0] != uid[2]
    assert (uid >= 0).all()


def test_uid_incomplete_rows_get_unique_negative_codes(toy):
    df = toy.assign(
        card1=["A", "A", None, "B"] + ["C"] * 6,
        addr1=[1, 1, 2, 2] + [3] * 6,
        day=[10] * 10,
        D1=[0, 0, 0, None] + [0] * 6,
    )
    cfg = {"uid": {"components": ["card1", "addr1"], "time_normalized": "D1"}}
    uid = build_uid(df, cfg)

    negatives = uid[uid < 0]
    assert len(negatives) == 2  # baris 2 (card1 NaN) dan 3 (D1 NaN)
    assert negatives.nunique() == 2  # tidak digabung menjadi satu klaster palsu


def test_uid_activity_features_nan_for_incomplete(toy):
    df = toy.assign(day=list(range(10)), DeviceInfo=["d"] * 10)
    uid = pd.Series([0, 0, 1, 1, -1, -2, 2, 2, 2, 3])
    out = uid_activity_features(df, uid, "TransactionAmt")

    assert np.isnan(out.loc[4, "uid_size"])  # UID tak lengkap -> NaN, bukan 0
    assert out.loc[0, "uid_size"] == pytest.approx(2.0)
    assert out.loc[6, "uid_size"] == pytest.approx(3.0)


def test_uid_amt_ratio_is_relative_to_client(toy):
    df = toy.assign(day=list(range(10)), DeviceInfo=["d"] * 10)
    uid = pd.Series([0, 0] + list(range(1, 9)))
    out = uid_activity_features(df, uid, "TransactionAmt")

    # UID 0: amt 10 dan 20 -> mean 15. Rasio baris 0 = 10/15
    assert out.loc[0, "uid_amt_ratio"] == pytest.approx(10.0 / 15.0, rel=1e-5)


def test_level12_features_never_touch_label(toy):
    """Level 1-2 harus bebas label: isFraud tidak boleh memengaruhi hasil."""
    graph = build_incidence(toy, ["card", "dev"])
    baseline = degree_features(graph)

    flipped = toy.assign(isFraud=[1] * 10)
    after = degree_features(build_incidence(flipped, ["card", "dev"]))

    pd.testing.assert_frame_equal(baseline, after)


def test_incidence_is_sparse_not_dense(toy):
    """Jaga agar tidak ada yang diam-diam mengubah ke dense."""
    graph = build_incidence(toy, ["card", "dev"])
    assert sp.issparse(graph.incidence)
    assert graph.incidence.format == "csr"
