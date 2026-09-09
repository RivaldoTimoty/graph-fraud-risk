"""Test fitur Level 3 — pengaman utama terhadap leakage label.

Tiga test paling kritis:
  test_changing_val_test_labels_does_not_change_features
  test_leave_one_out_excludes_own_label
  test_isolated_node_gets_global_prior

Kalau salah satu gagal, seluruh hasil Fase 4 tidak sah.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.graph.build import build_incidence
from src.graph.label_features import (
    build_level3_features,
    group_label_stats,
    neighbor_label_stats,
    smooth_rate,
    training_prior,
    two_hop_label_stats,
)

ALPHA = 20.0


@pytest.fixture
def toy():
    """6 transaksi: 0-2 berbagi card A, 3-5 berbagi card B.

    Label: [1, 0, 0, 1, 1, 0]. Mask training = 4 baris pertama.
    """
    df = pd.DataFrame({"card": ["A", "A", "A", "B", "B", "B"]})
    y = np.array([1, 0, 0, 1, 1, 0])
    mask = np.array([True, True, True, True, False, False])
    return build_incidence(df, ["card"]).incidence, y, mask


def test_changing_val_test_labels_does_not_change_features(toy):
    """ATURAN 1: label di luar mask tidak boleh memengaruhi fitur mana pun."""
    incidence, y, mask = toy
    community = np.array([0, 0, 0, 1, 1, 1])
    uid = np.array([0, 0, 1, 1, 2, 2])

    baseline = build_level3_features(incidence, y, mask, community, uid, ALPHA)

    flipped = y.copy()
    flipped[~mask] = 1 - flipped[~mask]
    after = build_level3_features(incidence, flipped, mask, community, uid, ALPHA)

    pd.testing.assert_frame_equal(baseline, after)


def test_leave_one_out_excludes_own_label(toy):
    """ATURAN 2: transaksi tidak melihat labelnya sendiri lewat tetangganya."""
    incidence, y, mask = toy
    pos_loo, lab_loo = neighbor_label_stats(incidence, y, mask, leave_one_out=True)
    pos_raw, lab_raw = neighbor_label_stats(incidence, y, mask, leave_one_out=False)

    # Transaksi 0 (fraud, dalam mask): tanpa LOO ikut menghitung dirinya sendiri.
    assert pos_raw[0] == pytest.approx(1.0)
    assert pos_loo[0] == pytest.approx(0.0)
    # Tetangganya (1 dan 2, non-fraud) tetap terhitung sebagai berlabel.
    assert lab_loo[0] == pytest.approx(2.0)


def test_leave_one_out_no_effect_outside_mask(toy):
    """Baris di luar mask: labelnya tidak pernah masuk, jadi LOO tidak mengubah."""
    incidence, y, mask = toy
    pos_loo, _ = neighbor_label_stats(incidence, y, mask, leave_one_out=True)
    pos_raw, _ = neighbor_label_stats(incidence, y, mask, leave_one_out=False)

    np.testing.assert_allclose(pos_loo[~mask], pos_raw[~mask])


def test_isolated_node_gets_global_prior():
    """ATURAN 3: tanpa tetangga berlabel -> prior global, bukan 0, bukan NaN."""
    df = pd.DataFrame({"card": ["A", "A", "B"]})
    incidence = build_incidence(df, ["card"]).incidence
    y = np.array([1, 0, 0])
    mask = np.array([True, True, False])  # transaksi 2 sendirian & tak berlabel

    prior = training_prior(y, mask)
    out = build_level3_features(incidence, y, mask, np.array([0, 0, 1]), np.array([0, 1, 2]), ALPHA)

    assert out.loc[2, "graph_nb_fraud_rate_1hop"] == pytest.approx(prior, rel=1e-6)
    assert out.loc[2, "graph_nb_labeled_count"] == pytest.approx(0.0)
    assert not out.isna().any().any()


def test_loo_matches_dense_hand_computation(toy):
    """LOO sparse harus identik dengan perhitungan dense eksplisit."""
    incidence, y, mask = toy
    pos, lab = neighbor_label_stats(incidence, y, mask)

    adjacency = (incidence @ incidence.T).toarray()
    np.fill_diagonal(adjacency, 0.0)  # inilah arti leave-one-out
    y_masked = y * mask

    np.testing.assert_allclose(pos, adjacency @ y_masked)
    np.testing.assert_allclose(lab, adjacency @ mask.astype(float))


def test_training_prior_uses_only_masked_rows(toy):
    _, y, mask = toy
    # y[mask] = [1,0,0,1] -> 0.5, sedangkan rata-rata seluruh y = 3/6 = 0.5.
    # Pakai mask berbeda supaya perbedaannya terlihat.
    other = np.array([True, True, False, False, False, False])
    assert training_prior(y, other) == pytest.approx(0.5)
    assert training_prior(y, np.array([False, True, True, False, False, False])) == 0.0


def test_smoothing_pulls_sparse_evidence_toward_prior():
    prior = 0.035
    # 1 dari 1 tetangga fraud: bukti lemah, harus jauh di bawah 1.0
    weak = smooth_rate(np.array([1.0]), np.array([1.0]), prior, ALPHA)
    assert weak[0] < 0.15

    # 500 dari 500: bukti kuat, harus mendekati 1.0
    strong = smooth_rate(np.array([500.0]), np.array([500.0]), prior, ALPHA)
    assert strong[0] > 0.95


def test_smoothing_returns_prior_when_no_evidence():
    prior = 0.035
    out = smooth_rate(np.array([0.0]), np.array([0.0]), prior, ALPHA)
    assert out[0] == pytest.approx(prior, rel=1e-6)


def test_group_stats_leave_one_out():
    group = np.array([0, 0, 0, 1, 1])
    y = np.array([1, 1, 0, 1, 0])
    mask = np.array([True, True, True, True, True])
    pos, lab = group_label_stats(group, y, mask)

    # Baris 0: grup punya 2 fraud, dikurangi dirinya sendiri -> 1
    assert pos[0] == pytest.approx(1.0)
    assert lab[0] == pytest.approx(2.0)
    # Baris 2 (non-fraud): tetangganya 2 fraud
    assert pos[2] == pytest.approx(2.0)


def test_group_stats_negative_ids_are_isolated():
    """UID tak lengkap (id negatif) tidak boleh membentuk klaster palsu."""
    group = np.array([-1, -2, 0, 0])
    y = np.array([1, 1, 1, 0])
    mask = np.array([True, True, True, True])
    pos, lab = group_label_stats(group, y, mask)

    assert lab[0] == pytest.approx(0.0)
    assert lab[1] == pytest.approx(0.0)
    assert lab[2] == pytest.approx(1.0)


def test_two_hop_loo_removes_direct_self_contribution(toy):
    """Kontribusi langsung diri sendiri harus hilang di 2-hop."""
    incidence, y, mask = toy
    pos, lab = two_hop_label_stats(incidence, y, mask)

    assert (pos >= 0).all() and (lab >= 0).all()
    # Transaksi 0 adalah satu-satunya fraud berlabel di kliknya; setelah koreksi
    # diri, jumlah positif 2-hop-nya tidak boleh mengandung dirinya lagi.
    assert pos[0] < lab[0] or lab[0] == 0


def test_label_mask_is_mandatory(toy):
    incidence, y, _ = toy
    with pytest.raises(TypeError):
        neighbor_label_stats(incidence, y)  # type: ignore[call-arg]


def test_label_mask_must_be_boolean(toy):
    incidence, y, mask = toy
    with pytest.raises(TypeError):
        neighbor_label_stats(incidence, y, mask.astype(int))


def test_label_mask_shape_is_validated(toy):
    incidence, y, _ = toy
    with pytest.raises(ValueError):
        neighbor_label_stats(incidence, y, np.array([True, False]))


def test_empty_mask_raises(toy):
    _, y, _ = toy
    with pytest.raises(ValueError):
        training_prior(y, np.zeros(len(y), dtype=bool))


def test_all_outputs_are_finite_and_in_range(toy):
    incidence, y, mask = toy
    out = build_level3_features(
        incidence, y, mask, np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 1, 1, 2, 2]), ALPHA
    )
    rates = [c for c in out.columns if "fraud_rate" in c]
    assert len(rates) == 4  # 1hop, 2hop, community, uid
    for col in rates:
        assert np.isfinite(out[col]).all()
        assert (out[col] >= 0).all() and (out[col] <= 1).all()
