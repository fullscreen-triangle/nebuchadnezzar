"""Folding ideas: SS assignment, coupling formula, Kuramoto locking, ternary addresses."""
from __future__ import annotations

import math

import numpy as np
import pytest

from circuit_kg import folding as F


def _loop_k(seq, K_scale=5.0):
    """The validation script's double loop, verbatim in structure."""
    c = [F.AA_COORDS[a] for a in seq]
    n = len(seq)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                sd = math.dist(c[i], c[j]); s = abs(i - j)
                cys = 2.0 * math.exp(-sd / 0.5) if seq[i] == seq[j] == "C" else 0.0
                K[i, j] = K_scale * (0.3 * math.exp(-s / 2) + 0.35 * (0.8 * math.exp(-(s - 4) ** 2)
                                     + 0.4 * math.exp(-(s - 3) ** 2))
                                     + 0.25 * math.exp(-sd / 0.2) * (1 - math.exp(-s / 8)) + 0.1 * cys)
    return K


def test_vectorised_coupling_equals_the_scripts_loop():
    seq = "TTCCPSIVARSNFNVCRLPG"
    assert np.allclose(F.k_seq(seq), _loop_k(seq))
    assert not np.allclose(F.k_seq(seq), F.k_seq(seq[::-1]))      # control: order matters


def test_identical_coupled_oscillators_lock_and_detuned_uncoupled_do_not():
    K = np.array([[0.0, 1.0], [1.0, 0.0]])
    s, _ = F.kuramoto(K, np.array([5.0, 5.2]), [5.0], steps=800)
    assert F.locked_pairs(s, 0)[0, 1]
    s, _ = F.kuramoto(K, np.array([1.0, 9.0]), [0.01], steps=800)
    assert not F.locked_pairs(s, 0)[0, 1]


def test_segments_give_one_group_per_sse_and_per_loop_stretch():
    lab = np.array(list("--HHHH-E-EE---HH"))
    groups, sse = F.segments(lab)
    assert [t for t, *_ in sse] == ["H", "E"]                    # lone E and short H are loop
    assert list(groups) == [0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4]


def test_ternary_address_is_a_bijection_and_bound_holds_but_converse_fails():
    g = (np.arange(3) + 0.5) / 3
    pts = np.array([(x, y, z) for x in g for y in g for z in g])
    pts = np.vstack([pts, [[0, 0, 0], [1, 1, 1]]])                # pin the bounding cube
    dig, side = F.ternary_address(pts, depth=3)
    assert len({tuple(d) for d in dig[:27]}) == 27
    rng = np.random.default_rng(1)
    x = rng.random((300, 3))
    dig, side = F.ternary_address(x, depth=9)
    pre = F.shared_prefix(dig)
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    assert (d <= math.sqrt(3) * side * 3.0 ** -(pre // 3) + 1e-12).all()
    # control: two points 1e-6 apart across the first boundary share no prefix
    dig, _ = F.ternary_address(np.array([[1 / 3 - 1e-6, .5, .5], [1 / 3 + 1e-6, .5, .5],
                                         [0, 0, 0], [1, 1, 1]]), depth=3)
    assert F.shared_prefix(dig)[0, 1] == 0


@pytest.mark.skipif(not (F.PDB_DIR / "1CRN.pdb").exists(), reason="run the folding stage first")
def test_ss_assignment_matches_author_records_and_shuffling_breaks_it():
    r = F.check_ss([F.PDB_DIR / "1CRN.pdb"])["1CRN"]
    assert r["q3"] >= 0.8 and r["control_shuffled_q3"] <= 0.5
