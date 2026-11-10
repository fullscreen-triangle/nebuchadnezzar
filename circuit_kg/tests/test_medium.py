"""The medium module: transform agreement, transfer test, crowding -- each with a control."""
from __future__ import annotations

import random

import pytest

from circuit_kg import medium as md
from circuit_kg.core import RAW


def test_crowding_only_shifts_reactions_that_change_particle_number():
    med = {"medium_window": {"accessible_volume_fraction": [0.7, 1.0]}}
    assert md.crowding_shift_range({"A": -1, "B": 1, "C00001": -1}, med, 2.577) == (0.0, 0.0)
    lo, hi = md.crowding_shift_range({"A": -1, "B": 1, "C": 1}, med, 2.577)   # control: +1 particle
    assert lo == 0.0 and hi == pytest.approx(2.577 * 0.35667, rel=1e-4)


def _synthetic_rows(seed, couple=True):
    """Reactions whose transformed values are a chemical value plus a transform that moves with pH."""
    rng = random.Random(seed)
    rows = []
    for j in range(40):
        g0, slope = rng.uniform(-30, 30), rng.uniform(-8, 8)
        for k, pH in enumerate((6.0, 6.5, 7.0, 7.5, 8.0)):
            t = slope * (pH - 7) + 0.4 * (pH - 7) ** 3
            t_rec = t if couple else rng.uniform(-8, 8)      # control: the recorded transform is wrong
            for ref in ("a", "b"):
                rows.append({"key": (("X%d" % j, 1),), "ref": ref, "pH": pH, "I": 0.25, "pMg": 14.0,
                             "T": 298.15, "t_syndrome": t_rec,
                             "g_prime": g0 + t + rng.gauss(0, 0.5)})
    return rows


def test_transfer_test_rewards_a_correct_transform_and_not_a_wrong_one():
    good = md.transfer_test(_synthetic_rows(1))["designs"]["extrapolation"]["all"]
    assert good["medium"]["mean"] < good["constant"]["mean"]
    assert good["medium"]["mean"] < good["affine-pH"]["mean"]       # cubic term defeats a line
    bad = md.transfer_test(_synthetic_rows(1, couple=False))["designs"]["extrapolation"]["all"]
    assert bad["medium"]["mean"] > good["medium"]["mean"] * 3


@pytest.mark.skipif(not md.TABLE.exists() or not (RAW / "tecrdb_transformed.csv").exists(),
                    reason="run the medium stage first")
def test_legendre_reproduces_equilibrator_transforms_and_a_perturbation_does_not():
    L, rows = md.Legendre(), md.tecrdb_rows()
    ok = md.reproduce_transform_column(L, rows)
    assert ok["compared"] == ok["rows"] and ok["max_abs_diff"] < 1e-6
    off = md.reproduce_transform_column(L, rows, perturb=lambda c: {**c, "pH": c["pH"] + 0.1})
    assert off["within_1e-6"] < 0.05 * off["compared"]
