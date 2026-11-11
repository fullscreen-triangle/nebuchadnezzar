"""Electron transfer as a ladder (P450 monograph, multi-hop ET chain paper; built here on
public data for cytosolic redox relays).

Potentials: for redox couples X_ox/X_red the electron potential is mu_e = -F E'.
Relative to the NADP couple, E_X - E_NADP = -Delta_r G'deg / (nF) for
NADPH + X_ox -> NADP+ + X_red, with Delta_r G deg from the Hess span of measured
TECRDB reactions and the Legendre transform at the declared conditions.

Hop rates: the Moser-Dutton ruler,
  log10 k_ET = 13 - (1.2 - 0.8 rho)(R - 3.6) - 3.1 (dG + lambda)^2 / lambda
with R the edge distance between redox centres (A), rho the packing density, dG and
lambda in eV.  R comes from PDB coordinates.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .core import Label, Verdict, absent, answered

F_KJ = 96.485                          # kJ / (mol V)
TEXTBOOK = {"pH": 7.0, "ionic_strength_M": 0.25, "T_K": 298.15, "pMg": 14.0}
REFERENCE = "NADP"
COUPLES = {   # name: ({ox: n}, {red: n}), two electrons each
    "NADP": ({"C00006": 1}, {"C00005": 1}),
    "NAD": ({"C00003": 1}, {"C00004": 1}),
    "glutathione": ({"C00127": 1}, {"C00051": 2}),
    "cystine": ({"C00491": 1}, {"C00097": 2}),
    "dehydroascorbate": ({"C05422": 1}, {"C00072": 1}),
    "FAD": ({"C00016": 1}, {"C01352": 1}),
    "lipoamide": ({"C00248": 1}, {"C00579": 1}),
    "thioredoxin": ({"C00343": 1}, {"C00342": 1}),
    "pyruvate/lactate": ({"C00022": 1}, {"C00186": 1}),
    "oxaloacetate/malate": ({"C00036": 1}, {"C00149": 1}),
}


def against_reference(name):
    """Stoichiometry of NADPH + X_ox -> NADP+ + X_red (products positive)."""
    ox, red = COUPLES[name]
    rox, rred = COUPLES[REFERENCE]
    st = {}
    for d, sign in ((rred, -1), (ox, -1), (rox, 1), (red, 1)):
        for k, v in d.items():
            st[k] = st.get(k, 0) + sign * v
    return {k: v for k, v in st.items() if v}


def couple_ladder(hess, L, cond=TEXTBOOK) -> dict:
    """E_X - E_NADP (mV) for every couple, or the blocker."""
    out = {}
    for name in COUPLES:
        if name == REFERENCE:
            out[name] = answered(0.0, reference=True).as_json()
            continue
        st = against_reference(name)
        v = hess.derive(st)
        if v.label is not Label.ANSWERED:
            out[name] = v.as_json()
            continue
        t, miss = L.reaction(st, cond["pH"], cond["ionic_strength_M"], cond["T_K"], cond["pMg"])
        if t is None:
            out[name] = absent("no pKa data for a participant", compounds=miss).as_json()
            continue
        dgp = v.payload + t
        sig = v.detail.get("sigma_birge", v.detail.get("sigma", 0.0))
        out[name] = answered(-dgp / (2 * F_KJ) * 1000, dG_prime_kJ=dgp, sigma_kJ=sig,
                             sigma_mV=sig / (2 * F_KJ) * 1000, tag="derived",
                             convention=["hess-span", "textbook-standard-conditions"]).as_json()
    return out


def moser_dutton(R, rho=0.76, dG=0.0, lam=0.7):
    R = max(R, 3.6)
    return 10 ** (13 - (1.2 - 0.8 * rho) * (R - 3.6) - 3.1 * (dG + lam) ** 2 / lam)


FLAVIN_RING = {"N1", "C2", "O2", "N3", "C4", "O4", "C4X", "N5", "C5X", "C6", "C7", "C8",
               "C9", "C9A", "N10", "C10"}


def flavin_to_cys(path: Path, cys, chain="A"):
    """Edge distance (A) from the FAD isoalloxazine ring to the nearest SG of the given cysteines."""
    ring, sg = [], []
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")) or line[21:22] != chain:
            continue
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        if line.startswith("HETATM") and line[17:20] == "FAD" and line[12:16].strip() in FLAVIN_RING:
            ring.append(xyz)
        elif line.startswith("ATOM") and line[17:20] == "CYS" and int(line[22:26]) in cys \
                and line[12:16].strip() == "SG":
            sg.append(xyz)
    if not ring or not sg:
        return None
    a, b = np.array(ring), np.array(sg)
    return float(np.linalg.norm(a[:, None] - b[None], axis=-1).min())
