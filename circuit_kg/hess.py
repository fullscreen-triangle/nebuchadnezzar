"""Delta_r G deg of reactions that were never measured, by Hess's law.

The measured TECRDB reactions N (compounds x reactions) with values g and
standard errors s fix the formation energies mu only up to the left null space
of N.  A target reaction b is determined exactly when b lies in the column
space of N; then b^T mu is the same for every mu that fits, and

  mu_hat = argmin || W^(1/2) (N^T mu - g) ||,   W = diag(1/s^2)
  Delta G(b) = b^T mu_hat,   Var = b^T (N W N^T)^+ b

(the reactant-contribution estimate).  Cycle inconsistency among the measured
reactions shows up as the fit's chi^2; it is reported, not hidden.

Verdicts: answered (in span), absent (a compound was never measured, or b is
outside the span).  Tag: supplied, convention "hess-span".
"""
from __future__ import annotations

import numpy as np

from .core import Label, Verdict, absent, answered


class HessSpan:
    def __init__(self, tecr: dict):
        self.keys = sorted(tecr["reactions"])
        self.compounds = sorted({k for key in self.keys for k, _ in key})
        self.idx = {c: i for i, c in enumerate(self.compounds)}
        N = np.zeros((len(self.compounds), len(self.keys)))
        for j, key in enumerate(self.keys):
            for k, v in key:
                N[self.idx[k], j] = v
        g = np.array([tecr["reactions"][k]["value"] for k in self.keys])
        s = np.array([tecr["reactions"][k]["sigma"] for k in self.keys])
        self.N, self.g, self.s = N, g, s
        A = (N / s).T                      # rows: reactions, scaled
        self.mu, *_ = np.linalg.lstsq(A, g / s, rcond=None)
        resid = A @ self.mu - g / s
        rank = np.linalg.matrix_rank(N)
        self.chi2, self.dof = float(resid @ resid), int(len(self.keys) - rank)
        # Birge ratio: the measured set is over-dispersed relative to its pooled sigma when
        # chi2/dof > 1 (cycle inconsistency); derived sigmas are reported scaled by it as well
        self.birge = float(np.sqrt(max(1.0, self.chi2 / self.dof))) if self.dof else 1.0
        self.cov = np.linalg.pinv(N @ np.diag(1 / s ** 2) @ N.T)
        # projector onto the column space of N, to test span membership
        U, S, _ = np.linalg.svd(N, full_matrices=False)
        self.U = U[:, S > 1e-9 * S[0]]

    def vector(self, st: dict):
        missing = sorted(k for k, v in st.items() if v and k not in self.idx)
        if missing:
            return None, missing
        b = np.zeros(len(self.compounds))
        for k, v in st.items():
            if v:
                b[self.idx[k]] += v
        return b, []

    def derive(self, st: dict) -> Verdict:
        b, missing = self.vector(st)
        if b is not None and not b.any():
            return Verdict(Label.EMPTY, detail={"reading": "no net chemical change (transport or "
                                                           "relabelling): nothing to derive"})
        if b is None:
            return absent("compound never measured in any TECRDB reaction",
                          remedy="a measurement involving it, or a group-contribution estimate",
                          compounds=missing)
        off = b - self.U @ (self.U.T @ b)
        if np.linalg.norm(off) > 1e-8 * max(1.0, np.linalg.norm(b)):
            return absent("reaction is not a combination of measured reactions",
                          remedy="a measurement closing the missing combination")
        value = float(b @ self.mu)
        sigma = float(np.sqrt(max(b @ self.cov @ b, 0.0)))
        return answered(value, sigma=sigma, sigma_birge=sigma * self.birge, tag="supplied",
                        convention=["hess-span"], reference_state="chemical (H+ dropped)",
                        stoich={k: float(v) for k, v in sorted(st.items()) if v})

    def fit(self) -> dict:
        return {"measured_reactions": len(self.keys), "compounds": len(self.compounds),
                "chi2": self.chi2, "dof": self.dof,
                "chi2_per_dof": self.chi2 / self.dof if self.dof else None,
                "birge_ratio": self.birge}


def derive_unmeasured(corpus: dict, match: dict, smap: dict, hess: HessSpan, max_combos=64) -> dict:
    """Hess's-law verdicts for small-molecule reactions TECRDB never measured directly."""
    import itertools
    from collections import defaultdict
    from .thermo import PROTON
    out = {}
    for rid, m in match.items():
        if m.get("blocker") != "no TECRDB measurement of this reaction":
            continue
        rx = corpus["reactions"][rid]
        parts = [(s, -c) for s, c, _ in rx["reactants"]] + [(s, c) for s, c, _ in rx["products"]]
        got, first = {}, None
        for combo in itertools.islice(itertools.product(*[smap[s]["kegg"] for s, _ in parts]),
                                      max_combos):
            st = defaultdict(float)
            for (s, c), k in zip(parts, combo):
                if k != PROTON:
                    st[k] += c
            v = hess.derive(dict(st))
            if v.label is Label.ANSWERED:
                got[combo] = v
            elif first is None:
                first = v
        if not got:
            out[rid] = first.as_json()
            continue
        values = {round(v.payload, 6) for v in got.values()}
        if len(values) > 1:
            out[rid] = Verdict(Label.UNDERDETERMINED, blocker="KEGG candidates give different "
                               "derived values", detail={"values": sorted(values)}).as_json()
            continue
        v = next(iter(got.values()))
        if any(smap[s]["tag"] != "stated" for s, _ in parts):
            v.detail["convention"] = v.detail["convention"] + ["conjugate-closure"]
        out[rid] = v.as_json()
    return out
