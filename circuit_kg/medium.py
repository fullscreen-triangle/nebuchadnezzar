"""The medium as the eluent: pH, ionic strength, pMg and temperature.

Chromatography as a contact ladder (maxwell) found that the eluent's numbers
transfer between laboratories while tabulated stationary-phase descriptors do
not.  Here the contact is the reaction's chemical-reference Delta_r G deg (H+
dropped, O balanced with water: the TECRDB convention of syndrome's stage A), and
the medium enters only through the Legendre transform of each compound:

  Delta_r G'deg(medium) = Delta_r G deg + sum_i nu_i * t_i(pH, I, T, pMg)

t_i is computed exactly as eQuilibrator's equilibrator_cache 0.8.1 does
(Compound.transform; CompoundMicrospecies.transform; _legendre_transform with
extended Debye-Hueckel, alpha(T) cubic, B = 1.6), from the microspecies table of
the local eQuilibrator compound cache.  The implementation is checked against
the per-row `transform` column syndrome computed with eQuilibrator itself
(`reproduce_transform_column`).

Crowding enters as a second medium number: an accessible volume fraction a of
the cytosol, so that mu = mu deg + RT ln(c/a).  With one a for every small
solute, it shifts a reaction's affinity by -RT ln(a) * sum(nu) (water excluded):
only reactions that change the number of solute particles feel it
(convention `accessible-volume`).
"""
from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import DERIVED, RAW
from .thermo import GRADES, T_WINDOW, WATER, canonical, parse_reaction

EQ_CACHE = Path(os.environ.get(
    "CIRCUIT_KG_EQ_CACHE",
    Path.home() / "AppData/Local/equilibrator/equilibrator/Cache/compounds.sqlite"))
TABLE = DERIVED / "legendre.json"

# equilibrator_cache.thermodynamic_constants (0.8.1)
R_EQ = 8.31e-3                 # kJ/mol/K -- eQuilibrator's value, kept for exact agreement
T_STD = 298.15
LN10 = math.log(10)
DG_MG, DH_MG = -455.3, -467.00
N_A = 6.02214076e23
VOLUME_PER_DALTON_NM3 = 0.73 / N_A * 1e21     # partial specific volume 0.73 cm3/g


def debye_hueckel(sqrt_I: float, T: float) -> float:
    alpha = 9.20483e-3 * T - 1.284668e-5 * T ** 2 + 4.95199e-8 * T ** 3
    return alpha / (1.0 / sqrt_I + 1.6)


def _ms_transform(ms, pH, I, T, pMg):
    z, nH, nMg, ddg = ms
    RT = R_EQ * T
    dg_mg = (T / T_STD) * DG_MG + (1.0 - T / T_STD) * DH_MG
    term = nH * RT * LN10 * pH + nMg * (RT * LN10 * pMg - dg_mg)
    if I > 0:
        term -= debye_hueckel(I ** 0.5, T) * (z * z - nH - 4 * nMg)
    return ddg * T_STD / T + term / RT


def build_table(kegg_ids, path=EQ_CACHE, out=TABLE) -> dict:
    """Microspecies (charge, nH, nMg, ddG/RT) and mass for each KEGG id, from the cache."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    reg = db.execute("select id from registries where namespace='kegg'").fetchone()[0]
    kegg_ids = sorted(set(kegg_ids))
    cid = {}
    for i in range(0, len(kegg_ids), 500):
        chunk = kegg_ids[i:i + 500]
        q = ",".join("?" * len(chunk))
        for acc, c in db.execute(f"select accession, compound_id from compound_identifiers "
                                 f"where registry_id=? and accession in ({q})", (reg, *chunk)):
            cid.setdefault(acc, c)
    ids = sorted(set(cid.values()))
    ms, mass = defaultdict(list), {}
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        q = ",".join("?" * len(chunk))
        for c, z, nH, nMg, ddg in db.execute(
                f"select compound_id, charge, number_protons, number_magnesiums, ddg_over_rt "
                f"from compound_microspecies where compound_id in ({q})", chunk):
            ms[c].append([z, nH, nMg, ddg])
        for c, m in db.execute(f"select id, mass from compounds where id in ({q})", chunk):
            mass[c] = m
    st = os.stat(path)
    table = {"source": {"path": str(path), "bytes": st.st_size, "mtime": int(st.st_mtime),
                        "formula": "equilibrator_cache 0.8.1 Compound.transform"},
             "compounds": {}}
    for acc in kegg_ids:
        c = cid.get(acc)
        rows = ms.get(c, [])
        ok = bool(rows) and all(r[3] is not None for r in rows)
        table["compounds"][acc] = {"microspecies": rows if ok else [],
                                   "mass": mass.get(c), "in_cache": c is not None}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table))
    return table


class Legendre:
    def __init__(self, table=None):
        table = table or json.loads(TABLE.read_text())
        self.c = table["compounds"]
        self.source = table["source"]

    def compound(self, kegg: str, pH, I, T, pMg) -> float | None:
        rec = self.c.get(kegg)
        if not rec or not rec["microspecies"]:
            return None
        x = [-_ms_transform(ms, pH, I, T, pMg) for ms in rec["microspecies"]]
        return -R_EQ * T * float(np.logaddexp.reduce(x))

    def reaction(self, st: dict, pH, I, T, pMg):
        total, missing = 0.0, []
        for k, v in st.items():
            if not v:
                continue
            t = self.compound(k, pH, I, T, pMg)
            if t is None:
                missing.append(k)
            else:
                total += v * t
        return (None, sorted(missing)) if missing else (total, [])


# ------------------------------------------------------------------ TECRDB rows

def tecrdb_rows(path=None) -> list[dict]:
    """Rows in the canonical orientation: transformed value g', syndrome's transform t, conditions."""
    path = path or RAW / "tecrdb_transformed.csv"
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                T, dg0, dgp, tr = (float(r[k]) for k in ("T", "dG0", "dG0_prime_measured", "transform"))
                pH, I, pMg = float(r["pH"]), float(r["I"]), float(r["pMg"])
            except ValueError:
                continue
            if not (T_WINDOW[0] <= T <= T_WINDOW[1]) or r["eval"] not in GRADES:
                continue
            if not all(map(math.isfinite, (dg0, dgp, tr))):
                continue
            st = parse_reaction(r["reaction"])
            if not st:
                continue
            chem = dgp - tr
            flip = 1 if abs(dg0 - chem) <= abs(dg0 + chem) else -1
            if abs(abs(dg0) - abs(chem)) > 1e-6 * max(1.0, abs(dg0)):
                continue
            key, sign = canonical(st)
            # `reaction` is written canonically by syndrome; key/sign re-canonicalise after
            # parse_reaction (which drops H+), so the row's orientation is sign * flip
            s = sign * flip
            out.append({"key": key, "st": {k: sign * v for k, v in st.items()},
                        "ref": r["ref"], "pH": pH, "I": I, "pMg": pMg, "T": T,
                        "g_prime": s * dgp, "t_syndrome": s * tr, "g0": sign * dg0})
    return out


def reproduce_transform_column(L: Legendre, rows, perturb=None) -> dict:
    """Recompute every row's transform from the microspecies table; compare with syndrome's."""
    diffs, missing = [], 0
    for r in rows:
        c = {k: r[k] for k in ("pH", "I", "T", "pMg")}
        if perturb:
            c = perturb(c)
        t, miss = L.reaction(dict(r["key"]), c["pH"], c["I"], c["T"], c["pMg"])
        if t is None:
            missing += 1
            continue
        diffs.append(abs(t - r["t_syndrome"]))
    d = np.array(diffs)
    return {"rows": len(rows), "compared": len(d), "missing_compound": missing,
            "max_abs_diff": float(d.max()), "median_abs_diff": float(np.median(d)),
            "within_1e-6": int((d < 1e-6).sum()), "within_0.1": int((d < 0.1).sum())}


# ------------------------------------------------------------------ transfer test

def _cond(r):
    return (round(r["pH"], 2), round(r["I"], 3), round(r["pMg"], 2), round(r["T"], 1))


def transfer_test(rows) -> dict:
    """Prediction M1 (predictions/medium-transfer.md): held-out medium conditions.

    Per reaction: 'medium' = one chemical value fitted on the training rows plus the
    held-out row's Legendre transform (1 number per reaction); 'constant' = mean
    training g' (ignores the medium, 1 number); 'affine-pH' = a + b pH on training
    rows (2 numbers).  Designs: interpolation (each condition whose pH lies strictly
    inside the other conditions' pH range, held out in turn) and extrapolation (the
    highest-pH and the lowest-pH condition held out, when strictly outside the rest).
    """
    by = defaultdict(list)
    for r in rows:
        by[r["key"]].append(r)
    # replicate noise: same reaction, same conditions, different studies
    rep = []
    for rs in by.values():
        g = defaultdict(lambda: defaultdict(list))
        for r in rs:
            g[_cond(r)][r["ref"]].append(r["g_prime"])
        for refs in g.values():
            means = [sum(v) / len(v) for v in refs.values()]
            rep += [abs(a - b) for i, a in enumerate(means) for b in means[i + 1:]]
    err = {d: defaultdict(list) for d in ("interpolation", "extrapolation")}
    medium_moves = {d: defaultdict(list) for d in err}
    per_rx = {d: defaultdict(lambda: defaultdict(list)) for d in err}
    n_reactions = defaultdict(set)
    for key, rs in by.items():
        conds = defaultdict(list)
        for r in rs:
            conds[_cond(r)].append(r)
        if len(conds) < 3:
            continue
        ph = {c: c[0] for c in conds}
        held = []
        for c in conds:
            others = [ph[o] for o in conds if o != c]
            if min(others) < ph[c] < max(others):
                held.append(("interpolation", [c]))
        top = max(ph.values()); bot = min(ph.values())
        for edge in (top, bot):
            cs = [c for c in conds if ph[c] == edge]
            rest = [ph[o] for o in conds if ph[o] != edge]
            if rest and (edge > max(rest) or edge < min(rest)):
                held.append(("extrapolation", cs))
        for design, cs in held:
            test = [r for c in cs for r in conds[c]]
            train = [r for c in conds if c not in cs for r in conds[c]]
            chem = np.mean([r["g_prime"] - r["t_syndrome"] for r in train])
            const = np.mean([r["g_prime"] for r in train])
            tph = np.array([r["pH"] for r in train])
            if np.ptp(tph) > 0:
                b, a = np.polyfit(tph, [r["g_prime"] for r in train], 1)
            else:
                a, b = const, 0.0
            tmean = np.mean([r["t_syndrome"] for r in train])
            for r in test:
                y = r["g_prime"]
                e = {"medium": abs(chem + r["t_syndrome"] - y), "constant": abs(const - y),
                     "affine-pH": abs(a + b * r["pH"] - y)}
                for m, v in e.items():
                    err[design][m].append(v)
                    per_rx[design][key][m].append(v)
                medium_moves[design]["moved"].append(abs(r["t_syndrome"] - tmean) > 1.0)
            n_reactions[design].add(key)

    def stats(v):
        v = np.asarray(v)
        return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v))} if len(v) else None

    from scipy.stats import wilcoxon

    def paired(d, other):
        """Per-reaction mean errors, medium vs other: who wins more reactions, and by how much."""
        a = np.array([np.mean(v["medium"]) for v in per_rx[d].values()])
        b = np.array([np.mean(v[other]) for v in per_rx[d].values()])
        w = wilcoxon(a, b)
        return {"reactions": int(len(a)), "medium_better": int((a < b).sum()),
                "other_better": int((a > b).sum()), "median_difference": float(np.median(a - b)),
                "wilcoxon_two_sided_p": float(w.pvalue)}

    out = {"replicate_noise": stats(rep), "designs": {}}
    for d, models in err.items():
        moved = np.array(medium_moves[d]["moved"])
        out["designs"][d] = {
            "reactions": len(n_reactions[d]),
            "paired_per_reaction": {o: paired(d, o) for o in ("constant", "affine-pH")},
            "all": {m: stats(v) for m, v in models.items()},
            "medium_moved_gt_1kJ": {m: stats(np.asarray(v)[moved]) for m, v in models.items()},
            "medium_moved_le_1kJ": {m: stats(np.asarray(v)[~moved]) for m, v in models.items()},
        }
    return out


# ------------------------------------------------------------------ the cytosol

def medium_grid(medium: dict):
    w = medium["medium_window"]
    pHs = np.linspace(*w["pH"], 5)
    Is = np.linspace(*w["ionic_strength_M"], 3)
    pMgs = np.linspace(*w["pMg"], 3)
    return [(float(p), float(i), w["T_K"], float(m)) for p in pHs for i in Is for m in pMgs]


def transformed_range(L: Legendre, st: dict, g0: float, medium: dict):
    """min/max of Delta_r G'deg over the medium window, or (None, missing)."""
    vals = []
    for pH, I, T, pMg in medium_grid(medium):
        t, miss = L.reaction(st, pH, I, T, pMg)
        if t is None:
            return None, miss
        vals.append(g0 + t)
    return (min(vals), max(vals)), []


def crowding_shift_range(st: dict, medium: dict, RT: float):
    """-RT ln(a) * sum(nu) over the accessible-volume window (water excluded)."""
    n = sum(v for k, v in st.items() if k != WATER)
    lo, hi = medium["medium_window"]["accessible_volume_fraction"]
    vals = [-RT * math.log(a) * n for a in (lo, hi)]
    return min(vals), max(vals)


def dilute_bound(L: Legendre, kegg_ids, c_max_M: float) -> dict:
    """Per-particle log-count error bound 2 n v (ideal-gas paper, Prop. occupation counts)."""
    n = c_max_M * N_A / 1e24                       # particles per nm^3
    vals = {k: 2 * n * L.c[k]["mass"] * VOLUME_PER_DALTON_NM3
            for k in kegg_ids if k in L.c and L.c[k].get("mass")}
    worst = max(vals, key=vals.get)
    return {"c_max_M": c_max_M, "species": len(vals), "max_bound": vals[worst], "worst": worst,
            "median_bound": float(np.median(list(vals.values())))}


# ------------------------------------------------------------------ direction (M2, M3)

def chemical_values(match: dict, tecr: dict, unmeasured: dict, slots: dict) -> dict:
    """Every Reactome reaction with a chemical-reference Delta_r G deg, oriented as written."""
    out = {}
    for rid, m in match.items():
        if m["label"] == "answered":
            key, s = tuple(map(tuple, m["key"])), m["sign"]
            rec = tecr["reactions"][key]
            out[rid] = {"g0": s * rec["value"], "sigma": rec["sigma"], "source": "measured",
                        "st": {k: s * c for k, c in key}}
    for source, d in (("hess", unmeasured), ("carrier-slot", slots)):
        for rid, v in d.items():
            if v["label"] == "answered":
                out[rid] = {"g0": v["payload"], "sigma": v["detail"]["sigma_birge"],
                            "source": source, "st": v["detail"]["stoich"]}
    return out


def _sign(lo, hi):
    return "forward" if hi < 0 else "reverse" if lo > 0 else None


def direction_with_medium(values: dict, L: Legendre, medium: dict, q: float, RT: float) -> dict:
    """Direction over the medium window x concentration window, with and without crowding.

    blocker 'concentration': the sign of Delta_r G'm (every participant at 1 mM) is fixed
    across the medium window, so only the concentration window leaves it open;
    blocker 'medium': the medium window alone admits both signs of Delta_r G'm.
    """
    lo_c, hi_c = (math.log(x) for x in medium["concentration_window_M"])
    out = {}
    for rid, v in values.items():
        st = {k: c for k, c in v["st"].items() if k != WATER and c}
        rng, miss = transformed_range(L, v["st"], v["g0"], medium)
        if rng is None:
            out[rid] = {"label": "absent", "blocker": "no pKa data for a participant",
                        "compounds": miss, "source": v["source"]}
            continue
        dmin, dmax = rng[0] - q * v["sigma"], rng[1] + q * v["sigma"]
        lnq_max = sum(c * (hi_c if c > 0 else lo_c) for c in st.values())
        lnq_min = sum(c * (lo_c if c > 0 else hi_c) for c in st.values())
        mm = RT * math.log(1e-3) * sum(st.values())
        cmin, cmax = crowding_shift_range(v["st"], medium, RT)
        plain = _sign(dmin + RT * lnq_min, dmax + RT * lnq_max)
        crowd = _sign(dmin + RT * lnq_min + cmin, dmax + RT * lnq_max + cmax)
        at_1mM = _sign(dmin + mm, dmax + mm)
        rec = {"source": v["source"], "dG_prime_range": [dmin, dmax],
               "RT_lnQ_range": [RT * lnq_min, RT * lnq_max], "crowding_range": [cmin, cmax],
               "dG_prime_m_sign_fixed": at_1mM,
               "conventions": ["medium-window", "concentration-window", "accessible-volume"]}
        if plain:
            rec.update(label="answered", payload=plain)
        else:
            rec.update(label="undecidable",
                       blocker="concentration" if at_1mM else "medium",
                       remedy="measured cytosolic concentrations" if at_1mM
                       else "a narrower medium window (measured pH, I, pMg)")
        rec["crowding_changes_verdict"] = crowd != plain
        out[rid] = rec
    return out
