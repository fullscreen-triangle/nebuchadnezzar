"""Join Reactome species to measured reaction thermodynamics.

Path: Reactome ChEBI -> KEGG compound (ChEBI accession table) -> TECRDB
reaction (KEGG stoichiometry) -> Delta_r G deg at the chemical reference state.

Provenance of each species' KEGG id:
  stated    the ChEBI record itself carries exactly one KEGG accession;
  supplied  "conjugate-closure": found through is_conjugate_acid_of /
            is_conjugate_base_of / is_tautomer_of (protonation states share a
            reaction-level KEGG id; TECRDB is protonation-agnostic);
  absent    no KEGG id on the record or its closure;
  underdetermined  several KEGG ids and nothing selects one -- all kept as
            candidates; a reaction match is accepted only if exactly one
            candidate combination matches a TECRDB reaction.

H+ (C00080) is dropped on both sides: the chemical reference state gives it
zero formation energy, and Reactome states it inconsistently.
"""
from __future__ import annotations

import csv
import gzip
import itertools
import math
import re
from collections import defaultdict, deque
from statistics import NormalDist

from .core import RAW, Tag

KEGG = re.compile(r"C\d{5}$")
PROTON = "C00080"
WATER = "C00001"
CLOSURE_RELATIONS = {"6", "7", "11"}     # conjugate acid, conjugate base, tautomer
KEGG_STATUS = {"1", "3"}                # ChEBI status ids admitted for an accession
T_WINDOW = (293.15, 313.15)             # as in the syndrome suite
GRADES = {"A", "B", "C"}
ALPHA = 0.01
MAX_COMBOS = 64


def _rows(name):
    with gzip.open(RAW / f"chebi_{name}.tsv.gz", "rt", encoding="utf-8") as f:
        yield from csv.DictReader(f, delimiter="\t")


class Chebi:
    """ChEBI id -> KEGG compound, with provenance."""

    def __init__(self):
        self.primary = {}
        for r in _rows("secondary_ids"):
            self.primary[int(r["secondary_id"])] = int(r["compound_id"])
        self.kegg = defaultdict(set)
        for r in _rows("database_accession"):
            if KEGG.match(r["accession_number"]) and r["status_id"] in KEGG_STATUS:
                self.kegg[int(r["compound_id"])].add(r["accession_number"])
        self.closure_adj = defaultdict(set)
        for r in _rows("relation"):
            if r["relation_type_id"] in CLOSURE_RELATIONS:
                a, b = int(r["init_id"]), int(r["final_id"])
                self.closure_adj[a].add(b)
                self.closure_adj[b].add(a)

    def resolve(self, chebi_id: int) -> dict:
        c = self.primary.get(chebi_id, chebi_id)
        direct = self.kegg.get(c, set())
        if len(direct) == 1:
            return {"kegg": sorted(direct), "tag": Tag.STATED.value}
        if len(direct) > 1:
            return {"kegg": sorted(direct), "tag": "underdetermined",
                    "blocker": "several KEGG accessions on the ChEBI record"}
        seen, found, q = {c}, set(), deque([c])
        while q:
            x = q.popleft()
            for y in self.closure_adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    found |= self.kegg.get(y, set())
                    q.append(y)
        if len(found) == 1:
            return {"kegg": sorted(found), "tag": Tag.SUPPLIED.value,
                    "convention": "conjugate-closure"}
        if found:
            return {"kegg": sorted(found), "tag": "underdetermined",
                    "blocker": "conjugate closure reaches several KEGG ids",
                    "convention": "conjugate-closure"}
        return {"kegg": [], "tag": Tag.ABSENT.value,
                "blocker": "no KEGG accession on the ChEBI record or its conjugate closure"}


def species_kegg(corpus: dict, chebi: Chebi) -> dict:
    """Map every small-molecule species to KEGG candidates."""
    out = {}
    for sid, sp in corpus["species"].items():
        if sp["kind"] != "small":
            continue
        res = [chebi.resolve(c) for c in sp["chebi"]]
        cands = sorted({k for r in res for k in r["kegg"]})
        tags = {r["tag"] for r in res}
        if not cands:
            tag = Tag.ABSENT.value
        elif len(cands) > 1:
            tag = "underdetermined"
        elif tags == {Tag.STATED.value}:
            tag = Tag.STATED.value
        else:
            tag = Tag.SUPPLIED.value
        out[sid] = {"kegg": cands, "tag": tag, "per_chebi": dict(zip(sp["chebi"], res))}
    return out


# ---------------------------------------------------------------- TECRDB

def parse_reaction(s: str) -> dict:
    out = {}
    for term in s.split():
        coef, cid = term.split("*")
        if cid != PROTON:
            out[cid] = out.get(cid, 0.0) + float(coef)
    return {k: v for k, v in out.items() if v}


def canonical(st: dict) -> tuple[tuple, int]:
    """Orientation-free key; returns (key, sign) with st = sign * key."""
    items = sorted(st.items())
    sign = 1 if items[0][1] < 0 else -1
    return tuple((k, sign * v) for k, v in items), sign


def _nonwater(st: dict):
    d = {k: v for k, v in st.items() if k != WATER}
    return canonical(d) if d else (None, 1)


def load_tecrdb(path=None, pH_window=(6.8, 7.6)) -> dict:
    """Per canonical reaction: value (kJ/mol), sigma, number of studies.

    Also collects the measured transformed Delta_r G'deg of rows whose pH lies
    in pH_window (study means; min/max kept), for direction verdicts.
    """
    path = path or RAW / "tecrdb_transformed.csv"
    study = defaultdict(list)
    prime = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                T, dg = float(r["T"]), float(r["dG0"])
            except ValueError:
                continue
            if not (T_WINDOW[0] <= T <= T_WINDOW[1]) or r["eval"] not in GRADES:
                continue
            if not math.isfinite(dg):
                continue
            st = parse_reaction(r["reaction"])
            if not st:
                continue
            key, sign = canonical(st)
            study[(key, r["ref"])].append(sign * dg)
            try:
                pH, dgp, tr = float(r["pH"]), float(r["dG0_prime_measured"]), float(r["transform"])
            except ValueError:
                continue
            if not (pH_window[0] <= pH <= pH_window[1] and math.isfinite(dgp)):
                continue
            # dG0_prime_measured is in the ORIGINAL TECRDB orientation; `reaction`
            # and dG0 are canonicalised (sometimes flipped).  Recover the flip from
            # dG0 = flip * (dG0' - transform).
            chem = dgp - tr
            flip = 1 if abs(dg - chem) <= abs(dg + chem) else -1
            if abs(abs(dg) - abs(chem)) > 1e-6 * max(1.0, abs(dg)):
                continue                      # row not self-consistent; leave it out
            prime[(key, r["ref"])].append(sign * flip * dgp)
    per = defaultdict(list)
    for (key, _), vals in study.items():
        per[key].append(sum(vals) / len(vals))
    dev2 = dof = 0.0
    for vals in per.values():
        if len(vals) >= 2:
            m = sum(vals) / len(vals)
            dev2 += sum((v - m) ** 2 for v in vals)
            dof += len(vals) - 1
    sigma_pool = math.sqrt(dev2 / dof)
    table = {key: {"value": sum(v) / len(v), "n": len(v),
                   "sigma": sigma_pool / math.sqrt(len(v))} for key, v in per.items()}
    pper = defaultdict(list)
    for (key, _), vals in prime.items():
        pper[key].append(sum(vals) / len(vals))
    prime_table = {key: {"min": min(v), "max": max(v), "mean": sum(v) / len(v), "n": len(v)}
                   for key, v in pper.items()}
    return {"reactions": table, "prime": prime_table, "pH_window": list(pH_window),
            "sigma_pool": sigma_pool,
            "q": {"gaussian": NormalDist().inv_cdf(1 - ALPHA / 2),
                  "chebyshev": ALPHA ** -0.5}}


def match_reactions(corpus: dict, smap: dict, tecr: dict) -> dict:
    """For each Reactome reaction, the TECRDB reaction it is, if determined."""
    table = tecr["reactions"]
    keys_mod_water = defaultdict(list)
    for key in table:
        nk, ns = _nonwater(dict(key))
        if nk:
            keys_mod_water[nk].append((key, ns))
    out = {}
    for rid, rx in corpus["reactions"].items():
        parts = [(s, -c) for s, c, _ in rx["reactants"]] + [(s, c) for s, c, _ in rx["products"]]
        if any(corpus["species"][s]["kind"] != "small" for s, _ in parts):
            out[rid] = {"label": "refused",
                        "blocker": "a participant is a macromolecule or unresolved entity; "
                                   "TECRDB states small-molecule reactions only"}
            continue
        missing = [s for s, _ in parts if not smap[s]["kegg"]]
        if missing:
            out[rid] = {"label": "absent", "blocker": "participant without KEGG id",
                        "species": missing}
            continue
        cand_lists = [smap[s]["kegg"] for s, _ in parts]
        combos = list(itertools.islice(itertools.product(*cand_lists), MAX_COMBOS + 1))
        if len(combos) > MAX_COMBOS:
            out[rid] = {"label": "underdetermined", "blocker": "too many KEGG combinations"}
            continue
        hits = {}
        for combo in combos:
            st = defaultdict(float)
            for (s, c), k in zip(parts, combo):
                if k != PROTON:
                    st[k] += c
            st = {k: v for k, v in st.items() if v}
            if not st:
                continue
            key, sign = canonical(st)
            if key in table:
                hits[key] = (sign, "exact")
                continue
            nk, ns2 = _nonwater(st)
            for full, ns in keys_mod_water.get(nk, []) if nk else []:
                # same reaction once water is balanced; the TECRDB form is the stated one.
                # nonwater(st) = ns2*nk and nonwater(full) = ns*nk, so st ~ (ns2*ns) * full
                hits[full] = (ns2 * ns, "water-balance")
        if not hits:
            out[rid] = {"label": "absent", "blocker": "no TECRDB measurement of this reaction"}
        elif len(hits) > 1:
            out[rid] = {"label": "underdetermined",
                        "blocker": "several TECRDB reactions match the KEGG candidates"}
        else:
            (key, (sign, how)), = hits.items()
            tag = Tag.STATED.value
            conv = []
            if how == "water-balance":
                conv.append("water-balance")
            if any(smap[s]["tag"] != Tag.STATED.value for s, _ in parts):
                conv.append("conjugate-closure" if all(len(smap[s]["kegg"]) == 1 for s, _ in parts)
                            else "kegg-candidate-selected-by-tecrdb-match")
            if conv:
                tag = Tag.SUPPLIED.value
            m = table[key]
            # dG of the Reactome orientation = sign * dG(canonical key)
            out[rid] = {"label": "answered", "key": [list(kv) for kv in key], "sign": sign,
                        "dG0": sign * m["value"], "sigma": m["sigma"], "n_studies": m["n"],
                        "tag": tag, "convention": conv}
    return out


# ---------------------------------------------------------------- cycles

def cycle_basis(columns: list[dict]) -> list[dict]:
    """Integer basis of ker S by exact rational elimination.

    columns: list of {species: coefficient}.  Returns [{column index: int}].
    """
    from fractions import Fraction
    species = sorted({k for c in columns for k in c})
    rows = [{} for _ in species]
    where = {k: i for i, k in enumerate(species)}
    for j, c in enumerate(columns):
        for k, v in c.items():
            rows[where[k]][j] = Fraction(v).limit_denominator(1000)
    pivots, reduced = [], []
    for row in rows:
        row = dict(row)
        for pc, pr in zip(pivots, reduced):
            if pc in row:
                f = row[pc]
                for k, v in pr.items():
                    nv = row.get(k, 0) - f * v
                    if nv:
                        row[k] = nv
                    else:
                        row.pop(k, None)
        if not row:
            continue
        pc = min(row)
        f = row[pc]
        row = {k: v / f for k, v in row.items()}
        for pr in reduced:
            if pc in pr:
                h = pr[pc]
                for k, v in row.items():
                    nv = pr.get(k, 0) - h * v
                    if nv:
                        pr[k] = nv
                    else:
                        pr.pop(k, None)
        pivots.append(pc)
        reduced.append(row)
    pivot_set = set(pivots)
    basis = []
    for free in (j for j in range(len(columns)) if j not in pivot_set):
        z = {free: Fraction(1)}
        for pc, pr in zip(pivots, reduced):
            if free in pr:
                z[pc] = -pr[free]
        den = math.lcm(*(v.denominator for v in z.values()))
        zi = {k: int(v * den) for k, v in z.items()}
        g = math.gcd(*zi.values())
        basis.append({k: v // g for k, v in zi.items() if v})
    return basis


def gamma(n: int, u: float = 2.0 ** -53) -> float:
    return n * u / (1 - n * u)


def judge_cycle(z: dict, value: list, sigma: list, q: float) -> dict:
    terms = [v * value[j] for j, v in z.items()]
    h = sum(terms)
    e_num = gamma(len(z)) * sum(abs(t) for t in terms)
    s = math.sqrt(sum(v * v * sigma[j] ** 2 for j, v in z.items()))
    e_data = q * s
    a = abs(h)
    verdict = ("CONSISTENT" if a <= e_num else
               "UNDECIDABLE" if a <= e_num + e_data else "INCONSISTENT")
    return {"length": len(z), "H": h, "eps_num": e_num, "eps_data": e_data,
            "Z": h / s if s else None, "verdict": verdict}


def falsifiable(support_tags) -> bool:
    """A cycle sum built only from species-derived energies (Hess span, Legendre transforms,
    estimates) is zero by construction, so its test cannot fail: refuse it
    (health-dcat-ap-plus-circuit thermo.py; Lemma: g = N^T f gives every cycle sum 0)."""
    return any(t in ("stated", "supplied") for t in support_tags)


def judge_cycle_guarded(z: dict, value: list, sigma: list, q: float, tags: list) -> dict:
    """judge_cycle behind the falsifiability guard."""
    if not falsifiable([tags[j] for j in z]):
        return {"verdict": "REFUSED", "blocker": "non-falsifiable: every energy on the cycle is derived"}
    return judge_cycle(z, value, sigma, q)
