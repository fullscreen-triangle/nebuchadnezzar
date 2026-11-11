"""Contacts with the stationary phase: K_m, retention factors, saturation, and test P1.

From *Chromatography as Two-Velocity Transport* (maxwell): a molecule that
alternates between a free (mobile) and a bound (stationary) state moves on long
scales with the free velocity divided by 1 + k, where k = bound/free. In the
circuit this is a capacitance: with binding fast, the node law
(c/RT) dmu/dt = sum nu J becomes (1+k)(c/RT) dmu/dt = sum nu J. For metabolite S,
summed over the enzymes that bind it,

  k_S = sum_E [E_free] / K_d(E,S)  >=  sum_E [E_free] / K_m(E,S)

since K_m = (k_off + k_cat)/k_on >= K_d. With [E_free] <= [E_total], the
K_m-based number is neither bound in both directions, so k is reported as
'retention per unit free enzyme' (1/K_m) plus a value at [E_total] under the
supplied abundance convention, tagged as an upper reading of the K_m term.

From *Chromatography as a Contact Ladder*: the carrier can be eliminated only
when contacts are local, i.e. the effect of one item's contact does not depend
on how many other items are present. Exclusion counting gives the Langmuir
form: an enzyme's sites are shared by every substrate, and the readout for one
substrate departs from its dilute limit by the factor 1/(1 + sum_S c_S/K_S).

Sources: SABIO-RK export API (human, wild type, K_m); PaxDb 9606 whole-organism
integrated abundances (ppm). Both are cached under data/raw/ with SOURCES records.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np

from .core import RAW

SABIO_URL = "https://sabiork.h-its.org/export-api/sabio/kinlaw-entry/json"
SABIO_QUERY = 'Organism:"Homo sapiens" AND Parametertype:Km AND EnzymeType:wildtype'
SABIO_DIR = RAW / "sabio"
PAXDB_URL = "https://pax-db.org/downloads/latest/datasets/9606/9606-WHOLE_ORGANISM-integrated.txt"
PAXDB_FILE = RAW / "paxdb-9606-whole-organism-integrated.txt"
UA = {"User-Agent": "circuit-kg-kernel"}

# convention protein-ppm-to-molar: total protein molecules in the cytosol, from
# about 200-300 g/L protein at a mean mass near 50 kDa
TOTAL_PROTEIN_MOLECULES_M = (2e-3, 6e-3)
DILUTE = 0.1          # sum c/K below this: within 10% of the dilute (local) limit


def _get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                return r.read()
        except Exception:
            time.sleep(2 ** k)
    raise RuntimeError(f"failed: {url}")


SABIO_ALL_QUERY = 'Organism:"Homo sapiens" AND EnzymeType:wildtype'
SABIO_ALL_DIR = RAW / "sabio_all"


def fetch_all(page_size=500) -> dict:
    """Every human wild-type kinetic law (K_m, k_cat, K_i, ... in one pass)."""
    return fetch(page_size, query=SABIO_ALL_QUERY, dest_dir=SABIO_ALL_DIR, with_paxdb=False)


def fetch(page_size=500, query=SABIO_QUERY, dest_dir=None, with_paxdb=True) -> dict:
    dest_dir = dest_dir or SABIO_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    pages, page, total = [], 1, None
    while True:
        dest = dest_dir / f"page-{page:03d}.json.gz"
        if dest.exists():
            body = gzip.decompress(dest.read_bytes())
        else:
            url = SABIO_URL + "?" + urllib.parse.urlencode(
                {"q": query, "page": page, "pageSize": page_size})
            body = _get(url)
            dest.write_bytes(gzip.compress(body))
        meta = json.loads(body)["meta"]
        total = meta["total_count"]
        pages.append({"page": page, "sha256": hashlib.sha256(body).hexdigest()})
        if page >= meta["total_pages"]:
            break
        page += 1
    rec = {"sabio": {"url": SABIO_URL, "query": query, "page_size": page_size,
                     "total_count": total, "pages": pages,
                     "retrieved": time.strftime("%Y-%m-%d")}}
    if with_paxdb:
        if not PAXDB_FILE.exists():
            PAXDB_FILE.write_bytes(_get(PAXDB_URL))
        rec["paxdb"] = {"url": PAXDB_URL,
                        "sha256": hashlib.sha256(PAXDB_FILE.read_bytes()).hexdigest()}
    (dest_dir / "SOURCES.json").write_text(json.dumps(rec, indent=1))
    return rec


# ------------------------------------------------------------------ parse

def km_records() -> list[dict]:
    """One record per (entry, substrate K_m): uniprot ids, KEGG ids of the substrate, K_m in M."""
    return [{**{k: v for k, v in r.items() if k not in ("value", "kind", "role")}, "km_M": r["value"]}
            for r in parameter_records(("Km",), roles=("Substrate",), src=SABIO_DIR)]


UNIT_OK = {"Km": "M", "Ki": "M", "kcat": "s^(-1)"}


def parameter_records(kinds=("Km", "kcat", "Ki"), roles=("Substrate", "Inhibitor"),
                      src=None) -> list[dict]:
    """One record per (entry, parameter): kind, value in normalised units, the species it refers
    to (KEGG ids and role), uniprot ids.  k_cat records carry no species when SABIO gives none."""
    out = []
    for f in sorted((src or SABIO_ALL_DIR).glob("page-*.json.gz")):
        for e in json.loads(gzip.decompress(f.read_bytes()))["data"]:
            ed = e.get("enzyme_description") or {}
            if (ed.get("wildtype") or "").lower() != "wildtype":
                continue
            uni = sorted({p["uniprot_id"] for p in ed.get("proteins") or [] if p.get("uniprot_id")})
            kegg = defaultdict(set)
            for x in (e.get("external_links") or {}).get("compound", []):
                if x["key"] == "KeggCompoundID":
                    kegg[x["id"]].add(x["value"])
            by_name = {}
            for s in e["reaction"]["species"]:
                by_name[(s["compound"]["name"], s["role"])] = s["compound"]["id"]
            cond = e.get("experimental_conditions") or {}
            pH = (cond.get("envvar_ph") or {}).get("start_value")
            for p in (e.get("kineticlaw") or {}).get("parameter", []):
                kind = (p.get("parameter_type") or {}).get("name")
                if kind not in kinds:
                    continue
                v = p.get("n_start_value")
                if not v or v <= 0 or (p.get("unit") or {}).get("n_name") != UNIT_OK.get(kind):
                    continue
                key = ((p.get("species") or {}).get("species_key") or "").split(" | ")
                role, name, keggs = None, None, []
                if len(key) == 3:
                    name, role = key[1], key[2]
                    cid = by_name.get((name, role))
                    keggs = sorted(kegg.get(cid, ())) if cid is not None else []
                if kind != "kcat" and (role not in roles or not keggs):
                    continue
                out.append({"entry": e["id"], "uniprot": uni, "kind": kind, "value": float(v),
                            "kegg": keggs, "name": name, "role": role, "pH": pH})
    return out


def km_table(records) -> dict:
    """(uniprot, kegg) -> geometric-mean K_m over entries, with count and log10 spread."""
    acc = defaultdict(list)
    for r in records:
        for u in r["uniprot"]:
            for k in r["kegg"]:
                acc[(u, k)].append(math.log10(r["km_M"]))
    return {key: {"log10_km": float(np.mean(v)), "n": len(v),
                  "spread": float(np.ptp(v)) if len(v) > 1 else 0.0} for key, v in acc.items()}


def paxdb_ppm() -> dict:
    out = {}
    for line in PAXDB_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        gene, _, ab = line.split("\t")[:3]
        out[gene.upper()] = max(out.get(gene.upper(), 0.0), float(ab))
    return out


# ------------------------------------------------------------------ join to the kernel

def kernel_contacts(corpus: dict, smap: dict, table: dict) -> dict:
    """For each catalysed reaction: its substrates' K_m with each catalyst protein, when measured."""
    sp = corpus["species"]
    out = {}
    for rid, r in corpus["reactions"].items():
        cats = [s for s, *_ in r["modifiers"] if sp[s]["kind"] != "small" and sp[s]["uniprot"]]
        if not cats:
            continue
        rows = []
        for s, *_ in r["reactants"]:
            if sp[s]["kind"] != "small":
                continue
            keggs = smap.get(s, {}).get("kegg", [])
            for c in cats:
                for u in sp[c]["uniprot"]:
                    hits = [table[(u, k)] for k in keggs if (u, k) in table]
                    if hits:
                        h = min(hits, key=lambda x: -x["n"])
                        rows.append({"substrate": s, "catalyst": c, "uniprot": u, **h})
        out[rid] = rows
    return out


def retention_and_saturation(corpus, contacts, ppm, gene_of, window) -> dict:
    """Retention per metabolite (1/K_m per unit enzyme, and at total enzyme), and per-enzyme
    departure from the dilute limit -- the ladder paper's locality diagnostic."""
    sp = corpus["species"]
    lo, hi = window
    per_metab = defaultdict(dict)                 # substrate -> uniprot -> log10 K_m
    per_enzyme = defaultdict(dict)                # uniprot -> substrate -> log10 K_m
    for rows in contacts.values():
        for x in rows:
            per_metab[x["substrate"]][x["uniprot"]] = x["log10_km"]
            per_enzyme[x["uniprot"]][x["substrate"]] = x["log10_km"]
    retention = {}
    for s, ks in per_metab.items():
        inv = sum(10 ** -v for v in ks.values())            # sum 1/K_m, per M of enzyme
        abund = [ppm.get((gene_of.get(u) or "").upper()) for u in ks]
        k_tot = None
        if all(a is not None for a in abund):
            k_tot = [sum(a * 1e-6 * T * 10 ** -v for a, v in zip(abund, ks.values()))
                     for T in TOTAL_PROTEIN_MOLECULES_M]
        retention[s] = {"name": sp[s]["name"], "enzymes": len(ks),
                        "sum_inverse_km_per_M": inv,
                        "k_at_total_enzyme_range": k_tot}
    sat = {}
    for u, ks in per_enzyme.items():
        inv = sum(10 ** -v for v in ks.values())
        c_star = DILUTE / inv                               # every substrate at the same c
        dev_1mM = 1 - 1 / (1 + 1e-3 * inv)
        label = ("local" if c_star >= hi else "nonlocal" if c_star <= lo else "window-dependent")
        sat[u] = {"substrates": len(ks), "c_star_M": c_star, "departure_at_1mM": dev_1mM,
                  "label": label}
    ks_all = [v["k_at_total_enzyme_range"] for v in retention.values() if v["k_at_total_enzyme_range"]]
    return {
        "retention": {
            "metabolites_with_km": len(retention),
            "with_abundance_for_every_enzyme": len(ks_all),
            "k_at_total_enzyme_median": (float(np.median([k[0] for k in ks_all])),
                                         float(np.median([k[1] for k in ks_all]))) if ks_all else None,
            "k_gt_1_at_upper_total": sum(k[1] > 1 for k in ks_all),
            "top": sorted(({"metabolite": v["name"], "k_range": v["k_at_total_enzyme_range"],
                            "enzymes": v["enzymes"]} for v in retention.values()
                           if v["k_at_total_enzyme_range"]),
                          key=lambda x: -x["k_range"][1])[:12]},
        "saturation": {
            "enzymes_with_km": len(sat),
            "labels": dict(_count(v["label"] for v in sat.values())),
            "shared_sites": sum(v["substrates"] > 1 for v in sat.values()),
            "departure_at_1mM_gt_10pct": sum(v["departure_at_1mM"] > 0.1 for v in sat.values()),
            "c_star_median_M": float(np.median([v["c_star_M"] for v in sat.values()])) if sat else None},
        "per_metabolite": retention, "per_enzyme": sat,
    }


def _count(it):
    d = defaultdict(int)
    for x in it:
        d[x] += 1
    return d


# ------------------------------------------------------------------ P1

DESCRIPTORS = ["n_residues", "mean_plddt", "fraction_buried", "contact_weight_per_residue",
               "exposed_atoms_per_residue"]


def protein_descriptors(acc: str):
    from .proteins import AF_DIR
    g, m = AF_DIR / f"AF-{acc}.graph.json", AF_DIR / f"AF-{acc}.json"
    if not (g.exists() and m.exists()):
        return None
    g, m = json.loads(g.read_text()), json.loads(m.read_text())
    if m.get("mean_plddt") is None:
        return None
    n = g["n_residues"]
    ex = list(g["exposed"].values())
    return [n, float(m["mean_plddt"]), sum(v == 0 for v in ex) / n,
            sum(w for *_, w in g["edges"]) / n, sum(ex) / n]


def _ridge(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda Xn: ((Xn - mu) / sd) @ w + y.mean()


def p1_test(table: dict, descriptors: dict, plant=None, alphas=(0.1, 1, 10, 100, 1000)) -> dict:
    """Leave-one-protein-out: substrate baseline vs baseline + ridge on protein descriptors.

    plant: optional function acc -> added log10 K_m (a positive control that the test can
    detect a protein-level effect carried by the descriptors).
    """
    from scipy.stats import wilcoxon
    rows = [(u, k, v["log10_km"]) for (u, k), v in table.items() if u in descriptors]
    if plant:
        rows = [(u, k, y + plant(u)) for u, k, y in rows]
    prots = sorted({u for u, _, _ in rows})
    by_sub = defaultdict(list)
    for u, k, y in rows:
        by_sub[k].append((u, y))

    def baseline(k, exclude):
        v = [y for u, y in by_sub[k] if u not in exclude]
        return float(np.mean(v)) if v else None

    def fit(train_prots, alpha):
        X, r = [], []
        for u, k, y in rows:
            if u in train_prots:
                b = baseline(k, {u})
                if b is not None:
                    X.append(descriptors[u]); r.append(y - b)
        return _ridge(np.array(X), np.array(r), alpha) if len(X) > 10 else None

    err_b, err_m, used = {}, {}, 0
    rng = np.random.default_rng(0)
    for p in prots:
        test = [(k, y) for u, k, y in rows if u == p]
        test = [(k, y, baseline(k, {p})) for k, y in test]
        test = [t for t in test if t[2] is not None]
        if not test:
            continue
        train = [q for q in prots if q != p]
        # inner choice of alpha: 5 random folds over the training proteins
        folds = np.array_split(rng.permutation(train), 5)
        best, best_err = alphas[0], math.inf
        for a in alphas:
            e = []
            for f in folds:
                fs = set(f)
                model = fit(set(train) - fs, a)
                if model is None:
                    continue
                for u, k, y in rows:
                    if u in fs:
                        b = baseline(k, {u, p})
                        if b is not None:
                            e.append(abs(y - b - model(np.array([descriptors[u]]))[0] + 0.0))
            if e and np.mean(e) < best_err:
                best, best_err = a, float(np.mean(e))
        model = fit(set(train), best)
        if model is None:
            continue
        r = float(model(np.array([descriptors[p]]))[0])
        err_b[p] = float(np.mean([abs(y - b) for _, y, b in test]))
        err_m[p] = float(np.mean([abs(y - b - r) for _, y, b in test]))
        used += 1
    a = np.array([err_m[p] for p in err_b]); b = np.array([err_b[p] for p in err_b])
    w = wilcoxon(a, b, alternative="less")
    return {"proteins": used, "pairs": len(rows), "mean_abs_error_baseline": float(b.mean()),
            "mean_abs_error_descriptors": float(a.mean()),
            "descriptors_better": int((a < b).sum()), "baseline_better": int((a > b).sum()),
            "wilcoxon_one_sided_p": float(w.pvalue)}
