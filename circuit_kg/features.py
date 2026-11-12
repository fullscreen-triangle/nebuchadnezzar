"""Residue-level evidence: UniProt modified residues and active sites, M-CSA catalytic
residues, DisProt disorder.  Checks S1 (slots), C1 (catalytic residues), D1 (disorder).

UniProt (CC BY 4.0): MOD_RES features with ECO evidence codes; ACT_SITE notes such
as "Phosphohistidine intermediate".  M-CSA (CC BY 4.0): catalytic residues per enzyme
with UniProt numbering.  DisProt (CC BY 4.0): experimentally observed disordered
regions.
"""
from __future__ import annotations

import csv
import io
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

import numpy as np

from .core import RAW
from .proteins import AF_DIR

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
FEATURES = RAW / "uniprot" / "features.json"
MCSA = RAW / "mcsa" / "residues.json"
DISPROT = RAW / "disprot" / "disprot.tsv"
PHOSPHO = re.compile(r"Phospho(serine|threonine|tyrosine|histidine|aspartate)", re.I)
EXPERIMENTAL = {"ECO:0000269", "ECO:0007744"}      # experimental; large-scale combinatorial evidence


def fetch_features(accessions, batch=100) -> dict:
    """accession -> {"mod_res": [...], "act_site": [...]}; the raw UniProt text is cached and
    parsed on every read, so a parser change never needs a refetch."""
    have = json.loads(FEATURES.read_text()) if FEATURES.exists() else {}
    have = {a: v for a, v in have.items() if v is None or "raw" in v}      # drop pre-raw caches
    todo = sorted({a.split("-")[0] for a in accessions} - set(have))
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        q = " OR ".join(f"accession:{a}" for a in chunk)
        url = UNIPROT + "?" + urllib.parse.urlencode(
            {"query": q, "fields": "accession,ft_mod_res,ft_act_site", "format": "tsv", "size": 500})
        body = ""
        for k in range(5):
            try:
                body = urllib.request.urlopen(urllib.request.Request(
                    url, headers={"User-Agent": "circuit-kg-kernel"}), timeout=120).read().decode()
                break
            except Exception:
                time.sleep(2 ** k)
        for row in csv.DictReader(io.StringIO(body), delimiter="\t"):
            have[row["Entry"]] = {"raw": {"mod_res": row.get("Modified residue", ""),
                                          "act_site": row.get("Active site", "")}}
        for a in chunk:
            have.setdefault(a, None)
    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    FEATURES.write_text(json.dumps(have))
    return {a: None if v is None else {k: _parse(t) for k, t in v["raw"].items()} for a, v in have.items()}


def _parse(text: str) -> list[dict]:
    """UniProt feature text: 'MOD_RES 473; /note="Phosphoserine; by MTOR"; /evidence="..."'.
    Notes may contain semicolons, so the text is split at each feature keyword, not at ';'."""
    out = []
    for part in re.split(r"(?=\b(?:MOD_RES|ACT_SITE) \d)", text or ""):
        m = re.match(r"(MOD_RES|ACT_SITE) (\d+)(?:\.\.(\d+))?;", part)
        if not m:
            continue
        note = re.search(r'/note="([^"]*)"', part)
        ev = re.findall(r"ECO:\d{7}", part)
        out.append({"pos": int(m.group(2)), "note": note.group(1) if note else "", "evidence": sorted(set(ev))})
    return out


# ------------------------------------------------------------------ S1

def slot_check(located: dict, mismatches: list, feats: dict, letters_of) -> dict:
    """Located slots against UniProt phospho MOD_RES; offsets for the residue mismatches."""
    hit = miss = 0
    ground = Counter()
    for sites in located.values():
        for acc, pos, res in sites:
            f = feats.get(acc) or {}
            mods = {m["pos"]: m for m in f.get("mod_res", []) if PHOSPHO.search(m["note"])}
            if pos in mods:
                hit += 1
                ev = set(mods[pos]["evidence"])
                ground["experimental" if ev & EXPERIMENTAL else "by similarity or other"] += 1
            else:
                miss += 1
    # offsets: per protein, the single shift that puts its mismatched named residues onto a
    # residue of the named type carrying a phospho MOD_RES
    by_acc = defaultdict(list)
    for m in mismatches:
        by_acc[m["acc"]].append(m)
    explained, per = 0, {}
    for acc, ms in by_acc.items():
        f = feats.get(acc) or {}
        mods = {m["pos"] for m in f.get("mod_res", []) if PHOSPHO.search(m["note"])}
        letters = letters_of(acc) or {}
        best = None
        for d in range(-120, 121):
            if d == 0:
                continue
            ok = [m for m in ms if letters.get(str(m["pos"] + d)) == m["named"] and m["pos"] + d in mods]
            if best is None or len(ok) > len(best[1]):
                best = (d, ok)
        if best and best[1]:
            explained += len(best[1])
            per[acc] = {"offset": best[0], "explained": len(best[1]), "mismatches": len(ms)}
    return {"located_sites": hit + miss, "with_phospho_mod_res": hit,
            "fraction": hit / (hit + miss) if hit + miss else None,
            "evidence_of_matched": dict(ground),
            "mismatches": len(mismatches), "explained_by_one_offset": explained, "offsets": per}


def phospho_intermediates(feats: dict, proteins) -> list[dict]:
    """Enzymes whose active site carries a phosphorylated intermediate (UniProt ACT_SITE notes)."""
    out = []
    for u in sorted(proteins):
        for a in (feats.get(u) or {}).get("act_site", []):
            if re.search(r"intermediate", a["note"], re.I) and re.search(r"phospho", a["note"], re.I):
                out.append({"uniprot": u, "pos": a["pos"], "note": a["note"]})
    return out


# ------------------------------------------------------------------ C1

def mcsa_residues() -> dict:
    """UniProt accession -> set of catalytic residue numbers (UniProt numbering)."""
    out = defaultdict(set)
    for e in json.loads(MCSA.read_text(encoding="utf-8")):
        for s in e.get("residue_sequences") or []:
            if s.get("uniprot_id") and s.get("resid"):
                out[s["uniprot_id"]].add(int(s["resid"]))
    return out


def contact_degree(acc):
    p = AF_DIR / f"AF-{acc}.graph.json"
    if not p.exists():
        return None, None
    g = json.loads(p.read_text())
    deg = defaultdict(float)
    for u, v, w in g["edges"]:
        if abs(u - v) > 1:
            deg[u] += w
            deg[v] += w
    return deg, g["letters"]


def catalytic_hubs(proteins) -> dict:
    from scipy.stats import mannwhitneyu
    cat = mcsa_residues()
    a, b, n_prot = [], [], 0
    for u in sorted(set(proteins) & set(cat)):
        deg, letters = contact_degree(u)
        if deg is None:
            continue
        sites = {r for r in cat[u] if str(r) in letters}
        if not sites:
            continue
        n_prot += 1
        types = {letters[str(r)] for r in sites}
        for k, L in letters.items():
            if L in types:
                (a if int(k) in sites else b).append(deg.get(int(k), 0.0))
    if len(a) < 3:
        return {"proteins": n_prot, "catalytic_residues": len(a)}
    t = mannwhitneyu(a, b, alternative="greater")
    rng = np.random.default_rng(0)
    ctrl = rng.choice(b, size=len(a), replace=False) if len(b) > len(a) else b
    tc = mannwhitneyu(ctrl, b, alternative="greater")
    return {"proteins": n_prot, "catalytic_residues": len(a), "same_type_residues": len(b),
            "median_contacts_catalytic": float(np.median(a)), "median_contacts_others": float(np.median(b)),
            "p_greater": float(t.pvalue), "control_random_same_type_p": float(tc.pvalue)}


# ------------------------------------------------------------------ D1

def disprot_regions(taxon="9606") -> dict:
    out = defaultdict(list)
    with open(DISPROT, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["ncbi_taxon_id"] == taxon and "disorder" in (r["term_name"] or "").lower():
                out[r["acc"]].append((int(r["start"]), int(r["end"])))
    return out


def disorder_test(located: dict, regions: dict, max_n=1500, K0=2.0) -> dict:
    """F4 with measured disorder: local order of named sites against same-type residues, split
    by DisProt-disordered or not, in proteins DisProt covers."""
    from scipy.stats import mannwhitneyu
    from . import folding as F
    named = defaultdict(set)
    for sites in located.values():
        for acc, pos, res in sites:
            named[acc].add(pos)
    rows = {"in": ([], []), "out": ([], [])}
    covered = 0
    for acc, poss in sorted(named.items()):
        if acc not in regions:
            continue
        p = AF_DIR / f"AF-{acc}.pdb.gz"
        if not p.exists():
            continue
        bb = F.read_backbone(p)
        if len(bb["seq"]) > max_n:
            continue
        K = F.k_af(acc, bb["num"])
        if K is None:
            continue
        covered += 1
        samples, Kn = F.kuramoto(K, F.omega(bb["seq"]), [K0])
        r = F.local_order(samples, 0, Kn)
        types = {bb["seq"][bb["num"].index(q)] for q in poss if q in bb["num"]}
        for i, (num, aa) in enumerate(zip(bb["num"], bb["seq"])):
            if aa not in types:
                continue
            where = "in" if any(a <= num <= b for a, b in regions[acc]) else "out"
            rows[where][0 if num in poss else 1].append(r[i])

    def mw(s, o):
        if len(s) < 3 or len(o) < 3:
            return {"n_sites": len(s), "n_others": len(o), "p_less": None}
        return {"n_sites": len(s), "n_others": len(o), "median_sites": float(np.median(s)),
                "median_others": float(np.median(o)),
                "p_less": float(mannwhitneyu(s, o, alternative="less").pvalue)}
    return {"proteins_with_sites_and_disprot": covered,
            "outside_disordered_regions": mw(*rows["out"]), "inside_disordered_regions": mw(*rows["in"])}
