"""Identifier and stoichiometry cross-references: Rhea, MetaNetX, Human-GEM.

Rhea (CC BY 4.0): curated, mass- and charge-balanced reactions on ChEBI, with a
ready-made Reactome mapping. A mapping names a directional Rhea reaction (LR,
RL) or a bidirectional one (BI); a directional mapping is a curated statement
that the Reactome reaction runs as written. Participants and coefficients are
read from Rhea's SPARQL endpoint (the coefficient is in the predicate name,
rh:contains1, rh:contains2, ...).

MetaNetX (MNXref, CC BY 4.0): reconciles compound identifiers. Used to supply
KEGG ids for ChEBI-identified species the ChEBI flat files leave without one
(convention `mnx-xref`).

Human-GEM (CC BY 4.0): an independent, mass-balanced genome-scale model. Its
cytosolic reactions are matched to kernel reactions by participant set.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

from .core import RAW

RHEA = RAW / "rhea"
MNX = RAW / "metanetx" / "chem_xref.filtered.tsv"
GEM = RAW / "humangem"
SPARQL = "https://sparql.rhea-db.org/sparql"
UA = {"User-Agent": "circuit-kg-kernel", "Accept": "application/sparql-results+json"}
PROTON, WATER = "C00080", "C00001"


# ------------------------------------------------------------------ Rhea

def rhea_reactome() -> dict:
    """Reactome stable number -> list of (rhea id, direction, master id)."""
    out = defaultdict(list)
    for line in (RHEA / "rhea2reactome.tsv").read_text().splitlines()[1:]:
        rid, direction, master, rx = line.split("\t")
        num = rx.split("-")[-1].split(".")[0]
        out[num].append((rid, direction, master))
    return dict(out)


def rhea_ec() -> dict:
    out = defaultdict(set)
    for line in (RHEA / "rhea2ec.tsv").read_text().splitlines()[1:]:
        rid, _, master, ec = line.split("\t")
        out[master].add(ec)
    return {k: sorted(v) for k, v in out.items()}


def _sparql(query: str) -> list[dict]:
    u = SPARQL + "?" + urllib.parse.urlencode({"query": "PREFIX rh:<http://rdf.rhea-db.org/>\n" + query})
    d = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=180).read())
    return [{k: v["value"] for k, v in b.items()} for b in d["results"]["bindings"]]


def rhea_participants(masters, cache=None) -> dict:
    """master id -> {"L": {accession: coef}, "R": {...}}; accession is CHEBI:n, GENERIC:n or POLYMER:n."""
    cache = cache or RHEA / "participants.json"
    have = json.loads(cache.read_text()) if cache.exists() else {}
    todo = sorted(set(masters) - set(have))
    for i in range(0, len(todo), 80):
        chunk = todo[i:i + 80]
        vals = " ".join(f"rh:{m}" for m in chunk)
        rows = _sparql(f"""SELECT ?r ?ord ?prop ?acc WHERE {{ VALUES ?r {{ {vals} }}
            ?r rh:side ?s . ?s rh:curatedOrder ?ord . ?s ?prop ?p .
            FILTER(STRSTARTS(STR(?prop), "http://rdf.rhea-db.org/contains") && ?prop != rh:contains)
            ?p rh:compound ?c . ?c rh:accession ?acc }}""")
        for m in chunk:
            have.setdefault(m, {"L": {}, "R": {}})
        for r in rows:
            m = r["r"].rsplit("/", 1)[-1]
            side = "L" if r["ord"] == "1" else "R"
            coef = r["prop"].rsplit("contains", 1)[-1]
            have[m][side][r["acc"]] = int(coef) if coef.isdigit() else coef
    cache.write_text(json.dumps(have))
    return have


# ------------------------------------------------------------------ MetaNetX

def mnx_tables() -> tuple[dict, dict]:
    """chebi id (int) -> MNX ids, and MNX id -> KEGG compound ids (C numbers only)."""
    chebi2mnx, mnx2kegg = defaultdict(set), defaultdict(set)
    for line in MNX.read_text(encoding="utf-8").splitlines():
        src, mnx = line.split("\t")[:2]
        if src.startswith("chebi:"):
            chebi2mnx[int(src.split(":")[1])].add(mnx)
        else:
            k = src.split(":", 1)[1]
            if re.fullmatch(r"C\d{5}", k):
                mnx2kegg[mnx].add(k)
    return chebi2mnx, mnx2kegg


def rescue_kegg(corpus: dict, smap: dict) -> dict:
    """KEGG ids through MetaNetX for small species the ChEBI flat files leave without one."""
    chebi2mnx, mnx2kegg = mnx_tables()
    out = {}
    for sid, s in corpus["species"].items():
        if s["kind"] != "small" or smap.get(sid, {}).get("kegg"):
            continue
        ks = sorted({k for c in s["chebi"] for m in chebi2mnx.get(int(c), ()) for k in mnx2kegg.get(m, ())})
        if ks:
            out[sid] = {"kegg": ks, "tag": "supplied", "convention": ["mnx-xref"]}
    return out


# ------------------------------------------------------------------ Human-GEM

def human_gem() -> list[dict]:
    """Cytosolic Human-GEM reactions as {id, st: {kegg: coef}, reversible} where every metabolite
    has a KEGG id; H+ left out."""
    import yaml
    meta = {}
    lines = (GEM / "metabolites.tsv").read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    ik = head.index("metKEGGID")
    for line in lines[1:]:
        f = line.split("\t")
        meta[f[0]] = f[ik].split(";")[0].strip() if len(f) > ik else ""
    doc = yaml.safe_load((GEM / "Human-GEM.yml").read_text(encoding="utf-8"))
    sections = {k: v for item in doc for k, v in (item if isinstance(item, list) else [item])} \
        if isinstance(doc, list) else doc
    rx = sections["reactions"]
    out = []
    for r in rx:
        r = dict(r)
        mets = dict(r.get("metabolites") or {})
        if not mets or not all(m.endswith("c") for m in mets):
            continue
        st, ok = {}, True
        for m, c in mets.items():
            k = meta.get(m)
            if not k:
                ok = False
                break
            if k != PROTON:
                st[k] = st.get(k, 0) + float(c)
        if ok and st:
            out.append({"id": r["id"], "st": st, "reversible": float(r.get("lower_bound", 0)) < 0})
    return out


def _key(st):
    return frozenset(k for k, v in st.items() if v and k != PROTON)


def compare_gem(corpus: dict, smap: dict, gem: list[dict]) -> dict:
    """G1: kernel small-molecule reactions against Human-GEM reactions with the same participants."""
    sp = corpus["species"]
    by = defaultdict(list)
    for g in gem:
        by[_key(g["st"])].append(g)
    agree = disagree = 0
    rev_agree = rev_total = 0
    examples = []
    candidates = 0
    for rid, r in corpus["reactions"].items():
        parts = [(s, -c) for s, c, _ in r["reactants"]] + [(s, c) for s, c, _ in r["products"]]
        if any(sp[s]["kind"] != "small" for s, _ in parts):
            continue
        st, ok = {}, True
        for s, c in parts:
            ks = smap.get(s, {}).get("kegg", [])
            if not ks:
                ok = False
                break
            if ks[0] != PROTON:
                st[ks[0]] = st.get(ks[0], 0) + c
        if not ok or not _key(st):
            continue
        candidates += 1
        hits = by.get(_key(st))
        if not hits:
            continue
        g = hits[0]
        same = all(abs(st.get(k, 0) - g["st"].get(k, 0)) < 1e-9 for k in _key(st)) or \
            all(abs(st.get(k, 0) + g["st"].get(k, 0)) < 1e-9 for k in _key(st))
        agree += same
        disagree += not same
        if not same and len(examples) < 8:
            examples.append(r["name"])
        rev_total += 1
        rev_agree += bool(r.get("reversible")) == g["reversible"]
    n = agree + disagree
    return {"kernel_small_molecule_reactions_with_kegg": candidates, "matched_by_participants": n,
            "coefficients_agree": agree, "coefficients_differ": disagree,
            "agreement": agree / n if n else None, "differing_examples": examples,
            "reversibility_agrees": rev_agree, "reversibility_compared": rev_total}


# ------------------------------------------------------------------ Rhea against the kernel

def rhea_against_kernel(corpus: dict, smap: dict, chebi, medium_direction: dict) -> dict:
    """Coverage, participant agreement (via ChEBI -> KEGG), and R1: curated direction against the
    fixed sign of Delta_r G'm at 1 mM."""
    rr = rhea_reactome()
    mapped = {rid: rr[rid.split("_")[1]] for rid in corpus["reactions"] if rid.split("_")[1] in rr}
    masters = {m for v in mapped.values() for _, _, m in v}
    parts = rhea_participants(masters)
    sp = corpus["species"]
    dirs = Counter(d for v in mapped.values() for _, d, _ in v)
    agree = disagree = generic = 0
    r1 = Counter()
    r1_examples = []
    for rid, maps in mapped.items():
        rx = corpus["reactions"][rid]
        rid_direction = {d for _, d, _ in maps}
        m = maps[0][2]
        p = parts.get(m, {"L": {}, "R": {}})
        if any(a.startswith(("GENERIC", "POLYMER")) for side in p.values() for a in side):
            generic += 1
        # participant agreement: Rhea ChEBI -> KEGG by conjugate closure, against the kernel's KEGG set
        rhea_keggs = set()
        for side in p.values():
            for acc in side:
                if acc.startswith("CHEBI:"):
                    rk = chebi.resolve(int(acc.split(":")[1])).get("kegg") or []
                    if len(rk) == 1:
                        rhea_keggs.add(rk[0])
        ours = {k for s, *_ in rx["reactants"] + rx["products"] for k in smap.get(s, {}).get("kegg", [])[:1]}
        ours -= {PROTON}
        rhea_keggs -= {PROTON}
        if ours and rhea_keggs:
            agree += ours == rhea_keggs
            disagree += ours != rhea_keggs
        # R1
        v = medium_direction.get(rid)
        if v and rid_direction & {"LR", "RL"} and v.get("dG_prime_m_sign_fixed"):
            # the mapping names the Rhea direction that matches the Reactome reaction as written
            r1["compared"] += 1
            ok = v["dG_prime_m_sign_fixed"] == "forward"
            r1["agree"] += ok
            if not ok and len(r1_examples) < 10:
                r1_examples.append(rx["name"])
    return {"kernel_reactions_mapped_to_rhea": len(mapped), "mapping_directions": dict(dirs),
            "rhea_masters": len(masters), "with_generic_or_polymer_participants": generic,
            "participant_sets_agree": agree, "participant_sets_differ": disagree,
            "R1_compared": r1["compared"], "R1_agree": r1["agree"],
            "R1_agreement": r1["agree"] / r1["compared"] if r1["compared"] else None,
            "R1_disagreeing_examples": r1_examples}
