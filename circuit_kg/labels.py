"""Labelled hierarchies on structural classes (X5) and stoichiometric signatures of
reaction classes (X6).

X5 (P450 monograph: family -> subfamily -> member addresses; spectroscopic atlas: a
refinement class should name the channel that separates it; empty-dictionary paper:
shuffle nulls).  Junctions get labels from public hierarchies -- EC at levels 1-4,
Pfam domain sets, gene symbols -- and each shared refinement class is asked which is
the coarsest label that splits it.  Concordance: do real EC labels split fewer shared
classes than the same labels shuffled among the same junctions?

X6 (atypical-reactions and heteroatom papers: a reaction class entails observable
signatures).  Fixed stoichiometric rules assign a class; several matching rules give
`underdetermined`; the class is compared with the catalyst's EC class.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict

import numpy as np

from .annotations import ec_prefix
from .core import DERIVED, Label, Verdict, absent, answered

FORMULAS = DERIVED / "formulas.json"


# ------------------------------------------------------------------ labels per junction

def junction_labels(corpus, ann) -> dict:
    """rid -> {EC1..EC4, pfam, gene}: a single value, a sorted tuple if several, None if none."""
    sp = corpus["species"]
    out = {}
    for rid, r in corpus["reactions"].items():
        prots = sorted({u.split("-")[0] for s, role in r["modifiers"] if role == "catalyst"
                        for u in sp[s]["uniprot"]})
        recs = [ann.get(u) for u in prots if ann.get(u)]
        lab = {}
        for lvl in (1, 2, 3, 4):
            vals = {p for a in recs for e in a["ec"] if (p := ec_prefix(e, lvl))}
            lab[f"EC{lvl}"] = (next(iter(vals)) if len(vals) == 1 else tuple(sorted(vals))) if vals else None
        pf = tuple(sorted({p for a in recs for p in a["pfam"]}))
        lab["pfam"] = pf or None
        genes = tuple(sorted({a["gene"] for a in recs if a.get("gene")}))
        lab["gene"] = genes or None
        out[rid] = lab
    return out


def shared_junction_classes(final: dict) -> list[list]:
    by = defaultdict(list)
    for n, c in final.items():
        if n[0] == "r":
            by[c].append(n[1])
    return [v for v in by.values() if len(v) > 1]


def splits(cls, lab) -> bool:
    vals = {lab[r] for r in cls if lab.get(r) is not None}
    return len(vals) >= 2


def concordance(classes, labels, key="EC1", n_shuffle=200, seed=0) -> dict:
    """Single-valued labels only (multi-EC catalysts count as unlabelled)."""
    lab = {r: (v if isinstance(v, str) else None)
           for r, d in labels.items() for v in [d.get(key)]}
    members = [r for c in classes for r in c if lab.get(r) is not None]
    real = sum(splits(c, lab) for c in classes)
    rng = np.random.default_rng(seed)
    vals = [lab[r] for r in members]
    null = []
    for _ in range(n_shuffle):
        perm = dict(zip(members, rng.permutation(vals)))
        null.append(sum(splits(c, perm) for c in classes))
    null = np.array(null)
    return {"label": key, "shared_classes": len(classes), "labelled_junctions": len(members),
            "classes_split_real": int(real), "classes_split_shuffled_mean": float(null.mean()),
            "classes_split_shuffled_min": int(null.min()),
            "empirical_p_real_le_shuffled": float(((null <= real).sum() + 1) / (n_shuffle + 1))}


ORDER = ("EC1", "EC2", "EC3", "EC4", "pfam", "gene")


def discriminating_dimension(classes, labels) -> dict:
    """For each shared class: the coarsest label that splits it (and whether it separates all)."""
    per, counts = [], Counter()
    for c in classes:
        hit = None
        for key in ORDER:
            lab = {r: labels[r][key] for r in c}
            if splits(c, lab):
                vals = [lab[r] for r in c]
                full = None not in vals and len(set(vals)) == len(c)
                hit = {"dimension": key, "separates_all": full}
                break
        counts[hit["dimension"] if hit else "underdetermined: no label splits it"] += 1
        per.append({"size": len(c), **(hit or {"dimension": None})})
    return {"classes": len(classes), "coarsest_splitting_label": dict(counts),
            "fully_separated": sum(1 for p in per if p.get("separates_all"))}


def bits_left(final: dict, labels: dict, key=None) -> float:
    """Mean binary distinctions still needed per item after splitting classes by a label."""
    groups = Counter()
    for n, c in final.items():
        extra = labels.get(n[1], {}).get(key) if (key and n[0] == "r") else None
        groups[(c, extra if not isinstance(extra, list) else tuple(extra))] += 1
    K = sum(groups.values())
    return sum(s * math.log2(s) for s in groups.values()) / K


# ------------------------------------------------------------------ formulas

def formulas(kegg_ids, cache_path) -> dict:
    """KEGG -> molecular formula from the eQuilibrator cache's InChI strings (no pickles)."""
    have = json.loads(FORMULAS.read_text()) if FORMULAS.exists() else {}
    todo = sorted(set(kegg_ids) - set(have))
    if todo:
        db = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
        reg = db.execute("select id from registries where namespace='kegg'").fetchone()[0]
        for i in range(0, len(todo), 500):
            chunk = todo[i:i + 500]
            q = ",".join("?" * len(chunk))
            rows = db.execute(f"select ci.accession, c.inchi from compound_identifiers ci join "
                              f"compounds c on c.id = ci.compound_id where ci.registry_id=? and "
                              f"ci.accession in ({q})", (reg, *chunk)).fetchall()
            for acc, inchi in rows:
                m = re.match(r"InChI=1S?/([^/]+)", inchi or "")
                have[acc] = m.group(1) if m else None
        for k in todo:
            have.setdefault(k, None)
        FORMULAS.write_text(json.dumps(have))
    return have


# ------------------------------------------------------------------ X6 signatures

COUPLES = [("C00003", "C00004"), ("C00006", "C00005"), ("C00016", "C01352"),
           ("C00127", "C00051"), ("C00343", "C00342"), ("C00007", None), ("C00027", None)]
COFACTORS = {"C00002", "C00008", "C00020", "C00009", "C00013", "C00001", "C00080",
             "C00003", "C00004", "C00006", "C00005"}
EC_NAME = {"1": "oxidoreductase", "2": "transferase", "3": "hydrolase", "4": "lyase",
           "5": "isomerase", "6": "ligase"}


def signature(rx, sp, smap, formula) -> Verdict:
    def keggs(side):
        out = []
        for s, *_ in rx[side]:
            if sp[s]["kind"] != "small":
                continue
            ks = smap.get(s, {}).get("kegg", [])
            out.append(set(ks))
        return out
    R, P = keggs("reactants"), keggs("products")
    if not R and not P:
        return Verdict(Label.REFUSED, blocker="no small-molecule participant: a macromolecular "
                       "event, outside the six stoichiometric classes")
    if any(not k for k in R + P):
        return absent("a small participant has no KEGG compound")
    has = lambda side, k: any(k in x for x in side)
    hits = set()
    for ox, red in COUPLES:
        if red is None:
            if has(R, ox):
                hits.add("1")
        elif (has(R, ox) and has(P, red)) or (has(R, red) and has(P, ox)):
            hits.add("1")
    core_R = [x for x in R if not x & COFACTORS]
    core_P = [x for x in P if not x & COFACTORS]
    water_in = has(R, "C00001")
    ntp_split = has(R, "C00002") and ((has(P, "C00008") and has(P, "C00009"))
                                      or (has(P, "C00020") and has(P, "C00013")))
    if water_in and len([x for x in P if "C00080" not in x]) > len([x for x in R if not x & {"C00001", "C00080"}]):
        hits.add("3")
    if ntp_split and len(core_R) > len(core_P) and "3" not in hits:
        hits.add("6")
    if (has(R, "C00002") and has(P, "C00008") and len(core_R) == len(core_P)) \
            or (has(R, "C00019") and has(P, "C00021")) or (has(R, "C00024") and has(P, "C00010")) \
            or (has(R, "C00029") and has(P, "C00015")):
        hits.add("2")
    if len(R) == 1 and len(P) == 1 and not (R[0] & P[0]):
        fr = {formula.get(k) for k in R[0]} - {None}
        fp = {formula.get(k) for k in P[0]} - {None}
        if fr & fp:
            hits.add("5")
    if len(R) == 1 and len(P) == 2 and not water_in:
        hits.add("4")
    if not hits:
        return absent("no signature rule applies")
    if len(hits) > 1:
        return Verdict(Label.UNDERDETERMINED, blocker="several class signatures match",
                       detail={"classes": sorted(EC_NAME[h] for h in hits)})
    h = hits.pop()
    return answered(h, name=EC_NAME[h])


def signature_test(corpus, smap, labels, formula, n_shuffle=200, seed=0) -> dict:
    sp = corpus["species"]
    sig = {rid: signature(r, sp, smap, formula) for rid, r in corpus["reactions"].items()}
    pairs = [(v.payload, labels[rid]["EC1"]) for rid, v in sig.items()
             if v.label is Label.ANSWERED and isinstance(labels[rid]["EC1"], str)]
    agree = sum(a == b for a, b in pairs)
    rng = np.random.default_rng(seed)
    ec = [b for _, b in pairs]
    null = np.array([sum(a == b for (a, _), b in zip(pairs, rng.permutation(ec)))
                     for _ in range(n_shuffle)])
    conf = Counter((EC_NAME[a], EC_NAME.get(b, b)) for a, b in pairs)
    return {"labels": dict(Counter(v.label.value for v in sig.values())),
            "classes_assigned": dict(Counter(EC_NAME[v.payload] for v in sig.values()
                                             if v.label is Label.ANSWERED)),
            "underdetermined_combinations": dict(Counter(
                "+".join(v.detail["classes"]) for v in sig.values()
                if v.label is Label.UNDERDETERMINED)),
            "compared_with_single_EC": len(pairs), "agree": agree,
            "agreement": agree / len(pairs) if pairs else None,
            "shuffled_mean_agreement": float(null.mean() / len(pairs)) if pairs else None,
            "empirical_p_real_le_shuffled_upper": float(((null >= agree).sum() + 1) / (n_shuffle + 1)),
            "confusion_top": [f"{a} vs EC {b}: {n}" for (a, b), n in conf.most_common(12)]}
