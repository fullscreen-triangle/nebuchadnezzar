"""Current dividers at shared intermediates (P450 monograph, atypical-reactions and
C-H activation papers: a junction's apparent rate is a partition among siblings).

At a branch species S consumed by several junctions, in the dilute limit each branch
draws flux in proportion to its specificity times its enzyme amount:

  share_i = (k_cat/K_m)_i [E_i] / sum_j (k_cat/K_m)_j [E_j]

The share is answered only when every sibling has all three numbers; otherwise the
verdict names the sibling and the missing number, because a slow branch cannot be
attributed to its own law without the others.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .core import Label, Verdict, absent, answered


def rate_tables(records):
    """(uniprot, kegg) -> geometric mean of k_cat/K_m (M^-1 s^-1), pairing a k_cat with the
    K_m values of the same SABIO-RK kinetic law (k_cat records name no substrate; the law's
    K_m records do).  Also returns the K_m table alone."""
    by_entry = defaultdict(lambda: {"kcat": [], "Km": []})
    km = defaultdict(list)
    for r in records:
        if r["kind"] == "kcat":
            by_entry[r["entry"]]["kcat"].append(r)
        elif r["kind"] == "Km" and r["role"] == "Substrate":
            by_entry[r["entry"]]["Km"].append(r)
            for u in r["uniprot"]:
                for k in r["kegg"]:
                    km[(u, k)].append(math.log(r["value"]))
    spec = defaultdict(list)
    for e in by_entry.values():
        if not e["kcat"] or not e["Km"]:
            continue
        kcat = math.exp(sum(math.log(x["value"]) for x in e["kcat"]) / len(e["kcat"]))
        for m in e["Km"]:
            for u in m["uniprot"]:
                for k in m["kegg"]:
                    spec[(u, k)].append(math.log(kcat / m["value"]))
    g = lambda d: {k: math.exp(sum(v) / len(v)) for k, v in d.items()}
    return g(spec), g(km)


def specificity(rid, substrate_keggs, corpus, spec, km, molar):
    """sum over a reaction's catalyst proteins of (k_cat/K_m)[E]; or the missing piece."""
    sp = corpus["species"]
    r = corpus["reactions"][rid]
    prots = sorted({u for s, role in r["modifiers"] if role == "catalyst" for u in sp[s]["uniprot"]})
    if not prots:
        return None, "uncatalysed: no catalyst, so no rate law to compare"
    total, missing = 0.0, []
    for u in prots:
        have = [spec[(u, k)] for k in substrate_keggs if (u, k) in spec]
        if not have:
            missing.append(f"{u}: k_cat or K_m for this substrate")
            continue
        E = molar.get(u)
        if E is None:
            missing.append(f"{u}: no PaxDb abundance")
            continue
        total += E * max(have)
    if total == 0.0:
        return None, "; ".join(missing)
    return total, None


def branch_points(corpus, smap, ambient):
    sp = corpus["species"]
    cons = defaultdict(list)
    for rid, r in corpus["reactions"].items():
        for s, *_ in r["reactants"]:
            if sp[s]["kind"] == "small" and s not in ambient:
                cons[s].append(rid)
    return {s: sorted(set(v)) for s, v in cons.items() if len(set(v)) > 1}


def dividers(corpus, smap, ambient, kc, km, molar) -> dict:
    """Healthy-cell dividers: Reactome's loss-of-function variant reactions are modifiers
    on their wild-type edge (modifiers.variant_edges), not competing siblings."""
    from .modifiers import is_variant
    out = {}
    for s, all_sibs in branch_points(corpus, smap, ambient).items():
        sibs = [r for r in all_sibs if not is_variant(corpus["reactions"][r]["name"])]
        if len(sibs) < 2:
            continue
        keggs = smap.get(s, {}).get("kegg", [])
        vals, gaps = {}, {}
        for rid in sibs:
            v, why = specificity(rid, keggs, corpus, kc, km, molar)
            if v is None:
                gaps[rid] = why
            else:
                vals[rid] = v
        if gaps:
            out[s] = absent("a sibling branch lacks k_cat, K_m or abundance",
                            remedy="measure the named sibling", siblings=len(sibs),
                            complete=len(vals), missing={corpus["reactions"][r]["name"]: w
                                                         for r, w in gaps.items()}).as_json()
            continue
        tot = sum(vals.values())
        shares = {corpus["reactions"][r]["name"]: v / tot for r, v in vals.items()}
        out[s] = answered(shares, siblings=len(sibs), tag="derived",
                          convention=["dilute-limit-share", "protein-ppm-to-molar"]).as_json()
    return out


def ladder_from_conductances(g):
    """Rung powers of a chain whose rungs have conductances g_i, with the last rung's
    conductance standing for the medium (ladder.rung_powers on resistances 1/g)."""
    from .ladder import rung_powers
    R = [1 / x for x in g[:-1]]
    return rung_powers(R, 1 / g[-1])
