"""Edge modifiers as an additive stack on the log-rate coordinate (P450 monograph,
pharmacogenomics and DDI papers; standard pharmacology underneath).

A junction's rate prefactor k carries a stack of named, typed shifts on ln k:
  competitive inhibitor   ln k -> ln k - ln(1 + [I]/K_i)
  inducer (enzyme total)  ln k -> ln k + ln E_fold
  mechanism-based loss    a sink current on the enzyme total (Kitz-Wilson)
Shifts add; a modifier whose constant is not in any source makes the stack `absent`
for that edge rather than silently defaulting to a factor of 1.

Enzyme totals that the stoichiometry does not conserve are classified by what breaks
them -- those are exactly the boundary currents (synthesis, degradation, assembly,
modification) on which induction and inactivation would act.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .core import Label, Tag, Verdict, absent, answered, combine


class Stack:
    """Named additive shifts on ln k, each with a range over the concentration window."""

    def __init__(self):
        self.items = []

    def add(self, name: str, kind: str, shift_range, tag: Tag, source: str | None = None):
        self.items.append({"name": name, "kind": kind,
                           "shift": list(shift_range) if shift_range else None,
                           "tag": Tag(tag).value, "source": source})

    def total(self) -> Verdict:
        if any(i["shift"] is None for i in self.items):
            miss = [i["name"] for i in self.items if i["shift"] is None]
            return absent("a modifier's constant is in no source", remedy="measure it",
                          missing=miss)
        lo = sum(i["shift"][0] for i in self.items)
        hi = sum(i["shift"][1] for i in self.items)
        return answered([lo, hi], tag=combine(*[i["tag"] for i in self.items]).value,
                        modifiers=len(self.items))


def inhibitor_shift(Ki: float, window) -> tuple[float, float]:
    """ln k shift range of a competitive inhibitor over its concentration window (<= 0)."""
    lo, hi = window
    return (-math.log1p(hi / Ki), -math.log1p(lo / Ki))


def ki_table(records) -> dict:
    """(uniprot, kegg) -> geometric-mean K_i (M) over SABIO-RK entries, inhibitor role."""
    acc = defaultdict(list)
    for r in records:
        if r["kind"] != "Ki" or r["role"] != "Inhibitor":
            continue
        for u in r["uniprot"]:
            for k in r["kegg"]:
                acc[(u, k)].append(math.log10(r["value"]))
    return {k: 10 ** (sum(v) / len(v)) for k, v in acc.items()}


def modifier_edges(corpus, smap, kis, window) -> dict:
    """Verdict per (reaction, modifier) for every inhibitor and stimulator edge."""
    sp = corpus["species"]
    out = {}
    for rid, r in corpus["reactions"].items():
        cats = [u for s, role in r["modifiers"] if role == "catalyst" for u in sp[s]["uniprot"]]
        for s, role in r["modifiers"]:
            if role == "catalyst":
                continue
            key = f"{rid}|{s}"
            if sp[s]["kind"] != "small":
                out[key] = Verdict(Label.REFUSED, blocker="a macromolecular modifier: binding "
                                   "partner, not a small-molecule inhibitor constant").as_json()
                out[key]["role"] = role
                continue
            if role == "stimulator":
                out[key] = absent("no activation constant in SABIO-RK for this pair",
                                  remedy="an activation constant K_a").as_json()
                out[key]["role"] = role
                continue
            keggs = smap.get(s, {}).get("kegg", [])
            if not keggs:
                out[key] = absent(f"no KEGG compound for {sp[s]['name']}").as_json()
                out[key]["role"] = role
                continue
            hits = sorted({kis[(u, k)] for u in cats for k in keggs if (u, k) in kis})
            if not hits:
                v = absent("no K_i in SABIO-RK for this enzyme-inhibitor pair",
                           remedy="a measured inhibition constant").as_json()
            else:
                Ki = math.exp(sum(map(math.log, hits)) / len(hits))
                lo, hi = inhibitor_shift(Ki, window)
                st = Stack()
                st.add(sp[s]["name"], "competitive-inhibitor", (lo, hi), Tag.DERIVED, "SABIO-RK K_i")
                tot = st.total().as_json()
                strength = ("negligible across the window" if -lo < math.log(1.1)
                            else "strong across the window" if -hi > math.log(2)
                            else "window-dependent")
                v = {**tot, "detail": {**tot.get("detail", {}), "K_i_M": Ki,
                                       "fold_at_window_top": math.exp(-lo),
                                       "fold_at_window_bottom": math.exp(-hi),
                                       "strength": strength}}
            v["role"] = role
            out[key] = v
    return out


BREAKERS = [("degradation", r"degrad|proteasom|ubiquitinat|cleav"),
            ("covalent modification", r"phosphorylat|dephosphorylat|acetylat|methylat|"
                                      r"sumoylat|neddylat|palmitoylat|glycosylat|autophosph"),
            ("assembly or dissociation", r"\bbind|binds|bound|dissociat|associat|complex|"
                                         r"dimeri|oligomer|recruit|form"),
            ("translocation or transport", r"transloc|transport|import|export")]


def classify_breakers(corpus) -> dict:
    """For each catalyst that also appears as a reactant or product somewhere, what the
    reactions that make or consume it are (by Reactome name)."""
    cats = {s for r in corpus["reactions"].values() for s, role in r["modifiers"]
            if role == "catalyst"}
    uses = defaultdict(list)
    for rid, r in corpus["reactions"].items():
        for s, *_ in r["reactants"] + r["products"]:
            if s in cats:
                uses[s].append(r["name"])
    per_enzyme, by_class = {}, Counter()
    for s, names in uses.items():
        kinds = set()
        for n in names:
            k = next((c for c, pat in BREAKERS if re.search(pat, n, re.I)), "other")
            kinds.add(k)
        per_enzyme[s] = sorted(kinds)
        for k in kinds:
            by_class[k] += 1
    return {"catalysts": len(cats), "catalysts_made_or_consumed": len(uses),
            "breaker_classes (an enzyme can have several)": dict(by_class),
            "only_one_class": dict(Counter(v[0] for v in per_enzyme.values() if len(v) == 1))}


def is_variant(name: str) -> bool:
    """Reactome's disease-variant reactions: a loss-of-function copy of a wild-type edge."""
    return name.startswith("Defective") or " does not " in name


def variant_edges(corpus) -> dict:
    """Each loss-of-function variant reaction linked to its wild-type edge (same small-molecule
    reactants), as a modifier on that edge: the variant's shift on ln k is -inf (no flux)."""
    sp = corpus["species"]

    def small_in(r):
        return frozenset(s for s, *_ in r["reactants"] if sp[s]["kind"] == "small")
    wild = defaultdict(list)
    for rid, r in corpus["reactions"].items():
        if not is_variant(r["name"]):
            wild[small_in(r)].append(rid)
    out = {}
    for rid, r in corpus["reactions"].items():
        if not is_variant(r["name"]):
            continue
        cands = wild.get(small_in(r), []) if small_in(r) else []
        if len(cands) == 1:
            out[rid] = answered(cands[0], modifier="loss-of-function", shift="-inf",
                                tag="derived").as_json()
        elif cands:
            out[rid] = Verdict(Label.UNDERDETERMINED, blocker="several wild-type edges share "
                               "its reactants", detail={"candidates": len(cands)}).as_json()
        else:
            out[rid] = absent("no wild-type edge with the same small-molecule reactants").as_json()
    return out
