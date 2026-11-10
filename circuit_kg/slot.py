"""Carrier and slot: thermodynamics for reactions TECRDB refused because a protein takes part.

After the P450 derivation (section "Carrier and slot"): a polypeptide's
backbone is a carrier and carries order; its side-chain slots carry the
specification.  In a reaction where the same protein entity is on both sides
and small molecules change, the protein is the carrier and its identity is
inert; the chemistry happens in slots.  The slot reaction is small-molecule
chemistry and can be put to TECRDB again:

  kinase       X + ATP  -> pX + ADP        (X in S, T, Y)
  phosphatase  pX + H2O -> X + Pi

Conventions (all supplied, all named on the verdict):
  carrier-slot      the carrier's own Delta G contribution is zero
  free-residue      a residue in a protein reacts as the free amino acid
  slot-from-name    new sites are read from Reactome species names
  hess-span         Delta G derived from measured reactions (hess.py)
"""
from __future__ import annotations

import re
from collections import Counter

from .core import Label, Verdict, absent, answered
from .thermo import PROTON

ATP, ADP, H2O, PI = "C00002", "C00008", "C00001", "C00009"
RESIDUE = {"S": ("C00065", "C01005"), "T": ("C00188", "C12147"), "Y": ("C00082", "C06501")}
THREE_TO_ONE = {"Ser": "S", "Thr": "T", "Tyr": "Y"}

_LIST = re.compile(r"(?<![A-Za-z0-9])p-?((?:\d*[STY]\d*|\d+)(?:,\s?(?:\d*[STY]\d*|\d+))*)(?=[-\s:)]|$)")
_PROSE = re.compile(r"\b(Ser|Thr|Tyr)\s?(\d+)")
_ON = re.compile(r"phosphorylated on ([STY])(\d+)")


def parse_sites(name: str) -> list[tuple]:
    """Phosphosites named in a Reactome species name: [(residue, position or None, gene or None)]."""
    sites = []
    for m in _LIST.finditer(name):
        g = re.match(r"-([A-Za-z0-9]+)", name[m.end():])
        gene = g.group(1) if g else None
        last = None
        for tok in re.split(r",\s?", m.group(1)):
            t = re.fullmatch(r"(\d*)([STY])(\d*)", tok)
            if t:
                count, res, pos = t.groups()
                last = res
                if pos:
                    sites.append((res, int(pos), gene))
                else:
                    sites.extend([(res, None, gene)] * (int(count) if count else 1))
            elif tok.isdigit() and last:
                sites.append((last, int(tok), gene))
    for m in _PROSE.finditer(name):
        sites.append((THREE_TO_ONE[m.group(1)], int(m.group(2)), None))
    for m in _ON.finditer(name):
        sites.append((m.group(1), int(m.group(2)), None))
    return sites


def _type_counts(sites):
    return Counter(r for r, _, _ in sites)


def slot_reaction(rx: dict, species: dict, smap: dict) -> dict:
    """Decompose one carrier reaction; returns {'verdict', 'slots', 'stoich'}."""
    R = [(s, c) for s, c, _ in rx["reactants"]]
    P = [(s, c) for s, c, _ in rx["products"]]
    mr = [s for s, _ in R if species[s]["kind"] != "small"]
    mp = [s for s, _ in P if species[s]["kind"] != "small"]
    if not (len(mr) == 1 and len(mp) == 1 and species[mr[0]]["uniprot"]
            and set(species[mr[0]]["uniprot"]) == set(species[mp[0]]["uniprot"])):
        return {"verdict": Verdict(Label.REFUSED, blocker="not a carrier reaction: the protein "
                                   "entity is not the same on both sides").as_json()}
    small = Counter()
    for s, c in R:
        if species[s]["kind"] == "small":
            k = smap.get(s, {}).get("kegg", [])
            small[k[0] if len(k) == 1 else s] -= c
    for s, c in P:
        if species[s]["kind"] == "small":
            k = smap.get(s, {}).get("kegg", [])
            small[k[0] if len(k) == 1 else s] += c
    small = {k: v for k, v in small.items() if v and k != PROTON}
    before = _type_counts(parse_sites(species[mr[0]]["name"]))
    after = _type_counts(parse_sites(species[mp[0]]["name"]))
    if set(small) == {ATP, ADP} and small[ATP] == -small[ADP] and small[ATP] < 0:
        n, kind, new = -small[ATP], "kinase", after - before
    elif set(small) == {H2O, PI} and small[H2O] == -small[PI] and small[H2O] < 0:
        n, kind, new = -small[H2O], "phosphatase", before - after
    else:
        return {"verdict": absent("carrier chemistry is not phosphoryl transfer",
                                  remedy="a slot template for this group transfer",
                                  small_molecule_change=small).as_json()}
    if sum(new.values()) != n:
        return {"verdict": Verdict(
            Label.UNDERDETERMINED,
            blocker=f"names state {sum(new.values())} changed site(s) for {n} {kind} step(s)",
            remedy="site annotation (PSI-MOD) instead of names",
            detail={"kind": kind, "named_change": dict(new)}).as_json()}
    st = Counter()
    for res, k in new.items():
        x, px = RESIDUE[res]
        sign = 1 if kind == "kinase" else -1
        st[x] -= sign * k
        st[px] += sign * k
    if kind == "kinase":
        st[ATP] -= n
        st[ADP] += n
    else:
        st[H2O] -= n
        st[PI] += n
    return {"kind": kind, "slots": dict(new), "stoich": {k: v for k, v in st.items() if v},
            "carrier": mr[0], "product": mp[0]}


def carrier_slot(corpus: dict, match: dict, smap: dict, hess) -> dict:
    """Slot verdicts for every TECRDB-refused reaction."""
    out = {}
    for rid, m in match.items():
        if m["label"] != "refused":
            continue
        d = slot_reaction(corpus["reactions"][rid], corpus["species"], smap)
        if "verdict" in d:
            out[rid] = d["verdict"]
            continue
        v = hess.derive(d["stoich"])
        conv = ["carrier-slot", "free-residue", "slot-from-name"]
        if v.label is Label.ANSWERED:
            v.detail["convention"] = conv + v.detail["convention"]
        v.detail.update({"kind": d["kind"], "slots": d["slots"]})
        out[rid] = v.as_json()
    return out
