"""Catalysts as closed ladders of enzyme states.

In the base graph a catalyst is a modifier: it enters a junction's prefactor
and carries no current, so its node law is 0 = 0 (underdetermined).  After the
P450 derivation's cycle-as-ladder, each catalysed junction r with catalyst E
is expanded into the minimal closed ladder

  r#bind:  E + reactants -> E.r          (binding rung)
  r#cat:   E.r -> E + products           (catalytic rung; returns to E)

The net stoichiometry of r is unchanged.  E now carries current, its node law
is non-trivial, and E + sum_r E.r is conserved: the enzyme total is a left
null vector of the expanded stoichiometric matrix.  The circulation of the
closed ladder is the turnover flux; its composite power is the coupling.

Convention `enzyme-cycle-2`: two rungs per catalysed junction (the consensus
P450 cycle has nine; Reactome states none).
"""
from __future__ import annotations

import copy

import numpy as np


def expand(corpus: dict) -> dict:
    out = copy.deepcopy(corpus)
    reactions, species = {}, out["species"]
    for rid, rx in corpus["reactions"].items():
        cats = [s for s, role in rx["modifiers"] if role == "catalyst"]
        if not cats:
            reactions[rid] = copy.deepcopy(rx)
            continue
        others = [[s, role] for s, role in rx["modifiers"] if role != "catalyst"]
        for k, e in enumerate(cats):
            es = f"ES:{rid}:{k}"
            species[es] = {"name": f"{species[e]['name']} . {rx['name'][:40]}",
                           "compartment": "cytosol", "chebi": [], "uniprot": species[e]["uniprot"],
                           "reactome": None, "kind": "enzyme-state", "n_parts": 0}
            reactions[f"{rid}#bind{k}"] = {
                "name": f"{rx['name']} [binding rung]", "reversible": False,
                "reactants": [[e, 1.0, "supplied"]] + copy.deepcopy(rx["reactants"]),
                "products": [[es, 1.0, "supplied"]], "modifiers": copy.deepcopy(others),
                "pathways": rx.get("pathways", []), "expanded_from": rid}
            reactions[f"{rid}#cat{k}"] = {
                "name": f"{rx['name']} [catalytic rung]", "reversible": False,
                "reactants": [[es, 1.0, "supplied"]],
                "products": [[e, 1.0, "supplied"]] + copy.deepcopy(rx["products"]),
                "modifiers": [], "pathways": rx.get("pathways", []), "expanded_from": rid}
    out["reactions"] = reactions
    out["criterion"] += "; catalysts expanded to two-rung enzyme cycles (enzyme-cycle-2)"
    return out


def enzyme_totals(kg) -> dict:
    """Check, for each catalyst, that E + sum E.r is a left null vector of N."""
    N, sp_nodes, jn = kg.stoichiometric_matrix()
    si = {n[1]: i for i, n in enumerate(sp_nodes)}
    members = {}
    for n in sp_nodes:
        sid = n[1]
        if sid.startswith("ES:"):
            rid, k = sid.split(":")[1], int(sid.split(":")[2])
            e = [s for s, *_ in kg.corpus["reactions"][f"{rid}#bind{k}"]["reactants"]][0]
            members.setdefault(e, []).append(sid)
    conserved, broken = 0, 0
    for e, es in members.items():
        if e not in si:
            continue
        y = np.zeros(len(sp_nodes))
        y[si[e]] = 1
        for x in es:
            y[si[x]] = 1
        if np.abs(y @ N).max() < 1e-12:
            conserved += 1
        else:
            broken += 1
    return {"catalysts": len(members), "enzyme_total_conserved": conserved,
            "enzyme_total_broken_by_stoichiometric_use": broken,
            "reading": "conserved: the catalyst's total is a structural invariant of the node laws; "
                       "broken: the catalyst is also made or consumed by some junction, so its total "
                       "is not conserved"}
