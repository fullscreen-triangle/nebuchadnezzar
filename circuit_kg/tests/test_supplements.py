"""Negative controls for the protein supplements."""
from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np

from circuit_kg import attach, enzyme, hess, slot
from circuit_kg.core import Label
from circuit_kg.graph import M, CircuitKG


def _tecr(cols, mu, noise=0.0, seed=0):
    rng = random.Random(seed)
    table = {}
    for col in cols:
        key = tuple(sorted(col.items()))
        table[key] = {"value": sum(v * mu[k] for k, v in col.items()) + rng.gauss(0, noise),
                      "sigma": 0.5, "n": 2}
    return {"reactions": table}


def test_hess_recovers_combination_and_refuses_outside_span():
    mu = {"A": 10.0, "B": -3.0, "C": 7.5, "D": 1.0, "E": -20.0}
    cols = [{"A": -1, "B": 1}, {"B": -1, "C": 1}, {"C": -1, "D": 1, "E": 1}]
    hs = hess.HessSpan(_tecr(cols, mu))
    target = {"A": -1, "D": 1, "E": 1}               # = sum of all three
    v = hess.HessSpan.derive(hs, target)
    assert v.label is Label.ANSWERED
    assert math.isclose(v.payload, mu["D"] + mu["E"] - mu["A"], abs_tol=1e-9)
    assert v.detail["sigma"] > 0
    # control: same compounds, not a combination of measured reactions
    assert hs.derive({"A": -1, "D": 1}).label is Label.ABSENT
    assert hs.derive({"A": -1, "Z": 1}).label is Label.ABSENT


def test_site_parser_formats_and_slot_reaction():
    assert slot.parse_sites("p-S317,467,T575-ULK1") == [("S", 317, "ULK1"), ("S", 467, "ULK1"),
                                                        ("T", 575, "ULK1")]
    assert [r for r, *_ in slot.parse_sites("p-3S,2T-PTEN")] == ["S", "S", "S", "T", "T"]
    assert slot.parse_sites("p-AMPK heterotrimer") == []          # control: 'p-A...' is not a site
    sp = {"p": {"kind": "macromolecule", "uniprot": ["X"], "name": "PROT [cytosol]"},
          "pp": {"kind": "macromolecule", "uniprot": ["X"], "name": "p-S10,S12-PROT [cytosol]"},
          "atp": {"kind": "small", "uniprot": [], "name": "ATP"},
          "adp": {"kind": "small", "uniprot": [], "name": "ADP"}}
    smap = {"atp": {"kegg": ["C00002"]}, "adp": {"kegg": ["C00008"]}}
    rx = {"reactants": [["p", 1, "stated"], ["atp", 2, "stated"]],
          "products": [["pp", 1, "stated"], ["adp", 2, "stated"]], "modifiers": []}
    d = slot.slot_reaction(rx, sp, smap)
    assert d["stoich"] == {"C00065": -2, "C01005": 2, "C00002": -2, "C00008": 2}
    rx["reactants"][1][1] = 1                                        # 1 ATP for 2 named sites
    rx["products"][1][1] = 1
    assert slot.slot_reaction(rx, sp, smap)["verdict"]["label"] == "underdetermined"


def _corpus():
    sp = lambda name, kind="small": {"name": name, "compartment": "cytosol", "chebi": [1],
                                    "uniprot": [] if kind == "small" else ["E1"], "reactome": None,
                                    "kind": kind, "n_parts": 0}
    return {"species": {"S": sp("S"), "P": sp("P"), "Q": sp("Q"), "E": sp("E", "macromolecule")},
            "reactions": {"r1": {"name": "r1", "reversible": False, "reactants": [["S", 1, "stated"]],
                                 "products": [["P", 1, "stated"]], "modifiers": [["E", "catalyst"]]},
                          "r2": {"name": "r2", "reversible": False, "reactants": [["P", 1, "stated"]],
                                 "products": [["Q", 1, "stated"]], "modifiers": [["E", "catalyst"]]}},
            "criterion": "test"}


MEDIUM = {"ambient_chebi": {}, "medium_floor_weight": 1.0, "modifier_weight": 1.0}


def test_enzyme_cycle_conserves_total_and_keeps_net_stoichiometry():
    c = _corpus()
    base = CircuitKG(c, MEDIUM)
    assert base.solve_node("E").label is Label.UNDERDETERMINED
    e = enzyme.expand(c)
    kg = CircuitKG(e, MEDIUM)
    t = enzyme.enzyme_totals(kg)
    assert t == {**t, "catalysts": 1, "enzyme_total_conserved": 1}
    assert kg.solve_node("E").label is Label.ABSENT                # now carries current
    N, sp_nodes, jn = kg.stoichiometric_matrix()
    i = {n[1]: k for k, n in enumerate(sp_nodes)}
    j = {n[1]: k for k, n in enumerate(jn)}
    net = N[:, j["r1#bind0"]] + N[:, j["r1#cat0"]]
    assert net[i["S"]] == -1 and net[i["P"]] == 1 and net[i["E"]] == 0
    # control: if E is also consumed by a junction, its total is not conserved
    c["reactions"]["r3"] = {"name": "r3", "reversible": False, "reactants": [["E", 1, "stated"]],
                            "products": [["Q", 1, "stated"]], "modifiers": []}
    t2 = enzyme.enzyme_totals(CircuitKG(enzyme.expand(c), MEDIUM))
    assert t2["enzyme_total_broken_by_stoichiometric_use"] == 1


def test_attachment_never_lowers_global_sigma_but_replacement_can():
    rng = random.Random(4)
    G = nx.Graph()
    items = list(range(30))
    for a in items:
        G.add_edge(a, M, weight=float(rng.randint(1, 3)), role="medium")
        for b in items[a + 1:]:
            if rng.random() < 0.12:
                G.add_edge(a, b, weight=float(rng.randint(1, 4)), role="x")
    before = {v: attach.global_sigma(G, v) for v in items}
    H = G.copy()                                       # attach: new vertices and edges only
    for k in range(10):
        u = ("res", k)
        H.add_edge(u, M, weight=1.0, role="medium")
        H.add_edge(u, rng.choice(items), weight=float(rng.randint(1, 5)), role="slot")
        if k:
            H.add_edge(u, ("res", k - 1), weight=2.0, role="contact")
    assert all(attach.global_sigma(H, v) >= before[v] - 1e-9 for v in items)
    # control: replacing a hub's contacts by a cheap stand-in lowers someone's sigma
    R = G.copy()
    hub = max(items, key=G.degree)
    for n in list(R[hub]):
        if n != M:
            R[hub][n]["weight"] = 0.1
    assert any(attach.global_sigma(R, v) < before[v] - 1e-9 for v in items)


def test_vectorised_sasa_matches_brute_force_and_isolated_sphere():
    from scipy.spatial import cKDTree
    from circuit_kg import proteins
    rng = np.random.default_rng(3)
    xyz = rng.uniform(0, 12, size=(60, 3))
    elem = np.array(rng.choice(["C", "N", "O", "S"], size=60))
    fast = proteins.sasa(xyz, elem, k=59)
    r = np.array([proteins.RADII[e] for e in elem]) + proteins.PROBE
    pts = proteins._sphere()
    for i in range(len(xyz)):
        q = xyz[i] + r[i] * pts
        others = np.delete(np.arange(len(xyz)), i)
        d = np.linalg.norm(q[:, None] - xyz[others][None], axis=2)
        exact = 4 * np.pi * r[i] ** 2 * (~(d < r[others][None]).any(1)).mean()
        assert abs(exact - fast[i]) < 1e-9
    # control: an isolated atom keeps its full sphere; a buried one loses area
    one = proteins.sasa(np.zeros((1, 3)), np.array(["C"]))
    assert abs(one[0] - 4 * np.pi * (1.70 + 1.4) ** 2) < 1e-9
    assert fast.min() < one[0]
