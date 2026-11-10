"""Negative controls: each test can fail, and a control shows it would."""
from __future__ import annotations

import json
import math
import random

import networkx as nx
import pytest

from circuit_kg import analyses, ladder, solve, thermo
from circuit_kg.core import DERIVED, RAW, RT, Label, Tag, Tagged, Verdict
from circuit_kg.graph import M

# ---------------------------------------------------------------- verdicts


def test_non_answered_verdict_carries_no_payload():
    with pytest.raises(ValueError):
        Verdict(Label.ABSENT, payload=1.0, blocker="x")
    with pytest.raises(ValueError):
        Verdict(Label.UNDECIDABLE)                 # no blocker named
    with pytest.raises(ValueError):
        Tagged(1.0, Tag.SUPPLIED)                  # supplied without convention
    Verdict(Label.EMPTY)                           # well posed, empty: allowed


# ---------------------------------------------------------------- cut keys

def _contact_graph(edges, medium_w=1.0):
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w, role="x")
    for n in list(G):
        G.add_edge(n, M, weight=medium_w, role="medium")
    return G


def _keys(G, r):
    H = G.subgraph([n for n in G if n != M])
    adj = {n: list(H[n]) for n in H}
    return {n: analyses.local_key(G, adj, n, r) for n in H}


def _random_graph(seed, n=40):
    rng = random.Random(seed)
    edges = [(("i", a), ("i", b), float(rng.choice([1, 1, 2, 3])))
             for a in range(n) for b in range(a + 1, n) if rng.random() < 0.08]
    return _contact_graph(edges)


def test_cut_key_relabel_invariant_and_weight_control():
    G = _random_graph(1)
    k = _keys(G, 2)
    perm = list(range(100)); random.Random(7).shuffle(perm)
    relabel = {n: ("j", perm[n[1]]) for n in G if n != M}
    k2 = _keys(nx.relabel_nodes(G, relabel), 2)
    for n, v in k.items():
        assert v == k2[relabel[n]]
    # control: change one internal edge weight next to v -> its sigma changes
    v = max(k, key=lambda n: G.degree(n))
    u = next(w for w in G[v] if w != M)
    G[v][u]["weight"] += 5.0
    assert _keys(G, 2)[v]["sigma"] != k[v]["sigma"]


def test_sigma_is_intensive_but_global_floor_version_is_not():
    G = _contact_graph([(i, i + 1, 2.0) for i in range(12)])
    r, v = 1, 0
    before = _keys(G, r)[v]
    # modify far outside B_{r+1}(v): new item hanging off node 10, cheap medium edge
    G.add_edge(10, "far", weight=2.0, role="x")
    G.add_edge("far", M, weight=0.25, role="medium")
    after = _keys(G, r)[v]
    assert before == after
    # control: normalising by the GLOBAL floor is not intensive
    gfloor = lambda g: min(d["weight"] for _, _, d in g.edges(data=True))
    G0 = _contact_graph([(i, i + 1, 2.0) for i in range(12)])
    assert 1 - gfloor(G0) / before["sigma"] != 1 - gfloor(G) / after["sigma"]


# ---------------------------------------------------------------- refinement

class _KG:   # minimal stand-in exposing what refine() reads
    def __init__(self, G):
        self.G = G
        for n in G:
            if n != M:
                G.nodes[n]["kind"] = "item"
                G.nodes[n]["name"] = str(n)

    def item_graph(self):
        return self.G.subgraph([n for n in self.G if n != M])


def test_refinement_respects_automorphism_orbits_and_is_monotone():
    # a star with two identical arms and one longer arm: the identical arms are an orbit
    edges = [("c", "a1", 1.0), ("a1", "a2", 1.0), ("c", "b1", 1.0), ("b1", "b2", 1.0),
             ("c", "d1", 1.0), ("d1", "d2", 1.0), ("d2", "d3", 1.0)]
    G = _contact_graph(edges)
    kg = _KG(G)
    keys = _keys(G, 1)
    rounds = analyses.refine(kg, keys)["rounds"]
    final = rounds[-1]
    assert final["a1"] == final["b1"] and final["a2"] == final["b2"]
    assert final["a1"] != final["d1"]          # control: non-orbit nodes are split
    for prev, cur in zip(rounds, rounds[1:]):
        parent = {}
        for n in cur:
            parent.setdefault(cur[n], set()).add(prev[n])
        assert all(len(p) == 1 for p in parent.values())


# ---------------------------------------------------------------- cycles

def _random_reactions(seed, n_species=8, n_rx=14):
    rng = random.Random(seed)
    sp = [f"C{i:05d}" for i in range(n_species)]
    cols = []
    while len(cols) < n_rx:
        a, b, c = rng.sample(sp, 3)
        col = {a: -1, b: 1} if rng.random() < 0.5 else {a: -1, b: 1, c: 1}
        if col not in cols:
            cols.append(col)
    mu0 = {s: rng.uniform(-500, 500) for s in sp}
    return cols, mu0


def _dg(col, mu0):
    return sum(c * mu0[s] for s, c in col.items())


def test_consistent_cycles_pass_and_planted_defect_is_flagged():
    cols, mu0 = _random_reactions(3)
    basis = thermo.cycle_basis(cols)
    assert basis, "fixture must contain cycles"
    value = [_dg(c, mu0) for c in cols]
    sigma = [0.5] * len(cols)
    q = 2.5758
    verdicts = [thermo.judge_cycle(z, value, sigma, q)["verdict"] for z in basis]
    assert all(v != "INCONSISTENT" for v in verdicts)
    value[0] += 25.0                                # planted defect, 50 sigma
    hit = [z for z in basis if 0 in z]
    assert hit
    assert all(thermo.judge_cycle(z, value, sigma, q)["verdict"] == "INCONSISTENT" for z in hit)


def test_species_level_errors_cancel_around_cycles():
    cols, mu0 = _random_reactions(5)
    basis = thermo.cycle_basis(cols)
    noisy = {s: v + random.Random(9).gauss(0, 30) for s, v in mu0.items()}
    for z in basis:
        h_true = sum(v * _dg(cols[j], mu0) for j, v in z.items())
        h_noisy = sum(v * _dg(cols[j], noisy) for j, v in z.items())
        assert abs(h_true) < 1e-8 and abs(h_noisy) < 1e-8
    # control: reaction-level error does not cancel
    j = next(iter(basis[0]))
    assert abs(sum(v * (_dg(cols[k], mu0) + (3.0 if k == j else 0)) for k, v in basis[0].items())) > 1


# ---------------------------------------------------------------- ladders

def test_ladder_powers_telescope_and_shares_are_flux_control_coefficients():
    G = [2.0, 0.5, 1.0, 4.0]
    G_M = 0.8
    R = [1 / g for g in G]
    R_M = 1 / G_M
    pis = ladder.rung_powers(R, R_M)
    assert math.isclose(ladder.composite(pis), 1 - R_M / (sum(R) + R_M))
    J = ladder.chain_flux(G, G_M)
    assert math.isclose(J, 10.0 / (sum(R) + R_M), rel_tol=1e-9)
    h = 1e-6
    for k, share in enumerate(ladder.shares(R, R_M)):
        Gp = list(G); Gp[k] *= math.exp(h)
        fd = (math.log(ladder.chain_flux(Gp, G_M)) - math.log(J)) / h
        assert math.isclose(fd, share, rel_tol=1e-4)
    # control: shares are not uniform when conductances differ
    assert len({round(s, 6) for s in ladder.shares(R, R_M)}) > 1


def test_mass_action_relaxation_matches_analytic_and_detailed_balance():
    # A (fixed) -> B -> C (fixed), J1 = k1(a - b/K1), J2 = k2(b - c/K2)
    k1, K1, k2, K2 = 2.0, 5.0, 0.7, 3.0
    a, c = 1.0, 0.2
    net = solve.Network({"A": {"fixed": math.log(a)}, "B": {"fixed": None},
                         "C": {"fixed": math.log(c)}},
                        {"r1": {"nu": {"A": -1, "B": 1}, "k": k1, "K": K1},
                         "r2": {"nu": {"B": -1, "C": 1}, "k": k2, "K": K2}})
    x, _ = solve.relax_mass_action(net)
    b = (k1 * a + k2 * c / K2) / (k1 / K1 + k2)
    assert math.isclose(math.exp(x["B"]), b, rel_tol=1e-9)
    # detailed balance: ends at the equilibrium ratio -> zero flux
    net.species["C"]["fixed"] = math.log(a * K1 * K2)
    x, _ = solve.relax_mass_action(net)
    assert abs(solve.mass_action_flux(net, x, "r1")) < 1e-9


# ---------------------------------------------------------------- real data

needs_data = pytest.mark.skipif(not (DERIVED / "reactome_cytosol.json").exists()
                                or not (RAW / "tecrdb_transformed.csv").exists(),
                                reason="run fetch and extract first")


@needs_data
def test_extraction_criterion_holds():
    c = json.loads((DERIVED / "reactome_cytosol.json").read_text())
    assert len(c["reactions"]) == 2914
    for r in c["reactions"].values():
        for s, *_ in r["reactants"] + r["products"] + r["modifiers"]:
            assert c["species"][s]["compartment"] == "cytosol"


@needs_data
def test_transformed_values_are_in_canonical_orientation():
    t = thermo.load_tecrdb()
    key, sign = thermo.canonical({"C00118": -1, "C00111": 1})      # GAP -> DHAP
    p = t["prime"][key]
    assert sign * p["mean"] < 0         # TPI favours DHAP: Delta_r G'deg ~ -7.5 kJ/mol
    chem = t["reactions"][key]["value"]
    assert sign * chem < 0


def test_rank_cut_keeps_small_genuine_singular_values():
    import numpy as np
    from circuit_kg.analyses import _null_rows
    # tall matrix whose second singular value is 1e-7 of the first: genuine, and below the
    # size-scaled cut 1e-9 * 3000 = 3e-6 that once invented a null vector here
    N = np.zeros((3000, 2))
    N[0, 0], N[1, 1] = 1.0, 1e-7
    blocked, rank = _null_rows(N)
    assert rank == 2 and blocked.all()
    # control: an exactly zero column is a genuine null vector and is not blocked
    N[1, 1] = 0.0
    blocked, rank = _null_rows(N)
    assert rank == 1 and blocked[0] and not blocked[1]
