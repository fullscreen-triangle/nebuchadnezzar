"""The circuit knowledge graph.

Vertices
  species  s   state: potential mu_s = mu0_s + RT ln c_s  (a solution, never stored)
  junction r   state: affinity A_r = -sum_s nu_sr mu_s and flux J_r
  medium   M   ground; ambient species are folded into it

Edges carry element laws, i.e. the differential equations:
  s --nu-- r   stoichiometric contact; contributes nu_sr * J_r to the node law of s
  s --mod-- r  modifier contact (catalyst / stimulator / inhibitor); carries no
               current, enters J_r's prefactor
  r --- M      junction's contact with the reservoir (ambient participants)
  s --- M      every item touches the medium (contact-graph axiom)

Junction element law (mass action in flux-force form; linear tangent G_r A_r):
  J_r = k_r * prod_{reactants} c_s^|nu|  * (1 - exp(-A_r / RT))
Node law (Kirchhoff current law = mass balance), with C_s = c_s / RT:
  C_s dmu_s/dt = sum_r nu_sr J_r            (+ G_ex,s (mu_res,s - mu_s) if exchanged)

A node's value is what solves its law from its contacts.  Without k_r, mu0_s
and concentrations the solve is ABSENT, and the verdict names what is missing.
"""
from __future__ import annotations

import json
from collections import defaultdict

import networkx as nx
import sympy as sp

from .core import MEDIA, RT, Label, Tag, Verdict, absent, answered

M = "M"


def load_medium(name: str = "cytosol") -> dict:
    return json.loads((MEDIA / f"{name}.json").read_text())


def species_node(sid):
    return ("s", sid)


def junction_node(rid):
    return ("r", rid)


class CircuitKG:
    def __init__(self, corpus: dict, medium: dict, thermo_match: dict | None = None):
        self.corpus, self.medium = corpus, medium
        self.thermo = thermo_match or {}
        ambient_chebi = {int(k) for k in medium["ambient_chebi"]}
        self.ambient = {sid for sid, s in corpus["species"].items()
                        if s["kind"] == "small" and set(s["chebi"]) & ambient_chebi}
        self.G = nx.Graph()
        self.G.add_node(M, kind="medium")
        w0, wmod = medium["medium_floor_weight"], medium["modifier_weight"]
        for rid, rx in corpus["reactions"].items():
            j = junction_node(rid)
            self.G.add_node(j, kind="junction", name=rx["name"])
            net = defaultdict(float)
            stoich_tag = {}
            for s, c, tag in rx["reactants"]:
                net[s] -= c
                stoich_tag[s] = tag
            for s, c, tag in rx["products"]:
                net[s] += c
                stoich_tag[s] = tag
            folded = 0.0
            ambient_parts = {}
            for s, nu in net.items():
                if s in self.ambient:
                    folded += abs(nu)
                    ambient_parts[s] = nu
                    continue
                n = species_node(s)
                self.G.add_node(n, kind="species", name=corpus["species"][s]["name"])
                role = "carrier-in-reaction" if nu == 0 else ("reactant" if nu < 0 else "product")
                self.G.add_edge(n, j, role=role, nu=nu, weight=abs(nu) if nu else 1.0,
                                tag=stoich_tag[s])
            for s, role in rx["modifiers"]:
                if s in self.ambient:
                    folded += wmod
                    continue
                n = species_node(s)
                self.G.add_node(n, kind="species", name=corpus["species"][s]["name"])
                if self.G.has_edge(n, j):   # also a participant: keep the stoichiometric edge
                    self.G.edges[n, j]["also_modifier"] = role
                    continue
                self.G.add_edge(n, j, role=role, nu=0.0, weight=wmod, tag=Tag.SUPPLIED.value,
                                convention="modifier-unit-weight")
            self.G.nodes[j]["ambient"] = ambient_parts
            self.G.add_edge(j, M, role="medium", weight=w0 + folded, folded=folded,
                            tag=Tag.SUPPLIED.value, convention="medium-floor+ambient-fold")
        for n, d in self.G.nodes(data=True):
            if d["kind"] == "species":
                self.G.add_edge(n, M, role="medium", weight=w0, folded=0.0,
                                tag=Tag.SUPPLIED.value, convention="medium-floor")

    # ----------------------------------------------------------- views
    def items(self):
        return [n for n in self.G if n != M]

    def item_graph(self):
        """The graph without the medium: what balls are taken in."""
        return self.G.subgraph(self.items())

    def stoichiometric_matrix(self):
        """Internal species x junctions (dense numpy) with index lists."""
        import numpy as np
        sp_nodes = sorted(n for n, d in self.G.nodes(data=True) if d["kind"] == "species")
        jn = sorted(n for n, d in self.G.nodes(data=True) if d["kind"] == "junction")
        si = {n: i for i, n in enumerate(sp_nodes)}
        ji = {n: i for i, n in enumerate(jn)}
        N = np.zeros((len(sp_nodes), len(jn)))
        for u, v, d in self.G.edges(data=True):
            if M in (u, v) or not d.get("nu"):
                continue
            s, r = (u, v) if u[0] == "s" else (v, u)
            N[si[s], ji[r]] = d["nu"]
        return N, sp_nodes, jn

    # ----------------------------------------------------------- laws
    def junction_law(self, rid: str):
        """The element law of junction r as a sympy equation J_r = f(A_r)."""
        j = junction_node(rid)
        k = sp.Symbol(f"k_{rid}", positive=True)
        A = sp.Symbol(f"A_{rid}")
        J = sp.Symbol(f"J_{rid}")
        prefactor = sp.Integer(1)
        affinity = sp.Integer(0)
        parts = dict(self.G.nodes[j]["ambient"])
        for n in self.G[j]:
            if n == M:
                continue
            d = self.G.edges[n, j]
            if d["nu"]:
                parts[n[1]] = d["nu"]
        for s, nu in parts.items():
            mu = sp.Symbol(f"mu_{s}")
            affinity -= nu * mu
            if nu < 0:
                prefactor *= sp.Symbol(f"c_{s}", positive=True) ** abs(nu)
        law = sp.Eq(J, k * prefactor * (1 - sp.exp(-A / RT)))
        return {"law": law, "affinity": sp.Eq(A, affinity),
                "tangent": sp.Eq(J, sp.Symbol(f"G_{rid}", positive=True) * A)}

    def node_law(self, sid: str):
        """KCL at species s: C_s dmu_s/dt = sum_r nu_sr J_r (+ exchange)."""
        n = species_node(sid)
        t = sp.Symbol("t")
        mu = sp.Function(f"mu_{sid}")(t)
        c = sp.Symbol(f"c_{sid}", positive=True)
        rhs = sp.Integer(0)
        for j in self.G[n]:
            if j == M:
                continue
            nu = self.G.edges[n, j]["nu"]
            if nu:
                rhs += nu * sp.Symbol(f"J_{j[1]}")
        return sp.Eq(c / RT * sp.Derivative(mu, t), rhs)

    def solve_node(self, sid: str, params: dict | None = None) -> Verdict:
        """Steady-state value of mu_s from its contacts, or what blocks it."""
        params = params or {}
        n = species_node(sid)
        if not any(self.G.edges[n, j]["nu"] for j in self.G[n] if j != M):
            return Verdict(Label.UNDERDETERMINED,
                           blocker="node law is identically 0 = 0: the species only modulates "
                                   "junctions, so no current constrains its state",
                           remedy="its total amount (a conserved quantity), stated externally")
        missing_k, missing_K, missing_c = [], [], set()
        for j in self.G[n]:
            if j == M or not self.G.edges[n, j]["nu"]:
                continue
            rid = j[1]
            if f"k_{rid}" not in params:
                missing_k.append(rid)
            if rid not in params.get("K", {}) and self.thermo.get(rid, {}).get("label") != "answered":
                missing_K.append(rid)
            for m in self.G[j]:
                if m not in (M, n) and self.G.edges[m, j]["nu"] and m[1] not in params.get("c", {}):
                    missing_c.add(m[1])
            for s in self.G.nodes[j]["ambient"]:
                if s not in params.get("c", {}):
                    missing_c.add(s)
        if not (missing_k or missing_K or missing_c):
            return Verdict(Label.REFUSED, blocker="numerical solve is done by circuit_kg.solve.relax "
                                                   "on a parameterised subnetwork")
        return absent("steady-state potential needs the contacts' laws and states",
                      remedy="supply rate constants k_r (e.g. SABIO-RK), K_r or Delta_r G deg, "
                             "and neighbour concentrations",
                      rate_constants_missing=len(missing_k),
                      equilibrium_constants_missing=len(missing_K),
                      concentrations_missing=len(missing_c))

    # ----------------------------------------------------------- export
    def to_json(self) -> dict:
        def nid(n):
            return n if n == M else f"{n[0]}:{n[1]}"
        nodes = [{"id": nid(n), **{k: v for k, v in d.items()}} for n, d in self.G.nodes(data=True)]
        edges = []
        for u, v, d in self.G.edges(data=True):
            e = {"source": nid(u), "target": nid(v), **d}
            if d["role"] in ("reactant", "product"):
                e["law"] = "contributes nu*J_r to the node law of the species"
            elif d["role"] == "medium":
                e["law"] = "reservoir contact (ground)"
            else:
                e["law"] = "modulates the prefactor of J_r; carries no current"
            edges.append(e)
        return {"nodes": nodes, "edges": edges,
                "junction_law": "J_r = k_r prod_reactants c^|nu| (1 - exp(-A_r/RT)),  "
                                "A_r = -sum nu_sr mu_s",
                "node_law": "c_s/RT dmu_s/dt = sum_r nu_sr J_r"}
