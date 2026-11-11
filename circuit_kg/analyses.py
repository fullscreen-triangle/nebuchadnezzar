"""What the circuit KG computes without a reasoner.

Every function returns plain JSON-able data; every per-item answer is a
Verdict.  Nothing here reads a name or an identifier to decide anything:
names are attached afterwards, for the reader.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict, deque

import networkx as nx
import numpy as np
from networkx.algorithms.flow import preflow_push

from . import thermo
from .core import RT, Label, Verdict, absent, answered
from .graph import M, CircuitKG

# ===================================================================== structure


def _null_rows(N: np.ndarray, tol=1e-9):
    """Rows of a basis of ker N that are identically zero = blocked columns."""
    if N.shape[1] == 0:
        return np.zeros(0, bool), 0
    if N.shape[0] == 0:
        return np.zeros(N.shape[1], bool), N.shape[1]
    # the right null space needs all n right singular vectors; U is never used, so the
    # m x m factor is only formed when m < n (then it is the small side)
    u, s, vt = np.linalg.svd(N, full_matrices=N.shape[0] < N.shape[1])
    # Rank by a relative cut of 1e-9 placed inside the spectral gap.  A cut scaled by
    # max(shape) (the numpy default form) fell above the smallest genuine singular value of
    # the enzyme-expanded matrix (4.6e-6 relative) and invented null vectors.
    rel = s / s[0] if len(s) and s[0] > 0 else s
    rank = int((rel > tol).sum())
    kept, dropped = rel[:rank], rel[rank:]
    if rank and len(dropped) and dropped.max() > 1e-12 and kept.min() < 1e-6:
        raise ValueError(f"no spectral gap at the rank cut: kept {kept.min():.2e}, "
                         f"dropped {dropped.max():.2e}")
    V0 = vt[rank:].T
    blocked = np.linalg.norm(V0, axis=1) < 1e-8 if V0.size else np.ones(N.shape[1], bool)
    _null_rows.last_gap = (float(kept.min()) if rank else None,
                           float(dropped.max()) if len(dropped) else None)
    return blocked, rank


def structure(kg: CircuitKG) -> dict:
    N, sp_nodes, jn = kg.stoichiometric_matrix()
    nz = (N != 0)
    produced, consumed = (N > 0).any(1), (N < 0).any(1)
    boundary = ~(produced & consumed)
    blocked_closed, rank = _null_rows(N)
    gap = _null_rows.last_gap
    blocked_open, rank_open = _null_rows(N[~boundary])
    roles = Counter()
    for n in sp_nodes:
        stoich = any(kg.G.edges[n, j]["nu"] for j in kg.G[n] if j != M)
        mod = any(kg.G.edges[n, j]["role"] in ("catalyst", "stimulator", "inhibitor")
                  or kg.G.edges[n, j].get("also_modifier") for j in kg.G[n] if j != M)
        roles[("participant" if stoich else "") + ("+" if stoich and mod else "")
              + ("modifier" if mod else "")] += 1
    return {
        "internal_species": len(sp_nodes), "junctions": len(jn),
        "ambient_species_folded": len(kg.ambient),
        "rank_N": rank,
        "spectral_gap": {"smallest_kept_relative": gap[0], "largest_dropped_relative": gap[1]},
        "conservation_laws_dim": len(sp_nodes) - rank,
        "steady_flux_modes_dim": len(jn) - rank,
        "boundary_species": int(boundary.sum()),
        "blocked_if_only_ambient_open": int(blocked_closed.sum()),
        "blocked_if_boundary_species_also_open": int(blocked_open.sum()),
        "species_roles": dict(roles),
        "blocked_ids_closed": [jn[i][1] for i in np.flatnonzero(blocked_closed)],
        "blocked_ids_open": [jn[i][1] for i in np.flatnonzero(blocked_open)],
        "reading": "A blocked junction carries zero flux in every steady state of the node "
                   "laws with the stated ground connections.  The count is a property of the "
                   "ODE system's structure, independent of every rate constant.",
    }


# ===================================================================== cycles


def cycle_consistency(match: dict, tecr: dict) -> dict:
    keys = sorted({tuple(map(tuple, v["key"])) for v in match.values() if v["label"] == "answered"})
    cols = [dict(k) for k in keys]
    basis = thermo.cycle_basis(cols) if cols else []
    value = [tecr["reactions"][k]["value"] for k in keys]
    sigma = [tecr["reactions"][k]["sigma"] for k in keys]
    if not basis:
        v = Verdict(Label.EMPTY, detail={"matched_distinct_reactions": len(keys)})
        return {"verdict": v.as_json(), "cycles": []}
    judged = {name: [thermo.judge_cycle(z, value, sigma, q) for z in basis]
              for name, q in tecr["q"].items()}
    for name, rows in judged.items():
        for z, row in zip(basis, rows):
            row["cycle"] = {" ".join(f"{c:g}*{k}" for k, c in keys[j]): int(v) for j, v in z.items()}
    return {"verdict": answered(len(basis)).as_json(),
            "counts": {n: dict(Counter(r["verdict"] for r in rows)) for n, rows in judged.items()},
            "cycles": judged["gaussian"]}


# ===================================================================== direction


def direction(kg: CircuitKG, match: dict, tecr: dict, medium: dict) -> dict:
    lo, hi = (math.log(x) for x in medium["concentration_window_M"])
    out = {}
    for rid, m in match.items():
        if m["label"] != "answered":
            out[rid] = Verdict(Label(m["label"]) if m["label"] in {l.value for l in Label}
                               else Label.ABSENT, blocker=m["blocker"]).as_json()
            continue
        key = tuple(map(tuple, m["key"]))
        p = tecr["prime"].get(key)
        if p is None:
            out[rid] = absent("no TECRDB transformed value in the pH window",
                              remedy="Legendre transform of Delta_r G deg (needs pKa data)").as_json()
            continue
        sign = m["sign"]
        dmin, dmax = sorted((sign * p["min"], sign * p["max"]))
        st = {k: sign * c for k, c in key if k != thermo.WATER}
        lnq_max = sum(c * (hi if c > 0 else lo) for c in st.values())
        lnq_min = sum(c * (lo if c > 0 else hi) for c in st.values())
        if dmax + RT * lnq_max < 0:
            d = "forward"
        elif dmin + RT * lnq_min > 0:
            d = "reverse"
        else:
            d = None
        detail = {"dG_prime_range": [dmin, dmax], "RT_lnQ_range": [RT * lnq_min, RT * lnq_max],
                  "conventions": ["concentration-window", "pH-window"]}
        out[rid] = (answered(d, **detail) if d else
                    Verdict(Label.UNDECIDABLE, blocker="the concentration window admits both signs "
                            "of the affinity", remedy="measured cytosolic concentrations",
                            detail=detail)).as_json()
    return out


# ===================================================================== cut keys


def _ball(adj: dict, v, r: int) -> set:
    seen, frontier = {v}, [v]
    for _ in range(r):
        nxt = []
        for u in frontier:
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    nxt.append(w)
        frontier = nxt
    return seen


def local_key(G: nx.Graph, adj: dict, v, r: int) -> dict:
    """sigma_r, delta_r (minimal minimiser size), beta_r and derived power at radius r."""
    ball = _ball(adj, v, r)
    F = nx.DiGraph()
    T = ("__sink__",)
    beta = math.inf
    for u in ball:
        for w, d in G[u].items():
            c = d["weight"]
            if w in ball:
                F.add_edge(u, w, capacity=F[u][w]["capacity"] + c if F.has_edge(u, w) else c)
                beta = min(beta, c)
            else:
                F.add_edge(u, T, capacity=F[u][T]["capacity"] + c if F.has_edge(u, T) else c)
                if w == M:
                    beta = min(beta, c)
    R = preflow_push(F, v, T, value_only=False)
    sigma = R.graph["flow_value"]
    seen, q = {v}, deque([v])
    while q:
        u = q.popleft()
        for w, d in R[u].items():
            if w not in seen and d["capacity"] - d["flow"] > 1e-12:
                seen.add(w)
                q.append(w)
    return {"sigma": sigma, "delta": len(seen), "beta": beta,
            "power": 1.0 - beta / sigma, "ball": len(ball)}


def cut_keys(kg: CircuitKG, radii=(1, 2)) -> dict:
    H = kg.item_graph()
    adj = {n: list(H[n]) for n in H}
    return {r: {n: local_key(kg.G, adj, n, r) for n in H} for r in radii}


# ===================================================================== refinement


def refine(kg: CircuitKG, keys1: dict, max_rounds=50) -> dict:
    """Colour refinement from the cut key; each round's partition refines the last.

    Initial colour: (kind, sigma_1, delta_1).  Update: (colour, sorted multiset of
    (edge role, edge weight, neighbour colour)).  Labels are canonical (ranked
    signatures), so a relabelling of items cannot change any class.
    """
    H = kg.item_graph()
    def canon(sig):
        order = {s: i for i, s in enumerate(sorted(set(sig.values())))}
        return {n: order[s] for n, s in sig.items()}
    colour = canon({n: (H.nodes[n]["kind"], round(keys1[n]["sigma"], 9), keys1[n]["delta"])
                    for n in H})
    rounds = [colour]
    for _ in range(max_rounds):
        sig = {n: (colour[n], tuple(sorted((d["role"], d["weight"], colour[w])
                                            for w, d in H[n].items()))) for n in H}
        new = canon(sig)
        if len(set(new.values())) == len(set(colour.values())):
            break
        colour = new
        rounds.append(colour)
    return {"rounds": rounds}


def lattice_report(kg: CircuitKG, rounds: list, top=15) -> dict:
    final = rounds[-1]
    counts = [len(set(c.values())) for c in rounds]
    by_class = defaultdict(list)
    for n, c in final.items():
        if n[0] == "r":
            by_class[c].append(n)
    multi = sorted((v for v in by_class.values() if len(v) > 1), key=len, reverse=True)
    parents = []
    for k in range(1, len(rounds)):
        par = {}
        for n in rounds[k]:
            par.setdefault(rounds[k][n], set()).add(rounds[k - 1][n])
        parents.append(all(len(p) == 1 for p in par.values()))
    return {"classes_per_round": counts,
            "each_round_refines_the_last": all(parents),
            "junction_classes_final": len(by_class),
            "junctions_in_shared_classes": sum(len(v) for v in multi),
            "largest_shared_classes": [
                {"size": len(v), "examples": sorted(kg.G.nodes[n]["name"] for n in v)[:4]}
                for v in multi[:top]]}


# ===================================================================== ladders


def chains(kg: CircuitKG, match: dict, min_len=3) -> list:
    """Maximal unbranched series chains junction -> species -> junction ..."""
    H = kg.item_graph()
    def stoich(n):
        return [(w, d["nu"]) for w, d in H[n].items() if d["nu"]]
    def link_species(s):
        st = stoich(s)
        prod = [j for j, nu in st if nu > 0]
        cons = [j for j, nu in st if nu < 0]
        return (prod[0], cons[0]) if len(st) == 2 and len(prod) == 1 and len(cons) == 1 else None
    def series_junction(j):
        st = stoich(j)
        return sum(nu < 0 for _, nu in st) == 1 and sum(nu > 0 for _, nu in st) == 1
    nxt = {}
    for s in H:
        if s[0] == "s":
            pc = link_species(s)
            if pc and series_junction(pc[0]) and series_junction(pc[1]):
                nxt[pc[0]] = (s, pc[1])
    has_prev = {j for _, j in nxt.values()}
    out = []
    for start in nxt:
        if start in has_prev:
            continue
        path, j = [start], start
        while j in nxt and len(path) < 10000:
            s, j = nxt[j]
            path.append(j)
        if len(path) >= min_len:
            drops = [match.get(j[1], {}).get("dG0") for j in path]
            out.append({
                "rungs": [kg.G.nodes[j]["name"] for j in path],
                "junctions": [j[1] for j in path],
                "rung_powers": absent("rung power needs junction conductances G_r",
                                      remedy="exchange fluxes or rate constants per junction").as_json(),
                "stated_dG0_chemical": drops})
    return sorted(out, key=lambda c: -len(c["rungs"]))


# ===================================================================== nodes


def node_verdicts(kg: CircuitKG) -> dict:
    labels, blockers = Counter(), Counter()
    for n, d in kg.G.nodes(data=True):
        if d.get("kind") != "species":
            continue
        v = kg.solve_node(n[1])
        labels[v.label.value] += 1
        det = v.detail
        missing = tuple(k for k in ("rate_constants_missing", "equilibrium_constants_missing",
                                    "concentrations_missing") if det.get(k))
        blockers[missing] += 1
    return {"labels": dict(labels),
            "missing_kinds": {"+".join(k) or "none": c for k, c in blockers.items()}}
