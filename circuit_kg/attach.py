"""Attach protein contact graphs to the circuit KG through named slots.

Attachment is an expansion: new vertices (residues) and new edges only, no
existing weight changed.  By the monotonicity theorem of positive occupation
propagation, the global separation cost of every item already in the graph
can then only stay or rise.  The control is the tempting alternative the
theorem does not cover: putting real-valued weights (a residue's relative
exposure, in [0, 1]) where counted ones were (`real_weight_replacement`).

Slot edge: junction -- residue vertex, weight 1 (counted; `slot-from-name`).
Residue vertices: counted heavy-atom contact weights; medium weight
max(1, exposed atoms) (`medium-floor-residue`).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

import networkx as nx
from networkx.algorithms.flow import preflow_push

from .graph import M, junction_node, species_node
from .proteins import AF_DIR
from .slot import parse_sites


def _manifest():
    """Per-accession metadata; read from the individual records so a partial fetch works too."""
    out = {}
    for f in AF_DIR.glob("AF-*.json"):
        if f.name.endswith(".graph.json"):
            continue
        m = json.loads(f.read_text())
        out[m["acc"]] = m
    return out


def _graph(acc):
    path = AF_DIR / f"AF-{acc}.graph.json"
    if path.exists():
        return json.loads(path.read_text())
    if (AF_DIR / f"AF-{acc}.pdb.gz").exists() and (AF_DIR / f"AF-{acc}.json").exists():
        from .proteins import summarise
        return summarise(acc)
    return None


def locate_slots(corpus: dict, slots: dict) -> dict:
    """Map each named, positioned new site of a carrier reaction to (acc, pos), and check the residue."""
    man = _manifest()
    gene_to_acc = defaultdict(set)
    for acc, m in man.items():
        if m.get("status") == "ok" and m.get("gene"):
            gene_to_acc[m["gene"].upper()].add(acc)
    out, checks, mismatches = {}, defaultdict(int), []
    for rid, v in slots.items():
        kind = v.get("detail", {}).get("kind")
        if kind not in ("kinase", "phosphatase"):
            continue
        rx = corpus["reactions"][rid]
        sp = corpus["species"]
        mr = [s for s, *_ in rx["reactants"] if sp[s]["kind"] != "small"]
        mp = [s for s, *_ in rx["products"] if sp[s]["kind"] != "small"]
        if not (mr and mp):
            continue
        before = set(parse_sites(sp[mr[0]]["name"]))
        after = set(parse_sites(sp[mp[0]]["name"]))
        changed = (after - before) if kind == "kinase" else (before - after)
        uni = sp[mr[0]]["uniprot"]
        located = []
        for res, pos, gene in changed:
            if pos is None:
                checks["site without position"] += 1
                continue
            if len(uni) == 1:
                acc = uni[0]
            elif gene and len(gene_to_acc.get(gene.upper(), set()) & set(uni)) == 1:
                acc = next(iter(gene_to_acc[gene.upper()] & set(uni)))
            else:
                checks["protein of the site not determined"] += 1
                continue
            g = _graph(acc)
            if g is None:
                checks["no AlphaFold contact graph"] += 1
                continue
            letter = g["letters"].get(str(pos))
            if letter is None:
                checks["position outside the model"] += 1
            elif letter != res:
                checks["residue in sequence differs from name"] += 1
                mismatches.append({"reaction": rid, "acc": acc, "pos": pos, "named": res,
                                   "in_sequence": letter})
            else:
                checks["residue matches sequence"] += 1
                located.append((acc, pos, res))
        if located:
            out[rid] = located
    return {"slots": out, "checks": dict(checks), "mismatches": mismatches}


def attach(kg, located: dict) -> dict:
    """Expand kg.G in place with the residue graphs of slot-bearing proteins."""
    G = kg.G
    proteins = sorted({acc for sites in located.values() for acc, _, _ in sites})
    n_res = n_edges = 0
    for acc in proteins:
        g = _graph(acc)
        for r in g["residues"] if "residues" in g else map(int, g["letters"]):
            node = ("res", f"{acc}:{r}")
            G.add_node(node, kind="residue", name=f"{acc} {g['letters'][str(r)]}{r}")
            exposed = g["exposed"][str(r)]
            G.add_edge(node, M, role="medium", weight=float(max(1, exposed)), exposed=exposed,
                       tag="supplied", convention="medium-floor-residue")
            n_res += 1
        for u, v, w in g["edges"]:
            G.add_edge(("res", f"{acc}:{u}"), ("res", f"{acc}:{v}"), role="contact",
                       weight=float(w), nu=0.0, tag="stated", convention="alphafold-contact-4.5A")
            n_edges += 1
    n_slot = 0
    for rid, sites in located.items():
        for acc, pos, res in sites:
            G.add_edge(junction_node(rid), ("res", f"{acc}:{pos}"), role="slot", weight=1.0, nu=0.0,
                       tag="supplied", convention="slot-from-name")
            n_slot += 1
    return {"proteins": len(proteins), "residues": n_res, "contact_edges": n_edges,
            "slot_edges": n_slot}


def global_sigma(G: nx.Graph, v) -> float:
    F = nx.DiGraph()
    for a, b, d in G.edges(data=True):
        for x, y in ((a, b), (b, a)):
            if F.has_edge(x, y):
                F[x][y]["capacity"] += d["weight"]
            else:
                F.add_edge(x, y, capacity=d["weight"])
    return preflow_push(F, v, M, value_only=True).graph["flow_value"]


def _component_of(G, v):
    H = G.subgraph([n for n in G if n != M])
    comp = nx.node_connected_component(H, v)
    return G.subgraph(comp | {M})


def monotonicity(kg_before_G: nx.Graph, kg_after_G: nx.Graph, items, replaced_G=None) -> dict:
    """Global separation cost of each item before/after (and under the replacement control)."""
    rows = []
    for v in items:
        b = global_sigma(_component_of(kg_before_G, v), v)
        a = global_sigma(_component_of(kg_after_G, v), v)
        r = (global_sigma(_component_of(replaced_G, v), v)
             if replaced_G is not None and v in replaced_G else None)
        rows.append((b, a, r))
    return {"items": len(rows),
            "min_sigma_before": min(b for b, _, _ in rows),
            "min_sigma_attach": min(a for _, a, _ in rows),
            "min_sigma_control": min((r for _, _, r in rows if r is not None), default=None),
            "attach_decreased": sum(a < b - 1e-9 for b, a, _ in rows),
            "attach_increased": sum(a > b + 1e-9 for b, a, _ in rows),
            "control_decreased": sum(r is not None and r < b - 1e-9 for b, _, r in rows),
            "control_increased": sum(r is not None and r > b + 1e-9 for b, _, r in rows)}


def exposure_test(located: dict) -> dict:
    """Are named phosphosites more exposed than other residues of the same type in the same proteins?"""
    from scipy.stats import mannwhitneyu
    named = defaultdict(set)
    for sites in located.values():
        for acc, pos, res in sites:
            named[acc].add(pos)
    site_vals, other_vals = [], []
    for acc, poss in named.items():
        g = _graph(acc)
        types = {g["letters"][str(p)] for p in poss}
        for k, letter in g["letters"].items():
            if letter not in types:
                continue
            (site_vals if int(k) in poss else other_vals).append(g["exposed"][k])
    if not site_vals:
        return {"sites": 0}
    u = mannwhitneyu(site_vals, other_vals, alternative="greater")
    med = lambda x: sorted(x)[len(x) // 2]
    return {"sites": len(site_vals), "same_type_residues": len(other_vals),
            "median_exposed_atoms_sites": med(site_vals), "median_exposed_atoms_others": med(other_vals),
            "buried_fraction_sites": sum(v == 0 for v in site_vals) / len(site_vals),
            "buried_fraction_others": sum(v == 0 for v in other_vals) / len(other_vals),
            "mann_whitney_U": float(u.statistic), "p_greater": float(u.pvalue)}


def real_weight_replacement(G: nx.Graph, corpus: dict, located: dict) -> nx.Graph:
    """Control: the tempting alternative to counted weights.  Each carrier-junction edge of a
    slot reaction gets the slot residue's relative exposure (exposed atoms / the protein's
    maximum) as its weight -- a real number in [0, 1].  The POP paper predicts that real
    weights lose the floor of one; counted attachment keeps it."""
    H = G.copy()
    sp = corpus["species"]
    for rid, sites in located.items():
        acc, pos, _ = sites[0]
        g = _graph(acc)
        top = max(g["exposed"].values()) or 1
        w = g["exposed"][str(pos)] / top
        j = junction_node(rid)
        rx = corpus["reactions"][rid]
        for s in {s for s, *_ in rx["reactants"] + rx["products"] if sp[s]["kind"] != "small"}:
            n = species_node(s)
            if H.has_edge(j, n):
                H[j][n]["weight"] = max(w, 1e-6)
    return H
