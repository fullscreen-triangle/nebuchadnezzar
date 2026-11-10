"""Completion from constraints, and the cost of separating what refinement cannot.

COMPLETE (from *Biochemical trajectory computing* and *Cellular observation
equations*, maxwell): given a source state, a target state and constraints,
return the completions. Here a completion is a set of junctions, each fired in
an allowed direction, that turns the source into the target. The other inputs
are the medium (ambient species, folded into ground) and the proteins that no
reaction makes (taken as present: convention `scope-from-source`; a complex is
made, so a complex holding a small ligand is not a free source of that ligand). The
three outcomes map onto the kernel's verdicts:

  no completion   -> certified-empty (relative to the allowed directions)
  one completion  -> answered (an order-determined ladder: every needed species
                     has exactly one producing junction)
  several         -> underdetermined (the branch points are named)

Forward scope first: the closure of the source under junctions whose inputs are
all available. Then backward from the target: keep only junctions in the scope
that produce a needed species, and require their inputs in turn.

Separation (from *Chromatography as Two-Velocity Transport*, Thm 'orthogonal
dimensions'): two analytes with equal retention cannot be separated by any column
length. Likewise, two nodes in one final refinement class cannot be separated by
any number of rounds. Locating one node among the s in its class needs at least
log2(s) further binary distinctions, which only a new observable can supply.
With n_c resolvable values per key, K items need at least log K / log n_c keys.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from .core import Label, Verdict, answered


def io(rx: dict, d: int):
    ins = rx["reactants"] if d > 0 else rx["products"]
    outs = rx["products"] if d > 0 else rx["reactants"]
    return [s for s, *_ in ins], [s for s, *_ in outs]


class Index:
    """Inputs and outputs of every allowed (junction, direction), for repeated closures."""

    def __init__(self, corpus: dict, allowed: dict, base: set):
        self.ins, self.outs, self.waiting = {}, {}, defaultdict(list)
        self.free = []
        for rid, dirs in allowed.items():
            for d in dirs:
                ins, outs = io(corpus["reactions"][rid], d)
                need = {s for s in ins if s not in base}
                self.ins[(rid, d)], self.outs[(rid, d)] = need, outs
                for s in need:
                    self.waiting[s].append((rid, d))
                if not need:
                    self.free.append((rid, d))


def scope(ix: Index, source: str, base: set, blocked=None):
    """Closure of source + base.  A blocked species is reached but never spent as an input,
    so a completion of the blocked target cannot pass through the target itself."""
    have = set(base)
    missing = {e: len(v) for e, v in ix.ins.items()}
    fired, queue = [], [source] if source is not None else []
    ready = list(ix.free)
    while queue or ready:
        while ready:
            e = ready.pop()
            fired.append(e)
            queue.extend(s for s in ix.outs[e] if s not in have)
        if queue:
            s = queue.pop()
            if s in have:
                continue
            have.add(s)
            if s == blocked:
                continue
            for e in ix.waiting.get(s, ()):
                missing[e] -= 1
                if missing[e] == 0:
                    ready.append(e)
    producers = defaultdict(list)
    for e in fired:
        for s in ix.outs[e]:
            producers[s].append(e)
    return have, producers


def complete(ix: Index, source: str, target: str, sc, base: set) -> Verdict:
    have, _ = sc
    if target not in have:
        return Verdict(Label.EMPTY, detail={"reading": "the target is outside the source's scope "
                                                       "under the allowed directions"})
    _, producers = scope(ix, source, base, blocked=target)      # no route through the target
    needed, used, seen = [target], set(), set()
    branch = {}
    while needed:
        x = needed.pop()
        if x in seen or x == source or x in base:
            continue
        seen.add(x)
        ps = producers[x]
        if len(ps) > 1:
            branch[x] = len(ps)
        for e in ps:
            used.add(e)
            needed.extend(ix.ins[e])
    if not branch:
        return answered(sorted(used), tag="supplied", convention=["scope-from-source"])
    return Verdict(Label.UNDERDETERMINED, blocker="more than one completion: a needed species "
                   "has several producing junctions", detail={"branch_points": len(branch),
                                                               "junctions": len(used)})


def allowed_directions(corpus: dict, medium_direction: dict | None, mode: str) -> dict:
    """as-annotated: Reactome direction only; reversible: both unless thermodynamics decides;
    at-1mM: as reversible, but also fix the sign where Delta_r G'm (1 mM) is fixed across the
    medium window (a counterfactual: what measured concentrations near 1 mM would add)."""
    out = {}
    for rid in corpus["reactions"]:
        if mode == "as-annotated":
            out[rid] = (1,)
            continue
        v = (medium_direction or {}).get(rid, {})
        sign = v.get("payload") if v.get("label") == "answered" else None
        if mode == "at-1mM" and sign is None:
            sign = v.get("dG_prime_m_sign_fixed")
        out[rid] = (1,) if sign == "forward" else (-1,) if sign == "reverse" else (1, -1)
    return out


def survey(corpus: dict, base: set, sources, targets, allowed: dict) -> dict:
    ix = Index(corpus, allowed, base)
    without = scope(ix, None, base)[0]          # what the medium and proteins make on their own
    labels = Counter()
    scope_sizes = []
    for s in sources:
        sc = scope(ix, s, base)
        scope_sizes.append(len((sc[0] - without) & targets))
        for t in targets:
            if t == s:
                continue
            if t in without:
                labels["reachable-without-the-source"] += 1
                continue
            labels[complete(ix, s, t, sc, base).label.value] += 1
    scope_sizes.sort()
    return {"sources": len(sources), "pairs": sum(labels.values()), "labels": dict(labels),
            "scope_median_targets": scope_sizes[len(scope_sizes) // 2] if scope_sizes else 0}


def separation(rounds: list, keys1: dict) -> dict:
    final = rounds[-1]
    K = len(final)
    sizes = Counter(final.values())
    bits = sum(s * math.log2(s) for s in sizes.values()) / K
    n_c = len({(round(k["sigma"], 9), k["delta"]) for k in keys1.values()})
    return {"items": K, "final_classes": len(sizes),
            "items_in_shared_classes": sum(s for s in sizes.values() if s > 1),
            "mean_bits_left_per_item": bits,
            "max_bits_left": math.log2(max(sizes.values())),
            "initial_key_values": n_c,
            "min_independent_keys_log_K_over_log_nc": math.log(K) / math.log(n_c)}
