"""COMPLETE: one completion answers, several are underdetermined, none is certified empty."""
from __future__ import annotations

from circuit_kg import complete as cp
from circuit_kg.core import Label


def _corpus(extra=()):
    rx = {"r1": (["A"], ["B"]), "r2": (["B"], ["C"]), "r3": (["X", "A"], ["Y"])}
    rx.update(dict(extra))
    return {"reactions": {k: {"reactants": [(s, 1, None) for s in i],
                              "products": [(s, 1, None) for s in o]} for k, (i, o) in rx.items()}}


def _verdict(corpus, s, t, mode="as-annotated", medium=None, base=frozenset({"M"})):
    allowed = cp.allowed_directions(corpus, medium, mode)
    ix = cp.Index(corpus, allowed, set(base))
    return cp.complete(ix, s, t, cp.scope(ix, s, set(base)), set(base))


def test_chain_answers_branch_underdetermines_missing_cosubstrate_is_empty():
    c = _corpus()
    v = _verdict(c, "A", "C")
    assert v.label is Label.ANSWERED and v.payload == [("r1", 1), ("r2", 1)]
    assert _verdict(c, "A", "Y").label is Label.EMPTY          # X never available
    branched = _corpus({"r4": (["A"], ["D"]), "r5": (["D"], ["C"])})
    v = _verdict(branched, "A", "C")
    assert v.label is Label.UNDERDETERMINED and v.detail["branch_points"] == 1


def test_direction_constraints_change_completions():
    c = _corpus()
    assert _verdict(c, "C", "A").label is Label.EMPTY           # as annotated: no way back
    assert _verdict(c, "C", "A", mode="reversible").label is Label.ANSWERED
    fixed = {"r2": {"label": "answered", "payload": "forward"}}
    assert _verdict(c, "C", "A", mode="reversible", medium=fixed).label is Label.EMPTY


def test_targets_made_without_the_source_are_set_aside():
    c = _corpus({"r6": (["M"], ["C"])})                          # the medium makes C itself
    s = cp.survey(c, {"M"}, ["A"], {"B", "C", "Y"}, cp.allowed_directions(c, None, "as-annotated"))
    assert s["labels"]["reachable-without-the-source"] == 1
    assert s["labels"]["answered"] == 1 and s["labels"]["certified-empty"] == 1
