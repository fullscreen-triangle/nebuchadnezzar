"""Extensions X1-X6: each test has a control that shows it could fail."""
from __future__ import annotations

import math

import numpy as np
import pytest

from circuit_kg import axes, branching, labels, modifiers, redox
from circuit_kg.core import Label, Tag, Verdict, combine


# ---------------------------------------------------------------- X1 axes

def test_basis_composes_as_max_and_absent_dominates():
    assert combine("stated", "stated") is Tag.STATED
    assert combine("stated", "supplied") is Tag.SUPPLIED          # never laundered down
    assert combine("supplied", "derived", "stated") is Tag.DERIVED
    assert combine("stated", "absent") is Tag.ABSENT


def test_gate_letters_only_on_empty_or_absent_and_rules_apply():
    Verdict(Label.ABSENT, blocker="x", gate="-")
    with pytest.raises(ValueError):
        Verdict(Label.ANSWERED, payload=1, gate="+")              # control: answered has no gate
    with pytest.raises(ValueError):
        Verdict(Label.EMPTY, gate="!")
    assert axes.gate_for({"label": "absent", "blocker": "no TECRDB measurement of this reaction"}) == "-"
    assert axes.gate_for({"label": "absent", "blocker": "participant without KEGG id"}) == "open"
    assert axes.gate_for({"label": "certified-empty"}) == "+"
    assert axes.gate_for({"label": "absent", "blocker": "something new"}) == "?"


def test_ground_verdicts_follow_the_provenance_chain():
    P = axes.Prov()
    P.record("m", P.activity("a1", "KineticAssay"))
    P.record("d", P.activity("a2", "RatioComputation", ["m"]))
    P.record("p", P.activity("a3", "StructurePrediction"))
    P.record("c", P.activity("a4", "Curation"))
    P.record("none")
    assert [P.ground(r) for r in ("m", "d", "p", "c", "none")] == \
        ["index", "composite", "icon", "collapsed", "symbol"]


def test_stage_is_invalidated_by_a_convention_change(tmp_path):
    f = tmp_path / "x.txt"; f.write_text("a")
    st = axes.stage({"x": f}, {"pH": [6.8, 7.6]})
    assert axes.is_current(st, {"pH": [6.8, 7.6]})
    assert not axes.is_current(st, {"pH": [7.0, 7.4]})           # control: conventions changed
    f.write_text("b")
    assert not axes.is_current(st, {"pH": [6.8, 7.6]})           # control: content changed


def test_subfloor_names_its_shortfall_spec_example():
    v = axes.judge_ladder([0.62, 0.55, 0.48, 0.41], 0.95)          # 1-(.38*.45*.52*.59)=0.9475
    assert v.label is Label.SUBFLOOR and v.detail["shortfall"] == pytest.approx(0.0025, abs=1e-4)
    assert axes.judge_ladder([0.62, 0.55, 0.48, 0.41], 0.90).label is Label.ANSWERED


def test_a_check_whose_control_passes_is_non_discriminating():
    assert axes.check("x", True, False)["status"] == "pass"
    assert axes.check("x", True, True)["status"] == "non-discriminating"


# ---------------------------------------------------------------- X2 modifiers

def test_inhibitor_shift_and_absent_stack():
    lo, hi = modifiers.inhibitor_shift(1e-6, (1e-6, 1e-2))
    assert lo == pytest.approx(-math.log(1 + 1e4)) and hi == pytest.approx(-math.log(2))
    st = modifiers.Stack()
    st.add("I1", "competitive-inhibitor", (lo, hi), Tag.DERIVED)
    assert st.total().label is Label.ANSWERED
    st.add("I2", "competitive-inhibitor", None, Tag.ABSENT)        # control: a missing constant
    assert st.total().label is Label.ABSENT


def test_variant_reactions_attach_to_their_wild_type_edge():
    corpus = {"species": {"S": {"kind": "small"}, "P": {"kind": "small"}, "E": {"kind": "m"}},
              "reactions": {"wt": {"name": "E converts S", "reactants": [["S", 1, ""]],
                                   "products": [["P", 1, ""]], "modifiers": []},
                            "mut": {"name": "Defective E does not convert S",
                                    "reactants": [["S", 1, ""]], "products": [["S", 1, ""]],
                                    "modifiers": []}}}
    v = modifiers.variant_edges(corpus)
    assert v["mut"]["label"] == "answered" and v["mut"]["payload"] == "wt"


# ---------------------------------------------------------------- X3 branching

def _branch_corpus():
    sp = {"S": {"kind": "small", "uniprot": [], "name": "S"},
          "E1": {"kind": "m", "uniprot": ["P1"], "name": "E1"},
          "E2": {"kind": "m", "uniprot": ["P2"], "name": "E2"}}
    rx = {f"r{i}": {"name": f"E{i} uses S", "reactants": [["S", 1, ""]], "products": [],
                    "modifiers": [[f"E{i}", "catalyst"]]} for i in (1, 2)}
    return {"species": sp, "reactions": rx}, {"S": {"kegg": ["C1"]}}


def test_divider_shares_follow_specificity_times_amount_and_name_missing_siblings():
    corpus, smap = _branch_corpus()
    spec = {("P1", "C1"): 1e5, ("P2", "C1"): 3e5}
    molar = {"P1": 2e-6, "P2": 2e-6}
    d = branching.dividers(corpus, smap, set(), spec, {}, molar)
    assert d["S"]["payload"]["E1 uses S"] == pytest.approx(0.25)
    del spec[("P2", "C1")]                                         # control: one sibling unmeasured
    d = branching.dividers(corpus, smap, set(), spec, {}, molar)
    assert d["S"]["label"] == "absent" and "E2 uses S" in d["S"]["detail"]["missing"]


def test_kcat_pairs_with_km_of_the_same_kinetic_law():
    recs = [{"entry": 1, "kind": "kcat", "value": 10.0, "uniprot": ["P1"], "kegg": [], "role": None},
            {"entry": 1, "kind": "Km", "value": 1e-4, "uniprot": ["P1"], "kegg": ["C1"], "role": "Substrate"},
            {"entry": 2, "kind": "Km", "value": 1e-3, "uniprot": ["P1"], "kegg": ["C2"], "role": "Substrate"}]
    spec, km = branching.rate_tables(recs)
    assert spec == {("P1", "C1"): pytest.approx(1e5)}             # entry 2 has no k_cat: no pair
    assert ("P1", "C2") in km


# ---------------------------------------------------------------- X4 redox

def test_moser_dutton_ruler_decays_with_distance_and_couple_stoichiometry():
    assert redox.moser_dutton(3.0) == redox.moser_dutton(3.6)     # contact floor
    assert redox.moser_dutton(3.6) == pytest.approx(10 ** (13 - 3.1 * 0.7))
    assert redox.moser_dutton(25.0) < 1e-1 < redox.moser_dutton(10.0)
    assert redox.against_reference("glutathione") == {"C00005": -1, "C00127": -1,
                                                      "C00006": 1, "C00051": 2}


# ---------------------------------------------------------------- X5, X6 labels

def test_concordant_labels_split_fewer_classes_than_shuffled_ones():
    classes = [[f"a{i}", f"b{i}"] for i in range(30)]
    aligned = {r: {"EC1": str(i % 6)} for i in range(30) for r in (f"a{i}", f"b{i}")}
    c = labels.concordance(classes, aligned, "EC1", n_shuffle=100)
    assert c["classes_split_real"] == 0 and c["empirical_p_real_le_shuffled"] < 0.05
    rng = np.random.default_rng(3)                                   # control: random labels
    rand = {r: {"EC1": str(rng.integers(6))} for cl in classes for r in cl}
    assert labels.concordance(classes, rand, "EC1", n_shuffle=100)["empirical_p_real_le_shuffled"] > 0.05


def _sig(reactants, products, formula=None):
    sp = {k: {"kind": "small"} for k in set(reactants) | set(products)}
    rx = {"reactants": [[k, 1, ""] for k in reactants], "products": [[k, 1, ""] for k in products]}
    return labels.signature(rx, sp, {k: {"kegg": [k]} for k in sp}, formula or {})


def test_stoichiometric_signatures():
    assert _sig(["C00002", "X"], ["C00008", "XP"]).payload == "2"           # kinase: transferase
    assert _sig(["C00001", "AB"], ["A", "B"]).payload == "3"                # hydrolase
    assert _sig(["G6P"], ["F6P"], {"G6P": "C6H13O9P", "F6P": "C6H13O9P"}).payload == "5"
    assert _sig(["G6P"], ["F6P"], {"G6P": "C6H13O9P", "F6P": "C6H11O9P"}).label is Label.ABSENT
    assert _sig(["C00004", "C00022"], ["C00003", "C00186"]).payload == "1"  # LDH: oxidoreductase
    assert _sig(["C00002", "X", "Y"], ["C00020", "C00013", "XY"]).payload == "6"   # ligase
