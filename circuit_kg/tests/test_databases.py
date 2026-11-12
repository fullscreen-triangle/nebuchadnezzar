"""The database round: each test has a control that shows it could fail."""
from __future__ import annotations

import json

import pytest

from circuit_kg import axes, features, loaders, localisation, medium, structures, thermo, xref
from circuit_kg.core import Tag, combine


def test_estimated_is_its_own_basis_and_ground():
    assert combine("stated", "estimated") is Tag.ESTIMATED
    assert combine("derived", "estimated") is Tag.ESTIMATED            # never reported as derived
    P = axes.Prov()
    P.record("m", P.activity("a", "EquilibriumMeasurement"))
    P.record("e", P.activity("b", "ComponentContribution", ["m"]))
    P.record("d", P.activity("c", "HessDerivation", ["m"]))
    assert P.ground("e") == "estimated" and P.ground("d") == "composite"   # control


def test_falsifiability_guard_refuses_derived_only_cycles():
    z = {0: 1, 1: -1}
    out = thermo.judge_cycle_guarded(z, [5.0, 5.0], [1.0, 1.0], 2.58, ["derived", "estimated"])
    assert out["verdict"] == "REFUSED"
    out = thermo.judge_cycle_guarded(z, [5.0, 5.0], [1.0, 1.0], 2.58, ["stated", "derived"])
    assert out["verdict"] != "REFUSED"                                  # control: one measurement


def test_human_gem_comparison_detects_a_coefficient_difference():
    sp = {"a": {"kind": "small"}, "b": {"kind": "small"}}
    corpus = {"species": sp, "reactions": {"r": {"name": "r", "reversible": True,
                                                 "reactants": [["a", 2, ""]], "products": [["b", 1, ""]]}}}
    smap = {"a": {"kegg": ["C1"]}, "b": {"kegg": ["C2"]}}
    same = xref.compare_gem(corpus, smap, [{"id": "g", "st": {"C1": -2, "C2": 1}, "reversible": True}])
    assert same["coefficients_agree"] == 1 and same["reversibility_agrees"] == 1
    diff = xref.compare_gem(corpus, smap, [{"id": "g", "st": {"C1": -1, "C2": 1}, "reversible": False}])
    assert diff["coefficients_differ"] == 1 and diff["reversibility_agrees"] == 0


def test_mnx_rescue_only_fills_species_without_kegg(monkeypatch):
    monkeypatch.setattr(xref, "mnx_tables", lambda: ({15: {"MNXM1"}, 16: {"MNXM2"}},
                                                     {"MNXM1": {"C00031"}, "MNXM2": {"C00022"}}))
    corpus = {"species": {"x": {"kind": "small", "chebi": [15]}, "y": {"kind": "small", "chebi": [16]}}}
    out = xref.rescue_kegg(corpus, {"y": {"kegg": ["C00022"]}})
    assert out == {"x": {"kegg": ["C00031"], "tag": "supplied", "convention": ["mnx-xref"]}}


def test_scope_check_counts_support(monkeypatch):
    monkeypatch.setattr(localisation, "hpa_cytosol", lambda: {"G1": {"cytosol": True, "reliability": "x"},
                                                              "G2": {"cytosol": False, "reliability": "x"}})
    monkeypatch.setattr(localisation, "compartments_cytosol", lambda: {"G2": 1.0})
    sp = {"E1": {"uniprot": ["P1"]}, "E2": {"uniprot": ["P2"]}, "E3": {"uniprot": ["P3"]}}
    corpus = {"species": sp, "reactions": {f"r{i}": {"modifiers": [[f"E{i}", "catalyst"]]} for i in (1, 2, 3)}}
    ann = {"P1": {"gene": "G1"}, "P2": {"gene": "G2"}, "P3": {"gene": "G3"}}
    out = localisation.scope_check(corpus, ann)
    assert out["proteins"] == {"supported": 1, "not supported": 1, "no data": 1}


def test_uniprot_feature_parsing_and_slot_offsets():
    text = ('MOD_RES 15; /note="Phosphoserine"; /evidence="ECO:0000269|PubMed:1"; '
            'MOD_RES 40; /note="Phosphothreonine"; /evidence="ECO:0000250|UniProtKB:P1"')
    mods = features._parse(text)
    assert [(m["pos"], m["note"], m["evidence"]) for m in mods] == \
        [(15, "Phosphoserine", ["ECO:0000269"]), (40, "Phosphothreonine", ["ECO:0000250"])]
    feats = {"P1": {"mod_res": mods, "act_site": []}}
    letters = {"15": "S", "40": "T", "12": "A", "37": "A"}
    mism = [{"acc": "P1", "pos": 12, "named": "S"}, {"acc": "P1", "pos": 37, "named": "T"}]
    out = features.slot_check({"r": [("P1", 15, "S")]}, mism, feats, lambda a: letters)
    assert out["fraction"] == 1.0 and out["offsets"]["P1"]["offset"] == 3
    assert out["explained_by_one_offset"] == 2
    out = features.slot_check({}, [{"acc": "P1", "pos": 12, "named": "Y"}], feats, lambda a: letters)
    assert out["explained_by_one_offset"] == 0                           # control: no Y anywhere


def test_contact_pair_shift_breaks_agreement():
    edges = [(1, 5, 2), (2, 9, 1), (3, 30, 4)]
    a = structures.pairs_of(edges)
    assert a == structures.pairs_of([(u, v) for u, v, _ in edges])
    assert not (a & structures.pairs_of(edges, shift=3))                 # control


def test_concentration_loader_and_measured_windows_decide_a_direction(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "CONC_DIR", tmp_path / "none")
    w, why = loaders.concentration_windows()
    assert w is None and why["label"] == "absent"
    d = tmp_path / "conc"; d.mkdir()
    (d / "x.csv").write_text("kegg,low_M,high_M,cell_type,source\nA,1e-3,2e-3,iBMK,test\nB,1e-4,2e-4,iBMK,test\n")
    monkeypatch.setattr(loaders, "CONC_DIR", d)
    w, why = loaders.concentration_windows()
    assert w["A"][:2] == (1e-3, 2e-3)

    class L0:
        def reaction(self, st, *a):
            return 0.0, []
    med = {"concentration_window_M": [1e-6, 1e-2],
           "medium_window": {"pH": [7, 7], "ionic_strength_M": [0.1, 0.1], "pMg": [3, 3], "T_K": 298.15,
                             "accessible_volume_fraction": [1.0, 1.0]}}
    vals = {"r": {"g0": -2.0, "sigma": 0.0, "source": "measured", "st": {"A": -1, "B": 1}}}
    glob = medium.direction_with_medium(vals, L0(), med, 2.58, 2.577)
    meas = medium.direction_with_medium(vals, L0(), med, 2.58, 2.577, windows={k: v[:2] for k, v in w.items()})
    assert glob["r"]["label"] == "undecidable"                           # control: the global window
    assert meas["r"]["label"] == "answered" and meas["r"]["payload"] == "forward"


def test_brenda_pairs_need_a_shared_reference(tmp_path):
    def entry(km_refs):
        return {"protein": {"1": {"id": "1", "organism": "Homo sapiens", "accessions": ["P1"], "references": ["1"]}},
                "turnover_number": [{"value": "glucose", "num_value": 10.0, "proteins": ["1"], "references": ["1"]}],
                "km_value": [{"value": "glucose", "num_value": 0.1, "proteins": ["1"], "references": km_refs}]}
    f = tmp_path / "b.json"
    f.write_text(json.dumps({"release": "x", "version": "2", "data": {"2.7.1.1": entry(["1"])}}))
    rows, _ = loaders.brenda_values(f)
    pairs = loaders.brenda_pairs(rows)
    assert pairs[("P1", "glucose")] == pytest.approx(10.0 / 1e-4)
    f.write_text(json.dumps({"release": "x", "version": "2", "data": {"2.7.1.1": entry(["2"])}}))
    assert loaders.brenda_pairs(loaders.brenda_values(f)[0]) == {}       # control: different paper


def test_feature_notes_may_contain_semicolons():
    text = ('MOD_RES 308; /note="Phosphothreonine; by PDPK1"; /evidence="ECO:0000269|PubMed:9"; '
            'MOD_RES 473; /note="Phosphoserine; by MTOR"; /evidence="ECO:0000269|PubMed:1"')
    mods = features._parse(text)
    assert [(m["pos"], m["note"]) for m in mods] == [(308, "Phosphothreonine; by PDPK1"),
                                                     (473, "Phosphoserine; by MTOR")]
    assert all(features.PHOSPHO.search(m["note"]) for m in mods)
