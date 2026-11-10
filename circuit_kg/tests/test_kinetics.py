"""Kinetics: K_m parsing, the locality diagnostic, and the P1 test -- each with a control."""
from __future__ import annotations

import gzip
import json
import math
import random

import numpy as np

from circuit_kg import kinetics as kn


def _entry(eid, wild, km, unit="M"):
    return {"id": eid,
            "enzyme_description": {"wildtype": wild, "proteins": [{"uniprot_id": "P1"}]},
            "reaction": {"species": [{"compound": {"id": 7, "name": "Glc"}, "role": "Substrate"}]},
            "external_links": {"compound": [{"id": 7, "key": "KeggCompoundID", "value": "C00031"}]},
            "experimental_conditions": {"envvar_ph": {"start_value": 7.4}},
            "kineticlaw": {"parameter": [
                {"parameter_type": {"name": "Km"}, "n_start_value": km, "unit": {"n_name": unit},
                 "species": {"species_key": "1 | Glc | Substrate"}}]}}


def test_km_records_keep_wild_type_molar_substrate_constants_only(tmp_path, monkeypatch):
    monkeypatch.setattr(kn, "SABIO_DIR", tmp_path)
    page = {"meta": {}, "data": [_entry(1, "Wildtype", 1e-4), _entry(2, "Mutant", 1e-3),
                                 _entry(3, "Wildtype", 5.0, unit="s^(-1)")]}
    (tmp_path / "page-001.json.gz").write_bytes(gzip.compress(json.dumps(page).encode()))
    recs = kn.km_records()
    assert [(r["entry"], r["kegg"], r["km_M"]) for r in recs] == [(1, ["C00031"], 1e-4)]
    t = kn.km_table(recs)
    assert math.isclose(t[("P1", "C00031")]["log10_km"], -4)


def test_locality_labels_follow_the_dilute_threshold():
    corpus = {"species": {"S": {"name": "S"}, "T": {"name": "T"}}}
    contacts = {"r1": [{"substrate": "S", "uniprot": "E1", "log10_km": -1.0}],    # K_m 100 mM
                "r2": [{"substrate": "S", "uniprot": "E2", "log10_km": -7.0},     # K_m 0.1 uM
                       {"substrate": "T", "uniprot": "E2", "log10_km": -7.0}]}
    out = kn.retention_and_saturation(corpus, contacts, {}, {}, (1e-6, 1e-2))["per_enzyme"]
    assert out["E1"]["label"] == "local"          # dilute up to c* = 10 mM
    assert out["E2"]["label"] == "nonlocal"       # control: shared, tight sites saturate at once
    assert out["E2"]["substrates"] == 2 and out["E2"]["departure_at_1mM"] > 0.99


def _p1_fixture(seed, effect):
    rng = random.Random(seed)
    desc = {f"P{i}": [rng.uniform(100, 1000), rng.uniform(50, 95), rng.random(),
                      rng.uniform(5, 15), rng.uniform(1, 6)] for i in range(60)}
    z = {u: (math.log10(d[0]) - 2.5) / 0.3 for u, d in desc.items()}
    table = {}
    for u in desc:
        for k in rng.sample([f"C{j}" for j in range(15)], 5):
            table[(u, k)] = {"log10_km": -4 + 0.1 * int(k[1:]) + effect * z[u] + rng.gauss(0, 0.4),
                             "n": 1, "spread": 0.0}
    return table, desc


def test_p1_finds_a_planted_descriptor_effect_and_not_a_missing_one():
    null = kn.p1_test(*_p1_fixture(1, 0.0), alphas=(1, 100))
    planted = kn.p1_test(*_p1_fixture(1, 0.6), alphas=(1, 100))
    assert planted["wilcoxon_one_sided_p"] < 0.01
    assert null["wilcoxon_one_sided_p"] > 0.05
    assert np.isfinite(null["mean_abs_error_baseline"])
