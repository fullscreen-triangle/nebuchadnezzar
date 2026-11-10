"""python -m circuit_kg {fetch|extract|build} [--radii 1 2]"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict

from .core import DERIVED, RESULTS


def build(radii) -> dict:
    from . import analyses, thermo
    from .graph import CircuitKG, load_medium
    t0 = time.time()
    timing = {}
    corpus = json.loads((DERIVED / "reactome_cytosol.json").read_text())
    medium = load_medium("cytosol")
    chebi = thermo.Chebi()
    smap = thermo.species_kegg(corpus, chebi)
    tecr = thermo.load_tecrdb(pH_window=tuple(medium["pH_window"]))
    match = thermo.match_reactions(corpus, smap, tecr)
    timing["thermo"] = time.time() - t0
    kg = CircuitKG(corpus, medium, match)
    timing["graph"] = time.time() - t0
    structure = analyses.structure(kg)
    timing["structure"] = time.time() - t0
    cycles = analyses.cycle_consistency(match, tecr)
    direction = analyses.direction(kg, match, tecr, medium)
    keys = analyses.cut_keys(kg, radii)
    timing["cut_keys"] = time.time() - t0
    ref = analyses.refine(kg, keys[radii[0]])
    lattice = analyses.lattice_report(kg, ref["rounds"])
    chains = analyses.chains(kg, match)
    nodes = analyses.node_verdicts(kg)
    timing["base"] = time.time() - t0

    # -------- protein supplements: Hess span, carrier/slot, enzyme cycles
    from . import enzyme, hess, slot
    hs = hess.HessSpan(tecr)
    unmeasured = hess.derive_unmeasured(corpus, match, smap, hs)
    slots = slot.carrier_slot(corpus, match, smap, hs)
    ecorpus = enzyme.expand(corpus)
    ekg = CircuitKG(ecorpus, medium, {})
    estructure = analyses.structure(ekg)
    etotals = enzyme.enzyme_totals(ekg)

    def per_reaction(ids):          # map expanded rung junctions back to original reactions
        return {ecorpus["reactions"][i].get("expanded_from", i) for i in ids}
    blocked_compare = {}
    for scen in ("closed", "open"):
        before = set(structure[f"blocked_ids_{scen}"])
        after = per_reaction(estructure[f"blocked_ids_{scen}"])
        blocked_compare[scen] = {"base": len(before), "expanded": len(after),
                                 "newly_blocked": len(after - before),
                                 "unblocked": len(before - after),
                                 "newly_blocked_examples": sorted(
                                     corpus["reactions"][r]["name"] for r in after - before)[:12]}
    for st_ in (structure, estructure):
        for k in ("blocked_ids_closed", "blocked_ids_open"):
            st_[k + "_count"] = len(st_.pop(k))
    enodes = analyses.node_verdicts(ekg)
    timing["supplements"] = time.time() - t0

    def labels(d):
        return dict(Counter(v["label"] for v in d.values()))

    def blockers(d):
        return dict(Counter(v.get("blocker") for v in d.values() if v["label"] != "answered"))

    answered_slots = [{"reaction": corpus["reactions"][r]["name"], "dG0": v["payload"],
                       "sigma": v["detail"]["sigma"], "kind": v["detail"]["kind"],
                       "slots": v["detail"]["slots"]}
                      for r, v in slots.items() if v["label"] == "answered"]
    answered_hess = [{"reaction": corpus["reactions"][r]["name"], "dG0": v["payload"],
                      "sigma": v["detail"]["sigma"]}
                     for r, v in unmeasured.items() if v["label"] == "answered"]
    supplements = {
        "hess_fit": hs.fit(),
        "unmeasured_small_molecule": {"labels": labels(unmeasured), "blockers": blockers(unmeasured),
                                      "answered": answered_hess},
        "carrier_slot": {"labels": labels(slots), "blockers": blockers(slots),
                         "kinds_answered": dict(Counter(v["detail"]["kind"] for v in slots.values()
                                                        if v["label"] == "answered")),
                         "answered": sorted(answered_slots, key=lambda x: x["reaction"])},
        "enzyme_cycles": {"reactions_after_expansion": len(ecorpus["reactions"]),
                          "structure": estructure, "totals": etotals, "node_solves": enodes,
                          "blocked_per_original_reaction": blocked_compare},
        "thermo_coverage": {
            "measured": sum(v["label"] == "answered" for v in match.values()),
            "hess_derived": sum(v["label"] == "answered" for v in unmeasured.values()),
            "carrier_slot_derived": len(answered_slots),
            "of": len(corpus["reactions"])},
    }
    (RESULTS / "supplements.json").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / "supplements.json").write_text(json.dumps(supplements, indent=1, default=str))
    timing["total"] = time.time() - t0

    def key_summary(r):
        rows = keys[r].values()
        return {"distinct_sigma": len({round(k["sigma"], 9) for k in rows}),
                "distinct_cut_keys": len({(round(k["sigma"], 9), k["delta"]) for k in rows}),
                "sigma_max": max(k["sigma"] for k in rows),
                "delta_max": max(k["delta"] for k in rows),
                "ball_max": max(k["ball"] for k in rows),
                "power_distinct": len({round(k["power"], 9) for k in rows})}

    dir_counts = Counter((v["label"], v.get("payload")) for v in direction.values())
    summary = {
        "corpus": {"reactions": len(corpus["reactions"]), "species": len(corpus["species"]),
                   "species_kinds": dict(Counter(s["kind"] for s in corpus["species"].values())),
                   "pathway_files": corpus["pathway_files"], "criterion": corpus["criterion"]},
        "medium": {k: medium[k] for k in ("name", "convention", "medium_floor_weight",
                                          "modifier_weight", "concentration_window_M", "pH_window")},
        "species_kegg_tags": dict(Counter(v["tag"] for v in smap.values())),
        "tecrdb": {"reactions_in_window": len(tecr["reactions"]),
                   "with_prime_in_pH_window": len(tecr["prime"]),
                   "sigma_pool_kJ": tecr["sigma_pool"], "q": tecr["q"]},
        "thermo_match": {"labels": dict(Counter(v["label"] for v in match.values())),
                         "blockers": dict(Counter(v["blocker"] for v in match.values()
                                                  if v["label"] != "answered")),
                         "answered_tags": dict(Counter(
                             f'{v["tag"]}:{"+".join(v["convention"]) or "-"}'
                             for v in match.values() if v["label"] == "answered")),
                         "distinct_tecrdb_reactions": len({tuple(map(tuple, v["key"]))
                                                           for v in match.values()
                                                           if v["label"] == "answered"})},
        "structure": structure,
        "cycle_consistency": cycles,
        "direction": {f"{l}:{p}" if p else l: c for (l, p), c in dir_counts.items()},
        "direction_answered": [{"reaction": corpus["reactions"][r]["name"], **v}
                               for r, v in direction.items()
                               if v["label"] in ("answered", "undecidable")],
        "cut_keys": {f"r={r}": key_summary(r) for r in radii},
        "refinement": lattice,
        "ladder_chains": {"count": len(chains), "longest": chains[:10]},
        "node_solves": nodes,
        "timing_s": timing,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (RESULTS / "graph.json").write_text(json.dumps(kg.to_json(), default=str))
    return summary


def attach_stage(n_items=60, seed=0) -> dict:
    """Attach AlphaFold contact graphs through named slots; check monotonicity and exposure."""
    import random
    from . import analyses, attach, hess, slot, thermo
    from .graph import M, CircuitKG, load_medium
    corpus = json.loads((DERIVED / "reactome_cytosol.json").read_text())
    medium = load_medium("cytosol")
    smap = thermo.species_kegg(corpus, thermo.Chebi())
    tecr = thermo.load_tecrdb(pH_window=tuple(medium["pH_window"]))
    match = thermo.match_reactions(corpus, smap, tecr)
    slots = slot.carrier_slot(corpus, match, smap, hess.HessSpan(tecr))
    kg = CircuitKG(corpus, medium, match)
    loc = attach.locate_slots(corpus, slots)
    G0 = kg.G.copy()
    info = attach.attach(kg, loc["slots"])
    R = attach.real_weight_replacement(kg.G, corpus, loc["slots"])
    near = set()
    for rid in loc["slots"]:
        j = ("r", rid)
        near.add(j)
        near.update(n for n in G0[j] if n != M)
    items = sorted(near)
    random.Random(seed).shuffle(items)
    items = sorted(items[:n_items])
    mono = attach.monotonicity(G0, kg.G, items, R)
    local = {}
    for r in (1, 2):
        H0 = G0.subgraph([n for n in G0 if n != M]); adj0 = {n: list(H0[n]) for n in H0}
        H1 = kg.G.subgraph([n for n in kg.G if n != M]); adj1 = {n: list(H1[n]) for n in H1}
        rows = [(analyses.local_key(G0, adj0, v, r)["sigma"], analyses.local_key(kg.G, adj1, v, r)["sigma"])
                for v in sorted(near)]
        local[f"r={r}"] = {"items": len(rows), "decreased": sum(a < b - 1e-9 for b, a in rows),
                           "increased": sum(a > b + 1e-9 for b, a in rows)}
    out = {"site_checks": loc["checks"], "reactions_with_located_slots": len(loc["slots"]),
           "attached": info, "global_sigma": mono, "local_sigma": local,
           "exposure": attach.exposure_test(loc["slots"])}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "attach.json").write_text(json.dumps(out, indent=1))
    return out


def medium_stage() -> dict:
    """The medium as the eluent: transform check, transfer test (M1), direction (M2, M3)."""
    from . import hess, slot, thermo
    from . import medium as md
    from .core import RT
    from .graph import load_medium
    corpus = json.loads((DERIVED / "reactome_cytosol.json").read_text())
    med = load_medium("cytosol")
    smap = thermo.species_kegg(corpus, thermo.Chebi())
    tecr = thermo.load_tecrdb(pH_window=tuple(med["pH_window"]))
    match = thermo.match_reactions(corpus, smap, tecr)
    hs = hess.HessSpan(tecr)
    unmeasured = hess.derive_unmeasured(corpus, match, smap, hs)
    slots = slot.carrier_slot(corpus, match, smap, hs)
    rows = md.tecrdb_rows()
    values = md.chemical_values(match, tecr, unmeasured, slots)
    metabolites = {k for v in smap.values() for k in v["kegg"]}
    ids = ({k for r in rows for k, _ in r["key"]} | {k for v in values.values() for k in v["st"]}
           | metabolites)
    have = json.loads(md.TABLE.read_text())["compounds"] if md.TABLE.exists() else {}
    if not ids <= set(have):
        if not md.EQ_CACHE.exists():
            raise SystemExit(f"no eQuilibrator compound cache at {md.EQ_CACHE} "
                             "(set CIRCUIT_KG_EQ_CACHE)")
        md.build_table(ids)
    L = md.Legendre()
    d = md.direction_with_medium(values, L, med, tecr["q"]["gaussian"], RT)
    decided = [v for v in d.values() if v["label"] == "answered"]
    out = {
        "legendre_source": L.source,
        "reproduce_syndrome_transform": md.reproduce_transform_column(L, rows),
        "controls": {
            "pH_plus_0.1": md.reproduce_transform_column(
                L, rows, perturb=lambda c: {**c, "pH": c["pH"] + 0.1}),
            "no_debye_hueckel": md.reproduce_transform_column(
                L, rows, perturb=lambda c: {**c, "I": 0.0})},
        "M1_transfer": md.transfer_test(rows),
        "M2_direction": {
            "reactions_with_chemical_value": len(values),
            "by_source": dict(Counter(v["source"] for v in values.values())),
            "labels": dict(Counter(f'{v["label"]}:{v.get("payload") or v.get("blocker")}'
                                   for v in d.values())),
            "labels_by_source": dict(Counter(f'{v["source"]}|{v["label"]}:'
                                             f'{v.get("payload") or v.get("blocker")}'
                                             for v in d.values())),
            "answered": sorted(({"reaction": corpus["reactions"][r]["name"], **v}
                                for r, v in d.items() if v["label"] == "answered"),
                               key=lambda x: x["reaction"]),
            "absent_compounds": dict(Counter(c for v in d.values() if v["label"] == "absent"
                                             for c in v["compounds"]))},
        "M3_crowding": {
            "decided_without_crowding": len(decided),
            "verdicts_changed_by_crowding": sum(v.get("crowding_changes_verdict", False)
                                                for v in d.values()),
            "dilute_bound_10mM": md.dilute_bound(L, sorted(metabolites), 1e-2)},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "medium.json").write_text(json.dumps({**out, "per_reaction": d}, indent=1,
                                                    default=str))
    return out


def complete_stage(n_sources=None) -> dict:
    """COMPLETE from source to target under three direction semantics; separation bound."""
    from . import analyses
    from . import complete as cp
    from .graph import CircuitKG, load_medium
    corpus = json.loads((DERIVED / "reactome_cytosol.json").read_text())
    med = load_medium("cytosol")
    mpath = RESULTS / "medium.json"
    if not mpath.exists() or "per_reaction" not in json.loads(mpath.read_text()):
        medium_stage()
    mdir = json.loads(mpath.read_text())["per_reaction"]
    kg = CircuitKG(corpus, med, {})
    sp = corpus["species"]
    used = defaultdict(set)
    for r in corpus["reactions"].values():
        for s, *_ in r["reactants"]:
            used[s].add("in")
        for s, *_ in r["products"]:
            used[s].add("out")
    # proteins present from the start: those no reaction makes (complexes are made, so a
    # complex holding a small ligand is not a free source of that ligand)
    base = set(kg.ambient) | {s for s, v in sp.items()
                              if v["kind"] != "small" and "out" not in used[s]}
    small = sorted(s for s in used if s not in base and sp[s]["kind"] == "small")
    sources = [s for s in small if "in" in used[s]][:n_sources]
    targets = set(small)
    surveys = {}
    for mode in ("as-annotated", "reversible", "at-1mM"):
        t = time.time()
        allowed = cp.allowed_directions(corpus, mdir, mode)
        surveys[mode] = cp.survey(corpus, base, sources, targets, allowed)
        surveys[mode]["fixed_direction_reactions"] = sum(len(v) == 1 for v in allowed.values())
        surveys[mode]["seconds"] = round(time.time() - t, 1)
    keys1 = analyses.cut_keys(kg, [1])[1]
    ref = analyses.refine(kg, keys1)
    out = {"base": {"ambient": len(kg.ambient), "unmade_macromolecules": len(base) - len(kg.ambient)},
           "small_species": len(small), "surveys": surveys,
           "separation": cp.separation(ref["rounds"], keys1)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "complete.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(prog="circuit_kg")


def kinetics_stage() -> dict:
    """K_m contacts: retention factors, the locality diagnostic, and test P1."""
    import numpy as np
    from . import kinetics as kn
    from . import thermo
    from .graph import load_medium
    from .proteins import AF_DIR
    corpus = json.loads((DERIVED / "reactome_cytosol.json").read_text())
    med = load_medium("cytosol")
    smap = thermo.species_kegg(corpus, thermo.Chebi())
    src = kn.fetch()
    table = kn.km_table(kn.km_records())
    contacts = kn.kernel_contacts(corpus, smap, table)
    gene_of = {}
    for f in AF_DIR.glob("AF-*.json"):
        if not f.name.endswith(".graph.json"):
            m = json.loads(f.read_text())
            gene_of[m["acc"]] = m.get("gene")
    rs = kn.retention_and_saturation(corpus, contacts, kn.paxdb_ppm(), gene_of,
                                     med["concentration_window_M"])
    desc = {u: d for u in {u for u, _ in table} if (d := kn.protein_descriptors(u)) is not None}
    p1 = kn.p1_test(table, desc)
    logn = {u: math.log10(d[0]) for u, d in desc.items()}
    m, s = np.mean(list(logn.values())), np.std(list(logn.values()))
    control = kn.p1_test(table, desc, plant=lambda u: 0.5 * (logn[u] - m) / s)
    with_rows = [r for r in contacts.values() if r]
    out = {
        "sources": {"sabio_total_entries": src["sabio"]["total_count"],
                    "km_pairs": len(table), "km_proteins": len({u for u, _ in table})},
        "kernel_join": {"catalysed_reactions": len(contacts),
                        "with_a_measured_km": len(with_rows),
                        "contacts": sum(len(r) for r in with_rows)},
        "retention": rs["retention"], "saturation": rs["saturation"],
        "P1": p1, "P1_positive_control_planted_size_effect": control,
        "P1_descriptors": kn.DESCRIPTORS,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "kinetics.json").write_text(json.dumps(
        {**out, "per_metabolite": rs["per_metabolite"], "per_enzyme": rs["per_enzyme"]},
        indent=1, default=str))
    return out


def main():
    ap = argparse.ArgumentParser(prog="circuit_kg")
    ap.add_argument("stage", choices=["fetch", "extract", "build", "proteins", "attach", "medium",
                                          "kinetics", "complete"])
    ap.add_argument("--radii", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.stage == "fetch":
        from .sources import fetch
        print(json.dumps(fetch(a.force), indent=1))
    elif a.stage == "extract":
        from .extract import extract
        c = extract()
        print(f"{len(c['reactions'])} reactions, {len(c['species'])} species "
              f"from {c['pathway_files']} pathway files")
    elif a.stage == "proteins":
        from . import proteins
        acc = sorted({u for sp in json.loads((DERIVED / "reactome_cytosol.json").read_text())
                      ["species"].values() for u in sp["uniprot"]})
        man = proteins.fetch_alphafold(acc)
        ok = [x for x, m in man.items() if m["status"] == "ok"]
        for x in ok:
            proteins.summarise(x)
        print(f"{len(ok)}/{len(acc)} AlphaFold models with contact graphs")
    elif a.stage == "attach":
        print(json.dumps(attach_stage(), indent=1))
    elif a.stage == "complete":
        print(json.dumps(complete_stage(), indent=1))
    elif a.stage == "kinetics":
        print(json.dumps(kinetics_stage(), indent=1, default=str))
    elif a.stage == "medium":
        o = medium_stage()
        o["M2_direction"].pop("answered")
        print(json.dumps({k: o[k] for k in ("reproduce_syndrome_transform", "M2_direction",
                                            "M3_crowding")}, indent=1))
        m1 = o["M1_transfer"]
        print("M1 replicate", m1["replicate_noise"])
        for dsg, x in m1["designs"].items():
            print(dsg, {m: round(s["mean"], 2) for m, s in x["all"].items()},
                  json.dumps(x["paired_per_reaction"]))
    else:
        s = build(a.radii)
        print(json.dumps({k: s[k] for k in ("thermo_match", "structure", "cut_keys",
                                            "node_solves", "timing_s")}, indent=1, default=str))
        sup = json.loads((RESULTS / "supplements.json").read_text())
        for k in ("hess_fit", "thermo_coverage"):
            print(k, json.dumps(sup[k]))
        for k in ("unmeasured_small_molecule", "carrier_slot"):
            print(k, json.dumps({x: sup[k][x] for x in ("labels", "blockers")}))
        e = sup["enzyme_cycles"]
        print("enzyme_cycles", json.dumps({"totals": e["totals"], "node_solves": e["node_solves"],
                                           "conservation_laws_dim": e["structure"]["conservation_laws_dim"]}))


if __name__ == "__main__":
    main()
