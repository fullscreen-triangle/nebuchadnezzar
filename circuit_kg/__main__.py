"""python -m circuit_kg {fetch|extract|build} [--radii 1 2]"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict

from .core import DERIVED, RAW, RESULTS


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


def folding_stage() -> dict:
    """Levinthal's folding ideas beside AlphaFold: experiments F1-F5 (predictions/PREDICTIONS.md)."""
    from . import attach, hess, slot, thermo
    from . import folding as F
    from .graph import load_medium
    timing, t0 = {}, time.time()
    out = {"ss_check_vs_author_records": F.check_ss(F.fetch_pdb())}
    models = F.high_confidence_models()
    out["F1_contacts"] = F.f1_contacts(F.high_confidence_models(hi=400, limit=None),
                                       F.PDB_DIR / "1CRN.pdb")
    timing["F1"] = time.time() - t0
    out["F2_eigencount"] = F.f2_eigencount(models)
    out["F3_phaselock"] = F.f3_phaselock(models)
    timing["F3"] = time.time() - t0
    corpus = json.loads((DERIVED / "reactome_cytosol.json").read_text())
    med = load_medium("cytosol")
    smap = thermo.species_kegg(corpus, thermo.Chebi())
    tecr = thermo.load_tecrdb(pH_window=tuple(med["pH_window"]))
    match = thermo.match_reactions(corpus, smap, tecr)
    slots = slot.carrier_slot(corpus, match, smap, hess.HessSpan(tecr))
    out["F4_sites"] = F.f4_sites(attach.locate_slots(corpus, slots)["slots"])
    timing["F4"] = time.time() - t0
    out["F5_ternary"] = F.f5_ternary(models)
    timing["total"] = time.time() - t0
    out["timing_s"] = timing
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "folding.json").write_text(json.dumps(out, indent=1))
    return out


def extend_stage() -> dict:
    """Extensions from the P450 monograph and dcat-ap-plus-p450: X1-X6 (predictions file)."""
    import numpy as np
    from . import analyses, annotations, axes, branching, hess, labels, modifiers, redox, slot
    from . import kinetics as kn
    from . import medium as md
    from . import thermo
    from .core import combine
    from .folding import PDB_DIR, fetch_pdb
    from .graph import CircuitKG, load_medium
    t0, timing = time.time(), {}
    corpus = json.loads((DERIVED / "reactome_cytosol.json").read_text())
    med = load_medium("cytosol")
    smap = thermo.species_kegg(corpus, thermo.Chebi())
    tecr = thermo.load_tecrdb(pH_window=tuple(med["pH_window"]))
    match = thermo.match_reactions(corpus, smap, tecr)
    hs = hess.HessSpan(tecr)
    kn.fetch_all()
    recs = kn.parameter_records()
    kc, km = branching.rate_tables(recs)
    kis = modifiers.ki_table(recs)
    sp = corpus["species"]
    ann = annotations.fetch({u for s in sp.values() for u in s["uniprot"]})
    ppm = kn.paxdb_ppm()
    total_M = float(np.mean(kn.TOTAL_PROTEIN_MOLECULES_M))
    molar = {u: ppm[a["gene"].upper()] * 1e-6 * total_M for u, a in ann.items()
             if a and a.get("gene") and a["gene"].upper() in ppm}
    kg = CircuitKG(corpus, med, match)
    out = {"data": {"sabio_records": dict(Counter(r["kind"] for r in recs)),
                    "uniprot": {"accessions": len(ann), "with_EC": sum(bool(a and a["ec"]) for a in ann.values())},
                    "proteins_with_molar_abundance": len(molar)}}
    timing["load"] = time.time() - t0

    # X2 -- modifier stack
    medges = modifiers.modifier_edges(corpus, smap, kis, med["concentration_window_M"])
    out["X2_modifiers"] = {
        "edges": len(medges),
        "labels_by_role": dict(Counter(f'{v["role"]}|{v["label"]}' for v in medges.values())),
        "strength_of_answered": dict(Counter(v["detail"]["strength"] for v in medges.values()
                                             if v["label"] == "answered")),
        "answered": [{"edge": k, **v["detail"]} for k, v in medges.items() if v["label"] == "answered"],
        "enzyme_totals": modifiers.classify_breakers(corpus),
        "loss_of_function_variants": dict(Counter(v["label"] for v in
                                                 modifiers.variant_edges(corpus).values()))}

    # X3 -- current dividers, and ladder floors on chains
    div = branching.dividers(corpus, smap, kg.ambient, kc, km, molar)
    miss = Counter()
    for v in div.values():
        for w in v.get("detail", {}).get("missing", {}).values():
            miss["uncatalysed sibling" if w.startswith("uncatalysed") else
                 "abundance" if "PaxDb" in w else "k_cat or K_m"] += 1
    chains = analyses.chains(kg, match)
    floors = []
    for ch in chains:
        g = []
        for rid in ch["junctions"]:
            rx = corpus["reactions"][rid]
            ks = [k for s, *_ in rx["reactants"] for k in smap.get(s, {}).get("kegg", [])]
            v, _ = branching.specificity(rid, ks, corpus, kc, km, molar)
            g.append(v)
        if all(x is not None for x in g):
            R = [1 / x for x in g]
            from .ladder import rung_powers
            pw = rung_powers(R, max(R))          # convention: medium-as-slowest-rung
            floors.append(axes.judge_ladder(pw, 0.95).as_json())
    out["X3_branching"] = {
        "branch_points": len(div),
        "labels": dict(Counter(v["label"] for v in div.values())),
        "missing_by_kind": dict(miss),
        "answered": {sp[s]["name"]: v["payload"] for s, v in div.items() if v["label"] == "answered"},
        "partial_coverage": float(np.mean([v["detail"]["complete"] / v["detail"]["siblings"]
                                           for v in div.values() if v["label"] == "absent"])),
        "ladder_chains": len(chains),
        "chains_with_all_rung_conductances": len(floors),
        "ladder_floor_verdicts (target 0.95, convention medium-as-slowest-rung)":
            dict(Counter(f["label"] for f in floors)), "floor_details": floors}
    timing["X2_X3"] = time.time() - t0

    # X4 -- redox ladder and the Moser-Dutton ruler
    rids = {k for c in redox.COUPLES.values() for side in c for k in side}
    have = json.loads(md.TABLE.read_text())["compounds"] if md.TABLE.exists() else {}
    if not rids <= set(have):
        md.build_table(set(have) | rids)
    L = md.Legendre()
    lad = redox.couple_ladder(hs, L)
    nad, gsh = lad.get("NAD", {}), lad.get("glutathione", {})
    fetch_pdb(("3GRS", "2J3N"))
    md_rows = {}
    for name, pid, acc, cys in (("glutathione reductase", "3GRS", "P00390", (58, 63)),
                                ("thioredoxin reductase 1", "2J3N", "Q16881", (59, 64))):
        R = redox.flavin_to_cys(PDB_DIR / f"{pid}.pdb", cys)
        kcats = [r["value"] for r in recs if r["kind"] == "kcat" and acc in r["uniprot"]]
        if R is None:
            md_rows[name] = {"label": "absent", "blocker": "no flavin-cysteine distance"}
            continue
        k_et, k_ctrl = redox.moser_dutton(R), redox.moser_dutton(25.0)
        if not kcats:
            md_rows[name] = {"label": "absent", "blocker": "no k_cat in SABIO-RK (human, wild type)",
                             "pdb": pid, "edge_distance_A": R, "k_ET_ruler_per_s": k_et,
                             "k_ET_control_25A_per_s": k_ctrl}
            continue
        kcat = float(np.median(kcats))
        md_rows[name] = {"pdb": pid, "edge_distance_A": R, "k_ET_ruler_per_s": k_et,
                         "k_cat_median_per_s": kcat, "k_cat_records": len(kcats),
                         **axes.check("ruler rate exceeds k_cat", k_et > kcat, k_ctrl > kcat)}
    out["X4_redox"] = {
        "conditions": redox.TEXTBOOK, "ladder_mV_relative_to_NADP": lad,
        "X4a_NAD_vs_NADP_within_10mV": (abs(nad["payload"]) <= 10) if nad.get("label") == "answered" else None,
        "X4b_GSSG_reduction_dG_within_10kJ_of_minus16": (
            gsh["detail"]["dG_prime_kJ"] < 0 and abs(gsh["detail"]["dG_prime_kJ"] + 16) <= 10)
            if gsh.get("label") == "answered" else None,
        "X4c_moser_dutton": md_rows}
    timing["X4"] = time.time() - t0

    # X5 -- labels on structural classes;  X6 -- stoichiometric signatures
    kg0 = CircuitKG(corpus, med, {})
    keys1 = analyses.cut_keys(kg0, [1])[1]
    final = analyses.refine(kg0, keys1)["rounds"][-1]
    classes = labels.shared_junction_classes(final)
    jl = labels.junction_labels(corpus, ann)
    out["X5_labels"] = {
        "concordance_EC1": labels.concordance(classes, jl, "EC1"),
        "concordance_EC3": labels.concordance(classes, jl, "EC3"),
        "discriminating_dimension": labels.discriminating_dimension(classes, jl),
        "bits_left_per_item": {k or "structure only": labels.bits_left(final, jl, k)
                               for k in (None, "EC1", "EC4", "pfam", "gene")}}
    formula = labels.formulas({k for v in smap.values() for k in v["kegg"]}, md.EQ_CACHE)
    out["X6_signatures"] = labels.signature_test(corpus, smap, jl, formula)
    timing["X5_X6"] = time.time() - t0

    # X1 -- verdict axes over everything the kernel has written, including this run
    unmeasured = hess.derive_unmeasured(corpus, match, smap, hs)
    slots = slot.carrier_slot(corpus, match, smap, hs)
    census = Counter()
    live = {"thermo_match": match, "hess": unmeasured, "carrier_slot": slots,
            "modifier_edges": medges, "dividers": div, "variants": modifiers.variant_edges(corpus),
            "redox_ladder": lad}
    if (RESULTS / "medium.json").exists():
        live["direction"] = json.loads((RESULTS / "medium.json").read_text()).get("per_reaction", {})
    for d in live.values():
        census.update(axes.gate_census(list(d.values())))
    census.update(axes.gate_census(floors))
    _, ground = axes.kernel_prov()
    mdir = json.loads((RESULTS / "medium.json").read_text()).get("per_reaction", {}) \
        if (RESULTS / "medium.json").exists() else {}
    basis = Counter(combine("stated" if v.get("source") == "measured" else "derived",
                            "supplied").value for v in mdir.values())
    conv = {"medium": med, "conventions": ["protein-ppm-to-molar", "dilute-limit-share",
                                           "medium-as-slowest-rung", "textbook-standard-conditions",
                                           "moser-dutton-defaults", "scope-from-source"]}
    st = axes.stage({"corpus": DERIVED / "reactome_cytosol.json",
                     "tecrdb": RAW / "tecrdb_transformed.csv", "legendre": md.TABLE,
                     "sabio": kn.SABIO_ALL_DIR / "SOURCES.json", "uniprot": annotations.TABLE},
                    conv)
    altered = json.loads(json.dumps(conv)); altered["medium"]["pH_window"] = [7.0, 7.4]
    out["X1_axes"] = {
        "gate_letters": dict(Counter(g for (_, g, _), n in census.items() for _ in range(n))),
        "gate_by_blocker": [{"label": l, "gate": g, "blocker": b, "n": n}
                            for (l, g, b), n in census.most_common(25)],
        "ground_by_fact_family": ground,
        "direction_verdict_basis": dict(basis),
        "stage": st,
        "stage_check": axes.check("stage is current under its own conventions",
                                  axes.is_current(st, conv), axes.is_current(st, altered))}
    out["timing_s"] = {k: round(v, 1) for k, v in timing.items()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "extend.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def databases_stage() -> dict:
    """Rhea, MetaNetX, Human-GEM, localisation, UniProt features, M-CSA, PDBe, DisProt, PaxDb
    HeLa, and the concentration and BRENDA loaders (predictions R1, X1, L1, S1, C1, P1, D1, G1)."""
    from . import annotations, attach, branching, features, hess, loaders, localisation, slot
    from . import kinetics as kn
    from . import medium as md
    from . import structures, thermo, xref
    from .core import RT
    from .graph import CircuitKG, load_medium
    from .proteins import AF_DIR
    t0, timing = time.time(), {}
    corpus = json.loads((DERIVED / "reactome_cytosol.json").read_text())
    med = load_medium("cytosol")
    chebi = thermo.Chebi()
    smap = thermo.species_kegg(corpus, chebi)
    tecr = thermo.load_tecrdb(pH_window=tuple(med["pH_window"]))
    match = thermo.match_reactions(corpus, smap, tecr)
    hs = hess.HessSpan(tecr)
    sp = corpus["species"]
    out = {}

    # X1 -- MetaNetX supplies KEGG ids; how many identifier misses does it close?
    rescued = xref.rescue_kegg(corpus, smap)
    smap2 = {**smap, **rescued}
    match2 = thermo.match_reactions(corpus, smap2, tecr)
    blocked = {r for r, v in match.items() if v.get("blocker") == "participant without KEGG id"}
    still = {r for r in blocked if match2[r].get("blocker") == "participant without KEGG id"}
    un1 = hess.derive_unmeasured(corpus, match, smap, hs)
    un2 = hess.derive_unmeasured(corpus, match2, smap2, hs)
    out["X1_metanetx"] = {
        "small_species_without_kegg_before": sum(1 for s, v in sp.items() if v["kind"] == "small"
                                                 and not smap.get(s, {}).get("kegg")),
        "species_given_kegg_by_mnx": len(rescued),
        "reactions_blocked_by_missing_kegg": len(blocked), "of_which_unblocked": len(blocked - still),
        "now_matched_to_tecrdb": sum(1 for r in blocked if match2[r]["label"] == "answered"),
        "now_hess_derived": sum(1 for r in blocked if un2.get(r, {}).get("label") == "answered"),
        "new_blockers_of_unblocked": dict(Counter(match2[r].get("blocker") for r in blocked - still)),
        "hess_answered_before_after": [sum(v["label"] == "answered" for v in un1.values()),
                                       sum(v["label"] == "answered" for v in un2.values())]}
    timing["X1"] = time.time() - t0

    # R1 -- Rhea
    mpath = RESULTS / "medium.json"
    mdir = json.loads(mpath.read_text()).get("per_reaction", {}) if mpath.exists() else {}
    out["R1_rhea"] = xref.rhea_against_kernel(corpus, smap, chebi, mdir)
    timing["R1"] = time.time() - t0

    # G1 -- Human-GEM
    out["G1_human_gem"] = xref.compare_gem(corpus, smap2, xref.human_gem())
    timing["G1"] = time.time() - t0

    # L1 -- localisation
    ann = annotations.fetch({u for s in sp.values() for u in s["uniprot"]})
    out["L1_localisation"] = localisation.scope_check(corpus, ann)

    # S1, phospho-intermediates, C1, D1 -- residue evidence
    slots = slot.carrier_slot(corpus, match, smap, hs)
    loc = attach.locate_slots(corpus, slots)
    feats = features.fetch_features({u for s in sp.values() for u in s["uniprot"]})

    def letters_of(acc):
        p = AF_DIR / f"AF-{acc}.graph.json"
        return json.loads(p.read_text())["letters"] if p.exists() else None
    out["S1_slots"] = features.slot_check(loc["slots"], loc["mismatches"], feats, letters_of)
    cat_prots = {u.split("-")[0] for r in corpus["reactions"].values() for s, role in r["modifiers"]
                 if role == "catalyst" for u in sp[s]["uniprot"]}
    out["phospho_enzyme_intermediates"] = features.phospho_intermediates(feats, cat_prots)
    out["C1_catalytic_hubs"] = features.catalytic_hubs(cat_prots)
    timing["S1_C1"] = time.time() - t0
    out["D1_disorder"] = features.disorder_test(loc["slots"], features.disprot_regions())
    timing["D1"] = time.time() - t0

    # P1 -- experimental structures
    p1 = structures.compare(sorted(cat_prots), limit=60)
    out["P1_structures"] = {k: v for k, v in p1.items() if k != "rows"}
    out["P1_rows"] = p1["rows"]
    timing["P1"] = time.time() - t0

    # concentrations and BRENDA -- loaders
    windows, why = loaders.concentration_windows()
    if windows is None:
        out["concentrations"] = why
    else:
        values = md.chemical_values(match, tecr, hess.derive_unmeasured(corpus, match, smap, hs),
                                    slots)
        L = md.Legendre()
        d0 = md.direction_with_medium(values, L, med, tecr["q"]["gaussian"], RT)
        d1 = md.direction_with_medium(values, L, med, tecr["q"]["gaussian"], RT,
                                      windows={k: v[:2] for k, v in windows.items()})
        out["concentrations"] = {"compounds_with_measured_windows": len(windows),
                                 "direction_before": dict(Counter(v["label"] for v in d0.values())),
                                 "direction_after": dict(Counter(v["label"] for v in d1.values()))}
    rows, why = loaders.brenda_values()
    out["brenda"] = why if rows is None else {"values": len(rows),
                                              "same_reference_pairs": len(loaders.brenda_pairs(rows))}

    # per-cell-type abundance: HeLa against whole organism, on the flux dividers
    recs = kn.parameter_records()
    spec, km = branching.rate_tables(recs)
    total_M = float(sum(kn.TOTAL_PROTEIN_MOLECULES_M) / 2)
    kg = CircuitKG(corpus, med, match)

    def molar_from(ppm):
        return {u: ppm[a["gene"].upper()] * 1e-6 * total_M for u, a in ann.items()
                if a and a.get("gene") and a["gene"].upper() in ppm}
    whole = branching.dividers(corpus, smap, kg.ambient, spec, km, molar_from(kn.paxdb_ppm()))
    hela = branching.dividers(corpus, smap, kg.ambient, spec, km,
                              molar_from(kn.paxdb_ppm(RAW / "paxdb-9606-hela-nagaraj-2011-cpc.txt")))
    out["abundance_hela_vs_whole_organism"] = {
        "dividers_answered": [sum(v["label"] == "answered" for v in whole.values()),
                              sum(v["label"] == "answered" for v in hela.values())],
        "shares": {sp[s]["name"]: {"whole_organism": whole[s].get("payload"), "hela": hela[s].get("payload")}
                   for s in whole if whole[s]["label"] == "answered" or hela.get(s, {}).get("label") == "answered"}}
    out["timing_s"] = {k: round(v, 1) for k, v in timing.items()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "databases.json").write_text(json.dumps(out, indent=1, default=str))
    return out


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
                                          "kinetics", "complete",
                                          "folding", "extend", "databases"])
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
    elif a.stage == "databases":
        o = databases_stage()
        o.pop("P1_rows", None)
        print(json.dumps(o, indent=1, default=str)[:15000])
    elif a.stage == "extend":
        o = extend_stage()
        print(json.dumps({k: v for k, v in o.items() if k not in ("X1_axes",)}, indent=1,
                         default=str)[:12000])
        print(json.dumps({k: v for k, v in o["X1_axes"].items() if k != "stage"}, indent=1))
    elif a.stage == "folding":
        print(json.dumps(folding_stage(), indent=1))
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
