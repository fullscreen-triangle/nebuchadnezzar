"""python -m circuit_kg {fetch|extract|build} [--radii 1 2]"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

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


def main():
    ap = argparse.ArgumentParser(prog="circuit_kg")
    ap.add_argument("stage", choices=["fetch", "extract", "build", "proteins", "attach"])
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
