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


def main():
    ap = argparse.ArgumentParser(prog="circuit_kg")
    ap.add_argument("stage", choices=["fetch", "extract", "build"])
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
    else:
        s = build(a.radii)
        print(json.dumps({k: s[k] for k in ("thermo_match", "structure", "cut_keys",
                                            "refinement", "node_solves", "timing_s")},
                         indent=1, default=str))


if __name__ == "__main__":
    main()
