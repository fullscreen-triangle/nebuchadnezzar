"""
LaTeX macros and the reproduction table, read from the results files (no number is typed in).

Writes ``data/numbers.tex`` (macros used in the text and captions) and
``tables/reproduction.tex`` (the research implementation against the extension).

Run from the nebuchadnezzar root::

    python circuit_kg/paper/journal/scripts/make_numbers.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RES = HERE.parents[1] / "results"

#: Short names for Reactome reactions named in captions (the Reactome name is kept in the data).
SHORT = {
    "reaction_70577": "argininosuccinate synthesis by ASS1",
    "reaction_6810464": "UDP-glucose hydrolysis by NUDT14",
}


def load(path: Path) -> dict:
    """A JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def n(x: float, digits: int = 0) -> str:
    """A number with LaTeX thousands separators."""
    s = f"{x:,.{digits}f}"
    return s.replace(",", "{,}")


def main() -> None:
    """Write the macros and the table."""
    e = load(HERE / "data" / "extension.json")
    s = load(RES / "summary.json")
    sup = load(RES / "supplements.json")
    med = load(RES / "medium.json")["M2_direction"]["labels_by_source"]
    k, cat, cmp_, ctl, sc, sch, sd = (e[x] for x in ("kernel", "catalogue", "comparison", "controls", "scaling",
                                                       "schema", "scidats"))
    dirs = k["directions"]
    answered = [d for d in dirs if d["label"] == "answered"]
    by_source = {src: sum(1 for d in answered if d["source"] == src) for src in ("measured", "hess")}
    kernel_answered = sum(v for key, v in med.items() if key.startswith(("measured|answered", "hess|answered")))
    margin = {
        src: min(-(d["dG_prime_range"][1] + d["RT_lnQ_range"][1]) for d in answered if d["source"] == src)
        for src in ("measured", "hess")
    }
    gates: dict[str, int] = {}
    for q in k["gates"].values():
        for g, c in q.items():
            gates[g] = gates.get(g, 0) + c
    pts = [(r["triples"], r["validate_s"]) for r in sc] + [(cat["triples_total"], cat["timing_s"]["validate"])]
    slope = float(np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0])
    core = "https://w3id.org/nfdi-de/dcat-ap-plus"
    chem = sum(v for key, v in sch["classes_by_schema"].items() if "/chemistry/" in key or key.endswith("/materials/"))
    flip = cmp_["direction_changes"][0] if cmp_["direction_changes"] else None
    surf = e["surface"]

    macros = {
        "xVerdicts": n(k["answers"]),
        "xTriples": n(cat["triples_total"]),
        "xStated": n(cat["triples_per_graph"]["stated"]),
        "xDerived": n(cat["triples_per_graph"]["derived"]),
        "xSupplied": n(cat["triples_per_graph"]["supplied"]),
        "xProv": n(cat["triples_per_graph"]["prov"]),
        "xCatalogue": n(cat["triples_per_graph"]["catalogue"]),
        "xTrigMB": f"{cat['trig_bytes'] / 1e6:.1f}",
        "xEmit": n(cat["timing_s"]["emit"]),
        "xSerialise": n(cat["timing_s"]["serialise"]),
        "xValidate": n(cat["timing_s"]["validate"]),
        "xStructureMin": f"{k['timing_s']['structure'] / 60:.0f}",
        "xGibbs": n(cat["instances_of_extension_classes"].get("GibbsEnergyOfReaction", 0)),
        "xConventions": n(cat["instances_of_extension_classes"].get("Convention", 0)),
        "xChemValues": n(len(dirs)),
        "xDirAnswered": n(len(answered)),
        "xDirAnsweredHess": n(by_source["hess"]),
        "xDirAnsweredMeasured": n(by_source["measured"]),
        "xDirKernel": n(kernel_answered),
        "xMarginMeasured": f"{margin['measured']:.2f}",
        "xMarginHess": f"{margin['hess']:.2f}",
        # crowding shift of a reaction that adds one solute particle, at the extension's temperature and a = 0.7
        "xCrowding": f"{-8.314462618e-3 * 298.15 * math.log(0.7):.2f}",
        "xDirConc": n(sum(1 for d in dirs if d["blocker"] == "concentration")),
        "xDirMedium": n(sum(1 for d in dirs if d["blocker"] == "medium")),
        "xGateNotRecorded": n(gates.get("-", 0)),
        "xGateOpen": n(gates.get("open", 0)),
        "xGateWitnessed": n(gates.get("+", 0)),
        "xGateUndecided": n(gates.get("?", 0)),
        "xScalingSlope": f"{slope:.2f}",
        "xScalingMin": n(sc[0]["triples"]),
        "xControlTriples": n(ctl["unaltered"]["triples"]),
        "xControlDefects": n(len(ctl["defects"])),
        "xControlCaught": n(sum(1 for v in ctl["defects"].values() if not v["conforms"])),
        "xNarrowPh": f"{cmp_['medium_b']['ph'][0]}--{cmp_['medium_b']['ph'][1]}",
        "xNarrowI": f"{cmp_['medium_b']['ionic_strength'][0]}--{cmp_['medium_b']['ionic_strength'][1]}",
        "xNarrowPmg": f"{cmp_['medium_b']['pmg'][0]}--{cmp_['medium_b']['pmg'][1]}",
        "xShared": n(cmp_["shared_iris"]),
        "xChanged": n(cmp_["changed_total"]),
        "xChangedReaction": SHORT.get(flip["reaction"], flip["name"]) if flip else "none",
        "xJoinMs": n(cmp_["timing_s"]["sparql_join"] * 1000),
        "xParquetRows": n(sd["rows"]),
        "xParquetKB": n(sd["bytes"] / 1000),
        "xParquetWrite": n(sd["timing_s"]["write"]),
        "xParquetRead": n(sd["timing_s"]["read"], 1),
        "xNetwork": n(sd["network_attempts"]),
        "xSchemaClasses": n(sch["classes"]),
        "xSchemaSlots": n(sch["slots"]),
        "xSchemaEnums": n(sch["enums"]),
        "xCoreClasses": n(sch["classes_by_schema"][core]),
        "xChemClasses": n(chem),
        "xPydantic": n(sch["pydantic_models"]),
        "xMaterialsClasses": n(sum(v for key, v in sch["classes_by_schema"].items() if key.endswith("/materials/"))),
        "MEDIUMREACTION": SHORT.get(surf["reaction"], surf["name"]),
        "NDIRECTIONS": n(len(dirs)),
    }
    lines = ["% generated by scripts/make_numbers.py from data/extension.json and results/*.json; do not edit"]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    (HERE / "data" / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    st, xs = s["structure"], k["structure"]
    tm = s["thermo_match"]["labels"]
    hs = sup["unmeasured_small_molecule"]["labels"]
    hf, xf = sup["hess_fit"], k["hess_fit"]
    xv = k["verdicts"]
    kernel_conc = sum(v for key, v in med.items() if key.startswith(("measured|undecidable:concentration",
                                                                   "hess|undecidable:concentration")))
    kernel_medium = sum(v for key, v in med.items() if key.startswith(("measured|undecidable:medium",
                                                                     "hess|undecidable:medium")))
    rows = [
        ("Internal species", st["internal_species"], xs["internal_species"]),
        ("Rank of $N$", st["rank_N"], xs["rank"]),
        ("Conservation laws", st["conservation_laws_dim"], xs["conservation_laws"]),
        ("Steady-state flux modes", st["steady_flux_modes_dim"], xs["steady_flux_modes"]),
        ("Blocked, medium open", st["blocked_if_only_ambient_open"], xs["blocked_if_only_medium_open"]),
        ("Blocked, boundary open", st["blocked_if_boundary_species_also_open"], xs["blocked_if_boundary_species_also_open"]),
        ("Measured \\DG: answered", tm["answered"], xv["reaction-energy"]["answered"]),
        ("\\quad refused", tm["refused"], xv["reaction-energy"]["refused"]),
        ("\\quad absent", tm["absent"], xv["reaction-energy"]["absent"]),
        ("Hess's law: answered", hs["answered"], xv["hess-energy"]["answered"]),
        ("\\quad absent", hs["absent"], xv["hess-energy"]["absent"]),
        ("\\quad certified empty", hs["certified-empty"], xv["hess-energy"]["certified-empty"]),
        ("Hess fit: $\\chi^2$ / degrees of freedom", f"{hf['chi2']:.1f} / {hf['dof']}", f"{xf['chi2']:.1f} / {xf['dof']}"),
        ("Species state: absent", s["node_solves"]["labels"]["absent"], xv["node-solve"]["absent"]),
        ("\\quad underdetermined", s["node_solves"]["labels"]["underdetermined"], xv["node-solve"]["underdetermined"]),
        ("Direction$^\\dagger$: answered", kernel_answered, len(answered)),
        ("\\quad concentrations decide", kernel_conc, sum(1 for d in dirs if d["blocker"] == "concentration")),
        ("\\quad medium decides", kernel_medium, sum(1 for d in dirs if d["blocker"] == "medium")),
    ]
    body = []
    for name, a, b in rows:
        fa = n(a) if isinstance(a, (int, float)) else a
        fb = n(b) if isinstance(b, (int, float)) else b
        mark = "" if fa == fb else "\\,$\\ast$"
        body.append(f"{name} & {fa} & {fb}{mark} \\\\")
    table = "\n".join(
        [
            "% generated by scripts/make_numbers.py; do not edit",
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{The extension's own run over the full corpus against the recorded results of the research "
            "implementation. $^\\dagger$Reactions with a chemical value from measurement or Hess's law. "
            "$\\ast$~Differs; see text.}",
            "\\label{tab:reproduction}",
            "\\footnotesize",
            "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrr@{}}",
            "\\toprule",
            "Quantity & Research & Extension \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular*}",
            "\\end{table}",
        ]
    )
    (HERE / "tables" / "reproduction.tex").write_text(table + "\n", encoding="utf-8")
    print("wrote data/numbers.tex and tables/reproduction.tex;", "slope", round(slope, 2))
    assert math.isfinite(slope)


if __name__ == "__main__":
    main()
