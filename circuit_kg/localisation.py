"""Independent checks on the cytosol scope (L1).

The corpus is cytosolic because Reactome says so. Two sources that do not depend
on Reactome say where the catalyst proteins are:
  Human Protein Atlas subcellular atlas (CC BY-SA 3.0): main and additional locations
    from immunofluorescence, with a reliability grade;
  COMPARTMENTS (Jensen lab, CC BY 4.0): integrated text-mining, knowledge and
    experiment, a 0-5 confidence score per protein and GO compartment.
A protein has cytosol support when HPA lists Cytosol among its locations or
COMPARTMENTS scores it at least 3 for GO:0005829.
"""
from __future__ import annotations

import io
import zipfile
from collections import Counter

from .core import RAW

HPA = RAW / "hpa" / "subcellular_location.tsv.zip"
COMP = RAW / "compartments" / "cytosol.tsv"


def hpa_cytosol() -> dict:
    z = zipfile.ZipFile(HPA)
    text = z.read(z.namelist()[0]).decode("utf-8")
    rows = text.splitlines()
    head = rows[0].split("\t")
    i_gene, i_rel = head.index("Gene name"), head.index("Reliability")
    i_main, i_add = head.index("Main location"), head.index("Additional location")
    out = {}
    for line in rows[1:]:
        f = line.split("\t")
        locs = (f[i_main] + ";" + f[i_add]).split(";")
        out[f[i_gene].upper()] = {"cytosol": "Cytosol" in locs, "reliability": f[i_rel]}
    return out


def compartments_cytosol() -> dict:
    out = {}
    for line in COMP.read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        if len(f) >= 5:
            g = f[1].upper()
            out[g] = max(out.get(g, 0.0), float(f[4]))
    return out


def scope_check(corpus: dict, ann: dict, min_score=3.0) -> dict:
    sp = corpus["species"]
    hpa, comp = hpa_cytosol(), compartments_cytosol()
    prots = {u.split("-")[0] for r in corpus["reactions"].values() for s, role in r["modifiers"]
             if role == "catalyst" for u in sp[s]["uniprot"]}
    per = {}
    for u in prots:
        g = ((ann.get(u) or {}).get("gene") or "").upper()
        h, c = hpa.get(g), comp.get(g)
        if h is None and c is None:
            per[u] = "no data"
        elif (h and h["cytosol"]) or (c is not None and c >= min_score):
            per[u] = "supported"
        else:
            per[u] = "not supported"
    rx = Counter()
    for r in corpus["reactions"].values():
        us = {u.split("-")[0] for s, role in r["modifiers"] if role == "catalyst" for u in sp[s]["uniprot"]}
        if not us:
            continue
        lab = {per[u] for u in us}
        rx["all catalysts supported" if lab == {"supported"} else
           "no catalyst supported" if "supported" not in lab else "some catalysts supported"] += 1
    counts = Counter(per.values())
    with_data = counts["supported"] + counts["not supported"]
    return {"catalyst_proteins": len(prots), "proteins": dict(counts),
            "supported_fraction_of_proteins_with_data": counts["supported"] / with_data if with_data else None,
            "reactions_by_catalyst_support": dict(rx),
            "sources": {"hpa_genes": len(hpa), "compartments_genes": len(comp), "min_score": min_score},
            "not_supported_examples": sorted(u for u, v in per.items() if v == "not supported")[:20]}
