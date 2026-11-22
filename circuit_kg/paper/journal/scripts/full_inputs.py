"""
Inputs of dcat-ap-plus-homo-sapien for the whole cytosolic corpus (provenance script for the manuscript).

This is the selection of ``dcat-ap-plus-homo-sapien/scripts/build_example.py`` widened to every
reaction of the kernel's corpus, every TECRDB row in the temperature and grade window (so that the
Hess span is the kernel's), and every compound of the microspecies table. It writes the three
files of the exchange format into ``--out``; nothing is written into either package.

* ``corpus.json``        all 2,914 cytosolic Reactome reactions and the species they reference,
                         with the KEGG compound of each small molecule and the basis of that mapping;
* ``measurements.csv``   TECRDB rows in 293.15-313.15 K with grades A-C (chemical reference state);
* ``microspecies.json``  the eQuilibrator microspecies of every compound in the kernel's table.

A species for which the kernel found several KEGG candidates is passed with all of them under
``conjugate-closure``; the extension's matcher accepts a measurement only when exactly one
combination of candidates matches, as the kernel does.

Usage (from the nebuchadnezzar root)::

    python circuit_kg/paper/journal/scripts/full_inputs.py --kernel circuit_kg --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

T_WINDOW = (293.15, 313.15)
GRADES = {"A", "B", "C"}
PROTON = "C00080"
COLUMNS = ("ref", "ec", "enzyme", "eval", "T", "pH", "I", "pMg", "reaction", "dG0", "dG0_prime_measured", "transform")


def main() -> None:
    """Write the three input files."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--kernel", type=Path, required=True, help="path to circuit_kg")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    kernel = a.kernel.resolve()
    sys.path.insert(0, str(kernel.parent))
    from circuit_kg import thermo  # type: ignore[import-not-found]

    corpus = json.loads((kernel / "data/derived/reactome_cytosol.json").read_text(encoding="utf-8"))
    smap = thermo.species_kegg(corpus, thermo.Chebi())
    medium = json.loads((kernel / "media/cytosol.json").read_text(encoding="utf-8"))
    ambient_chebi = {int(k) for k in medium["ambient_chebi"]}

    species = {}
    for sid, sp in corpus["species"].items():
        sp = dict(sp)
        if sp["kind"] == "small":
            m = smap.get(sid, {"kegg": [], "tag": "absent"})
            tag = "supplied" if m["tag"] == "underdetermined" else m["tag"]
            sp["kegg"] = {"ids": m["kegg"], "basis": tag}
            if tag == "supplied":
                sp["kegg"]["convention"] = "conjugate-closure"
            sp["ambient"] = bool(set(sp["chebi"]) & ambient_chebi)
        species[sid] = sp
    out_corpus = {
        "name": "cytosol",
        "title": "Cytosolic reactions of Homo sapiens (Reactome)",
        "source": {
            "resource": "Reactome, SBML export for Homo sapiens",
            "licence": "CC BY 4.0",
            "criterion": corpus["criterion"],
        },
        "species": species,
        "reactions": dict(sorted(corpus["reactions"].items())),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "corpus.json").write_text(json.dumps(out_corpus, ensure_ascii=False) + "\n", encoding="utf-8")

    keep = []
    with open(kernel / "data/raw/tecrdb_transformed.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                t = float(r["T"])
            except ValueError:
                continue
            if not (T_WINDOW[0] <= t <= T_WINDOW[1]) or r["eval"] not in GRADES:
                continue
            if {term.split("*")[1] for term in r["reaction"].split()} - {PROTON}:
                keep.append({k: r[k] for k in COLUMNS})
    with open(a.out / "measurements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUMNS))
        w.writeheader()
        w.writerows(keep)

    table = json.loads((kernel / "data/derived/legendre.json").read_text(encoding="utf-8"))
    (a.out / "microspecies.json").write_text(
        json.dumps({"source": "eQuilibrator compound cache (equilibrator-cache 0.8.1)", "compounds": table["compounds"]})
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(corpus['reactions'])} reactions, {len(species)} species, {len(keep)} TECRDB rows, "
          f"{len(table['compounds'])} compounds")


if __name__ == "__main__":
    main()
