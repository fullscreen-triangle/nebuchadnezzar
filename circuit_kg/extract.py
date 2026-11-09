"""Stream the Reactome SBML tarball once and keep the purely cytosolic reactions.

A reaction is kept when every species it references -- reactants, products and
modifiers -- lies in the compartment named "cytosol".  Reactions are
deduplicated by id across the per-pathway files; the pathways each reaction
appears in are recorded.  Output: data/derived/reactome_cytosol.json.
"""
from __future__ import annotations

import json
import re
import tarfile
import xml.etree.ElementTree as ET

from .core import DERIVED, REACTOME_TGZ, Tag

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
BQ = "{http://biomodels.net/biology-qualifiers/}"
IDENT = re.compile(r"identifiers\.org/([A-Za-z][\w.]*)[:/](.+)$")

SBO_ROLE = {"SBO:0000010": "reactant", "SBO:0000011": "product",
            "SBO:0000013": "catalyst", "SBO:0000459": "stimulator",
            "SBO:0000020": "inhibitor"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _identifiers(el, qualifier: str) -> list[tuple[str, str]]:
    out = []
    for q in el.iter(BQ + qualifier):
        for li in q.iter(RDF + "li"):
            m = IDENT.search(li.get(RDF + "resource", ""))
            if m:
                out.append((m.group(1).lower(), m.group(2)))
    return out


def _species_record(el, comp_name: dict) -> dict:
    is_ = _identifiers(el, "is")
    parts = _identifiers(el, "hasPart")
    chebi = sorted({int(v) for ns, v in is_ if ns == "chebi" and v.isdigit()})
    uniprot = sorted({v for ns, v in is_ + parts if ns == "uniprot"})
    reactome = next((v for ns, v in is_ if ns == "reactome"), None)
    if chebi and not uniprot and not parts:
        kind = "small"
    elif uniprot:
        kind = "macromolecule"
    else:
        kind = "other"
    return {"name": el.get("name"), "compartment": comp_name.get(el.get("compartment"), "?"),
            "chebi": chebi, "uniprot": uniprot, "reactome": reactome, "kind": kind,
            "n_parts": len(parts)}


def _reaction_record(el) -> dict:
    rec = {"name": el.get("name"), "reversible": el.get("reversible") == "true",
           "reactants": [], "products": [], "modifiers": []}
    for child in el:
        kind = _local(child.tag)
        if kind not in ("listOfReactants", "listOfProducts", "listOfModifiers"):
            continue
        for ref in child:
            sp = ref.get("species")
            if kind == "listOfModifiers":
                rec["modifiers"].append([sp, SBO_ROLE.get(ref.get("sboTerm"), "modifier")])
            else:
                st = ref.get("stoichiometry")
                tag = Tag.STATED.value if st is not None else Tag.SUPPLIED.value
                side = "reactants" if kind == "listOfReactants" else "products"
                rec[side].append([sp, float(st) if st is not None else 1.0, tag])
    return rec


def extract(path=REACTOME_TGZ, out=None) -> dict:
    out = out or DERIVED / "reactome_cytosol.json"
    species, reactions, pathways_of = {}, {}, {}
    n_files = 0
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            if not member.name.endswith(".sbml"):
                continue
            n_files += 1
            pathway = member.name.rsplit("/", 1)[-1].removesuffix(".sbml")
            comp_name, file_species = {}, {}
            for _, el in ET.iterparse(tar.extractfile(member), events=("end",)):
                tag = _local(el.tag)
                if tag == "compartment":
                    comp_name[el.get("id")] = el.get("name") or el.get("id")
                elif tag == "species":
                    file_species[el.get("id")] = _species_record(el, comp_name)
                    el.clear()
                elif tag == "reaction":
                    rid = el.get("id")
                    pathways_of.setdefault(rid, set()).add(pathway)
                    if rid not in reactions:
                        rec = _reaction_record(el)
                        refs = ([s for s, *_ in rec["reactants"]] + [s for s, *_ in rec["products"]]
                                + [s for s, _ in rec["modifiers"]])
                        if refs and all(file_species.get(s, {}).get("compartment") == "cytosol"
                                        for s in refs):
                            reactions[rid] = rec
                            for s in refs:
                                species.setdefault(s, file_species[s])
                    el.clear()
    for rid, rec in reactions.items():
        rec["pathways"] = sorted(pathways_of[rid])
    corpus = {"source": str(path), "pathway_files": n_files,
              "criterion": "every reactant, product and modifier in compartment 'cytosol'",
              "species": species, "reactions": reactions}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(corpus))
    return corpus
