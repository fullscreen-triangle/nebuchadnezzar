"""One-off streaming filters for two large public tables (kept out of the package; run from circuit_kg/)."""
import json, sys, urllib.request
UA = {"User-Agent": "circuit-kg-kernel"}
c = json.load(open("data/derived/reactome_cytosol.json"))
chebi = {f"chebi:{x}" for s in c["species"].values() for x in s["chebi"]}
# MetaNetX chem_xref: source  ID  description ; keep corpus ChEBI rows and every KEGG compound row
n = kept = 0
with urllib.request.urlopen(urllib.request.Request("https://www.metanetx.org/ftp/latest/chem_xref.tsv", headers=UA), timeout=600) as r, \
        open("data/raw/metanetx/chem_xref.filtered.tsv", "w", encoding="utf-8") as out:
    for raw in r:
        line = raw.decode("utf-8", "replace")
        n += 1
        if line.startswith("#"):
            continue
        src = line.split("\t", 1)[0]
        if src in chebi or src.startswith("kegg.compound:C") or src.startswith("keggC:"):
            out.write(line); kept += 1
print("metanetx rows", n, "kept", kept, flush=True)
# COMPARTMENTS integrated: ensp  gene  GO id  GO name  score ; keep cytosol GO:0005829
n = kept = 0
with urllib.request.urlopen(urllib.request.Request("https://download.jensenlab.org/human_compartment_integrated_full.tsv", headers=UA), timeout=600) as r, \
        open("data/raw/compartments/cytosol.tsv", "w", encoding="utf-8") as out:
    for raw in r:
        line = raw.decode("utf-8", "replace"); n += 1
        if "\tGO:0005829\t" in line:
            out.write(line); kept += 1
print("compartments rows", n, "kept", kept, flush=True)
