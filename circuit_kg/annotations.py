"""Public hierarchies for proteins: EC numbers, Pfam domains, protein families (UniProt).

Used as labelled hierarchies (idea from the P450 monograph's family -> subfamily ->
member addresses, built here from real hierarchies instead of composition): an EC
number truncated to 1, 2, 3 or 4 levels is a prefix address whose depth is the
level of resolution.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request

from .core import RAW

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,gene_primary,ec,xref_pfam,protein_families"
DEST = RAW / "uniprot"
TABLE = DEST / "annotations.json"


def _get(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "circuit-kg-kernel"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8")
        except Exception:
            time.sleep(2 ** k)
    raise RuntimeError(f"failed: {url}")


def fetch(accessions, batch=100) -> dict:
    """accession -> {gene, ec: [..], pfam: [..], family}; cached as one JSON file."""
    have = json.loads(TABLE.read_text()) if TABLE.exists() else {"records": {}, "batches": []}
    todo = sorted({a.split("-")[0] for a in accessions} - set(have["records"]))
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        q = " OR ".join(f"accession:{a}" for a in chunk)
        url = UNIPROT_URL + "?" + urllib.parse.urlencode(
            {"query": q, "fields": FIELDS, "format": "tsv", "size": 500})
        body = _get(url)
        have["batches"].append({"n": len(chunk), "sha256": hashlib.sha256(body.encode()).hexdigest()})
        for row in csv.DictReader(io.StringIO(body), delimiter="\t"):
            have["records"][row["Entry"]] = {
                "gene": row.get("Gene Names (primary)") or None,
                "ec": [e.strip() for e in (row.get("EC number") or "").split(";") if e.strip()],
                "pfam": [p for p in (row.get("Pfam") or "").split(";") if p],
                "family": (row.get("Protein families") or "").split(";")[0].strip() or None}
        for a in chunk:                        # mark the ones UniProt did not return
            have["records"].setdefault(a, None)
    have["source"] = {"url": UNIPROT_URL, "fields": FIELDS, "retrieved": time.strftime("%Y-%m-%d")}
    DEST.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(have))
    return have["records"]


def ec_prefix(ec: str, level: int) -> str | None:
    parts = ec.split(".")
    if len(parts) < level or any(p in ("-", "") or p.startswith("n") for p in parts[:level]):
        return None
    return ".".join(parts[:level])
