"""Fetch and record the external data the kernel reads.

Every file lands in data/raw with its origin, retrieval date and sha256 in
data/raw/SOURCES.json.  Nothing downstream reads the network.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import urllib.request

from .core import RAW, REACTOME_TGZ

CHEBI_BASE = "https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files"
CHEBI_FILES = ["compounds", "database_accession", "source", "relation",
               "relation_type", "secondary_ids"]

# TECRDB measurements reverse-transformed to the chemical reference state by
# the disease-circuit validation suite (stage A of run_realdata.py, which
# needs eQuilibrator).  Copied, not recomputed.
TECRDB_DERIVED = (REACTOME_TGZ.parents[4] / "health" / "syndrome" / "cause-for-concern"
                  / "docs" / "disease-circuit-tolerance" / "validation" / "realdata"
                  / "derived" / "tecrdb_transformed.csv")


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _record(entries: dict) -> None:
    path = RAW / "SOURCES.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    known.update(entries)
    path.write_text(json.dumps(known, indent=2))


def fetch(force: bool = False) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    entries = {}
    for name in CHEBI_FILES:
        dest = RAW / f"chebi_{name}.tsv.gz"
        url = f"{CHEBI_BASE}/{name}.tsv.gz"
        if force or not dest.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "circuit-kg-kernel"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
        entries[dest.name] = {"url": url, "retrieved": today,
                              "sha256": _sha256(dest), "bytes": dest.stat().st_size}
    dest = RAW / "tecrdb_transformed.csv"
    if force or not dest.exists():
        if not TECRDB_DERIVED.exists():
            raise FileNotFoundError(f"TECRDB derived table not found at {TECRDB_DERIVED}")
        shutil.copyfile(TECRDB_DERIVED, dest)
    entries[dest.name] = {"copied_from": str(TECRDB_DERIVED), "retrieved": today,
                          "sha256": _sha256(dest), "bytes": dest.stat().st_size,
                          "note": "TECRDB (component-contribution repo, sha256 b84fe77e...) "
                                  "reverse-transformed by syndrome run_realdata.py stage A"}
    entries["reactome"] = {"path": str(REACTOME_TGZ), "sha256": _sha256(REACTOME_TGZ),
                           "bytes": REACTOME_TGZ.stat().st_size}
    _record(entries)
    return entries
