"""Inputs the kernel cannot fetch on its own: measured concentrations and BRENDA.

Concentrations. Measured cytosolic or whole-cell concentrations replace the global
`concentration-window` species by species. The kernel reads every CSV under
data/raw/concentrations/ with the columns

    kegg, low_M, high_M, cell_type, source

(one row per compound; a single measurement x with standard deviation s is written
as low = x - 2s, high = x + 2s, floored at 1e-9 M). The tag is `supplied`, with the
cell type and source named. Without such a file every direction keeps the global
window and the verdict says so. Park et al. 2016 (absolute concentrations in iBMK
cells) is the intended first source; its tables sit behind the publisher's login.

BRENDA (CC BY 4.0). The bulk JSON is downloaded by a person who accepts the licence
at https://www.brenda-enzymes.org/download.php, then placed under data/raw/brenda/.
Values are read per EC number with their organism and UniProt accessions, from the
2.0.0 schema: turnover_number (1/s), km_value (mM), ki_value (mM); each value names its
substrate or inhibitor, protein ids and literature references. k_cat is paired with
K_m only when both share protein, substrate and reference (convention
`brenda-same-reference`), the same discipline as SABIO-RK's same-kinetic-law pairing.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from .core import RAW, absent

CONC_DIR = RAW / "concentrations"
BRENDA_DIR = RAW / "brenda"


# ------------------------------------------------------------------ concentrations

def concentration_windows():
    """KEGG -> (low_M, high_M, cell_type, source), or an absent verdict naming the file."""
    files = sorted(CONC_DIR.glob("*.csv")) if CONC_DIR.exists() else []
    if not files:
        return None, absent("no measured concentration table",
                            remedy=f"a CSV with columns kegg, low_M, high_M, cell_type, source "
                                   f"under {CONC_DIR}").as_json()
    out = {}
    for f in files:
        with open(f, newline="", encoding="utf-8") as h:
            for r in csv.DictReader(h):
                lo, hi = max(float(r["low_M"]), 1e-9), float(r["high_M"])
                if not lo <= hi:
                    continue
                k = r["kegg"].strip()
                prev = out.get(k)
                out[k] = (lo, hi, r.get("cell_type", ""), r.get("source", f.name)) if prev is None else \
                    (min(prev[0], lo), max(prev[1], hi), prev[2], prev[3])
    return out, None


# ------------------------------------------------------------------ BRENDA

def _num(item):
    for k in ("num_value", "value"):
        v = item.get(k)
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    lo, hi = item.get("min_value"), item.get("max_value")
    if lo is not None and hi is not None:
        return math.sqrt(float(lo) * float(hi))
    return None


def _subject(item):
    v = item.get("value") or ""
    m = re.search(r"\{([^}]*)\}", v)
    return (m.group(1) if m else item.get("substrate") or item.get("inhibitor") or v).strip().lower()


def brenda_values(path: Path | None = None):
    """Rows {ec, kind, subject, value (SI: 1/s or M), organism, accessions, refs, protein}."""
    if path is None and BRENDA_DIR.exists():
        path = next(iter(sorted(BRENDA_DIR.glob("*.json"))), None)
    if path is None:
        return None, absent("BRENDA is not downloaded",
                            remedy="accept the CC BY 4.0 licence at brenda-enzymes.org/download.php and "
                                   f"place the JSON under {BRENDA_DIR}").as_json()
    data = json.loads(Path(path).read_text(encoding="utf-8"))["data"]
    rows = []
    scale = {"turnover_number": 1.0, "km_value": 1e-3, "ki_value": 1e-3}
    for ec, e in data.items():
        prots = e.get("protein") or {}
        for kind, f in scale.items():
            for it in e.get(kind) or []:
                v = _num(it)
                if v is None or v <= 0:
                    continue
                for pid in it.get("proteins") or [None]:
                    p = prots.get(pid, {}) if pid else {}
                    rows.append({"ec": ec, "kind": kind, "subject": _subject(it), "value": v * f,
                                 "organism": p.get("organism"), "accessions": p.get("accessions") or [],
                                 "refs": tuple(sorted(it.get("references") or [])), "protein": pid})
    return rows, None


def brenda_pairs(rows) -> dict:
    """(accession or ec+organism, subject) -> k_cat/K_m from values that share protein, subject and a
    reference."""
    kc, km = defaultdict(list), defaultdict(list)
    for r in rows:
        key = (r["ec"], r["protein"], r["subject"])
        if r["kind"] == "turnover_number":
            kc[key].append(r)
        elif r["kind"] == "km_value":
            km[key].append(r)
    out = {}
    for key in set(kc) & set(km):
        for a in kc[key]:
            for b in km[key]:
                if set(a["refs"]) & set(b["refs"]):
                    for acc in a["accessions"] or [f"{a['ec']}|{a['organism']}"]:
                        out.setdefault((acc, key[2]), []).append(a["value"] / b["value"])
    return {k: math.exp(sum(map(math.log, v)) / len(v)) for k, v in out.items()}
