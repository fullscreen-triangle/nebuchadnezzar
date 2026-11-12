"""Experimental structures as a control for AlphaFold contact graphs (P1).

PDBe-KB / SIFTS (EMBL-EBI): the best experimental structures per UniProt entry and
the residue mapping between author numbering and UniProt numbering. Contacts are
computed exactly as for AlphaFold (heavy atoms within 4.5 A, counted), on the one
chain, and compared with the AlphaFold graph over the residues both contain.
Pairs closer than 3 in sequence are left out, since they are in contact in any fold.
"""
from __future__ import annotations

import gzip
import json
import time
import urllib.request

import numpy as np

from .core import RAW
from .proteins import AF_DIR, contact_graph

PDBE = "https://www.ebi.ac.uk/pdbe/api"
CHAINS = RAW / "pdb" / "chains"
UA = {"User-Agent": "circuit-kg-kernel"}


def _get(url):
    for k in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** k)
        except Exception:
            time.sleep(2 ** k)
    return None


def best_structure(acc: str, min_coverage: float):
    b = _get(f"{PDBE}/mappings/best_structures/{acc}")
    if not b:
        return None
    for s in json.loads(b).get(acc, []):
        if s.get("coverage", 0) >= min_coverage and s.get("experimental_method", "").startswith("X-ray"):
            return s
    return None


def sifts_offset(pdb: str, chain: str, acc: str):
    """UniProt number minus author number for the mapped segment(s) of one chain (one offset)."""
    b = _get(f"{PDBE}/mappings/uniprot/{pdb}")
    if not b:
        return None
    m = json.loads(b).get(pdb, {}).get("UniProt", {}).get(acc, {}).get("mappings", [])
    offs = {seg["unp_start"] - seg["start"]["author_residue_number"] for seg in m
            if seg.get("chain_id") == chain and seg["start"].get("author_residue_number") is not None}
    return offs.pop() if len(offs) == 1 else None


def chain_file(pdb: str, chain: str):
    CHAINS.mkdir(parents=True, exist_ok=True)
    out = CHAINS / f"{pdb}_{chain}.pdb.gz"
    if out.exists():
        return out
    body = _get(f"https://files.rcsb.org/download/{pdb.upper()}.pdb")
    if not body:
        return None
    lines = [l for l in body.decode("ascii", "replace").splitlines()
             if l.startswith("ATOM") and l[21] == chain and l[16] in " A"]
    if not lines:
        return None
    out.write_bytes(gzip.compress(("\n".join(lines) + "\nEND\n").encode()))
    return out


def pairs_of(edges, shift=0, min_sep=3):
    return {(u + shift, v + shift) for u, v, *_ in edges if abs(u - v) >= min_sep}


def compare(accessions, limit=60, min_coverage=0.8) -> dict:
    rows, tried = [], 0
    for acc in accessions:
        if len(rows) >= limit:
            break
        meta = AF_DIR / f"AF-{acc}.json"
        gpath = AF_DIR / f"AF-{acc}.graph.json"
        if not (meta.exists() and gpath.exists()):
            continue
        if (json.loads(meta.read_text()).get("mean_plddt") or 0) < 90:
            continue
        tried += 1
        s = best_structure(acc, min_coverage)
        if not s:
            continue
        pdb, chain = s["pdb_id"], s["chain_id"]
        off = sifts_offset(pdb, chain, acc)
        f = chain_file(pdb, chain)
        if off is None or f is None:
            continue
        try:
            g_exp = contact_graph(f)
        except Exception:
            continue
        af = json.loads(gpath.read_text())
        exp_edges = [(u + off, v + off) for (u, v) in g_exp["edges"]]
        common = {int(r) for r in af["letters"]} & {r + off for r in g_exp["residues"]}
        A = {p for p in pairs_of(af["edges"]) if p[0] in common and p[1] in common}

        def j(shift):
            E = {(u + shift, v + shift) for u, v in exp_edges if abs(u - v) >= 3}
            E = {p for p in E if p[0] in common and p[1] in common}
            return len(A & E) / len(A | E) if A | E else None
        exp_exposed = {int(k) + off: v for k, v in g_exp["exposed"].items()}
        both = [r for r in common if str(r) in af["exposed"] and r in exp_exposed]
        buried_agree = np.mean([(af["exposed"][str(r)] == 0) == (exp_exposed[r] == 0) for r in both]) if both else None
        rows.append({"acc": acc, "pdb": pdb, "chain": chain, "coverage": s["coverage"],
                     "resolution": s.get("resolution"), "common_residues": len(common),
                     "jaccard": j(0), "jaccard_shift3": j(3), "buried_agreement": buried_agree})
    J = [r["jaccard"] for r in rows if r["jaccard"] is not None]
    Js = [r["jaccard_shift3"] for r in rows if r["jaccard_shift3"] is not None]
    B = [r["buried_agreement"] for r in rows if r["buried_agreement"] is not None]
    return {"high_confidence_models_tried": tried, "compared": len(rows),
            "median_jaccard": float(np.median(J)) if J else None,
            "median_jaccard_control_shift3": float(np.median(Js)) if Js else None,
            "median_buried_agreement": float(np.median(B)) if B else None, "rows": rows}
