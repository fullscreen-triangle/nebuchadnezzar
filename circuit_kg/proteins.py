"""Proteins as contact graphs (AlphaFold DB models), attached to the circuit KG.

Protocol after the P450 derivation (its G5): one vertex per residue; the weight
between two residues is the number of heavy-atom pairs closer than 4.5 A
(counted); a residue's medium weight is the number of its heavy atoms with more
than 1 A^2 of solvent-accessible surface (Shrake-Rupley, probe 1.4 A, 192
points).  Counted weights keep the floor of individuation at one; real-valued
quantities (pLDDT, conductances, Delta G) are never used as cut weights.

Convention `medium-floor-residue`: a buried residue (no exposed atom) still
gets medium weight 1, so every item touches the medium and derived power stays
defined.  The raw exposed-atom count is kept beside it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

from .core import RAW

AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
AF_DIR = RAW / "alphafold"
CONTACT_CUTOFF = 4.5
EXPOSED_CUTOFF = 1.0
PROBE = 1.4
N_POINTS = 192
RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "SE": 1.90}
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
GZIP_MAGIC = bytes([0x1F, 0x8B])


# ------------------------------------------------------------------ fetch

def _get(url, tries=6):
    req = urllib.request.Request(url, headers={"User-Agent": "circuit-kg-kernel"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            # some responses arrive gzip-encoded without urllib decoding them
            return gzip.decompress(data) if data[:2] == GZIP_MAGIC else data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** k)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2 ** k)
    raise RuntimeError(f"failed: {url}")


def _fetch_one(acc: str) -> dict:
    dest = AF_DIR / f"AF-{acc}.pdb.gz"
    meta_path = AF_DIR / f"AF-{acc}.json"
    if dest.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())
    canonical = acc.split("-")[0]
    raw = _get(AF_API.format(acc=canonical))
    if raw is None:
        meta = {"acc": acc, "status": "absent", "blocker": "no AlphaFold DB entry"}
    else:
        entries = [e for e in json.loads(raw) if e.get("uniprotAccession") == canonical]
        if not entries:
            meta = {"acc": acc, "status": "absent", "blocker": "no AlphaFold DB entry"}
        else:
            e = entries[0]
            pdb = _get(e["pdbUrl"])
            if pdb is None:
                meta = {"acc": acc, "status": "absent", "blocker": "model file missing"}
            else:
                dest.write_bytes(gzip.compress(pdb))
                meta = {"acc": acc, "status": "ok", "url": e["pdbUrl"],
                        "sha256": hashlib.sha256(pdb).hexdigest(), "gene": e.get("gene"),
                        "sequence": e.get("uniprotSequence") or e.get("sequence"),
                        "start": int(e.get("uniprotStart", 1)), "end": int(e.get("uniprotEnd", 0)),
                        "version": e.get("latestVersion"), "mean_plddt": e.get("globalMetricValue"),
                        "isoform_mapped_to_canonical": acc != canonical,
                        "retrieved": time.strftime("%Y-%m-%d")}
    meta_path.write_text(json.dumps(meta))      # written last: marks the model complete
    return meta


def _fetch_safe(acc: str) -> dict:
    try:
        return _fetch_one(acc)
    except RuntimeError as e:          # transient: not cached, retried on the next run
        return {"acc": acc, "status": "failed", "blocker": str(e)}


def fetch_alphafold(accessions, workers=4) -> dict:
    AF_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    with ThreadPoolExecutor(workers) as ex:
        for meta in ex.map(_fetch_safe, sorted(accessions)):
            out[meta["acc"]] = meta
    (AF_DIR / "MANIFEST.json").write_text(json.dumps(out))
    return out


# ------------------------------------------------------------------ structure

def read_pdb(path):
    """Heavy atoms of the first model: arrays of coords, elements, residue numbers, names."""
    xyz, elem, resnum, resname = [], [], [], []
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            e = (line[76:78].strip() or line[12:14].strip()).upper()
            if e == "H":
                continue
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            elem.append(e)
            resnum.append(int(line[22:26]))
            resname.append(line[17:20])
    if not xyz:
        raise ValueError(f"no ATOM records in {path}")
    return np.array(xyz), np.array(elem), np.array(resnum), np.array(resname)


def _sphere(n=N_POINTS):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + 5 ** 0.5) * k
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


def sasa(xyz, elem, k=12, chunk=200_000):
    """Shrake-Rupley areas; a point is buried if any of its k nearest atoms covers it.

    Checked against the exact all-neighbour loop: identical on 300 sampled atoms
    of CTNNB1 (AF-P35222), and k=10 already gives the same total as k=16.
    """
    r = np.array([RADII.get(e, 1.8) for e in elem]) + PROBE
    pts = _sphere()
    owner = np.repeat(np.arange(len(xyz)), len(pts))
    P = (xyz[:, None, :] + r[:, None, None] * pts[None]).reshape(-1, 3)
    tree = cKDTree(xyz)
    free = np.empty(len(P), bool)
    kk = min(k, len(xyz))
    for a in range(0, len(P), chunk):
        d, j = tree.query(P[a:a + chunk], k=kk, distance_upper_bound=r.max(), workers=-1)
        d, j = d.reshape(len(d), -1), j.reshape(len(j), -1)
        valid = j < len(xyz)
        jj = np.where(valid, j, 0)
        cover = valid & (jj != owner[a:a + chunk, None]) & (d < r[jj] - 1e-9)
        free[a:a + chunk] = ~cover.any(1)
    frac = free.reshape(len(xyz), len(pts)).mean(1)
    return 4 * np.pi * r ** 2 * frac


def contact_graph(path) -> dict:
    """Residue contact graph with counted weights (P450 derivation, protocol G5)."""
    xyz, elem, resnum, resname = read_pdb(path)
    area = sasa(xyz, elem)
    residues = np.unique(resnum)
    exposed = {str(int(r)): int(((resnum == r) & (area > EXPOSED_CUTOFF)).sum()) for r in residues}
    letters = {str(int(r)): THREE.get(resname[resnum == r][0], "X") for r in residues}
    pairs = cKDTree(xyz).query_pairs(CONTACT_CUTOFF, output_type="ndarray")
    a, b = resnum[pairs[:, 0]], resnum[pairs[:, 1]]
    keep = a != b
    lo, hi = np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])
    edges = {}
    for u, v in zip(lo.tolist(), hi.tolist()):
        edges[(u, v)] = edges.get((u, v), 0) + 1
    return {"residues": [int(r) for r in residues], "letters": letters, "exposed": exposed,
            "edges": edges, "n_atoms": int(len(xyz))}


def summarise(acc: str) -> dict:
    """Per-protein numbers and per-residue exposure, cached as JSON next to the model."""
    cache = AF_DIR / f"AF-{acc}.graph.json"
    if cache.exists():
        return json.loads(cache.read_text())
    g = contact_graph(AF_DIR / f"AF-{acc}.pdb.gz")
    out = {"acc": acc, "n_residues": len(g["residues"]), "n_atoms": g["n_atoms"],
           "residues": g["residues"], "letters": g["letters"], "exposed": g["exposed"],
           "edges": [[u, v, w] for (u, v), w in g["edges"].items()],
           "exposed_atoms_total": int(sum(g["exposed"].values()))}
    tmp = cache.with_suffix(f".tmp{id(out)}")
    tmp.write_text(json.dumps(out))
    tmp.replace(cache)                       # atomic: a concurrent reader never sees half a file
    return out
