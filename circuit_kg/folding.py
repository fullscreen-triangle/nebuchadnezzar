"""Levinthal's folding ideas, implemented beside AlphaFold and tested against it.

Sources (levinthal main @ cf7fc86):
  models/publications/folding-partition-calculus  -- residues as Kuramoto oscillators,
      coupling K_ij, phase-lock groups as secondary structure, eigen-count theorem,
      contact prediction (Protocol 3);
  docs/chemistry/ternary-representation           -- interleaved base-3 addresses.

Two coupling matrices for the same protein:
  K_seq  the paper's sequence-only formula (copied from its validation script);
  K_AF   the AlphaFold residue contact graph (counted heavy-atom pairs < 4.5 A,
         proteins.contact_graph), so the paper's dynamics get a structure-given K.

Secondary structure comes from a DSSP-style assignment on the AlphaFold backbone
(Kabsch & Sander electrostatic H-bond energy, helix from consecutive 4-turns,
strands from bridges with a bridged neighbour), checked against the author
HELIX/SHEET records of experimental PDB entries (`check_ss`).
"""
from __future__ import annotations

import gzip
import json
import math
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .core import RAW
from .proteins import AF_DIR

PDB_DIR = RAW / "pdb"
VALIDATION_PDB = ("1CRN", "1UBQ", "2LZM", "3CHY")
AA_COORDS = {   # S-entropy coordinates (S_k, S_t, S_e), from the paper's validation script
    'I': (1.000, 0.636, 0.000), 'V': (0.967, 0.476, 0.000),
    'L': (0.922, 0.636, 0.000), 'F': (0.811, 0.774, 0.000),
    'C': (0.778, 0.289, 0.100), 'M': (0.711, 0.613, 0.000),
    'A': (0.700, 0.170, 0.000), 'G': (0.456, 0.000, 0.000),
    'T': (0.422, 0.334, 0.300), 'S': (0.411, 0.172, 0.300),
    'W': (0.400, 1.000, 0.100), 'Y': (0.356, 0.796, 0.200),
    'P': (0.322, 0.373, 0.000), 'H': (0.144, 0.555, 0.600),
    'N': (0.111, 0.322, 0.500), 'Q': (0.111, 0.499, 0.500),
    'D': (0.111, 0.304, 1.000), 'E': (0.111, 0.467, 1.000),
    'K': (0.067, 0.647, 1.000), 'R': (0.000, 0.676, 1.000),
}
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


# ------------------------------------------------------------------ structures

def read_backbone(path) -> dict:
    """N, CA, C, O per residue (first model, chain A or the first chain), CA B-factor."""
    opener = gzip.open if str(path).endswith(".gz") else open
    res, order, chain = {}, [], None
    with opener(path, "rt", encoding="ascii", errors="replace") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            ch = line[21]
            chain = chain or ch
            if ch != chain or line[16] not in " A":
                continue
            key = line[22:27]
            name = line[12:16].strip()
            if key not in res:
                res[key] = {"aa": THREE.get(line[17:20], "X"), "num": int(line[22:26])}
                order.append(key)
            if name in ("N", "CA", "C", "O"):
                res[key][name] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                if name == "CA":
                    res[key]["b"] = float(line[60:66])
    keep = [k for k in order if all(a in res[k] for a in ("N", "CA", "C", "O"))]
    arr = lambda a: np.array([res[k][a] for k in keep])
    return {"seq": "".join(res[k]["aa"] for k in keep), "num": [res[k]["num"] for k in keep],
            "N": arr("N"), "CA": arr("CA"), "C": arr("C"), "O": arr("O"),
            "b": np.array([res[k]["b"] for k in keep])}


def hbonds(bb) -> np.ndarray:
    """hb[a, d]: C=O of residue a accepts an H-bond from N-H of residue d (E < -0.5 kcal/mol)."""
    N, C, O = bb["N"], bb["C"], bb["O"]
    n = len(N)
    H = N.copy()
    v = C[:-1] - O[:-1]
    H[1:] = N[1:] + v / np.linalg.norm(v, axis=1)[:, None]
    d = lambda X, Y: np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
    with np.errstate(divide="ignore"):
        E = 0.084 * 332 * (1 / d(O, N) + 1 / d(C, H) - 1 / d(O, H) - 1 / d(C, N))
    hb = E < -0.5
    idx = np.arange(n)
    hb[np.abs(idx[:, None] - idx[None, :]) < 2] = False
    hb[:, 0] = False                                # no H on the first residue
    pro = np.array([a == "P" for a in bb["seq"]])
    hb[:, pro] = False                              # proline has no N-H
    return hb


def assign_ss(bb) -> np.ndarray:
    """Per-residue 'H', 'E' or '-' (DSSP core: 4-turn helices, bridged ladders)."""
    hb = hbonds(bb)
    n = hb.shape[0]
    lab = np.array(["-"] * n)
    turn4 = np.array([i + 4 < n and hb[i, i + 4] for i in range(n)])
    for i in range(1, n):
        if turn4[i - 1] and turn4[i]:
            lab[i:i + 4] = "H"
    bridged = np.zeros(n, bool)
    partners = defaultdict(set)
    H_ = lambda a, d: 0 <= a < n and 0 <= d < n and hb[a, d]
    for i in range(1, n - 1):
        for j in range(i + 3, n - 1):
            par = (H_(i - 1, j) and H_(j, i + 1)) or (H_(j - 1, i) and H_(i, j + 1))
            anti = (H_(i, j) and H_(j, i)) or (H_(i - 1, j + 1) and H_(j - 1, i + 1))
            if par or anti:
                bridged[i] = bridged[j] = True
                partners[i].add(j); partners[j].add(i)
    for i in np.flatnonzero(bridged):
        ladder = any(abs(p - q) == 1 for p in partners[i]
                     for k in (i - 1, i + 1) for q in partners.get(k, ()))
        if ladder and lab[i] != "H":
            lab[i] = "E"
    return lab


def segments(lab, min_h=4, min_e=2):
    """Group labels per residue: each SSE run one group, each loop stretch one group."""
    groups, seg, g = [], [], -1
    runs = []
    i = 0
    while i < len(lab):
        j = i
        while j < len(lab) and lab[j] == lab[i]:
            j += 1
        runs.append((lab[i], i, j))
        i = j
    sse = []
    clean = []
    for t, a, b in runs:
        ok = (t == "H" and b - a >= min_h) or (t == "E" and b - a >= min_e)
        clean.append(("S" if ok else "-", a, b, t))
        if ok:
            sse.append((t, a, b))
    out = np.zeros(len(lab), int)
    prev = None
    for t, a, b, _ in clean:
        if t == "S" or prev != "-":
            g += 1
        out[a:b] = g
        prev = t
    return out, sse


def pdb_ss_records(path) -> dict:
    """Author HELIX/SHEET records: residue number -> 'H' or 'E' (chain of the first ATOM)."""
    lab, chain = {}, None
    lines = Path(path).read_text(encoding="ascii", errors="replace").splitlines()
    for line in lines:
        if line.startswith("ATOM"):
            chain = line[21]; break
    for line in lines:
        if line.startswith("HELIX ") and line[19] == chain:
            for k in range(int(line[21:25]), int(line[33:37]) + 1):
                lab[k] = "H"
        if line.startswith("SHEET ") and line[21] == chain:
            for k in range(int(line[22:26]), int(line[33:37]) + 1):
                lab[k] = "E"
    return lab


def fetch_pdb(ids=VALIDATION_PDB):
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    for i in ids:
        p = PDB_DIR / f"{i}.pdb"
        if not p.exists():
            req = urllib.request.Request(f"https://files.rcsb.org/download/{i}.pdb",
                                         headers={"User-Agent": "circuit-kg-kernel"})
            p.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    return [PDB_DIR / f"{i}.pdb" for i in ids]


def check_ss(paths) -> dict:
    """Per-residue three-state agreement (Q3) with author records; control: shuffled residues."""
    rows, rng = {}, np.random.default_rng(0)
    for p in paths:
        bb = read_backbone(p)
        ref = pdb_ss_records(p)
        lab = assign_ss(bb)
        truth = np.array([ref.get(k, "-") for k in bb["num"]])
        perm = rng.permutation(len(lab))
        shuffled = {k: bb[k][perm] if k in ("N", "CA", "C", "O") else bb[k] for k in bb}
        rows[Path(p).stem] = {"residues": len(lab), "q3": float((lab == truth).mean()),
                              "helix_ref": int((truth == "H").sum()),
                              "helix_assigned": int((lab == "H").sum()),
                              "strand_ref": int((truth == "E").sum()),
                              "strand_assigned": int((lab == "E").sum()),
                              "control_shuffled_q3": float((assign_ss(shuffled) == truth).mean())}
    return rows


# ------------------------------------------------------------------ couplings

def k_seq(seq, K_scale=5.0) -> np.ndarray:
    """The paper's validation-script coupling (run_kuramoto), vectorised."""
    c = np.array([AA_COORDS.get(a, (0.5, 0.5, 0.5)) for a in seq])
    n = len(seq)
    s = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float)
    d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
    cc = np.array([a == "C" for a in seq])
    K = K_scale * (0.3 * np.exp(-s / 2) + 0.35 * (0.8 * np.exp(-(s - 4) ** 2)
                                                  + 0.4 * np.exp(-(s - 3) ** 2))
                   + 0.25 * np.exp(-d / 0.2) * (1 - np.exp(-s / 8))
                   + 0.1 * np.where(cc[:, None] & cc[None], 2 * np.exp(-d / 0.5), 0))
    np.fill_diagonal(K, 0)
    return K


def k_af(acc: str, nums) -> np.ndarray | None:
    p = AF_DIR / f"AF-{acc}.graph.json"
    if not p.exists():
        return None
    g = json.loads(p.read_text())
    pos = {r: i for i, r in enumerate(nums)}
    K = np.zeros((len(nums), len(nums)))
    for u, v, w in g["edges"]:
        if u in pos and v in pos:
            K[pos[u], pos[v]] = K[pos[v], pos[u]] = w
    return K


def omega(seq):
    return np.array([AA_COORDS.get(a, (0.5, 0.5, 0.5))[0] * 10 + 1 for a in seq])


# ------------------------------------------------------------------ dynamics

def kuramoto(K, w, K0s, steps=1500, dt=0.01, every=10, seed=0):
    """dphi_i/dt = w_i + K0 * sum_j Kn_ij sin(phi_j - phi_i), Kn = K / mean(K>0), for each K0.

    Returns the phases sampled every `every` steps over the last quarter: (samples, B, N)."""
    pos = K[K > 0]
    Kn = csr_matrix(K / pos.mean()) if len(pos) else csr_matrix(K)
    K0 = np.asarray(K0s, float)[:, None]
    rng = np.random.default_rng(seed)
    phi = np.tile(rng.uniform(0, 2 * np.pi, len(w)), (len(K0), 1))
    keep = []
    for t in range(steps):
        z = np.exp(1j * phi)
        field = (Kn @ z.T).T
        phi = phi + dt * (w + K0 * np.imag(np.conj(z) * field))
        if t >= steps * 3 // 4 and t % every == 0:
            keep.append(phi % (2 * np.pi))
    return np.array(keep), Kn


def locked_pairs(samples, b, tol=0.2):
    """Boolean N x N: wrapped phase difference below tol at every sampled time."""
    ok = None
    for ph in samples[:, b, :]:
        d = np.abs(np.angle(np.exp(1j * (ph[:, None] - ph[None, :]))))
        ok = d < tol if ok is None else ok & (d < tol)
    np.fill_diagonal(ok, False)
    return ok


def components(adj) -> np.ndarray:
    return connected_components(csr_matrix(adj), directed=False)[1]


def local_order(samples, b, Kn) -> np.ndarray:
    deg = np.asarray(Kn.sum(1)).ravel()
    vals = [np.abs(Kn @ np.exp(1j * ph)) / np.where(deg > 0, deg, 1) for ph in samples[:, b, :]]
    return np.mean(vals, axis=0)


# ------------------------------------------------------------------ ternary addresses

def ternary_address(xyz, depth=12):
    """Interleaved base-3 digits: position j picks axis j mod 3, the digit picks the third."""
    lo = xyz.min(0)
    side = float((xyz.max(0) - lo).max()) or 1.0
    u = np.clip((xyz - lo) / side, 0, 1 - 1e-12)
    digits = np.zeros((len(xyz), depth), int)
    frac = u.copy()
    for j in range(depth):
        a = j % 3
        dgt = np.floor(frac[:, a] * 3).astype(int)
        digits[:, j] = dgt
        frac[:, a] = frac[:, a] * 3 - dgt
    return digits, side


def shared_prefix(digits):
    eq = digits[:, None, :] == digits[None, :, :]
    return np.where(eq.all(-1), digits.shape[1], np.argmin(eq, axis=-1))


# ------------------------------------------------------------------ experiments

def high_confidence_models(lo=50, hi=250, limit=150, min_plddt=90):
    out = []
    for meta in sorted(AF_DIR.glob("AF-*.json")):
        if meta.name.endswith(".graph.json"):
            continue
        m = json.loads(meta.read_text())
        if m.get("status") != "ok" or (m.get("mean_plddt") or 0) < min_plddt:
            continue
        if not (AF_DIR / f"AF-{m['acc']}.graph.json").exists():
            continue
        bb = read_backbone(AF_DIR / f"AF-{m['acc']}.pdb.gz")
        if lo <= len(bb["seq"]) <= hi:
            out.append((m["acc"], bb))
        if limit and len(out) >= limit:
            break
    return out


def _precision_top(score, truth, k, min_sep=6):
    n = truth.shape[0]
    iu = np.triu_indices(n, k=min_sep)
    order = np.argsort(-score[iu], kind="stable")[:k]
    t = truth[iu]
    return float(t[order].mean()), float(t[order].sum() / max(1, t.sum())), float(t.mean())


def f1_contacts(models, crambin: Path) -> dict:
    """Protocol 3: top-L precision/recall of sequence-only scores against C-alpha < 8 A."""
    rng = np.random.default_rng(0)

    def scores(seq):
        K = k_seq(seq)
        h = np.array([AA_COORDS.get(a, (0.5, 0.5, 0.5))[0] for a in seq])
        return {"K_seq": K, "hydrophobic_product": np.outer(h, h),
                "random": rng.random(K.shape)}

    def run(bb):
        ca = bb["CA"]
        truth = np.linalg.norm(ca[:, None] - ca[None], axis=-1) < 8.0
        return {m: dict(zip(("precision_top_L", "recall_top_L", "base_rate"),
                            _precision_top(s, truth, len(bb["seq"]))))
                for m, s in scores(bb["seq"]).items()}
    out = {"1CRN": run(read_backbone(crambin))}
    rows = [run(bb) for _, bb in models]
    from scipy.stats import wilcoxon
    p = {m: np.array([r[m]["precision_top_L"] for r in rows]) for m in rows[0]}
    base = np.array([r["random"]["base_rate"] for r in rows])
    out["alphafold"] = {
        "proteins": len(rows),
        "precision_over_base_rate": {m: float(v.mean() / base.mean()) for m, v in p.items()},
        "paper_criterion_met": {m: int(sum(r[m]["precision_top_L"] >= 0.7
                                           and r[m]["recall_top_L"] >= 0.5 for r in rows))
                                for m in p},
        "K_seq_vs_random_two_sided_p": float(wilcoxon(p["K_seq"], p["random"]).pvalue)}
    return out


def f2_eigencount(models) -> dict:
    rows = []
    for acc, bb in models:
        lab = assign_ss(bb)
        _, sse = segments(lab)
        if len(sse) < 2:
            continue
        n = len(bb["seq"])
        Lbar = np.mean([b - a for _, a, b in sse])
        rec = {"acc": acc, "N": n, "M": len(sse)}
        for name, K in (("K_seq", k_seq(bb["seq"])), ("K_AF", k_af(acc, bb["num"]))):
            intra = [K[i, j] for _, a, b in sse for i in range(a, b) for j in range(i + 1, min(b, i + 5))]
            K0 = float(np.mean(intra))
            lam = np.linalg.eigvalsh(K)
            rec[name] = int((lam > K0 * math.exp(-0.5) * Lbar).sum())
        rows.append(rec)
    ratio = np.array([r["M"] / r["N"] for r in rows])
    for k, r in enumerate(rows):
        r["length_baseline"] = int(round(r["N"] * (ratio.sum() - ratio[k]) / (len(ratio) - 1)))
    M = np.array([r["M"] for r in rows])

    def stats(key):
        v = np.array([r[key] for r in rows])
        return {"exact_match": int((v == M).sum()), "mean_abs_error": float(np.abs(v - M).mean()),
                "median_count": float(np.median(v))}
    return {"proteins": len(rows), "median_M": float(np.median(M)),
            **{k: stats(k) for k in ("K_seq", "K_AF", "length_baseline")}}


K0_SWEEP = (0.5, 1, 2, 5, 10)


def f3_phaselock(models) -> dict:
    from scipy.stats import wilcoxon
    from sklearn.metrics import adjusted_rand_score as ari
    rows = []
    for acc, bb in models:
        truth, sse = segments(assign_ss(bb))
        if len(sse) < 2:
            continue
        n, w = len(bb["seq"]), omega(bb["seq"])
        g = truth.max() + 1
        blocks = np.minimum(np.arange(n) * g // n, g - 1)
        rec = {"acc": acc, "blocks": ari(truth, blocks)}
        for name, K in (("K_AF", k_af(acc, bb["num"])), ("K_seq", k_seq(bb["seq"]))):
            samples, Kn = kuramoto(K, w, K0_SWEEP)
            best_lock = best_static = -1.0
            iu = np.triu_indices(n, 1)
            kvals = Kn.toarray()[iu]
            for b in range(len(K0_SWEEP)):
                L = locked_pairs(samples, b)
                best_lock = max(best_lock, ari(truth, components(L)))
                m = int(np.triu(L, 1).sum())
                S = np.zeros((n, n), bool)
                if m:
                    top = np.argsort(-kvals, kind="stable")[:m]
                    S[iu[0][top], iu[1][top]] = True
                best_static = max(best_static, ari(truth, components(S | S.T)))
            rec[f"lock_{name}"], rec[f"static_{name}"] = best_lock, best_static
        rows.append(rec)

    def cmp(a, b):
        x, y = np.array([r[a] for r in rows]), np.array([r[b] for r in rows])
        return {"mean_a": float(x.mean()), "mean_b": float(y.mean()),
                "a_better": int((x > y).sum()), "b_better": int((x < y).sum()),
                "wilcoxon_one_sided_p_a_greater": float(wilcoxon(x, y, alternative="greater").pvalue)}
    return {"proteins": len(rows), "K0_sweep": list(K0_SWEEP),
            "lock_K_AF_vs_blocks": cmp("lock_K_AF", "blocks"),
            "lock_K_AF_vs_static_K_AF": cmp("lock_K_AF", "static_K_AF"),
            "lock_K_seq_vs_blocks": cmp("lock_K_seq", "blocks"),
            "static_K_AF_vs_blocks": cmp("static_K_AF", "blocks")}


def f4_sites(located: dict, max_n=1500, K0=2.0) -> dict:
    from scipy.stats import mannwhitneyu
    named = defaultdict(set)
    for sites in located.values():
        for acc, pos, res in sites:
            named[acc].add(pos)
    site_r, other_r, site_b, other_b = [], [], [], []
    for acc, poss in sorted(named.items()):
        p = AF_DIR / f"AF-{acc}.pdb.gz"
        if not p.exists():
            continue
        bb = read_backbone(p)
        if len(bb["seq"]) > max_n:
            continue
        K = k_af(acc, bb["num"])
        if K is None:
            continue
        samples, Kn = kuramoto(K, omega(bb["seq"]), [K0])
        r = local_order(samples, 0, Kn)
        types = {bb["seq"][bb["num"].index(q)] for q in poss if q in bb["num"]}
        for i, (num, aa) in enumerate(zip(bb["num"], bb["seq"])):
            if aa not in types:
                continue
            (site_r if num in poss else other_r).append(r[i])
            (site_b if num in poss else other_b).append(bb["b"][i])
    site_r, other_r, site_b, other_b = map(np.array, (site_r, other_r, site_b, other_b))

    def mw(a, b):
        if len(a) < 3 or len(b) < 3:
            return {"n_sites": int(len(a)), "n_others": int(len(b)), "p_less": None}
        return {"n_sites": int(len(a)), "n_others": int(len(b)),
                "median_sites": float(np.median(a)), "median_others": float(np.median(b)),
                "p_less": float(mannwhitneyu(a, b, alternative="less").pvalue)}
    return {"K0": K0, "local_order": mw(site_r, other_r),
            "plddt": mw(site_b, other_b),
            "local_order_plddt_lt_70": mw(site_r[site_b < 70], other_r[other_b < 70]),
            "local_order_plddt_ge_70": mw(site_r[site_b >= 70], other_r[other_b >= 70])}


def f5_ternary(models, depth=12, contact=8.0) -> dict:
    viol = total = shared = contacts = 0
    levels = []
    for _, bb in models:
        xyz = bb["CA"]
        dig, side = ternary_address(xyz, depth)
        pre = shared_prefix(dig)
        d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
        iu = np.triu_indices(len(xyz), 1)
        bound = math.sqrt(3) * side * 3.0 ** -(pre[iu] // 3)
        viol += int((d[iu] > bound + 1e-9).sum()); total += len(iu[0])
        m = max(0, int(math.floor(math.log(side / contact, 3))))       # cells >= contact on a side
        levels.append(m)
        c = d[iu] < contact
        contacts += int(c.sum())
        shared += int((pre[iu][c] >= 3 * m).sum())
    return {"proteins": len(models), "pairs": total, "bound_violations": viol,
            "contact_A": contact, "median_levels_with_cells_ge_contact": float(np.median(levels)),
            "contacts": contacts, "contacts_sharing_that_prefix": shared,
            "fraction_sharing": shared / contacts if contacts else None}
