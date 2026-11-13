"""Figures for the circuit_kg paper, drawn from results/*.json only (no number is typed in).

Run from the repository root:  python circuit_kg/paper/figures/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"

# reference palette, light mode (validated: slots 1-3 pass all-pairs CVD and normal-vision floors)
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ Figure: thermodynamic coverage
def coverage():
    s = load("summary.json")
    sup = load("supplements.json")
    med = load("medium.json")["M2_direction"]
    tm = s["thermo_match"]["labels"]
    cov = sup["thermo_coverage"]
    rows = [
        ("Refused: a macromolecular participant", tm["refused"]),
        ("Absent: no measurement or no identifier", tm["absent"]),
        ("Measured (TECRDB match)", cov["measured"]),
        ("of the absent: derived by Hess's law", cov["hess_derived"]),
        ("of the refused: derived by carrier/slot", cov["carrier_slot_derived"]),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 2.0))
    y = np.arange(len(rows))[::-1]
    vals = [v for _, v in rows]
    cols = [INK2, INK2, S1, S1, S1]
    ax.barh(y, vals, height=0.62, color=cols, edgecolor=SURF, linewidth=1.0)
    for yi, (lab, v) in zip(y, rows):
        ax.text(v + 25, yi, f"{v:,}", va="center", ha="left", color=INK, fontsize=8)
    ax.set_yticks(y, [r for r, _ in rows])
    ax.set_xlabel("reactions (of 2,914)")
    ax.set_xlim(0, max(vals) * 1.12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    xgrid(ax)
    ax.tick_params(axis="y", length=0)
    fig.savefig(HERE / "coverage.pdf")
    plt.close(fig)
    return {"direction_answered": med.get("answered") if isinstance(med, dict) else None}


# ------------------------------------------------------------------ Figure: M1 transfer
def transfer():
    m = load("medium.json")["M1_transfer"]
    noise = m["replicate_noise"]["mean"]
    models = [("medium", "medium (1 parameter)", S1), ("constant", "constant (1 parameter)", INK2),
              ("affine-pH", "affine in pH (2 parameters)", S2)]
    designs = [("interpolation", "interpolation"), ("extrapolation", "extrapolation")]
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    w = 0.26
    for i, (k, lab, c) in enumerate(models):
        xs = np.arange(len(designs)) + (i - 1) * w
        v = [m["designs"][d]["all"][k]["mean"] for d, _ in designs]
        ax.bar(xs, v, width=w - 0.03, color=c, label=lab, edgecolor=SURF, linewidth=1.0)
        for x, yv in zip(xs, v):
            ax.text(x, yv + 0.06, f"{yv:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.axhline(noise, color=INK, lw=0.8, ls=(0, (3, 2)), label=f"replicate noise ({noise:.2f})")
    ax.set_xticks(range(len(designs)), [d for _, d in designs])
    ax.set_ylabel(r"mean $|\Delta_r G'^\circ$ error$|$ (kJ mol$^{-1}$)")
    ax.set_ylim(0, 5.0)
    ygrid(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(HERE / "transfer.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Figure: completion outcomes
def completion():
    c = load("complete.json")["surveys"]
    sem = [("as-annotated", "as annotated", S1), ("reversible", "reversible unless decided", S2),
           ("at-1mM", "sign of $\\Delta_r G'_m$ fixed", S3)]
    outs = [("answered", "one completion"), ("underdetermined", "several completions"),
            ("certified-empty", "none (certified empty)"),
            ("reachable-without-the-source", "made without the source")]
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.0))
    for ax, (k, title) in zip(axes, outs):
        v = [c[s]["labels"][k] for s, _, _ in sem]
        bars = ax.bar(range(3), v, color=[col for *_, col in sem], width=0.7, edgecolor=SURF, linewidth=1.0)
        for b, yv in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, yv * 0.96, f"{yv:,}", ha="center", va="top",
                    fontsize=6.5, rotation=90, color="white")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_ylim(0, max(v) * 1.08)
        ax.ticklabel_format(axis="y", style="plain")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ygrid(ax)
    handles = [matplotlib.patches.Patch(color=col, label=lab) for _, lab, col in sem]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.1),
               fontsize=7.5)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(HERE / "completion.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Figure: refinement
def refinement():
    r = load("summary.json")["refinement"]["classes_per_round"]
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    ax.plot(range(len(r)), r, color=S1, lw=2, marker="o", ms=4, mec=SURF, mew=1.0)
    ax.annotate(f"{r[-1]:,}", (len(r) - 1, r[-1]), xytext=(-4, -12), textcoords="offset points",
                ha="right", fontsize=7.5)
    ax.annotate(f"{r[0]:,}", (0, r[0]), xytext=(6, 2), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("refinement round")
    ax.set_ylabel("classes")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ygrid(ax)
    fig.savefig(HERE / "refinement.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Figure: AlphaFold vs experiment
def structures():
    rows = load("databases.json")["P1_rows"]
    j = np.array([r["jaccard"] for r in rows])
    c = np.array([r["jaccard_shift3"] for r in rows])
    order = np.argsort(j)
    fig, ax = plt.subplots(figsize=(5.6, 2.1))
    x = np.arange(len(rows))
    for xi, a, b in zip(x, j[order], c[order]):
        ax.plot([xi, xi], [b, a], color=GRID, lw=0.8, zorder=1)
    ax.scatter(x, j[order], s=14, color=S1, edgecolor=SURF, linewidth=0.6, zorder=3,
               label="AlphaFold model against X-ray chain")
    ax.scatter(x, c[order], s=14, color=S2, edgecolor=SURF, linewidth=0.6, zorder=3,
               label="control: numbering shifted by 3")
    for v, col in ((np.median(j), S1), (np.median(c), S2)):
        ax.axhline(v, color=col, lw=0.8, ls=(0, (3, 2)))
    ax.text(len(rows) - 0.5, np.median(j) - 0.03, f"median {np.median(j):.2f}", ha="right", va="top",
            fontsize=7, color=INK)
    ax.text(len(rows) - 0.5, np.median(c) + 0.03, f"median {np.median(c):.2f}", ha="right",
            va="bottom", fontsize=7, color=INK)
    ax.set_xlabel(f"protein chains, ordered by agreement (n = {len(rows)})")
    ax.set_ylabel("Jaccard index of contact sets")
    ax.set_xticks([])
    ax.set_ylim(0, 1.02)
    ygrid(ax)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.62))
    fig.savefig(HERE / "structures.pdf")
    plt.close(fig)


if __name__ == "__main__":
    coverage()
    transfer()
    completion()
    refinement()
    structures()
    print("figures written to", HERE)
