"""
Figures of the two-column manuscript, drawn only from results files (no number is typed in).

Sources: ``circuit_kg/results/*.json`` (the research kernel, commit 15fb81a) and
``paper/journal/data/extension.json`` (written by ``scripts/extension_audit.py`` from
dcat-ap-plus-homo-sapien). Each figure is one row of four panels spanning both columns.

Run from the nebuchadnezzar root::

    python circuit_kg/paper/journal/figures/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RES = HERE.parents[2] / "results"
EXT = HERE.parent / "data" / "extension.json"

# Reference palette, light mode. Slots 1-3 were validated (all-pairs CVD and normal-vision floors).
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
BLUE_RAMP = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#a9c8ef", S1, "#123f75"])
DIVERGING = LinearSegmentedColormap.from_list("div", [S1, "#9cc0ea", "#efeeea", "#f4b496", S2])

WIDTH = 7.16  # inches: the text width of the two-column layout
HEIGHT = 1.92

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.titlesize": 7.2,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.2,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
        "pdf.fonttype": 42,
        "axes.titlelocation": "left",
        "axes.titlepad": 5,
    }
)


def load(name: str) -> dict:
    """A kernel results file."""
    return json.loads((RES / name).read_text(encoding="utf-8"))


def ext() -> dict:
    """The extension's measurements."""
    return json.loads(EXT.read_text(encoding="utf-8"))


def comma(ax, axis: str = "y") -> None:
    """Thousands separators on an axis."""
    fmt = ticker.FuncFormatter(lambda x, _: f"{int(round(x)):,}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def grid(ax, axis: str = "y") -> None:
    """A recessive grid behind the marks."""
    (ax.yaxis if axis == "y" else ax.xaxis).grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


TITLE_Y = 0.965  # panel titles share one baseline across the row


def panel(ax, letter: str, title: str, x: float | None = None) -> None:
    """Panel letter and title, left-aligned on the axes (or at ``x``), on the row's common baseline."""
    fig = ax.get_figure()
    x0 = ax.get_position().x0 if x is None else x
    fig.text(x0, TITLE_Y, rf"$\bf{{{letter}}}$  {title}", ha="left", va="top", fontsize=7.4, color=INK)


def row(
    projection3d: int | None = None,
    wspace: float = 0.62,
    left: float = 0.06,
    widths: tuple[float, float, float, float] = (1, 1, 1, 1),
    bottom: float = 0.2,
) -> tuple[plt.Figure, list]:
    """A figure of four panels in a row; one may be three-dimensional."""
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    gs = fig.add_gridspec(1, 4, left=left, right=0.99, bottom=bottom, top=0.83, wspace=wspace, width_ratios=widths)
    axes = []
    for i in range(4):
        if i == projection3d:
            axes.append(fig.add_subplot(gs[0, i], projection="3d"))
        else:
            axes.append(fig.add_subplot(gs[0, i]))
    fig._cells = [gs[0, i].get_position(fig).x0 for i in range(4)]  # type: ignore[attr-defined]
    return fig, axes


def save(fig: plt.Figure, name: str) -> None:
    """Write the figure as vector PDF."""
    fig.savefig(HERE / f"{name}.pdf")
    plt.close(fig)


# ============================================================ Figure 1: structure and coverage
def figure_structure() -> None:
    e = ext()
    s = load("summary.json")
    sup = load("supplements.json")
    x1 = load("extend.json")["X1_axes"]
    fig, (a, b, c, d) = row(left=0.1, wspace=0.85, widths=(1.2, 0.95, 1.0, 0.8))

    # (a) verdict status by question, from the extension's own run
    questions = [("reaction-energy", "measured $\\Delta_rG^\\circ$"), ("hess-energy", "Hess's law"),
                 ("direction", "direction"), ("node-solve", "species state")]
    labels = [("answered", "answered"), ("certified-empty", "empty"), ("absent", "absent"),
              ("underdetermined", "underdet."), ("undecidable", "undecid."), ("refused", "refused")]
    m = np.array([[e["kernel"]["verdicts"][q].get(lab, 0) for lab, _ in labels] for q, _ in questions], float)
    show = np.where(m > 0, np.log10(np.maximum(m, 1)), np.nan)
    a.imshow(show, cmap=BLUE_RAMP, vmin=0, vmax=4, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                v = int(m[i, j])
                a.text(j, i, f"{v:,}", ha="center", va="center", fontsize=6.0,
                       color="white" if show[i, j] > 2.6 else INK)
    a.set_xticks(range(len(labels)), [t for _, t in labels], rotation=40, ha="right", rotation_mode="anchor")
    a.set_yticks(range(len(questions)), [t for _, t in questions])
    a.tick_params(length=0)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.set_xticks(np.arange(-0.5, len(labels)), minor=True)
    a.set_yticks(np.arange(-0.5, len(questions)), minor=True)
    a.grid(which="minor", color=SURF, lw=1.2)
    a.tick_params(which="minor", length=0)
    panel(a, "a", "Verdict status per question")

    # (b) thermodynamic coverage
    tm = s["thermo_match"]["labels"]
    cov = sup["thermo_coverage"]
    rows = [("refused", tm["refused"], MUTED), ("absent", tm["absent"], MUTED),
            ("measured", cov["measured"], S1), ("Hess's law", cov["hess_derived"], S1),
            ("residue red.", cov["carrier_slot_derived"], S1)]
    y = np.arange(len(rows))[::-1]
    b.barh(y, [v for _, v, _ in rows], height=0.64, color=[c for *_, c in rows], edgecolor=SURF, lw=0.8)
    for yi, (_, v, _) in zip(y, rows, strict=True):
        b.text(v + 40, yi, f"{v:,}", va="center", fontsize=6.2, color=INK)
    b.set_yticks(y, [r for r, _, _ in rows])
    b.set_xlim(0, tm["refused"] * 1.3)
    b.set_xlabel("reactions (of 2,914)")
    b.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    comma(b, "x")
    grid(b, "x")
    b.tick_params(axis="y", length=0)
    panel(b, "b", "Thermodynamic coverage")

    # (c) absence codes
    gates = x1["gate_letters"]
    order = [("-", "not recorded ($-$)"), ("open", "unmappable (open)"), ("+", "witnessed ($+$)")]
    cols = [S2, MUTED, S1]
    y = np.arange(len(order))[::-1]
    c.barh(y, [gates[k] for k, _ in order], height=0.62, color=cols, edgecolor=SURF, lw=0.8)
    for yi, (k, _) in zip(y, order, strict=True):
        c.text(gates[k] + 12, yi, f"{gates[k]:,}", ha="left", va="center", fontsize=6.2, color=INK)
    c.set_yticks(y, [t for _, t in order])
    c.set_xlabel("empty or absent results")
    c.set_xlim(0, max(gates.values()) * 1.3)
    c.xaxis.set_major_locator(ticker.MultipleLocator(250))
    grid(c, "x")
    c.tick_params(axis="y", length=0)
    panel(c, "c", "Why an answer is missing")

    # (d) colour refinement
    r = s["refinement"]["classes_per_round"]
    d.plot(range(len(r)), r, color=S1, lw=1.6, marker="o", ms=3.2, mec=SURF, mew=0.8)
    d.axhline(8244, color=INK2, lw=0.7, ls=(0, (3, 2)))
    d.text(len(r) - 1, 8244 * 1.02, "vertices: 8,244", ha="right", va="bottom", fontsize=6.0, color=INK2)
    d.annotate(f"{r[-1]:,} classes", (len(r) - 1, r[-1]), xytext=(-2, -9), textcoords="offset points",
               ha="right", fontsize=6.0)
    d.set_xlabel("refinement round")
    d.set_ylabel("equivalence classes")
    d.set_ylim(0, 9400)
    d.xaxis.set_major_locator(ticker.MultipleLocator(2))
    comma(d)
    grid(d)
    panel(d, "d", "Refinement")
    save(fig, "fig_structure")


# ============================================================ Figure 2: medium and direction
def figure_medium() -> None:
    e = ext()
    med = load("medium.json")
    fig, (a, b, c, d) = row(projection3d=0, wspace=0.62, left=0.0, widths=(1.22, 1.0, 1.05, 0.9))

    # (a) Delta_r G'm over pH x pMg (three-dimensional), at the upper ionic strength
    surf = e["surface"]
    ph, pmg = np.array(surf["ph"]), np.array(surf["pmg"])
    X, Y = np.meshgrid(ph, pmg)
    keys = sorted(surf["ionic_strength"], key=float)
    z_hi = np.array(surf["ionic_strength"][keys[-1]])
    z_lo = np.array(surf["ionic_strength"][keys[0]])
    lim = float(abs(z_hi).max())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    a.plot_surface(X, Y, z_hi, facecolors=DIVERGING(norm(z_hi)), rstride=1, cstride=1, lw=0, antialiased=False,
                   shade=False)
    a.plot_surface(X, Y, np.zeros_like(z_hi), color=MUTED, alpha=0.18, lw=0, shade=False)

    def zero_line(z: np.ndarray) -> tuple[list[float], list[float]]:
        """The pH at which Delta_r G'm crosses zero, for each pMg row (linear interpolation)."""
        xs_, ys_ = [], []
        for i, yv in enumerate(pmg):
            zr = z[i]
            for j in range(len(ph) - 1):
                if zr[j] == 0 or zr[j] * zr[j + 1] < 0:
                    xs_.append(ph[j] + (ph[j + 1] - ph[j]) * zr[j] / (zr[j] - zr[j + 1]))
                    ys_.append(yv)
                    break
        return xs_, ys_

    for z, style, lw in ((z_hi, "-", 1.1), (z_lo, (0, (2.5, 1.5)), 0.8)):
        xs_, ys_ = zero_line(z)
        a.plot(xs_, ys_, np.zeros(len(xs_)), color=INK, lw=lw, ls=style, zorder=10)
    a.set_xlabel("pH", labelpad=-9)
    a.set_ylabel("pMg", labelpad=-9)
    a.set_zlabel(r"$\Delta_r G'_\mathrm{m}$ (kJ mol$^{-1}$)", labelpad=-8)
    a.tick_params(axis="x", pad=-4, labelsize=6.0)
    a.tick_params(axis="y", pad=-4, labelsize=6.0)
    a.tick_params(axis="z", pad=-2, labelsize=6.0)
    a.xaxis.set_major_locator(ticker.MaxNLocator(3))
    a.yaxis.set_major_locator(ticker.MaxNLocator(3))
    a.zaxis.set_major_locator(ticker.MaxNLocator(4))
    for axis in (a.xaxis, a.yaxis, a.zaxis):
        axis.pane.set_facecolor(SURF)
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = 0.4
        axis.line.set_linewidth(0.5)
    a.view_init(elev=22, azim=-58)
    a.set_box_aspect((1, 1, 0.85), zoom=0.98)
    panel(a, "a", "Sign of $\\Delta_rG'_\\mathrm{m}$ across the medium", x=0.012)

    # (b) held-out prediction (M1)
    m = med["M1_transfer"]
    noise = m["replicate_noise"]["mean"]
    models = [("medium", "Legendre transform", S1), ("constant", "constant", MUTED), ("affine-pH", "affine in pH", S2)]
    designs = ["interpolation", "extrapolation"]
    w = 0.25
    for i, (k, lab, col) in enumerate(models):
        xs = np.arange(2) + (i - 1) * w
        v = [m["designs"][dd]["all"][k]["mean"] for dd in designs]
        b.bar(xs, v, width=w - 0.03, color=col, label=lab, edgecolor=SURF, lw=0.8)
        for xx, yy in zip(xs, v, strict=True):
            b.text(xx, yy + 0.05, f"{yy:.2f}", ha="center", va="bottom", fontsize=6.0, color=INK)
    b.axhline(noise, color=INK, lw=0.7, ls=(0, (3, 2)), label="replicate noise")
    b.set_xticks(range(2), designs)
    b.set_ylabel(r"mean abs. error (kJ mol$^{-1}$)")
    b.set_ylim(0, 6.2)
    b.tick_params(axis="x", length=0)
    grid(b)
    b.legend(frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.1, ncol=1)
    panel(b, "b", "Held-out conditions (M1)")

    # (c) Delta_r G'm intervals of the reactions with a chemical value (extension run)
    dirs = [x for x in e["kernel"]["directions"] if x["dG_prime_m_range"]]
    dirs.sort(key=lambda x: sum(x["dG_prime_m_range"]) / 2)
    for i, x in enumerate(dirs):
        lo, hi = x["dG_prime_m_range"]
        if x["label"] == "answered":
            col = S3
        elif x["blocker"] == "concentration":
            col = S1
        else:
            col = S2
        c.plot([i, i], [lo, hi], color=col, lw=1.1, solid_capstyle="butt")
    c.axhline(0, color=INK, lw=0.6)
    c.set_xlabel(f"reactions, ordered by midpoint (n = {len(dirs)})")
    c.set_ylabel(r"$\Delta_rG'_\mathrm{m}$ range (kJ mol$^{-1}$)")
    c.set_xticks([])
    grid(c)
    handles = [Patch(color=S1, label="sign fixed: concentrations decide"), Patch(color=S2, label="sign open: medium decides"),
               Patch(color=S3, label="direction answered")]
    c.legend(handles=handles, frameon=False, loc="upper left", handlelength=0.9, borderaxespad=0.1)
    lim_c = max(abs(min(x["dG_prime_m_range"][0] for x in dirs)), abs(max(x["dG_prime_m_range"][1] for x in dirs)))
    c.set_ylim(-lim_c * 1.05, lim_c * 1.65)
    panel(c, "c", "What blocks each direction")

    # (d) direction verdicts by source of the chemical value (kernel, M2)
    lab = med["M2_direction"]["labels_by_source"]
    sources = [("measured", "measured"), ("hess", "Hess's\nlaw"), ("carrier-slot", "residue\nreduction")]
    stack = [("undecidable:concentration", "concentrations decide", S1), ("undecidable:medium", "medium decides", S2),
             ("answered:forward", "answered", S3)]
    bottom = np.zeros(len(sources))
    xs = np.arange(len(sources))
    for key, name, col in stack:
        v = np.array([lab.get(f"{s}|{key}", 0) for s, _ in sources], float)
        d.bar(xs, v, bottom=bottom, width=0.6, color=col, edgecolor=SURF, lw=0.8, label=name)
        bottom += v
    for xi, tot in zip(xs, bottom, strict=True):
        d.text(xi, tot + 1.5, f"{int(tot)}", ha="center", va="bottom", fontsize=6.2)
    d.set_xticks(xs, [t for _, t in sources])
    d.set_ylabel("reactions with a chemical value")
    d.set_ylim(0, max(bottom) * 1.2)
    d.tick_params(axis="x", length=0)
    grid(d)
    panel(d, "d", "By source (M2)")
    save(fig, "fig_medium")


# ============================================================ Figure 3: proteins, enzymes, expansion
def figure_proteins() -> None:
    db = load("databases.json")
    kin = load("kinetics.json")
    comp = load("complete.json")["surveys"]
    fig, (a, b, c, d) = row()

    # (a) AlphaFold against experiment (A1)
    rows = db["P1_rows"]
    j = np.array([r["jaccard"] for r in rows])
    sh = np.array([r["jaccard_shift3"] for r in rows])
    o = np.argsort(j)
    x = np.arange(len(rows))
    a.scatter(x, j[o], s=7, color=S1, edgecolor=SURF, lw=0.4, zorder=3, label="model vs. X-ray chain")
    a.scatter(x, sh[o], s=7, color=S2, edgecolor=SURF, lw=0.4, zorder=3, label="control: numbering +3")
    for v, col in ((np.median(j), S1), (np.median(sh), S2)):
        a.axhline(v, color=col, lw=0.6, ls=(0, (3, 2)), zorder=2)
        a.text(1, v + 0.035, f"median {v:.2f}", ha="left", va="bottom", fontsize=6.0, zorder=4,
               bbox={"boxstyle": "square,pad=0.1", "fc": SURF, "ec": "none"})
    a.set_xticks([])
    a.set_xlabel(f"protein chains, by agreement (n = {len(rows)})")
    a.set_ylabel("Jaccard index of contacts")
    a.set_ylim(0, 1.05)
    grid(a)
    a.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.52), handletextpad=0.2, borderaxespad=0.1)
    panel(a, "a", "Predicted vs. measured (A1)")

    # (b) enzyme saturation
    cs = np.array([v["c_star_M"] for v in kin["per_enzyme"].values() if v.get("c_star_M")])
    bins = np.arange(-10, -1.5, 0.5)
    counts, _, _ = b.hist(np.log10(cs), bins=bins, color=S1, edgecolor=SURF, lw=0.6)
    top = counts.max() * 1.42
    b.set_ylim(0, top)
    for xv, lab, ha, dx in ((-6, "1 µM", "right", -0.12), (-3, "1 mM", "left", 0.12)):
        b.axvline(xv, color=INK2, lw=0.6, ls=(0, (3, 2)))
        b.text(xv + dx, top * 0.95, lab, fontsize=6.0, color=INK2, va="top", ha=ha)
    med = float(np.median(cs))
    b.axvline(np.log10(med), color=S2, lw=0.9)
    b.text(np.log10(med) + 0.12, top * 0.95, "median\n" + f"{med * 1e6:.1f} µM", ha="left", va="top", fontsize=6.0,
           color=INK)
    b.set_xlabel(r"$\log_{10}\,c^\ast$ (M), 10% departure from linear")
    b.set_ylabel(f"enzymes (n = {len(cs)})")
    grid(b)
    panel(b, "b", "Enzymes saturate early")

    # (c) metabolite bound-to-free ratio
    ks = np.array([v["k_at_total_enzyme_range"][1] for v in kin["per_metabolite"].values()
                   if v.get("k_at_total_enzyme_range")])
    c.hist(np.log10(ks), bins=np.arange(-6, 2.01, 0.5), color=S3, edgecolor=SURF, lw=0.6)
    c.axvline(0, color=INK2, lw=0.6, ls=(0, (3, 2)))
    c.text(0.1, c.get_ylim()[1] * 0.9, "bound = free", fontsize=6.0, color=INK2)
    c.set_xlabel(r"$\log_{10}$ bound-to-free ratio $k$ (6 mM protein)")
    c.set_ylabel(f"metabolites (n = {len(ks)})")
    grid(c)
    panel(c, "c", "Metabolites stay free")

    # (d) network expansion outcomes
    sem = [("as-annotated", "as annotated", S1), ("reversible", "reversible", S2), ("at-1mM", "sign at 1 mM fixed", S3)]
    outs = [("answered", "one"), ("underdetermined", "several"), ("reachable-without-the-source", "no source"),
            ("certified-empty", "none")]
    w = 0.26
    for i, (k, lab, col) in enumerate(sem):
        xs = np.arange(len(outs)) + (i - 1) * w
        d.bar(xs, [comp[k]["labels"][o] for o, _ in outs], width=w - 0.03, color=col, label=lab, edgecolor=SURF, lw=0.6)
    d.set_yscale("log")
    d.set_ylim(100, 3e7)
    d.set_xticks(range(len(outs)), [t for _, t in outs], rotation=28, ha="right", rotation_mode="anchor")
    d.set_ylabel("source–target pairs")
    d.tick_params(axis="x", length=0)
    grid(d)
    d.legend(frameon=False, loc="upper left", ncol=1, handlelength=0.9, borderaxespad=0.1, labelspacing=0.25)
    panel(d, "d", "Production routes")
    save(fig, "fig_proteins")


# ============================================================ Figure 4: the extension in use
def figure_extension() -> None:
    e = ext()
    cat = e["catalogue"]
    fig, (a, b, c, d) = row(left=0.085, wspace=0.95, widths=(0.9, 0.9, 1.0, 0.95))

    # (a) triples per named graph
    names = [("stated", "stated"), ("supplied", "supplied"), ("derived", "derived"), ("prov", "provenance"),
             ("catalogue", "catalogue")]
    v = [cat["triples_per_graph"].get(k, 0) for k, _ in names]
    cols = [S1, S1, S2, MUTED, MUTED]
    y = np.arange(len(names))[::-1]
    a.barh(y, v, height=0.62, color=cols, edgecolor=SURF, lw=0.8)
    a.set_xscale("log")
    a.set_xlim(5, max(v) * 12)
    for yi, vv in zip(y, v, strict=True):
        a.text(vv * 1.25, yi, f"{vv:,}", va="center", fontsize=6.2)
    a.set_yticks(y, [t for _, t in names])
    a.set_xlabel("triples (log scale)")
    a.tick_params(axis="y", length=0)
    grid(a, "x")
    panel(a, "a", "Named graphs, whole corpus")

    # (b) scaling of emission and validation
    sc = e["scaling"] + [{"triples": cat["triples_total"], "emit_s": cat["timing_s"]["emit"],
                          "validate_s": cat["timing_s"]["validate"], "reactions": 2914}]
    t = np.array([r["triples"] for r in sc])
    b.plot(t, [r["validate_s"] for r in sc], color=S2, lw=1.4, marker="o", ms=3, mec=SURF, mew=0.6, label="SHACL validation")
    b.plot(t, [r["emit_s"] for r in sc], color=S1, lw=1.4, marker="s", ms=3, mec=SURF, mew=0.6, label="emission")
    b.set_xscale("log")
    b.set_yscale("log")
    b.xaxis.set_minor_formatter(ticker.NullFormatter())
    b.yaxis.set_minor_formatter(ticker.NullFormatter())
    b.set_xlabel("triples")
    b.set_ylabel("wall-clock time (s)")
    grid(b)
    b.xaxis.grid(True, color=GRID, lw=0.5)
    b.set_ylim(1, 3000)
    b.legend(frameon=False, loc="upper left", handlelength=1.2, borderaxespad=0.1)
    panel(b, "b", "Cost against size")

    # (c) planted defects
    ctl = e["controls"]
    items = [("unaltered catalogue", ctl["unaltered"]["results"])] + [
        (k, vv["results"]) for k, vv in ctl["defects"].items()
    ]
    short = {
        "unaltered catalogue": "none (control)",
        "V1: payload on a non-answered verdict": "payload, non-answer",
        "V2: blocker removed from an absent verdict": "blocker removed",
        "gate letter on an answered verdict": "code on an answer",
        "supplied basis without a convention": "convention removed",
        "label outside the vocabulary": "unknown label",
        "stage digest not a SHA-256": "bad stage digest",
    }
    y = np.arange(len(items))[::-1]
    vals = [vv for _, vv in items]
    c.barh(y, vals, height=0.62, color=[MUTED] + [S2] * (len(items) - 1), edgecolor=SURF, lw=0.8)
    for yi, vv in zip(y, vals, strict=True):
        c.text(vv + 0.08, yi, "conforms" if vv == 0 else f"{vv}", va="center", fontsize=6.0,
               color=INK2 if vv == 0 else INK)
    c.set_yticks(y, [short[k] for k, _ in items])
    c.set_xlabel("SHACL validation results")
    c.set_xlim(0, max(vals) + 1.6)
    c.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    c.tick_params(axis="y", length=0)
    grid(c, "x")
    panel(c, "c", "Planted defects are caught")

    # (d) two runs joined on verdict IRIs
    cmp_ = e["comparison"]
    pairs = [r for r in cmp_["joined"] if r["question"] == "Direction"]
    pairs.sort(key=lambda r: (r["a"] != r["b"], -r["n"]))
    lab = {"Undecidable": "undecidable", "Answered": "answered", "Absent": "absent", "Refused": "refused",
           "CertifiedEmpty": "empty"}
    names_d = [lab.get(r["a"], r["a"]) if r["a"] == r["b"] else f"{lab.get(r['a'], r['a'])} → {lab.get(r['b'], r['b'])}"
               for r in pairs]
    y = np.arange(len(pairs))[::-1]
    vals = [r["n"] for r in pairs]
    d.barh(y, vals, height=0.62, color=[S3 if r["a"] != r["b"] else MUTED for r in pairs], edgecolor=SURF, lw=0.8)
    d.set_xscale("log")
    d.set_xlim(0.7, max(vals) * 14)
    for yi, vv in zip(y, vals, strict=True):
        d.text(vv * 1.25, yi, f"{vv:,}", va="center", fontsize=6.2)
    d.set_yticks(y, names_d)
    d.set_xlabel("direction verdicts (log scale)")
    d.tick_params(axis="y", length=0)
    grid(d, "x")
    panel(d, "d", "Two media, one join")
    save(fig, "fig_extension")


if __name__ == "__main__":
    figure_structure()
    figure_medium()
    figure_proteins()
    figure_extension()
    print("figures written to", HERE)
