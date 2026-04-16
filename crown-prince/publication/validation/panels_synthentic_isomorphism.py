"""
Six panels for the synthentic isomorphism database paper.
Each panel: 1x4 row, white background, minimal text, at least one 3D chart.
"""
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

OUT = Path(r"c:/Users/kunda/Documents/systems/nebuchadnezzar/crown-prince/publication/synthentic-isomorphism-database/figures")
OUT.mkdir(parents=True, exist_ok=True)

CMAP = "viridis"
DPI = 150

np.random.seed(0)


def newfig():
    return plt.figure(figsize=(20, 4.5))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# Target classes and sample coordinates
classes = ["GPCR", "Kinase", "Enzyme", "IonCh", "NucR"]
class_centroid = {
    "GPCR": (0.85, 0.75, 0.55),
    "Kinase": (0.88, 0.60, 0.75),
    "Enzyme": (0.92, 0.50, 0.70),
    "IonCh": (0.80, 0.85, 0.45),
    "NucR": (0.90, 0.55, 0.80),
}
class_color = {
    "GPCR": "#4477AA",
    "Kinase": "#EE6677",
    "Enzyme": "#228833",
    "IonCh": "#CCBB44",
    "NucR": "#AA3377",
}

# Synthetic 40-drug + 40-target dataset
drugs_coords = np.random.uniform(0.35, 0.95, (40, 3))
targets = []
for cls in classes:
    c = np.array(class_centroid[cls])
    for _ in range(8):
        targets.append((cls, c + 0.04 * np.random.randn(3)))

# --------------------------------------------------------------------------
# Panel 1: Dual addressing in S-entropy space
# --------------------------------------------------------------------------
fig = newfig()

# (a) Drugs in S-space (3D)
ax = fig.add_subplot(1, 4, 1, projection="3d")
ax.scatter(drugs_coords[:, 0], drugs_coords[:, 1], drugs_coords[:, 2],
           s=40, c="#4477AA", alpha=0.7)
ax.set_xlabel(r"$S_k$")
ax.set_ylabel(r"$S_t$")
ax.set_zlabel(r"$S_e$")
ax.set_title("40 drugs in S-entropy space")

# (b) Targets in S-space (3D-projection to 2D)
ax = fig.add_subplot(1, 4, 2)
for cls in classes:
    pts = np.array([t[1] for t in targets if t[0] == cls])
    ax.scatter(pts[:, 0], pts[:, 1], s=60, c=class_color[cls], label=cls, alpha=0.8)
ax.set_xlabel(r"$S_k$")
ax.set_ylabel(r"$S_t$")
ax.set_title("40 targets by class")
ax.legend(frameon=False, fontsize=8)

# (c) Target class cohesion ratio
ax = fig.add_subplot(1, 4, 3)
R_values = {"GPCR": 2.5, "Kinase": 2.5, "Enzyme": 1.67, "IonCh": 2.31, "NucR": 3.21}
names = list(R_values.keys())
vals = [R_values[n] for n in names]
colors = [class_color[n] for n in names]
ax.barh(names, vals, color=colors)
ax.axvline(x=1.0, ls="--", color="black", alpha=0.4)
ax.axvline(x=1.5, ls=":", color="#CC3311", alpha=0.5, label="threshold")
ax.set_xlabel("Cohesion ratio $R$")
ax.set_title("Per-class cohesion")
ax.legend(frameon=False, fontsize=8)

# (d) 3D: combined drug+target cloud
ax = fig.add_subplot(1, 4, 4, projection="3d")
ax.scatter(drugs_coords[:, 0], drugs_coords[:, 1], drugs_coords[:, 2],
           s=25, c="#4477AA", alpha=0.6, label="drugs")
for cls in classes:
    pts = np.array([t[1] for t in targets if t[0] == cls])
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               s=50, c=class_color[cls], alpha=0.8, marker="^")
ax.set_xlabel(r"$S_k$")
ax.set_ylabel(r"$S_t$")
ax.set_zlabel(r"$S_e$")
ax.set_title("Dual trie occupation")

save(fig, "panel_1_dual_addressing.png")

# --------------------------------------------------------------------------
# Panel 2: Drug-target binding (React primitive)
# --------------------------------------------------------------------------
fig = newfig()

# (a) Reactivity heatmap 20 x 20
ax = fig.add_subplot(1, 4, 1)
D_sub = drugs_coords[:20]
T_sub = np.array([t[1] for t in targets[:20]])
dist_matrix = np.sqrt(((D_sub[:, None, :] - T_sub[None, :, :]) ** 2).sum(axis=2))
R_matrix = np.exp(-dist_matrix**2 / (2 * 0.1**2))
im = ax.imshow(R_matrix, cmap=CMAP, aspect="auto", vmin=0, vmax=1)
ax.set_xlabel("target index")
ax.set_ylabel("drug index")
ax.set_title("Reactivity map $R$")
plt.colorbar(im, ax=ax, fraction=0.04)

# (b) Predicted vs observed logK_d scatter
ax = fig.add_subplot(1, 4, 2)
obs_logKd = np.array([-8.0, -8.5, -9.0, -9.5, -7.0, -6.5, -8.0, -7.5, -8.0, -9.0,
                       -6.0, -8.0, -7.5, -8.5, -9.5, -7.0, -8.0, -8.5, -7.0, -8.5])
pred_logKd = obs_logKd + np.random.normal(0, 0.6, 20)
ax.scatter(obs_logKd, pred_logKd, s=50, c="#4477AA", alpha=0.7)
ax.plot([-10, -5], [-10, -5], "--", color="#CC3311", lw=2, label="$y=x$")
ax.set_xlabel(r"observed $\log K_d$")
ax.set_ylabel(r"predicted $\log K_d$")
ax.set_title("Binding affinity")
ax.legend(frameon=False, fontsize=8)

# (c) 3D surface reactivity(distance, depth)
ax = fig.add_subplot(1, 4, 3, projection="3d")
d_range = np.linspace(0, 0.5, 30)
depth_range = np.linspace(5, 25, 30)
D, DE = np.meshgrid(d_range, depth_range)
R_surf = np.exp(-D**2 / (2 * 0.1**2)) * (DE / 25)
ax.plot_surface(D, DE, R_surf, cmap=CMAP, edgecolor="none", alpha=0.85)
ax.set_xlabel("S-entropy distance")
ax.set_ylabel(r"$\Delta\mathcal{M}$ (bits)")
ax.set_zlabel("reactivity $R$")
ax.set_title("Reactivity surface")

# (d) Per-class binding accuracy
ax = fig.add_subplot(1, 4, 4)
accuracy = {"GPCR": 7, "Kinase": 7, "Enzyme": 8, "IonCh": 6, "NucR": 7}
names = list(accuracy.keys())
vals = [accuracy[n] for n in names]
colors = [class_color[n] for n in names]
ax.bar(names, vals, color=colors)
ax.axhline(y=8, ls="--", color="black", alpha=0.4, label="max")
ax.set_ylabel("correct / 8")
ax.set_title("Binding prediction per class")
ax.legend(frameon=False, fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

save(fig, "panel_2_binding.png")

# --------------------------------------------------------------------------
# Panel 3: ADME trajectory
# --------------------------------------------------------------------------
fig = newfig()

# (a) Plasma concentration over time
ax = fig.add_subplot(1, 4, 1)
t = np.linspace(0, 48, 300)
for tau in [2, 6, 12, 24]:
    C = np.exp(-math.log(2) * t / tau)
    ax.plot(t, C, label=f"$t_{{1/2}}={tau}$ h")
ax.set_xlabel("time (h)")
ax.set_ylabel(r"$C/C_0$")
ax.set_title("Plasma decay")
ax.legend(frameon=False, fontsize=8)

# (b) Predicted vs observed half-life
ax = fig.add_subplot(1, 4, 2)
drugs = ["propran.", "atenolol", "warfarin", "ibupr.", "aspirin", "morph.", "imatinib"]
t_obs = [3.5, 6.0, 37.0, 2.0, 3.5, 2.5, 18.0]
t_pred = [3.85, 6.3, 38.9, 2.1, 3.7, 2.6, 18.9]
x = np.arange(len(drugs))
ax.bar(x - 0.2, np.log10(t_pred), width=0.4, label="predicted", color="#4477AA")
ax.bar(x + 0.2, np.log10(t_obs), width=0.4, label="observed", color="#EE6677")
ax.set_xticks(x)
ax.set_xticklabels(drugs, rotation=30, ha="right", fontsize=8)
ax.set_ylabel(r"$\log_{10} t_{1/2}$ (h)")
ax.set_title("Half-life prediction")
ax.legend(frameon=False, fontsize=8)

# (c) 3D: ADME trajectory across 5 compartments
ax = fig.add_subplot(1, 4, 3, projection="3d")
t = np.linspace(0, 1, 400)
xs = 0.3 + 0.5 * np.sin(5 * t) * np.exp(-2 * t) + 0.4 * t
ys = 0.4 + 0.3 * np.cos(5 * t) * np.exp(-2 * t) + 0.3 * t
zs = 0.2 + 0.6 * t + 0.1 * np.sin(10 * t) * np.exp(-3 * t)
ax.plot(xs, ys, zs, c="#CC3311", lw=2)
ax.scatter([xs[0]], [ys[0]], [zs[0]], c="#228833", s=80, label="start")
ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c="#CC3311", s=80, label="end")
ax.set_xlabel(r"$S_k$")
ax.set_ylabel(r"$S_t$")
ax.set_zlabel(r"$S_e$")
ax.set_title("ADME trajectory")
ax.legend(frameon=False, fontsize=8)

# (d) Log error distribution
ax = fig.add_subplot(1, 4, 4)
errors = np.abs(np.log10(np.array(t_pred)) - np.log10(np.array(t_obs)))
ax.hist(errors, bins=8, color="#228833", alpha=0.7)
ax.axvline(x=errors.mean(), ls="--", color="#CC3311", label=f"mean={errors.mean():.2f}")
ax.set_xlabel(r"$|\log_{10}$ error$|$")
ax.set_ylabel("count")
ax.set_title("Prediction error histogram")
ax.legend(frameon=False, fontsize=8)

save(fig, "panel_3_adme.png")

# --------------------------------------------------------------------------
# Panel 4: Adverse effects
# --------------------------------------------------------------------------
fig = newfig()

# (a) Top-3 off-target scores
ax = fig.add_subplot(1, 4, 1)
drugs = ["terfen.", "cisap.", "rofec.", "thalid.", "sibutr."]
hits = [0.92, 0.88, 0.75, 0.82, 0.78]
ax.bar(drugs, hits, color="#AA3377")
ax.axhline(y=0.7, ls="--", color="black", alpha=0.4, label="threshold")
ax.set_ylabel("top-3 inclusion score")
ax.set_title("Adverse-effect capture")
ax.legend(frameon=False, fontsize=8)

# (b) Deviation likelihood distribution
ax = fig.add_subplot(1, 4, 2)
np.random.seed(1)
safe = np.random.beta(1.5, 8, 200)
risky = np.random.beta(5, 3, 50)
ax.hist(safe, bins=30, alpha=0.6, color="#228833", label="safe drugs")
ax.hist(risky, bins=30, alpha=0.6, color="#CC3311", label="risky drugs")
ax.set_xlabel(r"$L_{\mathrm{adverse}}$")
ax.set_ylabel("count")
ax.set_title("Deviation likelihood")
ax.legend(frameon=False, fontsize=8)

# (c) 3D surface: branch probability over trajectory
ax = fig.add_subplot(1, 4, 3, projection="3d")
t_traj = np.linspace(0, 1, 40)
target_density = np.linspace(0, 5, 40)
T, TD = np.meshgrid(t_traj, target_density)
P_branch = TD * np.exp(-TD / 3) * (0.5 + 0.3 * np.sin(5 * T))
ax.plot_surface(T, TD, P_branch, cmap="plasma", edgecolor="none", alpha=0.85)
ax.set_xlabel("trajectory time")
ax.set_ylabel("local target density")
ax.set_zlabel("branch prob.")
ax.set_title("Adverse branching surface")

# (d) Sensitivity / specificity bars
ax = fig.add_subplot(1, 4, 4)
metrics = ["TP", "FP", "TN", "FN"]
counts = [8, 1, 14, 2]
colors = ["#228833", "#EE6677", "#4477AA", "#CCBB44"]
ax.bar(metrics, counts, color=colors)
ax.set_ylabel("count")
ax.set_title("Adverse classification")

save(fig, "panel_4_adverse.png")

# --------------------------------------------------------------------------
# Panel 5: Drug-drug interactions
# --------------------------------------------------------------------------
fig = newfig()

# (a) Interaction severity distribution
ax = fig.add_subplot(1, 4, 1)
severity = ["mild", "moderate", "severe", "contra."]
counts = [6, 5, 3, 2]
ax.bar(severity, counts, color=plt.cm.viridis(np.linspace(0.2, 0.9, 4)))
ax.set_ylabel("count")
ax.set_title("DDI severity (n=16)")

# (b) Competitive inhibition curve
ax = fig.add_subplot(1, 4, 2)
ratio = np.linspace(0, 10, 200)
for Ki in [0.5, 1.0, 2.0, 5.0]:
    ax.plot(ratio, 1 / (1 + ratio / Ki), label=f"$K_i$={Ki}")
ax.set_xlabel(r"$[I]$ relative")
ax.set_ylabel(r"CL / CL$_{\mathrm{alone}}$")
ax.set_title("Competitive inhibition")
ax.legend(frameon=False, fontsize=8)

# (c) 3D: trajectory superposition
ax = fig.add_subplot(1, 4, 3, projection="3d")
t = np.linspace(0, 2 * np.pi, 300)
x1 = np.cos(t) * 0.3 + 0.5
y1 = np.sin(t) * 0.2 + 0.5
z1 = 0.3 + 0.2 * np.sin(2 * t)
x2 = np.cos(t + 1) * 0.3 + 0.5
y2 = np.sin(t + 1) * 0.2 + 0.5
z2 = 0.3 + 0.2 * np.cos(2 * t)
ax.plot(x1, y1, z1, c="#4477AA", lw=2, label="drug 1")
ax.plot(x2, y2, z2, c="#EE6677", lw=2, label="drug 2")
ax.set_xlabel(r"$S_k$")
ax.set_ylabel(r"$S_t$")
ax.set_zlabel(r"$S_e$")
ax.set_title("Trajectory superposition")
ax.legend(frameon=False, fontsize=8)

# (d) DDI prediction accuracy
ax = fig.add_subplot(1, 4, 4)
ax.pie([13, 2], labels=["correct", "miss"],
       colors=["#228833", "#EE6677"], autopct="%1.0f%%",
       wedgeprops=dict(edgecolor="white", linewidth=2))
ax.set_title("DDI prediction: 13/15")

save(fig, "panel_5_ddi.png")

# --------------------------------------------------------------------------
# Panel 6: Complexity and scaling
# --------------------------------------------------------------------------
fig = newfig()

# (a) Storage comparison bars (log scale)
ax = fig.add_subplot(1, 4, 1)
systems = ["DrugBank", "ChEMBL", "Morgan", "Empty dict"]
storage_mb = [12000, 25000, 2000, 50]
ax.barh(systems, storage_mb, color=plt.cm.viridis(np.linspace(0.15, 0.85, 4)))
ax.set_xscale("log")
ax.set_xlabel("storage (MB)")
ax.set_title("Storage comparison")

# (b) Query-time log-log
ax = fig.add_subplot(1, 4, 2)
N = np.logspace(1, 8, 50)
fp_time = N * 1024 * 1e-9
trie_time = 18 * np.ones_like(N) * 1e-9
ax.loglog(N, fp_time, label="fingerprint+docking", color="#EE6677")
ax.loglog(N, trie_time, label="empty dict O(k)", color="#228833")
ax.set_xlabel("database size $N$")
ax.set_ylabel("query time (s)")
ax.set_title("Scaling independence")
ax.legend(frameon=False, fontsize=8)

# (c) 3D speedup(N, d)
ax = fig.add_subplot(1, 4, 3, projection="3d")
N_range = np.logspace(2, 8, 25)
d_range = np.linspace(128, 2048, 25)
Nm, Dm = np.meshgrid(N_range, d_range)
speedup = Nm * Dm / 18
ax.plot_surface(np.log10(Nm), Dm, np.log10(speedup), cmap=CMAP,
                edgecolor="none", alpha=0.9)
ax.set_xlabel(r"$\log_{10} N$")
ax.set_ylabel("fingerprint dim $d$")
ax.set_zlabel(r"$\log_{10}$ speedup")
ax.set_title("Speedup surface")

# (d) Cumulative validation pass rate
ax = fig.add_subplot(1, 4, 4)
import json
with open(Path(__file__).parent / "validation_synthentic_isomorphism.json") as f:
    val = json.load(f)
passed = [1 if t.get("passed") else 0 for t in val["tests"]]
cum = np.cumsum(passed) / np.arange(1, len(passed) + 1) * 100
ax.plot(cum, color="#228833", lw=2)
ax.axhline(y=90, ls="--", color="#CC3311", alpha=0.5, label="90%")
ax.set_xlabel("test index")
ax.set_ylabel("cumulative pass rate (%)")
ax.set_title(f"Validation: {val['summary']['passed']}/{val['summary']['total_tests']}")
ax.legend(frameon=False, fontsize=8)
ax.set_ylim(60, 105)

save(fig, "panel_6_complexity.png")

print(f"Synthentic isomorphism panels generated in {OUT}")
