"""
Six panels for the pharmacokinetics paper.
Each panel: 1x4 row, white background, minimal text, at least one 3D chart.
No tables, no text-only/conceptual charts.
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

OUT = Path(r"c:/Users/kunda/Documents/systems/nebuchadnezzar/crown-prince/publication/partition-based-pharmacokinetics/figures")
OUT.mkdir(parents=True, exist_ok=True)

CMAP = "viridis"
DPI = 150


def newfig():
    fig = plt.figure(figsize=(20, 4.5))
    return fig


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Panel 1: Bioavailability F = F_abs (1-E_H)(1-E_G)
# ---------------------------------------------------------------------------
fig = newfig()

# (a) F vs E_H for several F_abs
ax = fig.add_subplot(1, 4, 1)
EH = np.linspace(0, 1, 200)
for F_abs in [0.5, 0.75, 0.9, 1.0]:
    ax.plot(EH, F_abs * (1 - EH), label=f"F_abs={F_abs}")
ax.set_xlabel("E_H (hepatic extraction)")
ax.set_ylabel("F")
ax.set_title("Bioavailability vs E_H")
ax.legend(frameon=False, fontsize=8)

# (b) Predicted vs observed F bars
ax = fig.add_subplot(1, 4, 2)
drugs = ["propranolol", "atorvastatin", "morphine", "verapamil", "aspirin"]
F_pred = [0.95*(1-0.68), 0.85*(1-0.50)*(1-0.65), 0.85*(1-0.65)*(1-0.05),
          0.95*(1-0.75)*(1-0.10), 0.90*(1-0.30)*(1-0.05)]
F_obs = [0.30, 0.14, 0.30, 0.20, 0.60]
x = np.arange(len(drugs))
ax.bar(x - 0.2, F_pred, width=0.4, label="predicted", color="#4477AA")
ax.bar(x + 0.2, F_obs, width=0.4, label="observed", color="#EE6677")
ax.set_xticks(x)
ax.set_xticklabels(drugs, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("F")
ax.set_title("Predicted vs observed F")
ax.legend(frameon=False, fontsize=8)

# (c) 3D F(E_H, E_G)
ax = fig.add_subplot(1, 4, 3, projection="3d")
EH = np.linspace(0, 1, 40)
EG = np.linspace(0, 1, 40)
EHm, EGm = np.meshgrid(EH, EG)
F = (1 - EHm) * (1 - EGm)
ax.plot_surface(EHm, EGm, F, cmap=CMAP, edgecolor="none", alpha=0.9)
ax.set_xlabel("E_H")
ax.set_ylabel("E_G")
ax.set_zlabel("F / F_abs")
ax.set_title("F surface")

# (d) Extraction ratio sensitivity
ax = fig.add_subplot(1, 4, 4)
EH = np.linspace(0, 0.99, 200)
ax.plot(EH, 100 * (1 - EH), color="#228833", lw=2)
ax.fill_between(EH, 0, 100*(1 - EH), color="#228833", alpha=0.15)
ax.set_xlabel("E_H")
ax.set_ylabel("F (% of F_abs)")
ax.set_title("First-pass loss")

save(fig, "panel_1_bioavailability.png")

# ---------------------------------------------------------------------------
# Panel 2: Volume of distribution V_d = V_p + Σ V_t K_p
# ---------------------------------------------------------------------------
fig = newfig()

# (a) V_d vs K_p (single tissue)
ax = fig.add_subplot(1, 4, 1)
Kp = np.logspace(-1, 3, 200)
ax.semilogx(Kp, 3 + 39 * Kp, label="water-rich tissue")
ax.semilogx(Kp, 3 + 33 * Kp, label="fat tissue")
ax.set_xlabel("K_p")
ax.set_ylabel("V_d (L)")
ax.set_title("V_d vs partition coefficient")
ax.legend(frameon=False, fontsize=8)

# (b) V_d for known drugs
ax = fig.add_subplot(1, 4, 2)
drugs = ["aminoglyc.", "warfarin", "digoxin", "propranolol", "amiodarone", "chloroq."]
Vd = [15, 8, 500, 280, 5000, 13000]
ax.barh(drugs, Vd, color=plt.cm.viridis(np.linspace(0.15, 0.85, len(drugs))))
ax.set_xscale("log")
ax.set_xlabel("V_d (L)")
ax.set_title("Drug V_d (log scale)")

# (c) 3D V_d(K_p_water, K_p_fat)
ax = fig.add_subplot(1, 4, 3, projection="3d")
Kw = np.linspace(0.1, 5, 40)
Kf = np.linspace(0.1, 50, 40)
Kwm, Kfm = np.meshgrid(Kw, Kf)
Vd = 3 + 39 * Kwm + 33 * Kfm
ax.plot_surface(Kwm, Kfm, np.log10(Vd), cmap=CMAP, edgecolor="none", alpha=0.9)
ax.set_xlabel("K_p (water)")
ax.set_ylabel("K_p (fat)")
ax.set_zlabel("log10 V_d")
ax.set_title("V_d surface")

# (d) Tissue contribution stack
ax = fig.add_subplot(1, 4, 4)
tissues = ["plasma", "muscle", "fat", "liver", "kidney", "brain"]
Vt = [3, 30, 13, 1.5, 0.3, 1.4]
Kp_lipo = [1, 1, 30, 5, 3, 4]
contribs = [v * k for v, k in zip(Vt, Kp_lipo)]
ax.bar(tissues, contribs, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(tissues))))
ax.set_ylabel("V_t · K_p (L)")
ax.set_title("Tissue contributions to V_d")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

save(fig, "panel_2_volume_distribution.png")

# ---------------------------------------------------------------------------
# Panel 3: Half-life t_1/2 = ln2 V_d / CL
# ---------------------------------------------------------------------------
fig = newfig()

# (a) t_1/2 vs CL for various V_d
ax = fig.add_subplot(1, 4, 1)
CL = np.logspace(-1, 2, 200)
for Vd in [10, 100, 500, 5000]:
    ax.loglog(CL, math.log(2) * Vd / CL, label=f"V_d={Vd} L")
ax.set_xlabel("CL (L/h)")
ax.set_ylabel("t_1/2 (h)")
ax.set_title("Half-life vs clearance")
ax.legend(frameon=False, fontsize=8)

# (b) Predicted vs observed for drugs
ax = fig.add_subplot(1, 4, 2)
drugs = ["digoxin", "warfarin", "propranolol", "atenolol", "amiodarone"]
Vd = [500, 8, 280, 50, 5000]
CL = [8.4, 0.2, 60, 9.6, 1.2]
t_pred = [math.log(2) * v / c for v, c in zip(Vd, CL)]
t_obs = [36, 37, 3.5, 6, 1500]
x = np.arange(len(drugs))
ax.bar(x - 0.2, t_pred, width=0.4, label="predicted", color="#4477AA")
ax.bar(x + 0.2, t_obs, width=0.4, label="observed", color="#EE6677")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(drugs, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("t_1/2 (h)")
ax.set_title("Predicted vs observed t_1/2")
ax.legend(frameon=False, fontsize=8)

# (c) 3D t_1/2(V_d, CL)
ax = fig.add_subplot(1, 4, 3, projection="3d")
Vd = np.linspace(5, 1000, 40)
CL = np.linspace(0.5, 50, 40)
Vm, Cm = np.meshgrid(Vd, CL)
t12 = math.log(2) * Vm / Cm
ax.plot_surface(Vm, Cm, np.log10(t12), cmap=CMAP, edgecolor="none", alpha=0.9)
ax.set_xlabel("V_d (L)")
ax.set_ylabel("CL (L/h)")
ax.set_zlabel("log10 t_1/2 (h)")
ax.set_title("Half-life surface")

# (d) Concentration decay
ax = fig.add_subplot(1, 4, 4)
t = np.linspace(0, 24, 300)
for tau in [2, 6, 12, 24]:
    C = np.exp(-math.log(2) * t / tau)
    ax.plot(t, C, label=f"t_1/2={tau} h")
ax.set_xlabel("time (h)")
ax.set_ylabel("C / C_0")
ax.set_title("Mono-exponential decay")
ax.legend(frameon=False, fontsize=8)

save(fig, "panel_3_half_life.png")

# ---------------------------------------------------------------------------
# Panel 4: Multiple dosing & steady state
# ---------------------------------------------------------------------------
fig = newfig()

# (a) Multiple-dose accumulation
ax = fig.add_subplot(1, 4, 1)
t = np.linspace(0, 72, 2000)
tau = 12  # h
khalf = math.log(2) / 6
C = np.zeros_like(t)
for n in range(0, 7):
    mask = t >= n * tau
    C[mask] += np.exp(-khalf * (t[mask] - n * tau))
ax.plot(t, C, color="#4477AA", lw=1.5)
ax.axhline(y=C[-200:].max(), ls="--", color="#EE6677", alpha=0.7, label="C_ss,max")
ax.axhline(y=C[-200:].min(), ls="--", color="#228833", alpha=0.7, label="C_ss,min")
ax.set_xlabel("time (h)")
ax.set_ylabel("C (relative)")
ax.set_title("Multiple-dose accumulation")
ax.legend(frameon=False, fontsize=8)

# (b) Css vs tau/t_1/2
ax = fig.add_subplot(1, 4, 2)
ratio = np.linspace(0.1, 4, 200)
acc = 1.0 / (1 - 0.5**(1/ratio))
ax.plot(ratio, acc, color="#CC3311", lw=2)
ax.set_xlabel("τ / t_1/2")
ax.set_ylabel("Accumulation factor")
ax.set_title("Accumulation vs dosing interval")

# (c) 3D C_ss(D, tau)
ax = fig.add_subplot(1, 4, 3, projection="3d")
D = np.linspace(50, 500, 40)
tau = np.linspace(4, 24, 40)
Dm, taum = np.meshgrid(D, tau)
Css = 0.5 * Dm / (10 * taum)
ax.plot_surface(Dm, taum, Css, cmap=CMAP, edgecolor="none", alpha=0.9)
ax.set_xlabel("Dose (mg)")
ax.set_ylabel("τ (h)")
ax.set_zlabel("C_ss,avg (mg/L)")
ax.set_title("Steady state surface")

# (d) Loading vs maintenance
ax = fig.add_subplot(1, 4, 4)
t = np.linspace(0, 48, 800)
khalf = math.log(2) / 8
C_load = np.exp(-khalf * t)  # one-time loading
C_maint = 1 - np.exp(-khalf * t)
ax.plot(t, C_load + C_maint - C_load*C_maint, label="loading + maintenance", color="#4477AA")
ax.plot(t, C_maint, label="maintenance only", color="#EE6677")
ax.set_xlabel("time (h)")
ax.set_ylabel("C (relative)")
ax.set_title("Loading dose effect")
ax.legend(frameon=False, fontsize=8)

save(fig, "panel_4_dosing.png")

# ---------------------------------------------------------------------------
# Panel 5: Hepatic clearance (well-stirred)
# ---------------------------------------------------------------------------
fig = newfig()

# (a) CL_H vs CL_int
ax = fig.add_subplot(1, 4, 1)
CL_int = np.logspace(0, 5, 200)
QH = 90
for fu in [0.05, 0.1, 0.5, 1.0]:
    CL_H = QH * fu * CL_int / (QH + fu * CL_int)
    ax.semilogx(CL_int, CL_H, label=f"f_u={fu}")
ax.axhline(y=QH, ls="--", color="black", alpha=0.5, label=f"Q_H={QH}")
ax.set_xlabel("CL_int (L/h)")
ax.set_ylabel("CL_H (L/h)")
ax.set_title("Hepatic clearance")
ax.legend(frameon=False, fontsize=8)

# (b) Extraction ratio vs CL_int
ax = fig.add_subplot(1, 4, 2)
for fu in [0.05, 0.1, 0.5, 1.0]:
    E = fu * CL_int / (QH + fu * CL_int)
    ax.semilogx(CL_int, E, label=f"f_u={fu}")
ax.set_xlabel("CL_int (L/h)")
ax.set_ylabel("E_H")
ax.set_title("Extraction ratio")
ax.legend(frameon=False, fontsize=8)

# (c) 3D CL_H(f_u, CL_int)
ax = fig.add_subplot(1, 4, 3, projection="3d")
fu = np.linspace(0.01, 1.0, 40)
CL_int = np.logspace(0, 5, 40)
fum, Cm = np.meshgrid(fu, CL_int)
CL_H = QH * fum * Cm / (QH + fum * Cm)
ax.plot_surface(fum, np.log10(Cm), CL_H, cmap=CMAP, edgecolor="none", alpha=0.9)
ax.set_xlabel("f_u")
ax.set_ylabel("log10 CL_int")
ax.set_zlabel("CL_H (L/h)")
ax.set_title("CL_H surface")

# (d) Flow vs capacity-limited categories
ax = fig.add_subplot(1, 4, 4)
drugs = ["warfarin", "diazepam", "phenytoin", "propranolol", "morphine", "lidocaine"]
E_H = [0.003, 0.03, 0.03, 0.68, 0.65, 0.7]
colors = ["#4477AA" if e < 0.3 else ("#CCBB44" if e < 0.7 else "#EE6677") for e in E_H]
ax.barh(drugs, E_H, color=colors)
ax.axvline(x=0.3, ls="--", color="black", alpha=0.4)
ax.axvline(x=0.7, ls="--", color="black", alpha=0.4)
ax.set_xlabel("E_H")
ax.set_title("Capacity → flow regimes")

save(fig, "panel_5_hepatic.png")

# ---------------------------------------------------------------------------
# Panel 6: Allometric scaling, renal clearance, two-compartment
# ---------------------------------------------------------------------------
fig = newfig()

# (a) Allometric CL ~ BW^0.75
ax = fig.add_subplot(1, 4, 1)
BW = np.logspace(-2, 3, 200)
ax.loglog(BW, 0.05 * (BW / 0.025) ** 0.75, label="exponent 3/4")
ax.loglog(BW, 0.05 * (BW / 0.025) ** 0.67, label="exponent 0.67", ls="--")
ax.loglog(BW, 0.05 * (BW / 0.025) ** 1.0, label="exponent 1.0", ls=":")
species_BW = [0.025, 0.25, 12, 70, 500]
species_CL = [0.05 * (b / 0.025) ** 0.75 for b in species_BW]
ax.scatter(species_BW, species_CL, s=60, color="#CC3311", zorder=5)
for n, b, c in zip(["mouse", "rat", "dog", "human", "horse"], species_BW, species_CL):
    ax.annotate(n, (b, c), fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Body weight (kg)")
ax.set_ylabel("CL (L/h)")
ax.set_title("Allometric scaling")
ax.legend(frameon=False, fontsize=8)

# (b) Renal clearance components
ax = fig.add_subplot(1, 4, 2)
substances = ["inulin", "creatinine", "glucose", "PAH", "penicillin"]
fu = [1, 1, 1, 0.6, 0.4]
GFR = 7.5
sec = [0, 0, 0, 30, 6]
F_reab = [0, 0, 1.0, 0, 0.2]
CL_R = [(f * GFR + s) * (1 - r) for f, s, r in zip(fu, sec, F_reab)]
ax.bar(substances, CL_R, color=plt.cm.viridis(np.linspace(0.15, 0.9, len(substances))))
ax.axhline(y=GFR, ls="--", color="black", alpha=0.5, label=f"GFR={GFR}")
ax.set_ylabel("CL_R (L/h)")
ax.set_title("Renal clearance")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
ax.legend(frameon=False, fontsize=8)

# (c) 3D two-compartment concentration C(t) for varying k12
ax = fig.add_subplot(1, 4, 3, projection="3d")
t = np.linspace(0, 24, 80)
k12_vals = np.linspace(0.2, 2.0, 40)
T, K = np.meshgrid(t, k12_vals)
k21, k10 = 0.5, 0.3
total = K + k21 + k10
disc = np.sqrt(np.maximum(total**2 - 4*k21*k10, 1e-12))
alpha = (total + disc) / 2
beta = (total - disc) / 2
A = (alpha - k21) / (alpha - beta)
B = (k21 - beta) / (alpha - beta)
C = A * np.exp(-alpha * T) + B * np.exp(-beta * T)
ax.plot_surface(T, K, C, cmap=CMAP, edgecolor="none", alpha=0.9)
ax.set_xlabel("time (h)")
ax.set_ylabel("k12")
ax.set_zlabel("C (relative)")
ax.set_title("Two-compartment C(t)")

# (d) Drug-drug competition
ax = fig.add_subplot(1, 4, 4)
ratio = np.linspace(0, 10, 200)
ax.plot(ratio, 1 / (1 + ratio), color="#CC3311", lw=2)
ax.set_xlabel("[Inhibitor] / K_i")
ax.set_ylabel("CL / CL_alone")
ax.set_title("Competitive inhibition")

save(fig, "panel_6_scaling_renal.png")

print(f"Pharmacokinetics panels generated in {OUT}")
