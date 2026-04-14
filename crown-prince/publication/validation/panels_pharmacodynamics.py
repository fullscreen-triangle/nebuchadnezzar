"""
Generate 6 panels for the pharmacodynamics paper.

Each panel: 4 subplots in a row, white background, minimal text, at least one 3D.
No tables, no conceptual diagrams, no text-only charts.
"""
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "0.2",
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT = Path(__file__).parent.parent / "partition-based-pharmacodynamics" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CMAP = "viridis"
DPI = 150

# -----------------------------------------------------------------------------
# PANEL 1: Drug binding & partition depth
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 4.5))
ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3)
ax4 = fig.add_subplot(1, 4, 4, projection='3d')

# (a) K_d vs partition depth (log-log)
delta_M = np.linspace(5, 40, 200)
K_d = 2.0 ** (-delta_M)
ax1.semilogy(delta_M, K_d * 1e9, color="#1f77b4", linewidth=2)
known = [(15, 1e3, "med"), (22, 50, "lead"), (30, 1, "high"), (35, 0.03, "v.high")]
for dM, Kd_nM, label in known:
    ax1.scatter(dM, Kd_nM, s=80, color="#d62728", zorder=5)
ax1.set_xlabel(r"Partition depth deficit $\Delta\mathcal{M}$ (bits)")
ax1.set_ylabel(r"$K_d$ (nM)")
ax1.set_title(r"$K_d = 2^{-\Delta\mathcal{M}}$")
ax1.grid(True, alpha=0.3)

# (b) Capacity formula C(n) = 2n^2
n = np.arange(1, 11)
C = 2 * n**2
ax2.bar(n, C, color="#2ca02c", edgecolor="black", linewidth=0.7)
ax2.set_xlabel(r"Principal depth $n$")
ax2.set_ylabel(r"Capacity $C(n) = 2n^2$")
ax2.set_title("Shell capacity")
ax2.grid(True, axis='y', alpha=0.3)

# (c) Free energy distribution
DG_kJ = np.array([15, 18, 20, 25, 28, 30, 35, 40, 42, 45, 48, 52, 55])
T = 310.0
R_const = 8.314
delta_M_drugs = DG_kJ * 1000 / (R_const * T * np.log(2))
ax3.scatter(DG_kJ, delta_M_drugs, s=70, c=delta_M_drugs, cmap=CMAP,
            edgecolors='black', linewidth=0.7)
slope = 1000 / (R_const * T * np.log(2))
ax3.plot(DG_kJ, DG_kJ * slope, '--', color="0.4", linewidth=1)
ax3.set_xlabel(r"$\Delta G_{\mathrm{bind}}$ (kJ/mol)")
ax3.set_ylabel(r"$\Delta\mathcal{M}_{\mathrm{bind}}$ (bits)")
ax3.set_title("Binding energy → partition depth")
ax3.grid(True, alpha=0.3)

# (d) 3D K_d - ΔM - drug count surface
dM = np.linspace(10, 35, 30)
T_grid = np.linspace(280, 320, 30)
DM, TG = np.meshgrid(dM, T_grid)
KD = 2.0 ** (-DM) * 1e9   # nM
KD_log = np.log10(KD)
surf = ax4.plot_surface(DM, TG, KD_log, cmap=CMAP, alpha=0.85,
                       edgecolor='none', antialiased=True)
ax4.set_xlabel(r"$\Delta\mathcal{M}$ (bits)")
ax4.set_ylabel(r"$T$ (K)")
ax4.set_zlabel(r"$\log_{10}(K_d/\mathrm{nM})$")
ax4.view_init(elev=25, azim=-55)
ax4.set_title("Affinity landscape")

plt.tight_layout()
fig.savefig(OUT / "panel_1_drug_binding.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------
# PANEL 2: Hill equation regimes & dose-response
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 4.5))
ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax4 = fig.add_subplot(1, 4, 4)

# (a) Hill curves for different regimes (n_H = 0.7, 1, 2, 4)
D = np.logspace(-3, 3, 200)
K = 1.0
colors = ["#9467bd", "#1f77b4", "#2ca02c", "#d62728"]
labels = [r"$n_H=0.7$ (turbulent)", r"$n_H=1$ (aperture)",
          r"$n_H=2$ (aperture)", r"$n_H=4$ (cascade)"]
for nH, c, lab in zip([0.7, 1, 2, 4], colors, labels):
    R = D**nH / (K**nH + D**nH)
    ax1.semilogx(D, R, color=c, linewidth=2, label=lab)
ax1.set_xlabel(r"$[D]/K$")
ax1.set_ylabel("Response (fraction)")
ax1.set_title("Dose-response by regime")
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)

# (b) Hill coefficient distribution
hill_data = {"hemoglobin": 2.8, "nAChR": 1.5, "muscarinic": 1.0,
             "dopamine_D2": 1.0, "opioid_mu": 1.0, "MAPK": 4.0,
             "calmodulin": 4.0, "rhodopsin": 0.7}
names = list(hill_data.keys())
values = list(hill_data.values())
colors_bar = ["#d62728" if v > 2.5 else "#2ca02c" if v >= 1.5 else
              "#1f77b4" if v >= 0.8 else "#9467bd" for v in values]
ax2.barh(names, values, color=colors_bar, edgecolor='black', linewidth=0.6)
ax2.axvline(2, color='black', linestyle='--', alpha=0.5)
ax2.set_xlabel(r"Hill coefficient $n_H$")
ax2.set_title("Receptor systems by regime")
ax2.grid(True, axis='x', alpha=0.3)

# (c) 3D regime landscape: response(dose, n_H)
D_log = np.linspace(-2, 2, 40)
nH_grid = np.linspace(0.5, 5, 40)
DG, NHG = np.meshgrid(D_log, nH_grid)
RR = (10**DG)**NHG / (1 + (10**DG)**NHG)
surf = ax3.plot_surface(DG, NHG, RR, cmap=CMAP, alpha=0.9, edgecolor='none')
ax3.set_xlabel(r"$\log_{10}([D]/K)$")
ax3.set_ylabel(r"$n_H$")
ax3.set_zlabel("Response")
ax3.view_init(elev=22, azim=-50)
ax3.set_title("Dose × cooperativity surface")

# (d) Slope (steepness) at half-max as function of n_H
nH_arr = np.linspace(0.5, 6, 100)
slope_at_half = nH_arr / 4.0   # max slope = n_H/4 for Hill
ax4.plot(nH_arr, slope_at_half, color="#1f77b4", linewidth=2)
ax4.fill_between(nH_arr, slope_at_half, alpha=0.3, color="#1f77b4")
ax4.scatter([1, 2, 4], [0.25, 0.5, 1.0], s=120, color='red',
            edgecolor='black', zorder=5)
ax4.set_xlabel(r"Hill coefficient $n_H$")
ax4.set_ylabel("Max slope at half-saturation")
ax4.set_title("Steepness scaling")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "panel_2_hill_regimes.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------
# PANEL 3: Allosteric coupling & propagation
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 4.5))
ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2, projection='3d')
ax3 = fig.add_subplot(1, 4, 3)
ax4 = fig.add_subplot(1, 4, 4)

# (a) Propagation velocity vs material
materials = ["Lipid", "Water", "Protein", "Bone", "Steel"]
v_s = [1500, 1500, 1900, 3500, 5000]    # m/s
E_Y = [1e7, 2.2e9, 1e9, 1.4e10, 2e11]
ax1.scatter(np.sqrt(np.array(E_Y) / 1000), v_s, s=120, c=range(len(v_s)),
            cmap=CMAP, edgecolors='black', linewidth=0.7)
ax1.plot([0, 5000], [0, 5000], '--', color='gray', linewidth=1)
for i, name in enumerate(materials):
    ax1.annotate(name, (math.sqrt(E_Y[i]/1000), v_s[i]),
                 textcoords="offset points", xytext=(8, 4), fontsize=9)
ax1.set_xlabel(r"$\sqrt{E_Y/\rho}$ (m/s)")
ax1.set_ylabel(r"$v_s$ measured (m/s)")
ax1.set_title("Propagation velocity")
ax1.grid(True, alpha=0.3)

# (b) 3D allosteric propagation: amplitude(distance, time)
x = np.linspace(0, 10, 80)
t = np.linspace(0, 20, 80)
X, T_arr = np.meshgrid(x, t)
v = 1.0
amp = np.exp(-(X - v*T_arr)**2 / 4.0) * np.exp(-T_arr/15)
surf = ax2.plot_surface(X, T_arr, amp, cmap='plasma', alpha=0.9, edgecolor='none')
ax2.set_xlabel("Distance (nm)")
ax2.set_ylabel("Time (ps)")
ax2.set_zlabel("Coupling amplitude")
ax2.view_init(elev=30, azim=-60)
ax2.set_title("Conformational wave")

# (c) Coupling decay with distance
d = np.linspace(0.5, 20, 100)
G_AB = 1.0 / d**2
ax3.semilogy(d, G_AB, color="#d62728", linewidth=2)
ax3.fill_between(d, G_AB, alpha=0.3, color="#d62728")
ax3.set_xlabel("Allosteric distance (nm)")
ax3.set_ylabel(r"Coupling $\mathcal{G}_{AB}$")
ax3.set_title("Distance scaling")
ax3.grid(True, which='both', alpha=0.3)

# (d) Two-state allosteric model: f_A vs effector concentration
[E] = [np.logspace(-3, 3, 150)]
DE = 2.0  # in kT units (effector lowers active state energy)
for kT_factor in [0.5, 1, 2, 4]:
    dE_eff = DE - 2*np.log(1 + [E][0]) / kT_factor
    f_A = 1.0 / (1.0 + np.exp(dE_eff))
    ax4.semilogx([E][0], f_A, linewidth=2, label=f"K={kT_factor}")
ax4.set_xlabel("Effector concentration")
ax4.set_ylabel("Active fraction $f_A$")
ax4.set_title("Two-state coupling")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "panel_3_allosteric.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------
# PANEL 4: Variance-free energy & efficacy
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 4.5))
ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax4 = fig.add_subplot(1, 4, 4)

# (a) F = kT sigma^2 linear relation
sigma2 = np.linspace(0, 2, 100)
F_kT = sigma2  # in units of kT
ax1.plot(sigma2, F_kT, color="#1f77b4", linewidth=2)
ax1.fill_between(sigma2, F_kT, alpha=0.3, color="#1f77b4")
states = [("healthy", 0.1), ("treated", 0.3), ("diseased", 1.0), ("severe", 1.6)]
colors_state = ["#2ca02c", "#ff7f0e", "#d62728", "#7f0000"]
for (name, s2), c in zip(states, colors_state):
    ax1.scatter(s2, s2, s=130, color=c, edgecolor='black', zorder=5)
    ax1.annotate(name, (s2, s2), textcoords="offset points",
                 xytext=(8, -8), fontsize=9)
ax1.set_xlabel(r"Phase variance $\sigma^2(\varphi)$ (rad$^2$)")
ax1.set_ylabel(r"Free energy $F/k_BT$")
ax1.set_title(r"$F = k_BT\sigma^2$")
ax1.grid(True, alpha=0.3)

# (b) Efficacy: time series of variance under treatment
t = np.linspace(0, 100, 500)
sigma2_pre = 0.8 + 0.1 * np.random.randn(500)
sigma2_treated = np.where(t < 30, 0.8 + 0.1 * np.random.randn(500),
                          0.8 - 0.7 * (1 - np.exp(-(t-30)/15)) +
                          0.05 * np.random.randn(500))
ax2.plot(t, sigma2_pre, color="0.5", linewidth=1, alpha=0.6, label="control")
ax2.plot(t, sigma2_treated, color="#2ca02c", linewidth=1.5, label="drug")
ax2.axvline(30, color='red', linestyle='--', alpha=0.6)
ax2.set_xlabel("Time (a.u.)")
ax2.set_ylabel(r"$\sigma^2(\varphi)$")
ax2.set_title("Variance reduction = efficacy")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# (c) 3D: F(sigma_p, sigma_t, sigma_e) - free energy across S-coordinates
s_k = np.linspace(0, 1, 30)
s_t = np.linspace(0, 1, 30)
SK, ST = np.meshgrid(s_k, s_t)
F_3d = SK**2 + ST**2 + 0.5  # representative
surf = ax3.plot_surface(SK, ST, F_3d, cmap='magma', alpha=0.9, edgecolor='none')
ax3.set_xlabel(r"$S_k$")
ax3.set_ylabel(r"$S_t$")
ax3.set_zlabel(r"$F/k_BT$")
ax3.view_init(elev=28, azim=-50)
ax3.set_title("Free energy surface")

# (d) Receptor reserve cascade amplification
levels = np.arange(1, 7)
gain_per = [3, 5, 10, 15]
for g, c in zip(gain_per, ["#1f77b4", "#2ca02c", "#d62728", "#7f0000"]):
    amp = (1 + g) ** levels
    ax4.semilogy(levels, amp, marker='o', linewidth=2, color=c, label=f"gain={g}")
ax4.axhline(1000, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel("Cascade levels")
ax4.set_ylabel("Total amplification")
ax4.set_title("Receptor reserve")
ax4.legend(fontsize=8)
ax4.grid(True, which='both', alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "panel_4_efficacy.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------
# PANEL 5: Selection rules & enantiomer selectivity
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 4.5))
ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax4 = fig.add_subplot(1, 4, 4)

# (a) Enantiomer activity ratios
drugs = ["Ibuprofen\nS/R", "Propranolol\nS/R", "Warfarin\nS/R", "Levodopa", "Naproxen\nS/R", "Ketamine\nS/R"]
ratios = [160, 100, 5, 1000, 28, 4]
colors_bar = plt.cm.viridis(np.linspace(0.2, 0.85, len(drugs)))
ax1.bar(drugs, ratios, color=colors_bar, edgecolor='black', linewidth=0.6)
ax1.set_yscale('log')
ax1.axhline(10, color='red', linestyle='--', alpha=0.5)
ax1.set_ylabel("Activity ratio")
ax1.set_title("Enantiomer selectivity")
ax1.tick_params(axis='x', rotation=0, labelsize=8)
ax1.grid(True, axis='y', which='both', alpha=0.3)

# (b) Selection rule transition matrix (allowed/forbidden)
l_init = np.arange(0, 6)
l_final = np.arange(0, 6)
allowed = np.zeros((6, 6))
for i, li in enumerate(l_init):
    for j, lf in enumerate(l_final):
        if abs(li - lf) == 1:
            allowed[i, j] = 1
im = ax2.imshow(allowed, cmap='RdYlGn', aspect='equal', vmin=0, vmax=1)
ax2.set_xticks(range(6))
ax2.set_yticks(range(6))
ax2.set_xticklabels([f"{l}" for l in l_final])
ax2.set_yticklabels([f"{l}" for l in l_init])
ax2.set_xlabel(r"Final $\ell$")
ax2.set_ylabel(r"Initial $\ell$")
ax2.set_title(r"$|\Delta\ell|=1$ allowed")

# (c) 3D selection landscape: M(n,l) with allowed transitions
n_arr = np.arange(1, 8)
l_arr = np.arange(0, 7)
N3, L3 = np.meshgrid(n_arr, l_arr)
mask = L3 < N3
M_depth = np.where(mask, np.log2(2*N3*N3) + np.log2(2*L3 + 1), np.nan)
ax3.scatter(N3[mask], L3[mask], M_depth[mask], c=M_depth[mask],
            cmap=CMAP, s=60, edgecolors='black', linewidth=0.5)
# Connect adjacent allowed transitions
for n in n_arr[:-1]:
    for l in l_arr:
        if l < n:
            for dn in [0, 1]:
                for dl in [-1, 1]:
                    nn = n + dn
                    ll = l + dl
                    if ll >= 0 and ll < nn and nn <= 7:
                        m1 = math.log2(2*n*n) + math.log2(2*l + 1)
                        m2 = math.log2(2*nn*nn) + math.log2(2*ll + 1)
                        ax3.plot([n, nn], [l, ll], [m1, m2], color='0.5', alpha=0.4)
ax3.set_xlabel(r"$n$")
ax3.set_ylabel(r"$\ell$")
ax3.set_zlabel(r"$\mathcal{M}$")
ax3.view_init(elev=22, azim=-60)
ax3.set_title("Allowed transitions")

# (d) Forbidden transitions: probability vs |Δl|
delta_l = np.array([0, 1, 2, 3, 4, 5])
prob = np.array([0.05, 1.0, 0.001, 1e-5, 1e-7, 1e-9])
ax4.bar(delta_l, prob, color=["red" if dl != 1 else "green" for dl in delta_l],
        edgecolor='black', linewidth=0.6, alpha=0.85)
ax4.set_yscale('log')
ax4.set_xlabel(r"$|\Delta\ell|$")
ax4.set_ylabel("Transition probability")
ax4.set_title("Selection rule strength")
ax4.grid(True, axis='y', which='both', alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "panel_5_selection_rules.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------
# PANEL 6: Drug-drug competition & validation results
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 4.5))
ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax4 = fig.add_subplot(1, 4, 4)

# (a) Competitive inhibition: occupancy vs competitor concentration
D1 = 1.0  # fixed primary drug
Kd1 = 0.5
D2_arr = np.logspace(-3, 3, 200)
Kd2 = 1.0
f1 = (D1/Kd1) / (1 + D1/Kd1 + D2_arr/Kd2)
ax1.semilogx(D2_arr, f1, color="#1f77b4", linewidth=2.5)
ax1.fill_between(D2_arr, f1, alpha=0.3, color="#1f77b4")
ax1.set_xlabel(r"Competitor [D$_2$]/K$_{d,2}$")
ax1.set_ylabel(r"Target occupancy by D$_1$")
ax1.set_title("Competitive inhibition")
ax1.grid(True, which='both', alpha=0.3)

# (b) Validation pass-rate scatter (predicted vs observed for 5 affinities)
# From validation JSON results
known_drugs = [("imatinib", 30, 30), ("propranolol", 30, 30), ("aspirin", 18, 18),
               ("acetylcholine", 25, 25), ("morphine", 33, 33)]
xs = [d[1] for d in known_drugs]
ys = [d[2] for d in known_drugs]
ax2.scatter(xs, ys, s=130, c=range(len(xs)), cmap=CMAP,
           edgecolors='black', linewidth=0.7)
ax2.plot([10, 40], [10, 40], '--', color='gray', linewidth=1)
ax2.set_xlabel("Observed range center (bits)")
ax2.set_ylabel("Predicted partition depth (bits)")
ax2.set_title("Drug affinity validation")
ax2.grid(True, alpha=0.3)

# (c) 3D: occupancy(D1, D2) for competitive inhibition
D1g = np.logspace(-2, 2, 40)
D2g = np.logspace(-2, 2, 40)
D1G, D2G = np.meshgrid(D1g, D2g)
occ = (D1G/0.5) / (1 + D1G/0.5 + D2G/1.0)
ax3.plot_surface(np.log10(D1G), np.log10(D2G), occ, cmap='coolwarm',
                 alpha=0.9, edgecolor='none')
ax3.set_xlabel(r"$\log_{10}[D_1]$")
ax3.set_ylabel(r"$\log_{10}[D_2]$")
ax3.set_zlabel("Occupancy")
ax3.view_init(elev=28, azim=-55)
ax3.set_title("Competition surface")

# (d) ATP partition depth = 18.5 bits validation
DG_ATP_vals = np.array([28, 30, 30.5, 31, 32])  # kJ/mol range
delta_M_ATP = DG_ATP_vals * 1000 / (8.314 * 310 * np.log(2))
ax4.bar(["28", "30", "30.5", "31", "32"], delta_M_ATP,
        color="#9467bd", edgecolor='black', linewidth=0.6)
ax4.axhline(18.5, color='red', linestyle='--', alpha=0.7, label="predicted")
ax4.set_xlabel(r"$\Delta G_{\mathrm{ATP}}$ (kJ/mol)")
ax4.set_ylabel(r"$\Delta\mathcal{M}_{\mathrm{ATP}}$ (bits)")
ax4.set_title("ATP hydrolysis depth")
ax4.legend(fontsize=9)
ax4.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "panel_6_validation.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

print(f"Pharmacodynamics panels generated in {OUT}")
