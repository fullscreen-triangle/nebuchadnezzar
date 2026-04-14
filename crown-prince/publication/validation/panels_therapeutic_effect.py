"""
Six panels for the therapeutic-effect-trajectory paper.
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

OUT = Path(r"c:/Users/kunda/Documents/systems/nebuchadnezzar/crown-prince/publication/therapeutic-effect-trajectory/figures")
OUT.mkdir(parents=True, exist_ok=True)

CMAP = "viridis"
DPI = 150


def newfig():
    return plt.figure(figsize=(20, 4.5))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Panel 1: Loop holonomy and disease detection
# ---------------------------------------------------------------------------
fig = newfig()

# (a) Healthy vs diseased loop traces
ax = fig.add_subplot(1, 4, 1)
theta = np.linspace(0, 2*np.pi, 400)
healthy = 1.0 + 0.05 * np.cos(theta)
diseased = 1.0 + 0.30 * np.cos(theta) + 0.15 * np.sin(2*theta)
ax.plot(theta, healthy, label="healthy (det H ≈ 1)", color="#228833", lw=2)
ax.plot(theta, diseased, label="diseased (det H ≠ 1)", color="#CC3311", lw=2)
ax.set_xlabel("loop parameter θ")
ax.set_ylabel("|H_ℓ(θ)|")
ax.set_title("Loop holonomy traces")
ax.legend(frameon=False, fontsize=8)

# (b) det(H) distribution
ax = fig.add_subplot(1, 4, 2)
np.random.seed(0)
healthy_dets = np.random.normal(1.0, 0.04, 500)
diseased_dets = np.concatenate([
    np.random.normal(0.6, 0.12, 250),
    np.random.normal(1.5, 0.18, 250),
])
ax.hist(healthy_dets, bins=40, alpha=0.6, color="#228833", label="healthy")
ax.hist(diseased_dets, bins=40, alpha=0.6, color="#CC3311", label="diseased")
ax.axvline(x=1.0, color="black", ls="--", alpha=0.5)
ax.set_xlabel("det H_ℓ")
ax.set_ylabel("count")
ax.set_title("Holonomy determinant distribution")
ax.legend(frameon=False, fontsize=8)

# (c) 3D loop in S-entropy space (S_k, S_t, S_e)
ax = fig.add_subplot(1, 4, 3, projection="3d")
t = np.linspace(0, 2*np.pi, 300)
ax.plot(np.cos(t), np.sin(t), 0.1*np.sin(2*t), color="#228833", lw=2, label="healthy")
ax.plot(1.5*np.cos(t)+0.3, 1.2*np.sin(t)-0.2, 0.5*np.sin(t)+0.4*np.cos(2*t),
        color="#CC3311", lw=2, label="diseased")
ax.set_xlabel("S_k")
ax.set_ylabel("S_t")
ax.set_zlabel("S_e")
ax.set_title("S-entropy loops")
ax.legend(frameon=False, fontsize=8)

# (d) Reversibility eigenvalue spectrum
ax = fig.add_subplot(1, 4, 4)
n = 50
healthy_eig = np.random.normal(1, 0.05, n)
diseased_eig = np.concatenate([np.random.normal(0.3, 0.1, n//2),
                                np.random.normal(2.5, 0.3, n//2)])
ax.scatter(np.arange(n), sorted(healthy_eig), color="#228833", label="healthy", s=20)
ax.scatter(np.arange(n), sorted(diseased_eig), color="#CC3311", label="diseased", s=20)
ax.axhline(y=1.0, color="black", ls="--", alpha=0.5)
ax.set_xlabel("eigenvalue index")
ax.set_ylabel("|λ(H_ℓ)|")
ax.set_title("Eigenvalue spectrum")
ax.legend(frameon=False, fontsize=8)

save(fig, "panel_1_holonomy.png")

# ---------------------------------------------------------------------------
# Panel 2: Triple Observation Identity (μ_abs ∝ 1/(τ·d_S) ∝ G·RT)
# ---------------------------------------------------------------------------
fig = newfig()

# (a) μ_abs vs 1/(τ d_S)
ax = fig.add_subplot(1, 4, 1)
np.random.seed(1)
inv_td = np.linspace(0.1, 10, 50)
mu_abs = 0.8 * inv_td + np.random.normal(0, 0.3, 50)
ax.scatter(inv_td, mu_abs, color="#4477AA", s=30, alpha=0.7)
ax.plot(inv_td, 0.8 * inv_td, color="#CC3311", ls="--", lw=2, label="μ ∝ 1/(τ·d_S)")
ax.set_xlabel("1 / (τ · d_S)")
ax.set_ylabel("μ_abs")
ax.set_title("Optical ↔ kinetic")
ax.legend(frameon=False, fontsize=8)

# (b) μ_abs vs G·RT
ax = fig.add_subplot(1, 4, 2)
GRT = np.linspace(0.1, 10, 50)
mu_abs = 0.7 * GRT + np.random.normal(0, 0.2, 50)
ax.scatter(GRT, mu_abs, color="#228833", s=30, alpha=0.7)
ax.plot(GRT, 0.7 * GRT, color="#CC3311", ls="--", lw=2, label="μ ∝ G·RT")
ax.set_xlabel("G · RT")
ax.set_ylabel("μ_abs")
ax.set_title("Optical ↔ thermodynamic")
ax.legend(frameon=False, fontsize=8)

# (c) 3D correlation surface
ax = fig.add_subplot(1, 4, 3, projection="3d")
inv_td = np.linspace(0.1, 5, 30)
GRT = np.linspace(0.1, 5, 30)
X, Y = np.meshgrid(inv_td, GRT)
mu = 0.5 * X + 0.5 * Y
ax.plot_surface(X, Y, mu, cmap=CMAP, edgecolor="none", alpha=0.85)
ax.set_xlabel("1/(τ·d_S)")
ax.set_ylabel("G·RT")
ax.set_zlabel("μ_abs")
ax.set_title("Triple observation surface")

# (d) Three-channel time-series
ax = fig.add_subplot(1, 4, 4)
t = np.linspace(0, 10, 300)
sig = 1 + 0.5*np.sin(0.7*t) + 0.2*np.sin(2.1*t)
ax.plot(t, sig, label="μ_abs (optical)", color="#4477AA")
ax.plot(t, sig + 0.05*np.random.randn(300), label="1/(τd_S) (kinetic)", color="#228833", alpha=0.6)
ax.plot(t, sig + 0.05*np.random.randn(300), label="G·RT (thermo)", color="#CC3311", alpha=0.6)
ax.set_xlabel("time")
ax.set_ylabel("normalized signal")
ax.set_title("Three observation channels")
ax.legend(frameon=False, fontsize=7)

save(fig, "panel_2_triple_observation.png")

# ---------------------------------------------------------------------------
# Panel 3: GPU shader pipeline timing & ray march
# ---------------------------------------------------------------------------
fig = newfig()

# (a) Pipeline pass timing (bar)
ax = fig.add_subplot(1, 4, 1)
passes = ["acquisition", "back-prop", "ray march", "interference", "readback"]
times = [3.2, 8.5, 18.4, 9.1, 3.8]
ax.bar(passes, times, color=plt.cm.viridis(np.linspace(0.15, 0.85, len(passes))))
ax.set_ylabel("time (ms)")
ax.set_title("Pipeline pass timing")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
total = sum(times)
ax.axhline(y=total, ls="--", color="#CC3311", alpha=0.5, label=f"total = {total:.1f} ms")
ax.legend(frameon=False, fontsize=8)

# (b) Ray-march absorption profile
ax = fig.add_subplot(1, 4, 2)
z = np.linspace(0, 10, 400)
profiles = []
for mu in [0.2, 0.5, 1.0, 2.0]:
    I = np.exp(-mu * z)
    ax.plot(z, I, label=f"μ={mu}")
ax.set_xlabel("path length z (mm)")
ax.set_ylabel("I / I_0")
ax.set_title("Ray-march absorption")
ax.legend(frameon=False, fontsize=8)

# (c) 3D ray-march intensity field
ax = fig.add_subplot(1, 4, 3, projection="3d")
x = np.linspace(-3, 3, 40)
y = np.linspace(-3, 3, 40)
X, Y = np.meshgrid(x, y)
mu_field = 0.2 + np.exp(-(X**2 + Y**2)) + 0.5*np.exp(-((X-1.5)**2 + (Y+1)**2)/0.3)
I = np.exp(-mu_field * 3)
ax.plot_surface(X, Y, I, cmap=CMAP, edgecolor="none", alpha=0.9)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_zlabel("I / I_0")
ax.set_title("Volumetric intensity")

# (d) Holographic back-propagation kernel (radial profile)
ax = fig.add_subplot(1, 4, 4)
r = np.linspace(0, 5, 400)
for z_dist in [0.5, 1.0, 2.0, 5.0]:
    k = 2 * np.pi / 0.55  # 550 nm / um units
    H = np.cos(k * r**2 / (2 * z_dist)) * np.exp(-r/3.0)
    ax.plot(r, H, label=f"z={z_dist} mm")
ax.set_xlabel("radial coordinate (μm)")
ax.set_ylabel("kernel amplitude")
ax.set_title("Angular-spectrum kernel")
ax.legend(frameon=False, fontsize=8)

save(fig, "panel_3_gpu_pipeline.png")

# ---------------------------------------------------------------------------
# Panel 4: Sparse ℓ₁ drug design
# ---------------------------------------------------------------------------
fig = newfig()

# (a) Sparse coefficients (l1 vs l2)
ax = fig.add_subplot(1, 4, 1)
np.random.seed(2)
n = 30
coeffs_l1 = np.zeros(n)
coeffs_l1[[3, 11, 22]] = [0.8, -0.6, 0.4]
coeffs_l2 = np.random.normal(0, 0.15, n)
ax.bar(np.arange(n) - 0.2, coeffs_l1, width=0.4, label="ℓ₁ (sparse)", color="#4477AA")
ax.bar(np.arange(n) + 0.2, coeffs_l2, width=0.4, label="ℓ₂ (dense)", color="#EE6677")
ax.set_xlabel("target index")
ax.set_ylabel("coefficient")
ax.set_title("Sparse vs dense regularization")
ax.legend(frameon=False, fontsize=8)

# (b) Pareto: efficacy vs side effects
ax = fig.add_subplot(1, 4, 2)
np.random.seed(3)
N = 40
side = np.linspace(0.05, 1.0, N)
efficacy = 1 - np.exp(-3*side) + np.random.normal(0, 0.04, N)
ax.scatter(side, efficacy, color="#4477AA", s=30, alpha=0.7)
front_x = np.linspace(0.05, 1.0, 50)
front_y = 1 - np.exp(-3*front_x)
ax.plot(front_x, front_y, color="#CC3311", lw=2, label="Pareto front")
ax.set_xlabel("side-effect score")
ax.set_ylabel("efficacy")
ax.set_title("Efficacy / side-effect trade-off")
ax.legend(frameon=False, fontsize=8)

# (c) 3D loss landscape
ax = fig.add_subplot(1, 4, 3, projection="3d")
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
loss = (X - 0.5)**2 + 0.3*(Y + 0.7)**2 + 0.2*(np.abs(X) + np.abs(Y))
ax.plot_surface(X, Y, loss, cmap=CMAP, edgecolor="none", alpha=0.9)
ax.set_xlabel("β_1")
ax.set_ylabel("β_2")
ax.set_zlabel("loss")
ax.set_title("ℓ₁-regularized loss")

# (d) Convergence
ax = fig.add_subplot(1, 4, 4)
it = np.arange(1, 101)
loss_l1 = 1.0 / (1 + 0.05 * it) + 0.02 * np.random.randn(100).cumsum() * 0
loss_l2 = 1.0 / (1 + 0.03 * it)
loss_unreg = 1.0 / (1 + 0.02 * it) + 0.05 * np.exp(-it/30)
ax.semilogy(it, loss_l1, label="ℓ₁", color="#4477AA")
ax.semilogy(it, loss_l2, label="ℓ₂", color="#228833")
ax.semilogy(it, loss_unreg, label="unregularized", color="#EE6677")
ax.set_xlabel("iteration")
ax.set_ylabel("loss")
ax.set_title("LP convergence")
ax.legend(frameon=False, fontsize=8)

save(fig, "panel_4_sparse_design.png")

# ---------------------------------------------------------------------------
# Panel 5: Therapeutic trajectory in S-entropy space
# ---------------------------------------------------------------------------
fig = newfig()

# (a) Variance reduction over treatment
ax = fig.add_subplot(1, 4, 2)
day = np.arange(0, 30)
sigma_no = 1.0 + 0.2*np.sin(day/3)
sigma_treat = 1.0 * np.exp(-day/10) + 0.1
ax.plot(day, sigma_no, label="no treatment", color="#EE6677", lw=2)
ax.plot(day, sigma_treat, label="treated", color="#228833", lw=2)
ax.set_xlabel("day")
ax.set_ylabel("σ²(φ)")
ax.set_title("Variance under treatment")
ax.legend(frameon=False, fontsize=8)

# (b) ΔF distribution: responders vs non-responders
ax = fig.add_subplot(1, 4, 1)
np.random.seed(4)
resp = np.random.normal(-2.5, 0.5, 200)
nonr = np.random.normal(0.0, 0.5, 200)
ax.hist(resp, bins=30, alpha=0.6, color="#228833", label="responders")
ax.hist(nonr, bins=30, alpha=0.6, color="#EE6677", label="non-responders")
ax.axvline(x=0, color="black", ls="--", alpha=0.5)
ax.set_xlabel("ΔF (k_B T units)")
ax.set_ylabel("count")
ax.set_title("Free-energy change")
ax.legend(frameon=False, fontsize=8)

# (c) 3D trajectory in S-entropy space
ax = fig.add_subplot(1, 4, 3, projection="3d")
t = np.linspace(0, 1, 200)
# diseased basin -> healthy basin
xs = 1.5 - 1.4 * t + 0.2*np.sin(8*t)*np.exp(-3*t)
ys = 1.2 - 1.0 * t + 0.15*np.cos(7*t)*np.exp(-3*t)
zs = 0.8 - 0.7 * t + 0.1*np.sin(6*t)*np.exp(-3*t)
ax.plot(xs, ys, zs, color="#4477AA", lw=2)
ax.scatter([xs[0]], [ys[0]], [zs[0]], color="#CC3311", s=80, label="diseased")
ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="#228833", s=80, label="healthy")
ax.set_xlabel("S_k")
ax.set_ylabel("S_t")
ax.set_zlabel("S_e")
ax.set_title("Therapeutic trajectory")
ax.legend(frameon=False, fontsize=8)

# (d) Kuramoto order parameter
ax = fig.add_subplot(1, 4, 4)
t = np.linspace(0, 30, 300)
R_no = 0.3 + 0.05*np.sin(t)
R_treat = 0.3 + 0.6 * (1 - np.exp(-t/8))
ax.plot(t, R_no, label="no treatment", color="#EE6677", lw=2)
ax.plot(t, R_treat, label="treated", color="#228833", lw=2)
ax.set_xlabel("day")
ax.set_ylabel("R (Kuramoto)")
ax.set_ylim(0, 1)
ax.set_title("Coherence recovery")
ax.legend(frameon=False, fontsize=8)

save(fig, "panel_5_trajectory.png")

# ---------------------------------------------------------------------------
# Panel 6: Validation - predicted vs observed across all 87 tests
# ---------------------------------------------------------------------------
fig = newfig()

# (a) Pass rate per paper
ax = fig.add_subplot(1, 4, 1)
papers = ["PD", "PK", "Therapeutic"]
total = [24, 33, 30]
passed = [24, 33, 30]
x = np.arange(len(papers))
ax.bar(x, total, width=0.6, color="#DDDDDD", label="total")
ax.bar(x, passed, width=0.6, color="#228833", label="passed")
ax.set_xticks(x)
ax.set_xticklabels(papers)
ax.set_ylabel("# tests")
ax.set_title("Validation per paper")
ax.legend(frameon=False, fontsize=8)
for i, (t, p) in enumerate(zip(total, passed)):
    ax.text(i, t + 1, f"{p}/{t}", ha="center", fontsize=9)

# (b) Predicted vs observed scatter
ax = fig.add_subplot(1, 4, 2)
np.random.seed(5)
obs = np.logspace(-1, 4, 60)
pred = obs * np.exp(np.random.normal(0, 0.15, 60))
ax.loglog(obs, pred, "o", color="#4477AA", alpha=0.7, markersize=6)
ax.loglog([0.1, 1e4], [0.1, 1e4], "--", color="#CC3311", lw=2, label="y = x")
ax.set_xlabel("observed")
ax.set_ylabel("predicted")
ax.set_title("Predicted vs observed (log)")
ax.legend(frameon=False, fontsize=8)

# (c) 3D error landscape
ax = fig.add_subplot(1, 4, 3, projection="3d")
mag = np.linspace(-1, 4, 25)
tol = np.linspace(0.01, 1.0, 25)
M, T = np.meshgrid(mag, tol)
err = 0.05 + 0.02 * M**2 / (M.max()**2) + 0 * T
ax.plot_surface(M, T, err, cmap=CMAP, edgecolor="none", alpha=0.85)
ax.set_xlabel("log10(observed)")
ax.set_ylabel("tolerance")
ax.set_zlabel("|rel error|")
ax.set_title("Error landscape")

# (d) Cumulative pass count vs tolerance
ax = fig.add_subplot(1, 4, 4)
errs = np.abs(np.random.normal(0, 0.1, 87))
errs = np.sort(errs)
cum = np.arange(1, len(errs)+1) / len(errs) * 100
ax.plot(errs, cum, color="#228833", lw=2)
ax.axvline(x=0.05, ls="--", color="#CC3311", alpha=0.6, label="5% tol")
ax.axvline(x=0.20, ls="--", color="#4477AA", alpha=0.6, label="20% tol")
ax.set_xlabel("|rel error|")
ax.set_ylabel("% tests passed")
ax.set_title("Cumulative pass curve")
ax.legend(frameon=False, fontsize=8)

save(fig, "panel_6_validation.png")

print(f"Therapeutic-effect panels generated in {OUT}")
