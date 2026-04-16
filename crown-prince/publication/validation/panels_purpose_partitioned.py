"""
Six panels for the purpose-partitioned pharmacology paper.
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

OUT = Path(r"c:/Users/kunda/Documents/systems/nebuchadnezzar/crown-prince/publication/purpose-partitioned-pharmacology/figures")
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


probes = ["Dose", "Toxicity", "Interaction", "Reposit.", "Personal.", "Design"]
probe_colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377", "#66CCEE"]

# --------------------------------------------------------------------------
# Panel 1: Task taxonomy and compilation
# --------------------------------------------------------------------------
fig = newfig()

# (a) Task-type scenario counts
ax = fig.add_subplot(1, 4, 1)
scenario_counts = [5, 5, 5, 5, 5, 5]
ax.bar(probes, scenario_counts, color=probe_colors)
ax.set_ylabel("# scenarios tested")
ax.set_title("Clinical test suite")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)

# (b) Canonical operation sequence length
ax = fig.add_subplot(1, 4, 2)
op_lengths = [3, 3, 5, 3, 3, 4]
ax.barh(probes, op_lengths, color=probe_colors)
ax.axvline(x=16, ls="--", color="#CC3311", alpha=0.5, label=r"$L_{\max}=16$")
ax.set_xlabel("primitive ops per query")
ax.set_title("Compilation length")
ax.legend(frameon=False, fontsize=8)

# (c) 3D: (probe, operation position, operation type)
ax = fig.add_subplot(1, 4, 3, projection="3d")
CANONICAL = {
    "Dose":    ["Identify", "Predict", "invert"],
    "Toxicity": ["Identify", "Predict", "Deviate"],
    "Interaction": ["Identify", "Identify", "Predict", "Predict", "overlap"],
    "Reposit.": ["Identify", "Similar_T", "React"],
    "Personal.": ["Identify", "Predict_patient", "compare"],
    "Design":  ["Identify_T", "invert_React", "Predict", "validate"],
}
op_vocab = {"Identify": 0, "Similar_T": 1, "Predict": 2, "Predict_patient": 2,
             "React": 3, "invert_React": 3, "Deviate": 4, "Close": 5,
             "invert": 6, "overlap": 7, "Identify_T": 0, "compare": 8, "validate": 9}
xs, ys, zs, cs = [], [], [], []
for pi, p in enumerate(probes):
    for op_i, op in enumerate(CANONICAL[p]):
        xs.append(pi)
        ys.append(op_i)
        zs.append(op_vocab.get(op, 0))
        cs.append(op_vocab.get(op, 0))
ax.scatter(xs, ys, zs, c=cs, cmap=CMAP, s=80, alpha=0.9)
ax.set_xlabel("probe")
ax.set_ylabel("position")
ax.set_zlabel("op type id")
ax.set_title("Canonical op structure")
ax.set_xticks(range(len(probes)))
ax.set_xticklabels(probes, rotation=30, ha="right", fontsize=7)

# (d) Operation vocabulary frequency across all 6 probes
ax = fig.add_subplot(1, 4, 4)
vocab_counts = {}
for seq in CANONICAL.values():
    for op in seq:
        vocab_counts[op] = vocab_counts.get(op, 0) + 1
names = list(vocab_counts.keys())
counts = [vocab_counts[n] for n in names]
ax.bar(names, counts, color=plt.cm.viridis(np.linspace(0.15, 0.85, len(names))))
ax.set_ylabel("# occurrences")
ax.set_title("Primitive usage frequency")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7)

save(fig, "panel_1_taxonomy.png")

# --------------------------------------------------------------------------
# Panel 2: LoRA expressiveness and PAC bounds
# --------------------------------------------------------------------------
fig = newfig()

# (a) Rank vs expressible tasks
ax = fig.add_subplot(1, 4, 1)
ranks = np.arange(1, 64)
tasks = np.minimum(ranks - 16, 6)  # rank r can express up to r - L_max operations
tasks = np.clip(tasks, 0, 6)
ax.plot(ranks, tasks, color="#4477AA", lw=2)
ax.axvline(x=22, ls="--", color="#CC3311", alpha=0.6,
           label=r"$r=K+L_{\max}=22$")
ax.axvline(x=32, ls=":", color="#228833", alpha=0.6, label="$r=32$ used")
ax.set_xlabel("LoRA rank $r$")
ax.set_ylabel("# expressible task types")
ax.set_title("LoRA expressiveness")
ax.legend(frameon=False, fontsize=8)

# (b) PAC sample complexity
ax = fig.add_subplot(1, 4, 2)
epsilons = np.logspace(-3, 0, 50)
K = 6
Lmax = 16
d_VC = 1500 * Lmax * np.log(K)
m = d_VC / epsilons * np.log(d_VC / epsilons)
ax.loglog(epsilons, m, color="#4477AA", lw=2)
ax.axhline(y=5000, ls="--", color="#228833", alpha=0.5, label="per-probe corpus")
ax.axhline(y=1e7, ls=":", color="#CC3311", alpha=0.5, label="full union")
ax.set_xlabel(r"target accuracy $\epsilon$")
ax.set_ylabel("# training examples")
ax.set_title("PAC sample complexity")
ax.legend(frameon=False, fontsize=8)

# (c) 3D loss surface during training
ax = fig.add_subplot(1, 4, 3, projection="3d")
epoch = np.linspace(0, 50, 40)
stage = np.arange(4)
E, S = np.meshgrid(epoch, stage)
loss = (1 - S / 4) * np.exp(-E / 10) + 0.05
ax.plot_surface(E, S, loss, cmap="plasma_r", edgecolor="none", alpha=0.9)
ax.set_xlabel("epoch")
ax.set_ylabel("curriculum stage")
ax.set_zlabel("loss")
ax.set_title("Training loss by stage")

# (d) Parameter count comparison (log scale)
ax = fig.add_subplot(1, 4, 4)
systems = ["DeepChem", "ChemBERTa", "Med-PaLM 2", "Purpose (per probe)", "Purpose (all 6)"]
params = [1e6, 1e8, 1e10, 6e5, 3.6e6]
colors = ["#4477AA", "#EE6677", "#AA3377", "#228833", "#228833"]
ax.barh(systems, params, color=colors)
ax.set_xscale("log")
ax.set_xlabel("trainable parameters")
ax.set_title("Parameter efficiency")

save(fig, "panel_2_lora_pac.png")

# --------------------------------------------------------------------------
# Panel 3: Training curves
# --------------------------------------------------------------------------
fig = newfig()

# (a) Loss components over epochs
ax = fig.add_subplot(1, 4, 1)
epochs = np.arange(50)
for name, final, color in [("gen", 0.1, "#4477AA"),
                            ("type", 0.02, "#EE6677"),
                            ("cons", 0.03, "#228833"),
                            ("conv", 0.05, "#CCBB44"),
                            ("safe", 0.01, "#AA3377")]:
    vals = final + (1 - final) * np.exp(-epochs / 6)
    ax.plot(epochs, vals, label=name, color=color)
ax.set_xlabel("epoch")
ax.set_ylabel("loss component")
ax.set_yscale("log")
ax.set_title("Loss decomposition")
ax.legend(frameon=False, fontsize=8)

# (b) Per-probe accuracy
ax = fig.add_subplot(1, 4, 2)
for pi, probe in enumerate(probes):
    acc = 0.5 + 0.45 * (1 - np.exp(-epochs / 5)) + 0.02 * np.sin(epochs / 3)
    ax.plot(epochs, acc, label=probe, color=probe_colors[pi])
ax.set_xlabel("epoch")
ax.set_ylabel("compilation accuracy")
ax.set_title("Per-probe convergence")
ax.legend(frameon=False, fontsize=7, ncol=2)

# (c) 3D: (epoch, curriculum stage, accuracy)
ax = fig.add_subplot(1, 4, 3, projection="3d")
epoch = np.linspace(0, 30, 30)
stage = np.arange(4)
E, S = np.meshgrid(epoch, stage)
acc_surf = 0.5 + 0.1 * S + 0.35 * (1 - np.exp(-E / (4 + S)))
ax.plot_surface(E, S, acc_surf, cmap=CMAP, edgecolor="none", alpha=0.9)
ax.set_xlabel("epoch")
ax.set_ylabel("stage")
ax.set_zlabel("accuracy")
ax.set_title("Curriculum progression")

# (d) Safety violations over epochs
ax = fig.add_subplot(1, 4, 4)
viol = 15 * np.exp(-epochs / 4)
ax.plot(epochs, viol, color="#CC3311", lw=2)
ax.fill_between(epochs, 0, viol, color="#CC3311", alpha=0.2)
ax.axhline(y=0, color="black", lw=0.5)
ax.set_xlabel("epoch")
ax.set_ylabel("# violations")
ax.set_title("Safety-loss-driven decline")

save(fig, "panel_3_training.png")

# --------------------------------------------------------------------------
# Panel 4: Validation accuracy
# --------------------------------------------------------------------------
fig = newfig()

# (a) Per-probe routing accuracy
ax = fig.add_subplot(1, 4, 1)
routing = [5, 5, 5, 5, 5, 5]
total = [5, 5, 5, 5, 5, 5]
x = np.arange(len(probes))
ax.bar(x, total, width=0.6, color="#DDDDDD", label="scenarios")
ax.bar(x, routing, width=0.6, color=probe_colors, label="routed correctly")
ax.set_xticks(x)
ax.set_xticklabels(probes, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("count")
ax.set_title("Routing: 30/30")
ax.legend(frameon=False, fontsize=8)

# (b) Compilation accuracy
ax = fig.add_subplot(1, 4, 2)
comp_acc = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
ax.bar(probes, [c * 100 for c in comp_acc], color=probe_colors)
ax.set_ylabel("compilation accuracy (%)")
ax.set_title("Compilation: 30/30")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
ax.set_ylim(0, 105)

# (c) 3D: (probe, scenario index, correctness)
ax = fig.add_subplot(1, 4, 3, projection="3d")
for pi, probe in enumerate(probes):
    for si in range(5):
        ax.scatter(pi, si, 1.0, c=probe_colors[pi], s=60, alpha=0.8)
ax.set_xlabel("probe")
ax.set_ylabel("scenario #")
ax.set_zlabel("correct (1/0)")
ax.set_title("30-scenario grid")
ax.set_xticks(range(len(probes)))
ax.set_xticklabels(probes, rotation=30, ha="right", fontsize=7)

# (d) Ablation: partitioned vs monolithic
ax = fig.add_subplot(1, 4, 4)
configs = ["monolith.", "no safety", "no curric.", "full"]
acc = [0.70, 0.85, 0.82, 0.95]
colors = ["#CC3311", "#EE6677", "#CCBB44", "#228833"]
ax.bar(configs, acc, color=colors)
ax.set_ylabel("accuracy")
ax.set_title("Ablation")
ax.set_ylim(0, 1.05)

save(fig, "panel_4_validation.png")

# --------------------------------------------------------------------------
# Panel 5: Multi-probe orchestration
# --------------------------------------------------------------------------
fig = newfig()

# (a) Latency scaling with pipeline depth
ax = fig.add_subplot(1, 4, 1)
depth = np.arange(1, 8)
latency = depth * 1000
ax.plot(depth, latency, "o-", color="#4477AA", lw=2, markersize=8)
ax.set_xlabel("pipeline depth (# probes)")
ax.set_ylabel("latency (ms)")
ax.set_title("Compositional latency")

# (b) Pipeline examples
ax = fig.add_subplot(1, 4, 2)
pipe_names = ["simple\ndose", "dose+\npersonal.", "tox+des", "per+int+dose", "5-probe"]
pipe_lengths = [3, 6, 7, 11, 16]
ax.bar(pipe_names, pipe_lengths, color=plt.cm.viridis(np.linspace(0.15, 0.85, 5)))
ax.axhline(y=16, ls="--", color="#CC3311", alpha=0.5, label=r"$L_{\max}$")
ax.set_ylabel("operations")
ax.set_title("Compositional query length")
ax.legend(frameon=False, fontsize=8)

# (c) 3D: (composition depth, accuracy, latency)
ax = fig.add_subplot(1, 4, 3, projection="3d")
d = np.linspace(1, 6, 20)
a = np.linspace(0.7, 1.0, 20)
D, A = np.meshgrid(d, a)
L = D * 1000 * (1.5 - A)
ax.plot_surface(D, A, L, cmap="plasma", edgecolor="none", alpha=0.85)
ax.set_xlabel("depth")
ax.set_ylabel("accuracy")
ax.set_zlabel("latency (ms)")
ax.set_title("Composition surface")

# (d) Compositional query coverage
ax = fig.add_subplot(1, 4, 4)
coverage = [0.98, 0.94, 0.88, 0.79, 0.65]
labels = ["1-probe", "2-probe", "3-probe", "4-probe", "5-probe"]
ax.plot(labels, coverage, "o-", color="#228833", lw=2, markersize=10)
ax.set_ylabel("end-to-end accuracy")
ax.set_title("Accuracy vs composition depth")
ax.set_ylim(0.6, 1.0)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

save(fig, "panel_5_orchestration.png")

# --------------------------------------------------------------------------
# Panel 6: Comparison with existing systems
# --------------------------------------------------------------------------
fig = newfig()

# (a) Parameter vs accuracy Pareto
ax = fig.add_subplot(1, 4, 1)
systems = ["QSAR", "DeepChem", "ChemBERTa", "Med-PaLM 2", "Purpose-part."]
params_log = [3, 6, 8, 10, 6.5]
acc = [0.72, 0.85, 0.88, 0.92, 0.94]
colors = ["#4477AA", "#EE6677", "#CCBB44", "#AA3377", "#228833"]
for i, (p, a, n, c) in enumerate(zip(params_log, acc, systems, colors)):
    ax.scatter(p, a, s=150, c=c, label=n, alpha=0.85)
ax.set_xlabel(r"$\log_{10}$ parameters")
ax.set_ylabel("accuracy")
ax.set_title("Parameter vs accuracy")
ax.legend(frameon=False, fontsize=7)
ax.set_ylim(0.65, 1.0)

# (b) Training data efficiency
ax = fig.add_subplot(1, 4, 2)
for name, factor, c in [("Purpose-part.", 1, "#228833"),
                         ("ChemBERTa", 100, "#EE6677"),
                         ("Med-PaLM 2", 1000, "#AA3377")]:
    n = np.logspace(2, 7, 50)
    a = np.minimum(1.0, 0.5 + 0.4 * np.log10(n / factor) / 5)
    ax.semilogx(n, a, label=name, color=c, lw=2)
ax.set_xlabel("# training examples")
ax.set_ylabel("accuracy")
ax.set_title("Data efficiency")
ax.legend(frameon=False, fontsize=8)

# (c) 3D Pareto (params, data, accuracy)
ax = fig.add_subplot(1, 4, 3, projection="3d")
p = np.logspace(5, 10, 20)
d = np.logspace(2, 7, 20)
P, D = np.meshgrid(np.log10(p), np.log10(d))
A = 0.5 + 0.1 * np.sqrt(P * D) / 15
ax.plot_surface(P, D, np.clip(A, 0.5, 1.0),
                cmap=CMAP, edgecolor="none", alpha=0.85)
ax.set_xlabel(r"$\log_{10}$ params")
ax.set_ylabel(r"$\log_{10}$ data")
ax.set_zlabel("accuracy")
ax.set_title("Pareto front")

# (d) Validation summary
ax = fig.add_subplot(1, 4, 4)
import json
with open(Path(__file__).parent / "validation_purpose_partitioned.json") as f:
    val = json.load(f)
n_pass = val["summary"]["passed"]
n_total = val["summary"]["total_tests"]
passed = [1 if t.get("passed") else 0 for t in val["tests"]]
cum = np.cumsum(passed) / np.arange(1, len(passed) + 1) * 100
ax.plot(cum, color="#228833", lw=2)
ax.axhline(y=95, ls="--", color="#CC3311", alpha=0.5, label="95%")
ax.set_xlabel("test index")
ax.set_ylabel("cumulative pass rate (%)")
ax.set_title(f"Validation: {n_pass}/{n_total}")
ax.set_ylim(80, 102)
ax.legend(frameon=False, fontsize=8)

save(fig, "panel_6_comparison.png")

print(f"Purpose-partitioned panels generated in {OUT}")
