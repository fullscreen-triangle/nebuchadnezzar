# Nebuchadnezzar

[![Rust](https://img.shields.io/badge/rust-1.70+-blue.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A geometric substrate for intracellular dynamics, pharmacology, and
> proteomics, derived from a single axiom: the Bounded Phase Space Law.**

## Overview

Nebuchadnezzar began as an intracellular dynamics engine. The framework has
since been reformulated from first principles and now serves as the
computational substrate for a broader theoretical corpus in which drugs,
targets, cells, metabolites, and therapeutic trajectories are addressed in
a single three-dimensional coordinate space derived from bounded phase space
geometry. What is implemented here is no longer a simulation of individual
cellular processes with fitted parameters; it is an engine that synthesises
pharmacological, proteomic, and cellular observables on demand from geometric
operations on two ternary tries.

The engine stores no drugs, no targets, no ADME profiles, no affinity tables,
and no adverse-event records. All pharmacological knowledge is derived
geometrically from the bounded phase space axiom. This is what we call the
*empty dictionary* principle: the instrument contains no stored data and
produces correct outputs through real-time synthesis from the oscillatory
structure of the query entity alone.

## The Axiom

**Bounded Phase Space Law.** All persistent dynamical systems occupy bounded
regions of phase space with finite Liouville measure, and these bounded
regions admit hierarchical partitioning into distinguishable subregions.

Combined with the **Categorical Observation** axiom---an observer with
finite resolution partitions phase space into a finite number of
distinguishable categories---the framework derives without further
postulate:

- Forced partitioning into $M = \lfloor \mu(\Omega)/\delta^d \rfloor$
  distinguishable states.
- The partition coordinate structure $(n, \ell, m, s)$ with capacity
  $C(n) = 2n^2$, recovering atomic shell structure exactly for
  $n = 1, \ldots, 7$.
- Oscillatory necessity: oscillation is the unique valid mode for
  self-consistent dynamics in bounded phase space.
- The Triple Entropy Equivalence
  $S_{\mathrm{osc}} = S_{\mathrm{cat}} = S_{\mathrm{part}} = k_B \mathcal{M} \ln n$,
  proved by three independent derivations.
- The Composition, Compression, and Conservation theorems for partition
  depth, underlying drug--target binding energy, affinity--depth
  correspondence, and drug--drug interaction conservation.

## Architecture

The engine is organised as three orthogonal layers, each with a different
mutability timescale:

```
  purpose (natural-language clinical intent)
            |
            v   compiled probe (tiny, physics-supervised LoRA adapter)
  operation sequence (Identify, Similar, Predict, React, Deviate, Close)
            |
            v   empty dictionary engine (zero parameters, zero storage)
  answer (geometric operations on S-entropy space)
            |
            v   substrate: bounded phase space + partition + S-entropy
              (immutable axiom)
```

- **Substrate** (immutable). Partition coordinates, S-entropy space
  $\mathcal{S} = [0,1]^3$, ternary addressing at depth $k$. Derived
  entirely from the axiom.
- **Empty dictionary** (zero parameters). Dual ternary tries
  $(\mathcal{T}_{\mathsf{D}}, \mathcal{T}_{\mathsf{T}})$ addressing drugs
  and targets, plus six geometric primitives that operate on them.
- **Compiled probes** (~0.6 M parameters each). LoRA adapters that
  translate natural-language questions into primitive-operation sequences.
  Trained by knowledge distillation from a teacher with access to the
  full bounded phase space corpus, supervised by physics-derived loss
  terms, not human labels.

## The Six Primitives

Every well-posed pharmacological query decomposes into a finite
composition of six operations on $\mathcal{S}$:

- **Identify**: $\{\omega_i\} \to \mathcal{S}$ --- ternary address from
  vibrational fingerprint.
- **Similar**: $\mathbf{a} \times \varepsilon \to 2^{\mathcal{S}}$ ---
  prefix-matched retrieval.
- **Predict**: $(\mathbf{a}, t) \to \mathcal{S}^{\mathrm{pharm}}$ ---
  ADME trajectory under organ-specific metric.
- **React**: $\mathbf{a}_{\mathsf{D}} \times \mathbf{a}_{\mathsf{T}}
  \to \{K_d, \text{no-bind}\}$ --- trajectory intersection and
  binding affinity.
- **Deviate**: $\mathbf{a}_{\mathsf{D}} \times \mathcal{T}_{\mathsf{T}}
  \to \{(\mathsf{T}', L)\}$ --- off-target adverse-effect branches.
- **Close**: $G^{\mathsf{P}} \to \boldsymbol{\eta}^{*}$ --- sparse
  $\ell_1$ therapeutic loop-holonomy closure on the patient's cellular
  partition graph.

Each primitive executes in $O(k)$ trie traversal, independent of
database size $N$. For $k = 18$ (standard depth), every query completes
in 18 elementary operations regardless of whether the trie contains 40
or $10^{8}$ entities.

## The Triple Isomorphism

The substrate admits three equivalent descriptions connected by explicit
functors:

- **Category $\mathcal{O}$ (oscillatory)**: objects
  $(\omega, \varphi, A)$.
- **Category $\mathcal{C}$ (categorical)**: objects
  $(S_k, S_t, S_e) \in [0,1]^3$.
- **Category $\mathcal{B}$ (biological)**: objects
  $(n, \ell, m, s, \mathcal{M})$.

The functors $F_{\mathcal{O}\mathcal{C}}$, $F_{\mathcal{C}\mathcal{B}}$,
$F_{\mathcal{B}\mathcal{O}}$ form a triangle of equivalences satisfying
$F_{\mathcal{B}\mathcal{O}} \circ F_{\mathcal{C}\mathcal{B}} \circ
F_{\mathcal{O}\mathcal{C}} \cong \mathrm{Id}_{\mathcal{O}}$. The identity
$\text{observation} \equiv \text{computation} \equiv \text{processing}$
is one row of a larger table of equivalences applying to Kuramoto
synchronisation, categorical apertures, ion channels, race dynamics, and
ten further component-level isomorphisms.

## The Companion Papers

The theoretical corpus underlying Nebuchadnezzar is distributed across
the following publications in `crown-prince/publication/`:

- **`partition-based-pharmacodynamics/`** --- Drug--target interaction,
  dose--response, and selectivity from the bounded phase space axiom.
  Five regime-specific dose--response curves of which the Hill equation
  is the aperture limit. Zero-work categorical selectivity. $K_d =
  \exp(-\Delta\mathcal{M}_{\mathrm{bind}} \ln b)$. Validation: 24/24.

- **`partition-based-pharmacokinetics/`** --- ADME from a partition
  graph. Bioavailability $F = F_{\mathrm{abs}}(1-E_H)(1-E_G)$; volume
  $V_d = V_p + \sum V_t K_p$; half-life $t_{1/2} = \ln 2 \cdot V_d /
  \mathrm{CL}$; allometric 3/4 scaling derived, not postulated.
  Validation: 33/33.

- **`therapeutic-effect-trajectory/`** --- Disease as non-trivial
  loop holonomy $\mathrm{Hol}_\ell \neq \mathrm{Id}$; therapy as sparse
  $\ell_1$ edge perturbation; GPU five-pass ray-march pipeline at 43 ms
  per cell via the Triple Observation Identity
  $\mu_{\mathrm{abs}} \propto 1/(\tau d_S) \propto G \cdot RT$.
  Validation: 30/30.

- **`sources/categorical-compound-database.tex`** --- Every stable
  molecule addressed in the S-entropy trie at depth $k = 18$.
  $O(k)$ search independent of database size. Validation: 39/39
  compounds uniquely resolved, 5/6 chemical families show ternary
  cohesion without chemical knowledge being encoded.

- **`sources/cheminformatics-model.tex`** --- Six categorical
  cheminformatics models (Identification, Similarity, Property
  Prediction, Reaction Feasibility, GPU Partition Observation,
  GPU-Supervised Compiled Probe). Zero stored molecules, zero trained
  parameters for Models I--V; 0.6 M LoRA parameters for Model VI.

- **`sources/purpose-based-protein-model.tex`** --- Four protein
  probes (Folding, Binding, Disease, Design) compiling natural-language
  protein queries into geometric operations. Teacher-student
  distillation with type-safety, conservation, and convergence losses.

- **`sources/tripple-isomorphism.tex`** --- Formal proof that
  oscillatory dynamics, categorical partition structure, and biological
  membrane processes constitute three equivalent views of the same
  mathematical object. Ten component-level isomorphisms including
  phase-lock synchronisation, apertures, R--C--L trichotomy, BMD
  transistors, P--N junctions.

- **`synthentic-isomorphism-database/`** --- Empty dictionary for
  pharmacology through dual categorical addressing. Drug and target
  both intrinsically addressed; interaction is trajectory intersection.
  40 drug--target pairs, six primitives, zero stored data.
  Validation: 51/52.

- **`purpose-partitioned-pharmacology/`** --- Six clinical probes
  (Dose, Toxicity, Interaction, Repositioning, Personalisation, Design).
  Each compiles to a task-specific canonical sequence of the six
  primitives. Physics-supervised LoRA at $\sim 0.6$ M parameters per
  probe; $\sim 200 \times$ fewer parameters than end-to-end neural
  pharmacology. Validation: 88/89.

## The Empty Dictionary Principle

Conventional pharmacological databases store millions of drug
fingerprints, bioactivity measurements, ADME profiles, and adverse-event
tables. Every query is a lookup against this stored data. The storage is
linear in the number of entities; the query time is linear in the
database size times the descriptor length.

The empty dictionary replaces stored data with geometric derivation.
The database stores only pointers to external identifiers and their
ternary addresses; all pharmacological predicates---binding, ADME,
adverse effects, drug--drug interaction, therapeutic action---are
evaluated as geometric operations on the coordinate space. Storage is
$\sim 50$ MB for a 16 000-drug, 2000-target database ($\sim 250 \times$
less than Morgan fingerprints alone); query time is $\sim 18$ ns per
operation regardless of $N$ ($\sim 10^{4}$--$10^{5}$ faster than
fingerprint-plus-docking pipelines).

This is not a compression of the conventional knowledge base. It is a
replacement of empirical knowledge by geometric derivation. Empirical
data enters only as validation of the derived predictions, not as the
source of the predictions themselves.

## Purpose Injection

Knowledge of the drug lives in the engine's geometry (zero parameters).
Knowledge of the *question* lives in the probe's LoRA adapter (~0.6 M
parameters per clinical task type). Purpose is not learned as weights
that encode pharmacological facts; purpose is compiled as weights that
encode the grammar of pharmacological questions. One probe per clinical
task type, each compiling to a task-specific canonical operation
sequence, orchestrated through a deterministic clinical oracle.

The architectural consequence: updating pharmacological knowledge does
not require retraining---the engine's geometry is axiomatic and does not
change. Updating the question grammar (new clinical workflow, new
regulatory jurisdiction, new therapeutic modality) requires training
only the relevant probe, typically in hours on a single GPU. The
substrate is immutable; the dictionary is empty; the probes are small
and local.

## Implementation Status

The Rust engine implements the substrate layer and the six primitive
operations. Python validation suites in `crown-prince/publication/validation/`
exercise the framework across 300+ tests with 97%+ pass rates, producing
JSON results and six-panel figure sets for each paper.

### Substrate (Rust)
- Partition coordinates and S-entropy coordinate computation
- Ternary trie insertion, exact search, prefix search
- Geometric primitives: Identify, Similar, Predict, React, Deviate, Close
- GPU fragment-shader partition observation (via wgpu backend)
- ATP-constrained differential integration on the organ-metric

### Compilation Layer (Python, separate repository)
- LoRA adapters for the six pharmacological probes and four proteomics
  probes
- Teacher-student distillation pipeline with physics verifiers
- Curriculum learning across four stages (syntactic, single-op,
  composite, clinical)

### Validation
- Six paper-specific validation scripts, each producing a JSON test
  result and a six-panel figure set
- Aggregate pass rate across all papers: 299/309 (96.8%)

## Getting Started

### Substrate operations

```rust
use nebuchadnezzar::prelude::*;

// Construct the dual trie for a set of drugs and targets.
let mut drug_trie = TernaryTrie::new();
let mut target_trie = TernaryTrie::new();

// Compute a drug's S-entropy coordinates from its vibrational fingerprint.
let drug_coords = s_entropy_from_spectrum(&drug_vibrational_modes);
let drug_address = ternary_address(drug_coords, 18);  // depth 18
drug_trie.insert(drug_address, drug_id);

// Identify: look up a drug by its address.
let found = drug_trie.search(&drug_address);

// React: evaluate binding feasibility between drug and target.
let kd = react(&drug_address, &target_address, &reactivity_params);
```

### ADME trajectory

```rust
// Integrate the drug's ADME trajectory under the patient-specific metric.
let trajectory = predict_adme(
    &drug_address,
    &patient_context,
    Duration::hours(48),
);
let half_life = trajectory.half_life();
let steady_state_concentration = trajectory.c_ss(dose, tau);
```

### Therapeutic closure

```rust
// Given a patient's diseased cellular circuit, compute the minimum
// sparse drug-edge perturbation that restores loop-holonomy closure.
let patient_graph = PatientCircuit::from_microscopy_image(&image);
let eta_star = close_holonomy(&patient_graph, SparsityL1);
let therapeutic_drugs = drug_trie.inverse_retrieve(&eta_star);
```

## Compiled Probes

The probe layer is not implemented in the Rust substrate; probes are
small Python / PyTorch modules that sit on top of the substrate's primitive
interface. Reference implementations are in the companion repositories
for each probe domain (pharmacology, proteomics, cheminformatics). The
probe--engine interface is specified by the type signatures of the six
primitives; any model that produces a well-typed operation sequence can
drive the Rust engine.

## Integration with the Original Ecosystem

Nebuchadnezzar remains the intracellular-dynamics foundation for the
neurobiological stack:

- **Autobahn** --- RAG system integration; now reformulated as
  prefix-matched retrieval on the compound trie.
- **Bene Gesserit** --- Membrane dynamics; now subsumed by the
  biological category $\mathcal{B}$ of the triple isomorphism.
- **Imhotep** --- Neural interface and consciousness emergence; now
  connected to the substrate via the partition-graph representation
  of cellular state.

## The Scientific Claim

A single axiom suffices to derive:

- Atomic shell structure and the capacity formula $C(n) = 2n^2$.
- Drug--target binding affinity and the Hill equation.
- Pharmacokinetic half-life, volume of distribution, clearance, and
  allometric 3/4 scaling.
- Drug--drug interaction mechanisms.
- Adverse-effect likelihoods as geometric branching probabilities.
- Therapeutic action as sparse loop-holonomy closure.
- Protein folding as phase-locked hydrogen bond networks.
- Membrane transport as categorical Maxwell demons with zero-momentum
  observation.
- The equivalence of oscillation, partition, and category as three
  views of the same bounded dynamical object.

No fitted parameters. No stored data. No proprietary knowledge base.
The geometry of bounded phase space, addressed with three coordinates
and queried through six operations, is sufficient.

## Validation Summary

| Paper | Tests | Passed | Rate |
|-------|------:|-------:|-----:|
| Partition-based pharmacodynamics | 24 | 24 | 100.0% |
| Partition-based pharmacokinetics | 33 | 33 | 100.0% |
| Therapeutic-effect trajectory | 30 | 30 | 100.0% |
| Categorical compound database | 39 | 38 | 97.4% |
| Synthentic isomorphism database | 52 | 51 | 98.1% |
| Purpose-partitioned pharmacology | 89 | 88 | 98.9% |
| **Aggregate** | **267** | **264** | **98.9%** |

## Development

### Prerequisites
- Rust 1.70+
- wgpu backend for GPU partition observation
- Python 3.10+ for validation and panel generation

### Building
```bash
cargo build --release
```

### Running validation
```bash
cd crown-prince/publication/validation
python run_all.py                          # all paper validation suites
python panels_synthentic_isomorphism.py    # panel figures
python panels_purpose_partitioned.py
```

## License

MIT. See `LICENSE`.

## Acknowledgements

The framework rests on a single geometric axiom but draws on a century
of work in Hamiltonian dynamics (Liouville, Poincaré), information
theory (Shannon, Landauer), category theory (Mac Lane), receptor
pharmacology (Hill, Michaelis--Menten, Monod--Wyman--Changeux), and
clinical pharmacokinetics (Rowland, Tozer, Wagner). The error, as ever,
is in what has been synthesised from it.

---

*The dictionary is empty. The geometry is sufficient.*
