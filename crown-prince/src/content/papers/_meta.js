// Manifest of all papers. Populated entries have content; the rest are stubs.

export const PAPERS = [
  {
    id: "pharmacodynamics",
    n: "01",
    title: "Pharmacodynamics from the Bounded Phase Space Law",
    short: "Pharmacodynamics",
    kicker: "PD",
    abstract:
      "Drug–target interaction, dose–response, and selectivity derived inline from a single geometric axiom. K_d = exp(−ΔM·ln b); five regime-specific dose–response curves of which the Hill equation is the aperture limit; zero-work categorical selectivity.",
    tests: "24 / 24",
    status: "stub",
    primitives: ["React", "Predict"],
  },
  {
    id: "pharmacokinetics",
    n: "02",
    title: "Pharmacokinetics from the Bounded Phase Space Law",
    short: "Pharmacokinetics",
    kicker: "PK",
    abstract:
      "ADME from a partition graph. Bioavailability F = F_abs(1−E_H)(1−E_G), volume V_d = V_p + Σ V_t K_p, half-life t_{1/2} = ln 2 · V_d/CL, allometric 3/4 scaling, well-stirred liver model — all corollaries of the axiom rather than postulates.",
    tests: "33 / 33",
    status: "stub",
    primitives: ["Predict"],
  },
  {
    id: "therapeutic-effect",
    n: "03",
    title: "Therapeutic-Effect Trajectory Mechanism",
    short: "Therapeutic Effect",
    kicker: "TX",
    abstract:
      "Disease as non-trivial loop holonomy on the cellular partition graph; therapy as sparse ℓ_1 edge perturbation. GPU five-pass ray-march pipeline at 43 ms per cell via the Triple Observation Identity μ_abs ∝ 1/(τ d_S) ∝ G·RT.",
    tests: "30 / 30",
    status: "stub",
    primitives: ["Close", "Predict"],
  },
  {
    id: "categorical-compound-database",
    n: "04",
    title: "Categorical Compound Database",
    short: "Compound Database",
    kicker: "CCD",
    abstract:
      "Every stable molecule addressed in a ternary trie at depth k = 18. O(k) search independent of database size; ternary similarity as shared-prefix length; 39/39 NIST compounds uniquely resolved; 5/6 chemical families show ternary cohesion without chemical knowledge being encoded.",
    tests: "39 / 39",
    status: "stub",
    primitives: ["Identify", "Similar"],
  },
  {
    id: "cheminformatics-models",
    n: "05",
    title: "Categorical Cheminformatics Models",
    short: "Cheminformatics Models",
    kicker: "CM",
    abstract:
      "Six categorical models (Identification, Similarity, Property Prediction, Reaction Feasibility, GPU Partition Observation, GPU-Supervised Compiled Probe). Models I–V have zero parameters; Model VI is a 0.6 M LoRA adapter trained on physical observables.",
    tests: "—",
    status: "stub",
    primitives: ["Identify", "Similar", "Predict", "React"],
  },
  {
    id: "purpose-based-protein-model",
    n: "06",
    title: "Purpose-Based Protein Models",
    short: "Protein Models",
    kicker: "PPM",
    abstract:
      "Four protein probes — Folding, Binding, Disease, Design — compiling natural-language protein queries into geometric operations on S-entropy space. Teacher-student distillation with type-safety, conservation, and convergence losses.",
    tests: "—",
    status: "stub",
    primitives: ["Identify", "Predict", "React"],
  },
  {
    id: "tripartite-isomorphism",
    n: "07",
    title: "Tripartite Isomorphism Architecture of Membrane Processes",
    short: "Triple Isomorphism",
    kicker: "TRI",
    abstract:
      "Formal proof that oscillatory dynamics, categorical partition structure, and biological membrane processes constitute three equivalent views of the same mathematical object via explicit functors. Ten component-level isomorphisms.",
    tests: "—",
    status: "stub",
    primitives: ["—"],
  },
  {
    id: "synthentic-isomorphism-database",
    n: "08",
    title: "The Synthentic Isomorphism Database",
    short: "Synthentic Isomorphism",
    kicker: "SID",
    abstract:
      "An empty dictionary for pharmacology through dual categorical addressing. Drug and target both intrinsically addressed; interaction is trajectory intersection. Storage 50 MB vs DrugBank 12 GB; query O(k) independent of N. Six primitives, zero stored data.",
    tests: "51 / 52",
    status: "live",
    primitives: ["Identify", "Similar", "Predict", "React", "Deviate", "Close"],
  },
  {
    id: "purpose-partitioned-pharmacology",
    n: "09",
    title: "Purpose-Partitioned Pharmacology",
    short: "Purpose-Partitioned",
    kicker: "PPP",
    abstract:
      "Six clinical probes (Dose, Toxicity, Interaction, Repositioning, Personalisation, Design). Each compiles to a task-specific canonical sequence of the six primitives. ~0.6 M LoRA parameters per probe; ~200× fewer than end-to-end neural pharmacology.",
    tests: "88 / 89",
    status: "stub",
    primitives: ["Identify", "Predict", "React", "Deviate", "Close"],
  },
];

export function paperById(id) {
  return PAPERS.find((p) => p.id === id);
}
