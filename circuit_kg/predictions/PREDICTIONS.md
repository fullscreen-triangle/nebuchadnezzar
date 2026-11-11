# Predictions, written before the analyses were run

Ideas taken from the chromatography papers in `maxwell` (branch
`reorganise-and-ideal-gas-derivation`): *Chromatography as a Contact Ladder*,
*Chromatography as Two-Velocity Transport*, and *The Ideal Gas Laws from a
Bounded Phase Space*. Each prediction states what would count against it. This
file is committed before the code that tests it is run; results go in
`results/`, never back into this file.

Mapping used throughout: a metabolite is the item, a stay bound to a protein is
a contact, free diffusion is transit, the cytosol is the carrier, and the medium
(pH, ionic strength, pMg, temperature, accessible volume) is the eluent.

## M1 — the medium transfers (analogue of the paper's P1, "transit is free")

On TECRDB reactions measured at three or more distinct medium conditions, hold
out one condition at a time and predict the measured Delta_r G'deg there:

- **medium**: one chemical-reference value per reaction, fitted on the other
  conditions, plus the held-out row's Legendre transform;
- **constant**: the mean training Delta_r G'deg, ignoring the medium (one number);
- **affine-pH**: a + b·pH fitted on the training rows (two numbers).

Prediction: the medium model's mean absolute error approaches the replicate
noise (the same reaction under the same conditions, measured by different
studies), and is below the constant model's. Where the held-out transform
differs from the training mean by more than 1 kJ/mol, the advantage over the
constant model is largest. In extrapolation (the highest or lowest pH held out),
the medium model is no worse than affine-pH.

Against: medium-model error above twice the replicate noise; or no advantage over
the constant model where the medium moved by more than 1 kJ/mol; or affine-pH
better than the medium model in extrapolation.

## M2 — the concentration window, not the medium, blocks reaction direction

For every Reactome reaction with a chemical Delta_r G deg (measured, Hess-derived
or carrier/slot), transform it over the cytosolic medium window. Prediction: more
than the current 3 reactions get a direction. For most of the rest, the sign of
Delta_r G'm (all participants at 1 mM) is fixed across the medium window, so the
remaining blocker is the concentration window and not the medium.

Against: for most reactions with a value, the medium window alone admits both signs
of Delta_r G'm.

## M3 — crowding is a small medium number

An accessible-volume fraction a in [0.7, 1.0] shifts the affinity of reactions that
change the number of solute particles by at most RT ln(1/0.7) per particle, about
0.9 kJ/mol. Prediction: including it changes at most a few direction verdicts.
The dilute-solution error bound 2·n·v from the ideal-gas paper stays below 10^-2
per particle for every metabolite at 10 mM.

Against: crowding changes more than 5% of the decided verdicts, or the dilute
bound exceeds 10^-2 for any metabolite in the window.

## P1 — whole-protein descriptors do not carry reaction-specific contacts (analogue of the paper's failed P4)

The paper found that tabulated stationary-phase descriptors, measured with neutral
test solutes, did not transfer column-specific retention. Our AlphaFold summaries
are the same kind of whole-object descriptor. For enzymes with measured K_m
(SABIO-RK, wild type, Homo sapiens), under leave-one-protein-out:

- **baseline**: the substrate's mean log10 K_m over the other proteins;
- **descriptors**: the baseline plus a ridge map from protein descriptors (number of
  residues, mean pLDDT, fraction of residues with no exposed atom, mean contact weight
  per residue, exposed atoms per residue) to the residual.

Prediction: the descriptors do not reduce the held-out error. On per-protein mean
absolute errors, a one-sided Wilcoxon test of descriptors < baseline gives p ≥ 0.05.

Against: p < 0.05 with a lower mean error for the descriptor model.

# Folding ideas from levinthal, tested with AlphaFold

Sources: levinthal `models/publications/folding-partition-calculus` and
`docs/chemistry/ternary-representation` (main @ cf7fc86). The coupling matrix is
built two ways. K_seq is the paper's sequence-only formula, copied from its
validation script. K_AF is the AlphaFold residue contact graph: counted heavy-atom
pairs within 4.5 Å. Secondary-structure elements (SSEs) come from a DSSP-style
backbone hydrogen-bond assignment (helix ≥ 4 residues, strand ≥ 2). That
assignment is first checked against author HELIX/SHEET records of experimental
PDB entries.

Proteins: AlphaFold models in the cache with mean pLDDT ≥ 90 and 50-250 residues,
the first 150 by accession.

## F1 — Protocol 3 (contacts from sequence). Not a prediction: already run

This was run in the scratchpad before this file section was written (see the
session record). Recorded here so that the kernel reproduces it. Result then:
precision at top-L is 0.81x the base rate on 607 models, with a hydrophobicity
product at 2.4x.

## F2 — eigenvalue count equals the number of SSEs (the paper's Thm "Secondary Structure Count")

λ_c = K0·e^(-1/2)·L̄, where L̄ is the mean SSE length and K0 is the mean of K over
pairs inside one SSE with 1 ≤ |i-j| ≤ 4 (the intra-block coupling of the paper's
proof).

Hypothesis (the paper's): the number of eigenvalues of K above λ_c equals M, the
number of SSEs. Tested for K_seq and K_AF.

Against: the exact-match rate is no higher than that of a length-only baseline,
M̂ = round(N × the mean M/N of the other proteins).

## F3 — phase-lock groups are SSEs (the paper's Thm "Phase-Lock Groups Are Secondary Structure")

- **Dynamics:** the paper's equation dφ_i/dt = ω_i + Σ_j K_ij sin(φ_j - φ_i), with
  ω_i = 10·S_k + 1 as in its script. K is scaled to K0 × K / mean(K > 0), with
  K0 swept over {0.5, 1, 2, 5, 10}. Integration: dt = 0.01, 1500 steps.
- **Groups:** two residues are locked if their wrapped phase difference stays below
  0.2 rad over the last quarter of the run. Groups are the connected components of
  locked pairs.
- **Truth:** each SSE is one group, and each contiguous loop stretch is one group.
- **Score:** the adjusted Rand index, best over the K0 sweep.
- **Baselines:** equal contiguous blocks, with the true number of groups as the block
  count (one choice, truth-informed). And static: components of the top-m pairs of
  K, where m is the number of locked pairs in the same run (best over the same sweep).

Hypothesis (the paper's): the phase-lock groups from K_AF beat the blocks baseline,
per protein. One-sided Wilcoxon p < 0.05.

Against: p ≥ 0.05. Also reported: whether the dynamics beat the static K
components (does synchronisation add anything to the coupling itself?), and the
same tests for K_seq.

## F4 — phosphosites sit in weakly synchronised regions, explained by disorder

The 206 named, sequence-checked phosphosites from the attach stage are used, on
proteins with an AlphaFold graph and at most 1,500 residues. Kuramoto dynamics run
on K_AF at K0 = 2 (fixed in advance). The local order of residue i is the time
mean of |Σ_j K_ij e^{i(φ_j - φ_i)}| / Σ_j K_ij.

- **Prediction 1:** named sites have lower local order than other residues of the
  same type in the same proteins (one-sided Mann-Whitney p < 0.05).
- **Prediction 2:** within the pLDDT strata < 70 and ≥ 70, the difference is not
  significant (p ≥ 0.05 in each). In other words, synchronisation adds nothing to
  disorder.

Against: prediction 1 has p ≥ 0.05, or prediction 2 fails, in which case
synchronisation carries information beyond pLDDT.

## F5 — the ternary address on AlphaFold coordinates (measured, not predicted)

Cα coordinates are normalised to each protein's bounding cube and given an
interleaved base-3 address (the ternary paper's mapping theorem: position j mod 3
picks the axis, the digit picks the third). Measured:

- that the distance-preservation bound holds for every pair (it must);
- the fraction of Cα contacts (< 8 Å) whose addresses share the prefix of the
  deepest level whose cells are still at least 8 Å on a side. This is the cost of
  using the prefix as a proximity key, since neighbours split across cell
  boundaries.

# Extensions from the P450 monograph and the dcat-ap-plus-p450 specification

Sources: levinthal `cytochrome/publications` (the 19 manuscripts) and
`dcat-ap-plus-p450/docs/specifications/components-graph.md`. The numbers in those
papers are not carried over. Their structures are implemented here on the kernel's
own public data: Reactome, TECRDB with eQuilibrator transforms, SABIO-RK (all human
wild-type kinetic laws), PaxDb, UniProt (EC, Pfam, families), AlphaFold, and PDB.

## Measured, not predicted (instrumentation)

- **X1 — verdict axes.** Basis composes as max (stated < supplied < derived). Every
  empty or absent verdict gets a gate letter (+ witnessed absent, − not recorded,
  open = a name the source never uses, ? undecided). Every kernel fact gets a ground
  verdict from its provenance chain (symbol / collapsed / index / icon / composite).
  A `subfloor` verdict names the shortfall when a ladder's composite power is below
  its declared target. Results carry a stage digest of their content and
  conventions. Reported: the census of each axis over the kernel's outputs.
- **X2 — modifier stack.** Inhibitor and stimulator edges get additive log-rate
  modifiers from SABIO-RK K_i (factor 1/(1+[I]/K_i) over the concentration window),
  and `absent` where no K_i exists. Each enzyme total that is not conserved is
  classified by what breaks it. Reported: coverage and the classification.
- **X3 — branching.** At every small-molecule branch point, each consuming branch's
  share at the dilute limit is (k_cat/K_m)_i·[E_i] / Σ_j (k_cat/K_m)_j·[E_j]. The
  verdict is answered only when every sibling has k_cat, K_m and abundance;
  otherwise the missing sibling is named. Reported: coverage.

## X4 — the redox ladder (predicted)

Couple potentials differences ΔE'° = −ΔG'°/(nF) come from the Hess span with the
Legendre transform at pH 7, I = 0.25 M, 298.15 K.

- **X4a.** NAD+/NADH and NADP+/NADPH lie within 10 mV of each other. Textbook:
  −320 and −324 mV.
- **X4b.** NADPH reduces glutathione: ΔG'° of NADPH + GSSG -> NADP+ + 2 GSH is
  negative and within 10 kJ/mol of −16 kJ/mol. Textbook: E°' −324 against −240 mV.

Against: either falls outside its band, or the value is not derivable (then
`absent`, not a failure of the prediction, and recorded as such).

## X4c — the Moser–Dutton ruler on a cytosolic relay (predicted, weak by design)

For glutathione reductase (PDB 3GRS) and thioredoxin reductase 1 (PDB 2J3N), the
edge distance from the FAD isoalloxazine ring to the redox-active cysteine sulfur
gives a ruler rate (λ = 0.7 eV, ΔG = 0, packing density 0.76; convention
`moser-dutton-defaults`).

- Prediction: the ruler rate exceeds the enzyme's measured k_cat (SABIO-RK, human
  wild type), so intramolecular electron transfer is not rate-limiting.
- Control: the same calculation at 25 Å must fall below k_cat. If it does not, the
  check cannot fail and is reported as non-discriminating.

## X5 — EC labels and structural classes (predicted)

After colour refinement, 537 junctions sit in shared classes. Junctions get the EC
class (first digit) of their catalyst where one exists.

- Prediction: EC labels are concordant with structure. Real labels split fewer
  shared classes than labels shuffled among the same junctions (200 shuffles,
  one-sided empirical p < 0.05).
- Against: p ≥ 0.05. The labels would then add distinctions no more aligned with
  structure than random ones.

Also reported (measured): for every shared class, the coarsest label — EC level
1-4, Pfam, gene — that splits it, or `underdetermined` naming none.

## X6 — stoichiometric signatures of reaction classes (predicted)

Each reaction gets a class from its stoichiometry alone, by fixed rules:

- oxidoreductase: an NAD(P)(H), FAD or glutathione/thioredoxin couple changes state;
- hydrolase: water consumed and the number of small-molecule products exceeds the
  number of reactants excluding water;
- ligase: a nucleoside triphosphate is split to NDP+Pi or NMP+PPi alongside a bond
  formation;
- transferase: a group donor/acceptor pair (ATP -> ADP with a phosphorylated
  product, or SAM -> SAH, or CoA esters);
- isomerase: one small molecule in, one out, with identical formula;
- lyase: one small molecule in, two out, no water.

When several rules match, the verdict is `underdetermined`.

- Prediction: on reactions whose catalyst has a single EC class, the signature
  class agrees with that EC class in more than 60% of classifiable reactions, and
  above a shuffle null (EC labels permuted, 200 times, one-sided p < 0.05).
- Against: agreement at or below 60%, or p ≥ 0.05.
