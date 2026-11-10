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
