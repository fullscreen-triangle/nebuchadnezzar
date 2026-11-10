# Outcomes of the predictions in PREDICTIONS.md

Numbers are from `python -m circuit_kg medium` (writes `results/medium.json`).
PREDICTIONS.md was committed (de6f210) before any of this code ran.

## Check before M1: the transform itself

The per-compound Legendre transform is written from equilibrator_cache 0.8.1 and
evaluated on the local eQuilibrator compound cache. It reproduces syndrome's
`transform` column on all 3,276 rows in the window of the Thermodynamics of
Enzyme-Catalyzed Reactions Database (TECRDB), with a largest difference of
6.8e-13 kJ/mol. Controls: shifting pH by 0.1 leaves 66 rows within
1e-6 kJ/mol, and dropping the Debye-Hueckel term leaves 59.

## M1 — the medium transfers: supported

The replicate noise (same reaction, same conditions, different studies; 171
pairs) is 1.38 kJ/mol mean and 0.71 median.

| design | reactions | medium | constant | affine-pH |
|---|---|---|---|---|
| interpolation | 137 | 1.56 | 2.11 | 1.40 |
| extrapolation | 154 | 2.59 | 3.32 | 4.37 |

Each entry is the mean absolute error in kJ/mol.

- **Medium vs constant.** Paired per reaction, the medium model beats the constant
  model in 84 of 137 interpolated reactions (Wilcoxon p = 8e-4) and in 95 of 154
  extrapolated ones (p = 5e-6).
- **Where the medium moved.** Where the held-out transform moved by more than
  1 kJ/mol, the interpolation errors are 1.95 (medium) against 2.94 (constant).
  Where it moved less, they are equal: 1.08 against 1.08.
- **Against replicate noise.** The medium model's mean error is 1.13 times the
  replicate noise in interpolation and 1.88 times in extrapolation, below the
  stated 2x limit.
- **Medium vs affine-pH.** In extrapolation the medium model is no worse than
  affine-pH: 82 against 72 reactions, p = 0.27. Its mean error is lower, because
  the straight line sometimes runs away, but its median error is higher (1.57
  against 1.44).
- **Not predicted.** In interpolation, affine-pH (two numbers per reaction) beats
  the medium model (one number): 77 against 58 reactions, p = 0.009.

## M2 — direction: the count fails; the blocker attribution holds

Of 201 reactions with a chemical value (75 measured, 30 Hess, 96 carrier/slot):

- answered forward: 2 (pyrophosphate hydrolysis by two cytosolic pyrophosphatases), against 3 before;
- undecidable because of the concentration window: 156;
- undecidable because of the medium window: 43.

The prediction "more than 3 answered" fails. The old 3 used the measured
Delta_r G'deg study means in the pH window, with no uncertainty and no ionic
strength or pMg range. Adenosine monophosphate hydrolysis by a nucleotidase rested on one study. Here, adding
q·sigma and the I and pMg windows removes it. The two survivors clear zero by only
0.28 kJ/mol.

The second part holds: for 156 of 199 undecided reactions (78%), Delta_r G'm at
1 mM has a fixed sign across the medium window, so only concentrations block
them. This includes all 96 carrier/slot reactions.

## M3 — crowding and dilution: fails on both counts

- **Crowding.** It changes 2 of the 2 decided verdicts, which is above the 5% limit.
  Both PPi hydrolyses add a solute particle (one pyrophosphate and one water give two phosphates). Their 0.28 kJ/mol
  margin is below the 0.92 kJ/mol crowding shift at a = 0.7. With crowding in the
  medium, no reaction in the kernel has a decided direction.
- **Dilution.** The bound 2·n·v at 10 mM has a median of 0.0035 per particle, but
  exceeds 1e-2 for large compounds. The worst is C05893 (1,873 Da) at 0.027. The
  ideal-solution form of mu is safe for ordinary metabolites at 10 mM, but not for
  molecules this size at the top of the concentration window.

## P1 — whole-protein descriptors do not carry K_m: supported

Numbers are from `python -m circuit_kg kinetics` (writes `results/kinetics.json`).
SABIO-RK (export API, human, wild type, K_m): 7,458 entries give 5,682 substrate
K_m records and 3,559 (protein, Kyoto Encyclopedia of Genes and Genomes (KEGG) compound) pairs. Of these, 272 proteins with
AlphaFold descriptors and 1,228 pairs entered the leave-one-protein-out test.

| model | mean abs error, log10 K_m |
|---|---|
| substrate baseline | 0.775 |
| baseline + 5 protein descriptors (ridge) | 0.780 |

The descriptor model is better for 105 proteins and worse for 127. The one-sided
Wilcoxon test gives p = 0.94, so the prediction holds.

Positive control: adding a synthetic effect of 0.5 standard deviations of log10(residues) to
log10 K_m is detected (p = 0.007), so the test can find a protein-level effect
that the descriptors carry.

## Not predicted, measured: retention and locality

These analyses are computed, not tested; none of them was written down in advance.

- **Retention.** 370 of the 1,631 catalysed kernel reactions have a measured K_m
  for a substrate (596 contacts). Of 168 metabolites with a K_m, 161 have a
  PaxDb abundance for every binding enzyme. Their median retention factor
  k = sum [E]/K_m is 0.001-0.003 under the protein-ppm-to-molar convention, so
  metabolites are almost entirely free. Their capacitance is barely raised by
  binding. Only 8 exceed k = 1 at the upper total: 5-PP-IP5 (9-26),
  S-adenosylmethionine, F1,6BP, reduced nicotinamide adenine
  dinucleotide phosphate and four others.
- **Locality.** The enzymes are not dilute. For 223 enzymes, the concentration at
  which they leave the dilute (local) regime, c* = 0.1 / sum(1/K_m), has median
  2.4 uM; 76 leave it below 1 uM. None stays dilute across the whole 1 uM-10 mM
  window, and 221 depart by more than 10% at 1 mM. 106 enzymes have two or more
  measured substrates sharing the same sites.
- **In chromatographic terms.** The cytosolic column is overloaded: analyte far
  exceeds sites. For the metabolite the stationary phase is negligible (k << 1),
  but for the enzyme the contacts are non-local (Langmuir, not linear). The
  ladder paper's condition for dropping the carrier fails on the enzyme side.
  The enzyme-substrate nodes cannot be folded into the free species'
  capacitance; they have to stay explicit with conserved enzyme totals, as in
  the enzyme expansion.

## Not predicted, measured: completion and separation

These numbers are from `python -m circuit_kg complete` (writes `results/complete.json`).

**Completion.** The question is whether the kernel turns source S into target P
using only S, the medium (14 ambient species) and the 2,241 proteins that no
reaction makes. The survey covers 540 small-molecule sources and 779 targets, so
420,120 pairs.

| directions allowed | one completion | several | none (certified empty) | made without S |
|---|---|---|---|---|
| as annotated by Reactome | 659 | 634 | 362,739 | 56,088 |
| both, unless thermodynamics decides (2 fixed) | 352 | 1,464 | 258,690 | 159,614 |
| both, with the sign of Delta_r G'm at 1 mM (158 fixed) | 390 | 1,510 | 276,937 | 141,283 |

- **Reading the outcomes.** A unique completion is an order-determined ladder in
  the ladder paper's sense. Several completions means order-determination fails.
- **Most pairs have no completion.** Most reactions need a second small molecule
  that a single source cannot supply.
- **Reversibility.** Allowing reactions to run backwards halves the unique
  completions and lets the medium and proteins make many more targets on their
  own.
- **Measured directions.** Fixing the 156 directions that measured concentrations
  near 1 mM would decide adds 38 unique completions (+11%) and removes 18,331
  targets from "made without S".
- **Limitation.** A completion may not pass through its own target. Alternatives
  that pass through other cycles among intermediates still count as "several", so
  that column is an upper count.

**Separation.** Refinement leaves 6,792 classes over 8,244 items, with 1,988 items
in shared classes.

- Locating an item within its class needs 0.65 binary distinctions on average,
  and at most 7.1 (the largest class). This information is absent from the graph's
  structure, as with two analytes of equal retention, which no column length
  separates.
- With 63 values of the initial cut key, at least log K / log n_c = 2.2
  independent keys are needed to give 8,244 items distinct positions. Refinement
  rounds supply them through neighbours, but cannot split a class whose members
  are structurally identical.

# Folding ideas from levinthal, tested with AlphaFold

Numbers are from `python -m circuit_kg folding` (writes `results/folding.json`, 15 min).
The predictions were committed in d77d09c, before `folding.py` was written.

## Check before F2-F4: the secondary-structure assignment

The assignment is DSSP-style, from backbone hydrogen bonds. Against the authors'
helix and sheet records in four experimental structures, per-residue agreement is 0.87
(1CRN), 0.83 (1UBQ), 0.87 (2LZM) and 0.91 (3CHY). Shuffling the residues drops it
to 0.24-0.42. The assignment gives fewer strand residues than the author records.

## F1 — contacts from sequence: reproduced

On 607 AlphaFold models (pLDDT ≥ 90, 50-400 residues), the paper's coupling K_seq
reaches top-L precision 0.81 times the base rate. That is worse than random
(two-sided Wilcoxon p = 7e-11). The hydrophobicity product reaches 2.36 times. No
protein meets the paper's criterion (precision ≥ 0.7, recall ≥ 0.5) with any score.
On the experimental crambin structure 1CRN, K_seq has precision 0.11 against a
base rate of 0.07.

## F2 — eigenvalue count equals the number of SSEs: fails

On 149 proteins, the median number of secondary-structure elements (SSEs) is 12.

| count | exact matches | mean absolute error |
|---|---|---|
| eigenvalues of K_seq above λ_c | 0 | 7.1 |
| eigenvalues of K_AF above λ_c | 3 | 9.1 |
| length-only baseline | 23 | 2.2 |

Both couplings give too many eigenvalues above λ_c (medians 18 and 20). The
excess holds even for K_AF, which is built from the real structure.

## F3 — phase-lock groups are SSEs: fails

Agreement with the SSE/loop segmentation is measured by the adjusted Rand index,
taking the best value over the K0 sweep.

- **Against blocks.** Phase-lock groups from K_AF score 0.067. Equal contiguous
  blocks score 0.486 and are better for 148 of 149 proteins (p = 1.0 for the
  hypothesis).
- **Sequence coupling.** K_seq phase-lock groups score 0.001.
- **Dynamics against the coupling itself.** The dynamics do not beat the static
  coupling: the static components of the top K_AF pairs score 0.057, and the
  dynamics win 80 proteins and lose 66 (p = 0.14). Synchronisation adds nothing
  significant to the contact graph it runs on.

## F4 — phosphosites are less synchronised, and disorder accounts for it: both parts hold, the second weakly

Kuramoto runs on K_AF at K0 = 2. The comparison set is 158 named sites against
3,870 same-type residues in the same proteins.

- **Prediction 1 holds.** Sites have lower local order: median 0.912 against
  0.936, one-sided Mann-Whitney p = 0.012. They also have lower pLDDT: 53.7 against
  57.9, p = 0.043.
- **Prediction 2 holds.** Within pLDDT strata the difference is not significant:
  p = 0.39 below 70 (106 sites) and p = 0.13 at 70 or above (52 sites).
- **Caveat on prediction 2.** It rests on an absence of significance. The ≥ 70
  stratum is small, and its medians still point the same way (0.933 against
  0.953). A small residual effect is not excluded.

## F5 — ternary addresses (measured)

- **The bound.** On 150 AlphaFold models, the distance-preservation bound holds for
  all 2,480,054 Cα pairs, with no violations.
- **The proximity key.** The deepest level whose cells are at least 8 Å on a side
  is one full triple of digits (median), i.e. 27 cells. Only 54.6% of the 122,559
  Cα contacts share that prefix. As a proximity key the address is one-sided:
  sharing the prefix bounds the distance, but 45% of contacting pairs fall either
  side of a cell boundary.

## What this means for the kernel

AlphaFold's counted contact graph stays the protein representation. On every test
here, the oscillator layer added nothing to it: not secondary structure, not
eigen-counts, and not phosphosite placement beyond what pLDDT already says. The
sequence-only coupling carries less contact information than a hydrophobicity
product. The ternary address is usable as a hierarchical key only together with
its neighbouring cells.
