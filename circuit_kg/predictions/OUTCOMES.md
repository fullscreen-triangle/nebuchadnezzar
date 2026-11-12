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

# Extensions from the P450 monograph and the dcat-ap-plus-p450 specification

Numbers are from `python -m circuit_kg extend` (writes `results/extend.json`). The
predictions were committed in cddf153, before the code. Data: SABIO-RK, all human
wild-type kinetic laws (5,682 K_m, 3,089 k_cat and 771 K_i records); UniProt
annotations for 3,207 proteins, 1,481 of them with an Enzyme Commission (EC) number; PaxDb abundance for
3,048 proteins.

## X1 — verdict axes (measured)

- **Gate letters.** Every empty or absent verdict from the live producers now carries
  one: thermo match, Hess, carrier/slot, direction, modifiers, dividers, variants and
  the redox ladder. Of 1,681 such verdicts, 686 are not recorded (`-`), 605 are a name
  the source never uses (`open`) and 390 are witnessed absent (`+`). None is undecided.
- **Ground verdicts by fact family.**
  - Index: TECRDB rows, SABIO-RK parameters, PaxDb and Protein Data Bank (PDB) coordinates.
  - Composite: every ΔG the kernel pools or derives, direction verdicts, retention
    factors and PDB distances.
  - Icon: AlphaFold models and the contact graphs and enzyme cycles built on them.
  - Collapsed: Reactome stoichiometry, UniProt EC and eQuilibrator microspecies, which
    are curation with no contact recorded.
  - Symbol: the potential μ, which is never stored.
- **Basis of direction verdicts.** 75 are supplied (a measured ΔG under the supplied
  medium window) and 126 derived. None can be reported as stated.
- **Stage digest.** It is current under its own conventions. The control, with the pH
  window changed, makes it stale, so the check is discriminating.
- **Subfloor.** The `subfloor` verdict reproduces the specification's example: a
  composite of 0.9475 against a target of 0.95 gives a shortfall of 0.0025. On real data
  no ladder chain has all its rung conductances (0 of 17), so it never fires there.

## X2 — modifier stack (measured)

- **Inhibitor edges (175).** 147 are macromolecular binding partners. They are refused,
  since no K_i applies to them. 26 small molecules have no K_i in SABIO-RK for their
  enzyme (absent, `-`), and 2 have no KEGG compound (`open`).
- **Stimulator edges (105).** 52 are macromolecular. The 53 small molecules have no
  activation constant in SABIO-RK. No modifier edge is answered.
- **Broken enzyme totals.** The stoichiometry does not conserve 434 enzyme totals. The
  reactions that make or consume those enzymes are:
  - assembly or dissociation (266);
  - covalent modification (140);
  - degradation (124);
  - other (103).

  An enzyme can have several kinds. Where only one kind applies, the counts are 91, 52,
  49 and 45. Induction and inactivation would act on these boundary currents.
- **Disease variants.** Reactome has 58 loss-of-function variant reactions. 23 attach
  to one wild-type edge as a modifier, 19 match several (underdetermined) and 16 match
  none.

## X3 — current dividers (measured)

- **Coverage.** Of 168 healthy-cell branch points (variant siblings removed), 3 are
  answered. The others lack k_cat/K_m for a sibling (487 sibling gaps) or have an
  uncatalysed sibling (135). On average 7.6% of a branch point's siblings are complete.
- **The three answered branch points:**
  - fructose 1,6-bisphosphate: fructose-1,6-bisphosphatase 0.67, aldolase 0.33;
  - galactose: galactokinase 0.98, aldose reductase 0.02;
  - glucose: aldose reductase 0.52, hexokinases 0.48.
- **The glucose share is not credible.** At millimolar glucose the hexokinases are
  saturated (K_m about 0.1 mM), so the dilute limit does not hold. The abundances are
  also whole-organism figures. The failed share repeats the locality failure from the
  kinetics stage.

## X4 — redox ladder: X4a and X4b hold, X4c is absent

The couples include nicotinamide adenine dinucleotide phosphate (NADP), as NADP+ and
reduced NADP (NADPH). They include nicotinamide adenine dinucleotide (NAD), as NAD+ and
reduced NAD (NADH). They also include glutathione disulfide (GSSG), reduced glutathione
(GSH) and flavin adenine dinucleotide (FAD). Potentials are relative to the NADP couple. They come
from the Hess span and the Legendre transform at pH 7, I = 0.25 M and 298.15 K. The
textbook differences are approximate.

| couple | E − E_NADP (mV) | sigma (mV) | textbook difference (mV) |
|---|---|---|---|
| NAD+/NADH | +1.6 | 7.5 | +4 |
| GSSG/GSH | +71.7 | 12.7 | about +84 |
| lipoamide | +25.1 | 29.1 | about +34 |
| thioredoxin | +57.9 | 28.2 | about +54 |
| pyruvate/lactate | +138.7 | 10.6 | +139 |
| oxaloacetate/malate | +156.6 | 9.1 | +158 |

- **X4a holds.** The NAD and NADP couples lie 1.6 mV apart.
- **X4b holds.** NADPH + GSSG → NADP+ + 2 GSH has ΔG'° = −13.8 kJ/mol, inside the
  predicted −16 ± 10.
- **Not derivable.** Cystine and dehydroascorbate were never measured in TECRDB
  (`open`). FAD is not a combination of measured reactions (`-`).
- **X4c is absent, not tested.** The flavin-to-cysteine edge distance is 3.42 Å in
  glutathione reductase (3GRS) and 3.33 Å in thioredoxin reductase 1 (2J3N). Both
  distances give a ruler rate of 6.8e10 per second. The control at 25 Å gives 0.014 per second, so the
  check would discriminate, but SABIO-RK has no human wild-type k_cat for either enzyme.

## X5 — EC labels are concordant with structural classes: holds

An EC number truncated to its first level is written EC level 1 (EC1), and so on to
EC level 4 (EC4).

- **Concordance.** Real EC1 labels split 4 of the 147 shared junction classes (210
  labelled junctions). Shuffled labels split 53.7 on average, with a minimum of 41
  (empirical p = 0.005). At EC level 3 the counts are 4 against 61.8.
- **Coarsest splitting label.** For 100 classes no label splits them
  (underdetermined). For the rest it is the gene (18), EC1 (11), Pfam (8), EC4 (8), EC2
  (1) or EC3 (1). 32 classes are separated completely.
- **Bits left per item.** 0.649 with structure alone, then 0.641 with EC1, 0.629 with
  EC4, 0.627 with Pfam and 0.619 with the gene.

## X6 — stoichiometric signatures match EC classes: holds

- **Verdicts.** 798 answered, 42 underdetermined (mostly lyase+transferase, 28), 653
  absent and 1,421 refused (macromolecular events).
- **Agreement.** On 634 reactions with a single-EC catalyst the signature agrees in
  593 (93.5%), against 32.5% for shuffled EC labels (empirical p = 0.005).
- **Largest confusions.** Lyase against EC oxidoreductase (8) and hydrolase against EC
  transferase (8).


# Databases (Rhea, MetaNetX, localisation, UniProt features, M-CSA, PDBe, DisProt, Human-GEM)

`python -m circuit_kg databases` -> `results/databases.json` (about 12 min, most of it
the experimental-structure downloads). Five predictions hold and three fail.

| test | outcome | number |
|---|---|---|
| R1 curated direction against the 1 mM sign | fails | 45 of 62 agree (72.6%), predicted above 80% |
| X1 MetaNetX closes identifier misses | fails | 34 of 197 unblocked (17%), predicted a quarter |
| L1 catalysts have cytosol support | holds | 1,947 of 2,047 proteins with data (95.1%) |
| S1 slots sit on curated phospho-residues | holds | 168 of 206 (81.6%); one offset explains 5 of 9 mismatches |
| C1 catalytic residues are contact hubs | holds | median 55 against 43 contacts, p = 2.7e-18; control p = 0.46 |
| P1 AlphaFold contacts match experiment | holds | median Jaccard 0.90 on 60 chains; shift-by-3 control 0.14 |
| D1 disorder explains the phosphosite effect | fails | outside DisProt disorder the sites still sit lower, p = 0.0012 |
| G1 Human-GEM coefficients agree | holds | 324 of 324; only 23 of them have a coefficient other than 1 |

## R1 — fails, and the failures are uphill reactions

The disagreements include triose-phosphate isomerase, glyceraldehyde-3-phosphate
dehydrogenase, lactate oxidation by lactate dehydrogenase, ethanol oxidation, creatine
kinase forming phosphocreatine and the thiolase condensation of two acetyl-CoA. All of
them run uphill at 1 mM and run in the cell because product is drawn away. The 1 mM sign is therefore not
a direction; the concentration ratio is. These 17 reactions are where measured concentration windows
(`measured-window`) would decide the direction. Coverage: 460 kernel reactions map to
Rhea (403 left-to-right, 69 right-to-left), and the participant sets agree in 421 of 459.

## X1 — fails, and the blocker moves

MetaNetX gives 36 more species a KEGG compound and unblocks 34 of the 197 reactions.
One of them gains a TECRDB match; 33 are now blocked by "no TECRDB measurement of this
reaction". The Hess-derived count stays at 30. The gate letter of those 33 changes from
`open` (a name the source never uses) to `-` (not recorded). The limit is measurement,
not identifiers.

## L1 — holds

Of 2,134 catalyst proteins, 1,947 have cytosol support in the Human Protein Atlas or in
COMPARTMENTS (confidence at least 3), 100 have none and 87 have no data. By reaction:
1,404 have all catalysts supported, 79 some and 78 none. The unsupported list includes
the three fibrinogen chains (P02671, P02675, P02679), which are secreted. These 78
reactions are candidates for a scope error in the cytosol filter.

## S1 — holds

168 of the 206 located phosphosites coincide with a UniProt phospho modified residue;
141 of those carry experimental evidence in the Evidence and Conclusion Ontology (ECO),
codes ECO:0000269 or ECO:0007744. One offset per protein explains 5 of the 9 residue
mismatches: +39 in Q04637 (3 sites), +2 in Q14155 and −48 in tau (P10636, an isoform
numbering). A first run reported 19%, because the parser cut each note at its first
semicolon. Notes name the kinase after one, as the mechanistic target of rapamycin
(MTOR) in "Phosphoserine; by MTOR". The fixed parser has a regression test, and the
cache now keeps UniProt's raw text.

UniProt also names 31 phosphorylated enzyme intermediates among the catalysts
(phosphohistidine, phosphocysteine, phosphoserine). They are the enzymes whose
`enzyme-cycle-2` expansion should carry a covalent intermediate.

## C1 — holds

Over 50 enzymes in the Mechanism and Catalytic Site Atlas (M-CSA), 255 catalytic residues have a median of 55 heavy-atom contacts.
The 3,773 residues of the same amino-acid types have a median of 43 (one-sided
Mann-Whitney p = 2.7e-18). A random same-type sample of equal size gives p = 0.46.

## P1 — holds

Of 313 high-confidence AlphaFold models, 60 were compared with the first X-ray chain
covering at least 80% of them (PDBe best structures, numbering from Structure Integration with Function, Taxonomy and Sequences (SIFTS)). The median
Jaccard of contact sets is 0.90, the shift-by-3 control is 0.14, and the buried and
exposed calls agree for 95% of residues. The AlphaFold contact graphs used by `attach`
and `folding` stand in for measured ones on these proteins.

## D1 — fails

Over 40 proteins with located sites and DisProt entries, named sites outside the
disordered regions still have lower local order than same-type residues (median 0.88
against 0.94, one-sided p = 0.0012, 34 sites). Inside disordered regions they do not
differ (p = 0.37, 26 sites). Measured disorder does not account for the weak F4
effect. DisProt is incomplete, so "outside" is not the same as "ordered".

## G1 — holds, weakly discriminating

505 kernel small-molecule reactions have KEGG ids for every participant. 324 match a
Human-GEM cytosolic reaction by participant set, and all 324 have the same coefficients.
Only 23 have a coefficient other than 1, so the agreement rests on those 23.
Reversibility agrees in 244 of 324.

## Loaders, not yet answering

- **Concentrations.** `absent`: no file under `data/raw/concentrations/`. The supplementary
  tables of Park et al. (2016) need a Nature login and the Human Metabolome Database (HMDB) refuses scripted downloads.
- **BRaunschweig ENzyme DAtabase (BRENDA).** `absent`: the JSON release needs its licence accepted on the download page.
- **HeLa abundance.** With HeLa abundances (PaxDb) 1 branch point is answered, against 3 with whole-organism ones.
  Galactose goes to galactokinase 0.96 (whole organism 0.98).
