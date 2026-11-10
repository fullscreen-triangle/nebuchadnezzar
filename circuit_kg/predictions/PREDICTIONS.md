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
