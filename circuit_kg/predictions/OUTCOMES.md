# Outcomes of the predictions in PREDICTIONS.md

Numbers are from `python -m circuit_kg medium` (writes `results/medium.json`).
PREDICTIONS.md was committed (de6f210) before any of this code ran.

## Check before M1: the transform itself

The per-compound Legendre transform, written from equilibrator_cache 0.8.1 and
evaluated on the local eQuilibrator compound cache, reproduces syndrome's
`transform` column on all 3,276 TECRDB rows in the window, with a largest
difference of 6.8e-13 kJ/mol. Controls: shifting pH by 0.1 leaves 66 rows within
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

- answered forward: 2 (PPi hydrolysis by PPA1 and by LHPP), against 3 before;
- undecidable because of the concentration window: 156;
- undecidable because of the medium window: 43.

The prediction "more than 3 answered" fails. The old 3 used the measured
Delta_r G'deg study means in the pH window, with no uncertainty and no ionic
strength or pMg range. AMP hydrolysis (NT5C1B) rested on one study. Here, adding
q·sigma and the I and pMg windows removes it. The two survivors clear zero by only
0.28 kJ/mol.

The second part holds: for 156 of 199 undecided reactions (78%), Delta_r G'm at
1 mM has a fixed sign across the medium window, so only concentrations block
them. This includes all 96 carrier/slot reactions.

## M3 — crowding and dilution: fails on both counts

- **Crowding.** It changes 2 of the 2 decided verdicts, which is above the 5% limit.
  Both PPi hydrolyses add a solute particle (PPi + H2O -> 2 Pi). Their 0.28 kJ/mol
  margin is below the 0.92 kJ/mol crowding shift at a = 0.7. With crowding in the
  medium, no reaction in the kernel has a decided direction.
- **Dilution.** The bound 2·n·v at 10 mM has a median of 0.0035 per particle, but
  exceeds 1e-2 for large compounds. The worst is C05893 (1,873 Da) at 0.027. The
  ideal-solution form of mu is safe for ordinary metabolites at 10 mM, but not for
  molecules this size at the top of the concentration window.
