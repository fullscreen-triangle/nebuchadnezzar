# circuit_kg

An experimental kernel that turns Reactome's purely cytosolic reactions into a
circuit knowledge graph. Its edges carry element laws (differential equations).
Its node values are what solves those laws from the node's contacts. Each answer
is a verdict: `answered`, `certified-empty`, `absent`, `underdetermined`,
`undecidable` or `refused`. A verdict other than `answered` names its blocker.

## Run

```bash
python -m circuit_kg fetch      # ChEBI flat files (about 14 MB) + TECRDB derived table -> data/raw
python -m circuit_kg extract    # Reactome SBML tarball -> data/derived/reactome_cytosol.json
python -m circuit_kg build      # -> results/summary.json, graph.json, supplements.json
python -m circuit_kg proteins   # AlphaFold DB models -> residue contact graphs (data/raw/alphafold)
python -m circuit_kg attach     # attach protein graphs through named slots -> results/attach.json
python -m circuit_kg medium     # Legendre transforms, test M1, direction over the medium -> results/medium.json
python -m circuit_kg kinetics   # SABIO-RK K_m + PaxDb: retention, locality, test P1 -> results/kinetics.json
python -m circuit_kg complete   # source-to-target completions; separation left after refinement -> results/complete.json
python -m circuit_kg folding    # levinthal's oscillator folding ideas tested against AlphaFold -> results/folding.json
python -m circuit_kg extend     # verdict axes, modifiers, dividers, redox ladder, labels -> results/extend.json
python -m circuit_kg databases  # Rhea, MetaNetX, Human-GEM, localisation, residue features, PDBe -> results/databases.json
python -m pytest circuit_kg/tests -q
```

`extract` streams every pathway file once and takes about 100 s. `build` takes
about 13 min. Most of that is the singular value decompositions (SVDs) of the
stoichiometric matrix, before and after the enzyme expansion. `medium` needs the
local eQuilibrator compound cache (set `CIRCUIT_KG_EQ_CACHE` if it is not in the
default place). It takes about 30 s. `kinetics` fetches 15 pages from the System for the Analysis of Biochemical
Pathways - Reaction Kinetics database (SABIO-RK)
once and then takes about 5 min. `complete` takes about 45 s. `folding` takes about 15 min.
`extend` fetches 22 SABIO-RK pages and the UniProt annotations once, then takes
about a minute. `databases` takes about 12 min, most of it downloading 60
experimental structures.

The ideas behind `medium`, `kinetics` and `complete` come from the chromatography
and ideal-gas papers in `maxwell`. The ideas behind `extend` come from the
cytochrome P450 monograph and the dcat-ap-plus-p450 specification in `levinthal`. The `databases` stage adds curated databases
that check or replace a supplied convention. Their predictions were committed before the
code ran (`predictions/PREDICTIONS.md`), and the results are in
`predictions/OUTCOMES.md`.

## The graph

| Element | Meaning |
|---|---|
| species vertex | potential mu = mu0 + RT ln c, never stored |
| junction vertex | one reaction, with affinity A_r and flux J_r |
| medium vertex `M` | ground; the species listed in `media/cytosol.json` are folded into it |
| species-junction edge | stoichiometric contact; adds nu * J_r to the species' node law |
| modifier edge | catalyst, stimulator or inhibitor; enters J_r's prefactor and carries no current |
| edge to `M` | every item touches the medium |

Write the gas constant times temperature as RT (2.577 kJ/mol at 310 K). The
junction law is mass action in flux-force form:
J_r = k_r * prod c^|nu| * (1 - exp(-A_r/RT)). Its tangent is G_r * A_r.
The node law is Kirchhoff's current law: (c/RT) dmu/dt = sum nu * J_r.

## Conventions

Each supplied convention is named where it is used:

- `ambient-reservoir`: the cofactor list in `media/cytosol.json`.
- `medium-floor`: unit medium edges.
- `modifier-unit-weight`: unit weight on every modifier edge.
- `conjugate-closure`: a Kyoto Encyclopedia of Genes and Genomes (KEGG) compound id reached through conjugate acids, conjugate bases
  and tautomers.
- `water-balance`: a match in the Thermodynamics of Enzyme-Catalyzed Reactions
  Database (TECRDB) that holds once water is balanced.
- `concentration-window`: 1 uM to 10 mM.
- `pH-window`: 6.8 to 7.6.
- `hess-span`: a Delta G derived from measured reactions by Hess's law.
- `carrier-slot`: a protein that is the same on both sides of a reaction adds
  nothing to its Delta G; only the modified residues (slots) do.
- `free-residue`: a residue in a protein reacts like the free amino acid.
- `slot-from-name`: modified residues are read from Reactome species names.
- `enzyme-cycle-2`: each catalysed reaction becomes two steps, binding then
  catalysis, which return the enzyme.
- `medium-floor-residue`: a buried residue still gets a medium edge of weight 1.
- `medium-window`: pH 6.8 to 7.6, ionic strength 0.1 to 0.25 M, pMg 2.8 to 3.5,
  at 298.15 K. Chemical-reference values are Legendre-transformed over this window.
- `accessible-volume`: mu = mu0 + RT ln(c/a), with one a in [0.7, 1] for every
  small solute.
- `protein-ppm-to-molar`: PaxDb parts per million times 2 to 6 mM of total
  protein molecules.
- `scope-from-source`: a completion may use the source, the medium, and the
  proteins that no reaction makes.
- `dilute-limit-share`: a branch's share of flux is (k_cat/K_m)[E] over the sum
  of its siblings'. Each k_cat is paired with the K_m values of the same SABIO-RK
  kinetic law, because SABIO-RK records no substrate on k_cat.
- `loss-of-function-variant`: Reactome's "Defective ..." reactions are modifiers
  on their wild-type edge, not competing siblings.
- `textbook-standard-conditions`: redox couples at pH 7, ionic strength 0.25 M,
  298.15 K, no Mg.
- `moser-dutton-defaults`: reorganisation energy 0.7 eV, driving force 0,
  packing density 0.76.
- `medium-as-slowest-rung`: when a ladder chain's floor is judged, its medium
  resistance is taken equal to its slowest rung's.
- `mnx-xref`: a KEGG compound for a ChEBI id reached through a MetaNetX identifier.
  It is used only for species that `conjugate-closure` left without one.
- `brenda-same-reference`: a k_cat from the BRaunschweig ENzyme DAtabase (BRENDA) is paired with a K_m only when both come
  from the same protein, the same substrate and the same reference.
- `measured-window`: a per-compound concentration window read from
  `data/raw/concentrations/*.csv` (columns `kegg,low_M,high_M,cell_type,source`).
  It replaces `concentration-window` for the compounds it lists. Without such a
  file the stage reports `absent` and names the file.

A value predicted by a model (component contribution, a learned k_cat) carries the
basis `estimated`. It composes above `derived`, and its ground verdict is
`estimated`, never `composite`. A cycle whose every energy is derived or estimated
is refused as non-falsifiable (`thermo.judge_cycle_guarded`).

## Modules

| File | Content |
|---|---|
| `extract.py` | cytosol filter over the SBML tarball |
| `thermo.py` | ChEBI to KEGG to TECRDB join, pooled replicate sigma, exact cycle basis, three-valued cycle test |
| `graph.py` | the graph, its sympy laws, node-solve verdicts |
| `solve.py` | Gauss-Seidel node relaxation, linear and mass action |
| `ladder.py` | rung powers from resistances; shares equal flux control coefficients |
| `analyses.py` | structure, cycles, direction, local cut keys and derived power, refinement lattice, series chains |
| `hess.py` | weighted least-squares formation energies; Delta G and its sigma for any reaction in the span of measured ones |
| `slot.py` | phosphosite parser and carrier/slot decomposition of reactions refused by TECRDB |
| `enzyme.py` | catalysts expanded into two-step enzyme cycles; checks that each enzyme total is conserved |
| `proteins.py` | AlphaFold DB fetch, Shrake-Rupley surface areas, residue contact graphs with counted weights |
| `attach.py` | residue graphs attached through slot edges; separation-cost monotonicity check against a real-weight control; exposure test |
| `medium.py` | per-compound Legendre transform (eQuilibrator's formula and microspecies), held-out-condition test, direction over the medium window, crowding |
| `kinetics.py` | SABIO-RK K_m and PaxDb abundance; retention factor per metabolite; departure from the dilute limit per enzyme; test P1 |
| `folding.py` | DSSP-style secondary structure, sequence and AlphaFold couplings, Kuramoto phase-lock groups and local order, ternary addresses; experiments F1-F5 |
| `complete.py` | forward scope and backward completion under three direction semantics; bits left after refinement |
| `axes.py` | gate letters for empty answers, ground verdicts from provenance, stage digests, ladder floors, check markers |
| `annotations.py` | UniProt EC numbers, Pfam domains and families for every protein |
| `modifiers.py` | additive log-rate modifier stacks, inhibitor constants, what breaks enzyme totals, disease-variant edges |
| `branching.py` | flux shares at branch species from k_cat/K_m and abundance |
| `redox.py` | redox couple ladder from the Hess span; Moser-Dutton ruler on PDB distances |
| `labels.py` | EC, Pfam and gene labels on structural classes with a shuffle null; stoichiometric signatures |
| `xref.py` | Rhea mappings, directions and participants (SPARQL); MetaNetX KEGG rescue; Human-GEM comparison |
| `localisation.py` | Human Protein Atlas and COMPARTMENTS cytosol support for every catalyst |
| `features.py` | UniProt modified residues and active sites; slot check; M-CSA catalytic residues; DisProt disorder |
| `structures.py` | PDBe best structure and SIFTS numbering; AlphaFold contacts against the experimental chain |
| `loaders.py` | measured concentration windows and the BRENDA JSON release, each `absent` until its file is present |

Edge weights used for cuts are always counts (stoichiometry, contacts, exposed
atoms). Real-valued quantities such as Delta G or a conductance stay in the
edge laws and never become cut weights.

The TECRDB table is copied from the disease-circuit-tolerance validation suite.
Its `dG0_prime_measured` column keeps the original TECRDB orientation, while
`reaction` and `dG0` are canonicalised. `thermo.load_tecrdb` recovers the flip
from dG0 = flip * (dG0' - transform).
