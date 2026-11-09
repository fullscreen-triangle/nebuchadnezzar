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
python -m circuit_kg build      # -> results/summary.json, results/graph.json
python -m pytest circuit_kg/tests -q
```

`extract` streams every pathway file once and takes about 100 s. `build` takes
about 60 s, most of it in one dense singular value decomposition (SVD) of the stoichiometric matrix.

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

## Modules

| File | Content |
|---|---|
| `extract.py` | cytosol filter over the SBML tarball |
| `thermo.py` | ChEBI to KEGG to TECRDB join, pooled replicate sigma, exact cycle basis, three-valued cycle test |
| `graph.py` | the graph, its sympy laws, node-solve verdicts |
| `solve.py` | Gauss-Seidel node relaxation, linear and mass action |
| `ladder.py` | rung powers from resistances; shares equal flux control coefficients |
| `analyses.py` | structure, cycles, direction, local cut keys and derived power, refinement lattice, series chains |

The TECRDB table is copied from the disease-circuit-tolerance validation suite.
Its `dG0_prime_measured` column keeps the original TECRDB orientation, while
`reaction` and `dG0` are canonicalised. `thermo.load_tecrdb` recovers the flip
from dG0 = flip * (dG0' - transform).
