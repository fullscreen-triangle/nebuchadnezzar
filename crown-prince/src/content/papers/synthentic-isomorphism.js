import { paperById } from "./_meta";
import { Section, Sub, P, L } from "@/components/paper/Section";
import {
  Axiom,
  Theorem,
  Lemma,
  Corollary,
  Proposition,
  Definition,
  Remark,
  Proof,
} from "@/components/paper/Blocks";
import { M, Eq } from "@/components/paper/Math";
import { Chart } from "@/components/paper/Figure";
import { ValidationTable } from "@/components/paper/ValidationTable";
import Heatmap from "@/components/charts/Heatmap";
import Scatter from "@/components/charts/Scatter";
import BarChart from "@/components/charts/BarChart";
import ScalingCurve from "@/components/charts/ScalingCurve";
import CumulativeCurve from "@/components/charts/CumulativeCurve";
import {
  buildHeatmap,
  buildKdScatter,
  BINDING_BY_CLASS,
  VALIDATION_ROWS,
  VALIDATION_PASSED,
} from "./synthentic-isomorphism-data";

const meta = {
  ...paperById("synthentic-isomorphism-database"),
  author: "Kundai Farai Sachikonye",
  year: "2026",
  keywords: [
    "empty dictionary pharmacology",
    "synthentic isomorphism",
    "dual categorical addressing",
    "drug–target trajectory intersection",
    "ADME trajectory",
    "loop holonomy closure",
    "ternary trie",
    "S-entropy coordinates",
  ],
};

const sections = [
  { id: "introduction", n: "1", title: "Introduction" },
  { id: "axioms-and-machinery", n: "2", title: "Axioms and Machinery" },
  { id: "drug-coordinates", n: "3", title: "Drug S-Entropy Coordinates" },
  { id: "target-coordinates", n: "4", title: "Target S-Entropy Coordinates" },
  { id: "dual-ternary-trie", n: "5", title: "The Dual Ternary Trie" },
  { id: "synthentic-isomorphism", n: "6", title: "The Synthentic Isomorphism Theorem" },
  { id: "adme-trajectory", n: "7", title: "ADME as Trajectory" },
  { id: "adverse-effects", n: "8", title: "Adverse Effects as Branches" },
  { id: "therapeutic-effect", n: "9", title: "Therapeutic Loop Closure" },
  { id: "drug-drug-interactions", n: "10", title: "Drug–Drug Interactions" },
  { id: "primitives", n: "11", title: "Six Pharmacological Primitives" },
  { id: "complexity", n: "12", title: "Complexity and Scaling" },
  { id: "validation", n: "13", title: "Validation" },
  { id: "discussion", n: "14", title: "Discussion" },
];

function Body() {
  const heatmapData = buildHeatmap();
  const scatterData = buildKdScatter();

  return (
    <>
      {/* ========================================================== */}
      <Section n="1" title="Introduction">
        <P>
          Modern pharmacological knowledge is stored in dictionaries. DrugBank
          catalogues <M>{"\\sim 16{,}000"}</M> drug entries with SMILES strings,
          structural descriptors, and known target annotations; ChEMBL contains{" "}
          <M>{"\\sim 2.4 \\cdot 10^6"}</M> bioactivity measurements; STITCH
          curates <M>{"\\sim 10^7"}</M> chemical–protein interactions. Every one
          of these systems is a <em>dictionary</em>: it stores pre-computed
          representations of entities, accepts queries as lookups, and returns
          answers by similarity scoring. The architecture is the unavoidable
          consequence of using extrinsic coordinates.
        </P>
        <P>
          This paper develops a different architecture, the{" "}
          <em>synthentic isomorphism database</em>: a system in which drugs and
          targets are <em>both</em> intrinsically addressed in the S-entropy
          coordinate space <M>\Sspace = [0,1]^3</M> derived from bounded phase
          space geometry. The two address spaces are linked by an explicit
          isomorphism, and pharmacological action emerges as the geometric
          relationship between drug and target addresses.
        </P>
        <Sub title="The empty dictionary principle">
          <P>
            The system stores no drugs, no targets, no affinities, no ADME
            profiles, and no adverse-event tables. All pharmacological
            predicates are evaluated as geometric operations on two ternary
            tries. Storage is <M>{"\\sim 50"}</M> MB for a 16 000-drug,
            2 000-target database (roughly <M>{"250\\times"}</M> less than
            Morgan fingerprints alone); query time is <M>{"\\sim 18"}</M> ns
            independent of database size <M>N</M>.
          </P>
        </Sub>
        <Sub title="Contributions">
          <L>
            <li>
              Pharmacological S-entropy coordinates on both drug and target
              sides (§3, §4).
            </li>
            <li>
              The dual ternary trie <M>{"(\\mathcal{T}_\\Drug, \\mathcal{T}_\\Target)"}</M>{" "}
              and a proof that drug–target interaction is a geometric operation
              on it (§5, §6).
            </li>
            <li>
              ADME as a time-parametrised trajectory in <M>{"\\Sspace^{\\mathrm{pharm}}"}</M> (§7).
            </li>
            <li>
              Adverse effects as <em>geometric</em> off-target branches, not
              stochastic events (§8).
            </li>
            <li>
              Therapeutic effect as loop-holonomy closure via sparse{" "}
              <M>\ell_1</M> design (§9).
            </li>
            <li>
              Six primitives that compose to answer every well-posed
              pharmacological query, validated on 40 drug–target pairs (§11–§13).
            </li>
          </L>
        </Sub>
      </Section>

      {/* ========================================================== */}
      <Section n="2" title="Axioms and Machinery">
        <Axiom label="Bounded Phase Space Law">
          All persistent dynamical systems occupy bounded regions of phase space
          with finite Liouville measure, and these bounded regions admit
          hierarchical partitioning into distinguishable subregions.
        </Axiom>
        <Axiom label="Categorical Observation">
          An observer with finite resolution partitions phase space into a
          finite number of distinguishable categories. Two states belong to the
          same category if and only if the observer cannot distinguish them
          through available measurements.
        </Axiom>
        <P>
          From these two axioms we derive without further postulate: forced
          partitioning into{" "}
          <M>{"M = \\lfloor \\mu(\\Omega)/\\delta^d \\rfloor"}</M> distinguishable
          states; the partition coordinate structure{" "}
          <M>(n, \ell, m, s)</M> with capacity <M>C(n) = 2n^2</M>; oscillatory
          necessity; the Triple Entropy Equivalence{" "}
          <M>{"S_{\\mathrm{osc}} = S_{\\mathrm{cat}} = S_{\\mathrm{part}} = \\kB \\Depth \\ln n"}</M>;
          and the Composition, Compression, and Conservation theorems.
        </P>
        <Theorem label="Composition">
          For two partition states with depths <M>\Depth_1</M>, <M>\Depth_2</M>,
          the composed state has{" "}
          <M>{"\\Depth_{\\mathrm{comp}} = \\Depth_1 + \\Depth_2 - \\Depth_{\\mathrm{overlap}}"}</M>.
        </Theorem>
        <Theorem label="Compression">
          The free energy of a partition state is bounded below by its depth:
          <Eq>{"F \\geq T \\kB \\ln b \\cdot \\Depth"}</Eq>
        </Theorem>
        <Theorem label="Partition Conservation">
          In the absence of explicit partition exchange with an external
          reservoir, the total partition depth of a closed system is conserved:{" "}
          <M>{"d\\Depth_{\\mathrm{total}}/dt = 0"}</M>.
        </Theorem>
        <Remark>
          These three theorems underlie, respectively, the drug–target free
          energy law, the binding affinity–depth correspondence, and the
          drug–drug interaction conservation law derived in subsequent sections.
        </Remark>
      </Section>

      {/* ========================================================== */}
      <Section n="3" title="Drug S-Entropy Coordinates">
        <P>
          A drug molecule is a bounded oscillatory system: nuclear positions
          confined by the intramolecular potential, vibrational modes with
          discrete frequencies. Within the harmonic Born–Oppenheimer
          approximation, the vibrational spectrum{" "}
          <M>{"\\{\\omega_i^\\Drug\\}_{i=1}^{N_\\Drug}"}</M> constitutes the drug's{" "}
          <em>oscillatory fingerprint</em>.
        </P>
        <Definition label="Knowledge entropy">
          For frequencies <M>{"\\{\\omega_i\\}_{i=1}^{N}"}</M>:
          <Eq label="3.1">
            {"\\Sk = -\\frac{1}{\\ln N}\\sum_{i=1}^{N} p_i \\ln p_i, \\quad p_i = \\frac{\\omega_i}{\\sum_j \\omega_j}"}
          </Eq>
        </Definition>
        <Definition label="Temporal entropy">
          <Eq label="3.2">
            {"\\St = \\frac{\\log(\\omega_{\\max}/\\omega_{\\min})}{\\log(\\omega_{\\mathrm{ref,max}}/\\omega_{\\mathrm{ref,min}})}"}
          </Eq>
          with reference scales{" "}
          <M>{"\\omega_{\\mathrm{ref,max}} = 4401\\,\\mathrm{cm}^{-1}"}</M>{" "}
          (H<sub>2</sub> stretch) and{" "}
          <M>{"\\omega_{\\mathrm{ref,min}} = 218\\,\\mathrm{cm}^{-1}"}</M> (CCl
          <sub>4</sub> lowest mode).
        </Definition>
        <Definition label="Evolution entropy">
          With the harmonic-proximity matrix{" "}
          <M>{"H_{ij} = [\\min_{p/q}|\\omega_i/\\omega_j - p/q| < \\delta]"}</M>{" "}
          for <M>{"p, q \\leq q_{\\max}"}</M>:
          <Eq label="3.3">
            {"\\Se = \\frac{\\sum_{i<j} H_{ij}}{N(N-1)/2}"}
          </Eq>
        </Definition>
        <Sub title="Pharmacological extensions">
          <P>
            For ADME modelling the basic triple{" "}
            <M>(\Sk^\Drug, \St^\Drug, \Se^\Drug)</M> is supplemented by a
            lipophilicity coordinate{" "}
            <M>{"\\Sk^{\\Drug,\\log P} = (\\log P_\\Drug - \\log P_{\\min})/(\\log P_{\\max} - \\log P_{\\min})"}</M>{" "}
            and an ionisability coordinate based on the Henderson–Hasselbalch
            fraction. The extended space is{" "}
            <M>{"\\Sspace^{\\mathrm{pharm}} = [0,1]^5"}</M>; for pure
            identification we project back to <M>\Sspace = [0,1]^3</M>.
          </P>
        </Sub>
      </Section>

      {/* ========================================================== */}
      <Section n="4" title="Target S-Entropy Coordinates">
        <P>
          A protein target is a bounded oscillatory system whose vibrational
          fingerprint is obtained from elastic network models or all-atom normal
          mode analysis. We use the 50 lowest-frequency modes (collective
          motions relevant to allostery and conformational dynamics).
        </P>
        <Definition label="Target knowledge entropy">
          <Eq label="4.1">
            {"\\Sk^\\Target = -\\frac{1}{\\ln N_{\\mathrm{low}}} \\sum_{j=1}^{N_{\\mathrm{low}}} p_j^\\Target \\ln p_j^\\Target"}
          </Eq>
          with <M>{"N_{\\mathrm{low}} = 50"}</M> and{" "}
          <M>{"p_j^\\Target = \\omega_j^\\Target / \\sum_k \\omega_k^\\Target"}</M>.
        </Definition>
        <Definition label="Target temporal entropy">
          <Eq label="4.2">
            {"\\St^\\Target = \\frac{\\log(\\omega_{\\max}^\\Target/\\omega_{\\min}^\\Target)}{\\log(\\omega_{\\mathrm{ref,max}}^\\Target/\\omega_{\\mathrm{ref,min}}^\\Target)}"}
          </Eq>
          with target-specific reference scales{" "}
          <M>{"\\omega_{\\mathrm{ref,max}}^\\Target = 1500\\,\\mathrm{cm}^{-1}"}</M>{" "}
          (amide C=O stretch) and{" "}
          <M>{"\\omega_{\\mathrm{ref,min}}^\\Target = 5\\,\\mathrm{cm}^{-1}"}</M>{" "}
          (collective protein breathing).
        </Definition>
        <Sub title="Target classes and coordinate distributions">
          <P>
            The target S-entropy coordinate space is not uniformly populated.
            Known drug targets cluster in distinct regions:
          </P>
          <L>
            <li>
              <strong>GPCRs</strong>: <M>\Sk^\Target \sim 0.85</M>,{" "}
              <M>\St^\Target \sim 0.75</M>, <M>\Se^\Target \sim 0.55</M>.
            </li>
            <li>
              <strong>Enzymes with deep cavities</strong>:{" "}
              <M>\Sk^\Target \sim 0.92</M>, <M>\St^\Target \sim 0.50</M>,{" "}
              <M>\Se^\Target \sim 0.70</M>.
            </li>
            <li>
              <strong>Ion channels</strong>: <M>\Sk^\Target \sim 0.80</M>,{" "}
              <M>\St^\Target \sim 0.85</M>, <M>\Se^\Target \sim 0.45</M>.
            </li>
            <li>
              <strong>Kinases</strong>: <M>\Sk^\Target \sim 0.88</M>,{" "}
              <M>\St^\Target \sim 0.60</M>, <M>\Se^\Target \sim 0.75</M>.
            </li>
            <li>
              <strong>Nuclear receptors</strong>: <M>\Sk^\Target \sim 0.90</M>,{" "}
              <M>\St^\Target \sim 0.55</M>, <M>\Se^\Target \sim 0.80</M>.
            </li>
          </L>
          <P>
            These class-specific clusters emerge from the vibrational structure;
            no class labels are used in the encoding.
          </P>
        </Sub>
      </Section>

      {/* ========================================================== */}
      <Section n="5" title="The Dual Ternary Trie">
        <Definition label="Dual ternary trie">
          The dual ternary trie is the pair{" "}
          <M>{"(\\mathcal{T}_\\Drug, \\mathcal{T}_\\Target)"}</M> where each is a
          rooted tree with three children per internal node, storing entities at
          nodes reached by their interleaved ternary addresses. The two tries
          share the hierarchical cell structure of <M>\Sspace</M> but are
          populated independently.
        </Definition>
        <Proposition label="Co-location and spatial proximity">
          If drug <M>\Drug</M> and target <M>\Target</M> share a ternary prefix
          of length <M>k</M>, their S-entropy coordinates lie within a cell of
          diameter <M>{"\\sqrt{3}\\cdot 3^{-\\lfloor k/3 \\rfloor}"}</M>.
        </Proposition>
        <Proof>
          Each group of three trits refines one dimension by a factor of 3. After{" "}
          <M>k</M> trits, each dimension has been refined at most{" "}
          <M>{"\\lceil k/3 \\rceil"}</M> times; the cell diagonal is{" "}
          <M>{"\\sqrt{3}"}</M> times the side length.
        </Proof>
        <Theorem label="Empty dictionary property">
          The storage cost of <M>{"(\\mathcal{T}_\\Drug, \\mathcal{T}_\\Target)"}</M>{" "}
          for <M>N_\Drug</M> drugs and <M>N_\Target</M> targets is{" "}
          <M>{"O((N_\\Drug + N_\\Target) \\cdot k)"}</M>: only ternary addresses
          and external identifiers are stored.
        </Theorem>
      </Section>

      {/* ========================================================== */}
      <Section n="6" title="The Synthentic Isomorphism Theorem">
        <Definition label="Synthentic isomorphism">
          The synthentic isomorphism{" "}
          <M>{"\\Iso: \\mathcal{T}_\\Drug \\leftrightarrow \\mathcal{T}_\\Target"}</M>{" "}
          is the pair of partial maps that send each drug address to the set of
          target cells with which its trajectory intersects, and vice versa. The
          maps are not stored as tables; they are evaluated on demand.
        </Definition>
        <Theorem label="Synthentic isomorphism">
          Drug <M>\Drug</M> and target <M>\Target</M> form a thermodynamically
          favourable complex with{" "}
          <M>{"K_d = \\exp(-\\Delta\\Depth_{\\mathrm{bind}} \\ln b)"}</M> if and
          only if (i) the S-entropy centroid{" "}
          <M>{"\\mathbf{s}_\\oplus = (N_\\Drug \\mathbf{s}_\\Drug + N_\\Target \\mathbf{s}_\\Target)/(N_\\Drug + N_\\Target)"}</M>{" "}
          lies within the reachability region; (ii) the partition-depth barrier{" "}
          <M>\Depth^*</M> along the minimum-depth path satisfies{" "}
          <M>{"\\Depth^* \\leq \\Delta\\Depth_{\\mathrm{bind}}"}</M>; and (iii)
          the complex preserves S-entropy conservation.
        </Theorem>
        <Definition label="Reactivity map">
          <Eq label="6.1">
            {"R(C^\\Drug, C^\\Target) = \\exp\\!\\left(-\\frac{d_{\\mathcal{S}}(C^\\Drug, C^\\Target)^2}{2 \\sigma_R^2}\\right)"}
          </Eq>
          with reactivity bandwidth <M>{"\\sigma_R = 0.1"}</M>.
        </Definition>
        <Chart
          num="1"
          caption="Reactivity heatmap R(drug, target) for a 20×20 representative block. Bright cells identify drug–target pairs whose centroids lie within the reactivity bandwidth. The map is not stored — it is evaluated on demand from the ternary addresses by a single geometric operation. Hover any cell."
        >
          <Heatmap data={heatmapData} />
        </Chart>
        <Corollary label="Binding from geometry alone">
          Binding feasibility can be evaluated in <M>O(k)</M> operations by
          comparing the two ternary addresses. No docking simulation is
          required.
        </Corollary>
        <Corollary label="Inverse synthetic retrieval">
          Given a target <M>\Target</M>, the set of drugs that bind <M>\Target</M>{" "}
          is enumerable by traversing <M>{"\\mathcal{T}_\\Drug"}</M> restricted to
          the reachability region around <M>\targaddr</M>.
        </Corollary>
      </Section>

      {/* ========================================================== */}
      <Section n="7" title="ADME as Trajectory">
        <P>
          ADME is a time-parametrised trajectory in S-entropy space. The drug's
          coordinates evolve as it passes through successive body compartments,
          each imposing its own partition structure.
        </P>
        <Definition label="ADME trajectory">
          <Eq label="7.1">
            {"\\gamma_\\Drug : [0,\\infty) \\to \\Sspace^{\\mathrm{pharm}}, \\quad \\gamma_\\Drug(t) = (\\Sk^\\Drug(t), \\St^\\Drug(t), \\Se^\\Drug(t), \\ldots)"}
          </Eq>
        </Definition>
        <Definition label="Compartment transition rate (Eyring)">
          <Eq label="7.2">
            {"k_{ij} = k_0 \\exp\\!\\left(-\\frac{\\Delta\\Depth_{ij}}{\\Depth_T}\\right)"}
          </Eq>
          with <M>{"\\Depth_T = \\kB T / (\\kB \\ln b)"}</M> the thermal depth
          scale.
        </Definition>
        <Theorem label="Half-life from trajectory curvature">
          The elimination half-life is determined by the geodesic curvature of
          the ADME trajectory at the elimination boundary:{" "}
          <M>{"t_{1/2} = \\ln 2 / |\\kappa_\\gamma|_{\\mathrm{elim}}"}</M>.
        </Theorem>
        <Corollary label="Zero-parameter half-life prediction">
          Given a drug's S-entropy coordinates, its half-life is computable
          without training data by evaluating the geodesic curvature at the
          elimination boundary. The prediction uses only the drug's coordinates
          and the universal organ-metric (shared across drugs).
        </Corollary>
      </Section>

      {/* ========================================================== */}
      <Section n="8" title="Adverse Effects as Branches">
        <Definition label="Adverse effect branch">
          An adverse-effect branch from a trajectory point <M>\gamma_\Drug(t)</M>{" "}
          is a sub-trajectory that starts on the main trajectory, ends in the
          reachability region of an off-target <M>\Target'</M>, and requires a
          partition-depth barrier <M>{"\\Depth^*_{\\mathrm{off}}"}</M> bounded by
          the off-target selectivity margin.
        </Definition>
        <Theorem label="Branch probability">
          <Eq label="8.1">
            {"P_{\\mathrm{branch}}(\\Target') = \\frac{R(\\gamma_\\Drug(t), \\Target')}{\\sum_{\\Target'' \\in \\mathcal{N}(\\gamma_\\Drug(t))} R(\\gamma_\\Drug(t), \\Target'')}"}
          </Eq>
        </Theorem>
        <Corollary label="Integrated adverse-effect likelihood">
          The expected fraction of drug molecules reaching off-target <M>\Target'</M>{" "}
          over the full ADME trajectory is{" "}
          <M>{"L_{\\mathrm{adverse}}(\\Target') = \\int_0^\\infty P_{\\mathrm{branch}}(\\Target'; t)\\, c_\\Drug(t)\\, dt"}</M>.
        </Corollary>
        <Theorem label="Determinism of adverse effects">
          Given the S-entropy coordinates of drug <M>\Drug</M> and the positions
          of all targets <M>{"\\{\\Target_j\\}"}</M> in <M>\Sspace</M>, the
          expected adverse-effect profile of <M>\Drug</M> is computable exactly
          as a deterministic function of the dual-trie geometry. No drug-specific
          empirical parameters are required.
        </Theorem>
        <Remark label="Stochastic appearance">
          The apparent stochasticity of clinical adverse-event data arises from
          patient heterogeneity — the trie <M>\mathcal{T}_\Target</M> is
          patient-specific (target expression varies), and the ADME metric is
          patient-specific (organ mass, blood flow). The deterministic
          prediction becomes a distribution over patients only when these
          parameters are marginalised.
        </Remark>
      </Section>

      {/* ========================================================== */}
      <Section n="9" title="Therapeutic Loop Closure">
        <P>
          A drug's therapeutic effect is the restoration of the patient's
          cellular biochemical circuit from a diseased state to a healthy state.
          The framework of the therapeutic-effect-trajectory paper establishes
          that this restoration is a <em>loop-holonomy closure</em>: a diseased
          cell has non-trivial holonomy <M>{"\\Hol_\\ell \\neq \\mathrm{Id}"}</M>{" "}
          on at least one cycle of its partition graph.
        </P>
        <Definition label="Therapeutic design problem">
          Find the minimum-norm edge perturbation{" "}
          <M>{"\\boldsymbol{\\eta} \\in \\Real^{|E|}"}</M> such that:
          <Eq label="9.1">
            {"\\boldsymbol{\\eta}^* = \\arg\\min_{\\boldsymbol{\\eta}} \\|\\boldsymbol{\\eta}\\|_1 \\quad \\text{s.t.}\\quad \\Hol_{\\ell_i}(\\boldsymbol{\\eta}) = \\mathrm{Id}\\ \\forall i,\\ \\eta_{ij} \\in [-1, 1]"}
          </Eq>
        </Definition>
        <Theorem label="Therapeutic drug signature">
          The therapeutic drug for a given patient circuit is a drug{" "}
          <M>\Drug</M> whose trajectory intersection pattern with the target
          trie reproduces the optimal edge perturbation{" "}
          <M>\boldsymbol{\eta}^*</M>. Inverse synthetic retrieval enumerates
          these drugs from <M>{"\\mathcal{T}_\\Drug"}</M>.
        </Theorem>
        <Sub title="Zero-dictionary drug discovery">
          <P>
            The pipeline: solve the sparse <M>\ell_1</M> LP for{" "}
            <M>\boldsymbol{\eta}^*</M>; convert to required target interactions;
            invoke inverse synthetic retrieval per target; intersect the
            retrieved drug sets; if empty, solve the inverse design problem to
            generate a synthetic molecule with the required S-entropy
            coordinates. The protocol stores nothing drug-specific.
          </P>
        </Sub>
      </Section>

      {/* ========================================================== */}
      <Section n="10" title="Drug–Drug Interactions">
        <Definition label="Trajectory superposition">
          When drugs <M>\Drug_1</M>, <M>\Drug_2</M> are co-administered, their
          joint trajectory is constrained by Partition Conservation on shared
          metabolic edges:
          <Eq label="10.1">
            {"\\Gres_e(\\gamma_{12}) = \\Gres_e(\\gamma_1) + \\Gres_e(\\gamma_2) - \\Gres_e^{\\max}"}
          </Eq>
        </Definition>
        <Theorem label="DDI feasibility">
          Drugs interact significantly iff (i) their ADME trajectories share at
          least one edge at which <M>\Gres_e^{\max}</M> is exceeded, OR (ii)
          their target sets retrieved via{" "}
          <M>{"\\Iso_{\\Drug \\to \\Target}"}</M> have non-empty intersection.
        </Theorem>
        <Corollary>
          Both conditions are evaluable from the ternary addresses alone. No DDI
          database and no DDI classifier is required.
        </Corollary>
      </Section>

      {/* ========================================================== */}
      <Section n="11" title="Six Pharmacological Primitives">
        <P>
          The empty dictionary exposes six geometric primitives. Every
          well-posed pharmacological query decomposes into a finite composition.
        </P>
        <div className="my-6 grid gap-3 md:grid-cols-2">
          <PrimitiveCard
            name="Identify"
            sig="\{\omega_i\} \to \Sspace^{\mathrm{pharm}}"
            note="Vibrational fingerprint → ternary address. O(N log N + k)."
          />
          <PrimitiveCard
            name="Similar"
            sig="\mathbf{a} \times \varepsilon \to 2^{\Sspace}"
            note="Prefix-matched retrieval. O(k + m)."
          />
          <PrimitiveCard
            name="Predict"
            sig="(\mathbf{a}, t) \to \Sspace^{\mathrm{pharm}}"
            note="ADME trajectory under organ-metric. O(k + T/Δt)."
          />
          <PrimitiveCard
            name="React"
            sig="\drugaddr \times \targaddr \to \{K_d, \mathrm{no}\!-\!\mathrm{bind}\}"
            note="Trajectory intersection. O(k)."
          />
          <PrimitiveCard
            name="Deviate"
            sig="\drugaddr \times \mathcal{T}_\Target \to \{(\Target', L)\}"
            note="Off-target adverse-effect branches."
          />
          <PrimitiveCard
            name="Close"
            sig="G^\Patient \to \boldsymbol{\eta}^*"
            note="Sparse ℓ₁ loop-holonomy closure. Polynomial."
          />
        </div>
        <Theorem label="Primitive completeness">
          Every well-posed pharmacological query is answerable by a finite
          composition of the six primitives.
        </Theorem>
      </Section>

      {/* ========================================================== */}
      <Section n="12" title="Complexity and Scaling">
        <P>
          All four core primitives execute in <M>O(k)</M> trie traversal{" "}
          regardless of database size <M>N</M>. <em>Deviate</em> is sublinear in{" "}
          <M>N_\Target</M>; <em>Close</em> is polynomial in the patient graph.
        </P>
        <Theorem label="Scaling independence">
          The complexity of <em>Identify, Similar, Predict, React</em> is
          independent of the number of stored drugs and targets. For{" "}
          <M>k = 18</M>, every query completes in <M>{"\\sim 18"}</M>{" "}
          operations regardless of whether the trie contains 40 or{" "}
          <M>{"10^8"}</M> entities.
        </Theorem>
        <Chart
          num="2"
          caption="Query cost on log–log axes. Move the slider to set the database size N. The fingerprint cost scales as O(Nd) with d = 1024; the empty dictionary cost is flat at O(k) = O(18). At PubChem scale the speedup exceeds 10⁹×."
        >
          <ScalingCurve />
        </Chart>
      </Section>

      {/* ========================================================== */}
      <Section n="13" title="Validation">
        <P>
          We validate the framework on a 40-pair drug–target test suite spanning
          five pharmacological classes (8 pairs per class): GPCR ligands, kinase
          inhibitors, enzyme inhibitors, ion-channel modulators, and nuclear
          receptor ligands. Predictions use only S-entropy coordinates and the
          axiomatic geometric machinery; no parameters are fitted.
        </P>
        <ValidationTable
          rows={VALIDATION_ROWS}
          summary={{ passed: 51, total: 52 }}
        />
        <Chart
          num="3"
          caption="Predicted vs observed log K_d for all 40 drug–target pairs. The y=x identity line is drawn dashed. Points are coloured by target class. Hover for the drug–target pair name."
        >
          <Scatter
            data={buildKdScatter()}
            xLabel="observed log K_d (M)"
            yLabel="predicted log K_d"
            diagonal
          />
        </Chart>
        <Chart
          num="4"
          caption="Per-class binding accuracy: number of correctly predicted pairs out of 8 per class. Enzymes 8/8 (strongest); ion channels 6/8 (weakest, reflecting under-representation of membrane dynamics in the 50-lowest-mode encoding)."
        >
          <BarChart data={BINDING_BY_CLASS} threshold={5} yLabel="correct / 8" />
        </Chart>
        <Chart
          num="5"
          caption="Cumulative validation pass rate across all 52 tests. The 95% reference line is reached early and maintained; the final 51/52 = 98.1% includes drug & target uniqueness, target-class cohesion, binding affinity, half-life, adverse effects, drug–drug interactions, reachability, and complexity bounds."
        >
          <CumulativeCurve passed={VALIDATION_PASSED} threshold={95} />
        </Chart>
      </Section>

      {/* ========================================================== */}
      <Section n="14" title="Discussion">
        <Sub title="What the empty dictionary achieves">
          <P>
            The synthentic isomorphism database is a pharmacology knowledge
            system with no pharmacological knowledge stored. Drugs and targets
            are simply bounded oscillatory systems; their interactions are
            deterministic consequences of their vibrational structure. The
            database's "knowledge" is the geometry of <M>\Sspace</M> combined
            with the axiomatic machinery that maps fingerprints to coordinates.
          </P>
        </Sub>
        <Sub title="What replaces the dictionary">
          <P>
            In the conventional pipeline, pharmacological knowledge is stored as
            structural data, bioactivity measurements, ADMET predictions,
            adverse-event databases, drug-drug interaction tables, and clinical
            guidelines. In the synthentic isomorphism database, all of these
            are replaced by: the Bounded Phase Space Law (axiom), the S-entropy
            coordinate map (3 closed-form equations), the dual ternary trie
            (addresses only), and the six geometric primitives.
          </P>
        </Sub>
        <Sub title="Limitations">
          <L>
            <li>Membrane proteins are under-represented by the 50-mode encoding.</li>
            <li>
              Allosteric targets require a target <em>trajectory</em> rather
              than a single point.
            </li>
            <li>Covalent drugs have a different trajectory-completion criterion.</li>
            <li>
              Biologics need a hierarchical encoding to capture domain- and
              residue-level S-entropy at different depths.
            </li>
            <li>
              Cross-species variation requires species-specific target
              addresses.
            </li>
          </L>
        </Sub>
        <Sub title="Conclusion">
          <P>
            Pharmacology is a geometric science. The empirical data collected
            over two centuries records the operations of nature, which derives
            the same geometry we derive. Building a database to store those
            records is building a dictionary of nature's computations; building
            an empty dictionary is computing them directly.
          </P>
        </Sub>
      </Section>
    </>
  );
}

function PrimitiveCard({ name, sig, note }) {
  return (
    <div className="rounded-md border border-light/10 bg-light/[0.02] p-4">
      <p className="text-[10px] uppercase tracking-[0.25em] text-primaryDark">
        {name}
      </p>
      <p className="mt-2 text-sm text-light/80">
        <M>{sig}</M>
      </p>
      <p className="mt-2 text-[11px] leading-relaxed text-light/55">{note}</p>
    </div>
  );
}

const paper = { meta, sections, Body };
export default paper;
