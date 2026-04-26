import Head from "next/head";

export default function Framework() {
  return (
    <>
      <Head>
        <title>Framework — Crown Prince</title>
      </Head>
      <section className="mx-auto max-w-3xl px-6 py-12 md:px-10">
        <p className="text-[11px] uppercase tracking-[0.3em] text-primaryDark">Framework</p>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight">
          One axiom. Three coordinates. Six primitives.
        </h1>

        <div className="mt-10 space-y-10 text-sm leading-relaxed text-light/80">
          <Block heading="Bounded Phase Space Law">
            All persistent dynamical systems occupy bounded regions of phase
            space with finite Liouville measure, and these bounded regions admit
            hierarchical partitioning into distinguishable subregions. Combined
            with finite observational resolution, the axiom forces partitioning
            into <span className="mono accent">M = ⌊μ(Ω)/δᵈ⌋</span> distinguishable
            states, oscillatory necessity (oscillation is the only valid mode
            for self-consistent bounded dynamics), and the partition coordinate
            structure <span className="mono accent">(n, ℓ, m, s)</span> with
            shell capacity <span className="mono accent">C(n) = 2n²</span>.
          </Block>

          <Block heading="S-entropy coordinates">
            Every bounded oscillatory system maps to a point in{" "}
            <span className="mono accent">𝒮 = [0, 1]³</span> through three
            coordinates derived from its vibrational spectrum:
            <ul className="mt-3 space-y-2 pl-4">
              <li>
                <span className="mono accent">S_k</span> — knowledge entropy:
                normalised Shannon entropy of the frequency distribution.
              </li>
              <li>
                <span className="mono accent">S_t</span> — temporal entropy:
                logarithmic ratio of the frequency span to a reference span.
              </li>
              <li>
                <span className="mono accent">S_e</span> — evolution entropy:
                fraction of mode pairs in rational frequency proximity.
              </li>
            </ul>
          </Block>

          <Block heading="Ternary addressing">
            Three-dimensionality makes base-3 the natural encoding. An
            interleaved ternary string of length <span className="mono">k</span>{" "}
            addresses one of <span className="mono">3ᵏ</span> cells in the
            hierarchical partition of <span className="mono">𝒮</span>. Trie
            traversal is <span className="mono accent">O(k)</span> regardless of
            the number of stored entities — the empty dictionary stores only
            ternary addresses, never the entities themselves.
          </Block>

          <Block heading="Six primitives">
            Every well-posed pharmacological query decomposes into a finite
            composition of:
            <ul className="mt-3 space-y-1.5 pl-4 font-mono text-xs">
              <li>Identify · Similar · Predict · React · Deviate · Close</li>
            </ul>
            Each primitive is a deterministic geometric operation on the dual
            trie. Compiled probes (~0.6 M parameters each) translate
            natural-language clinical questions into primitive sequences. The
            engine stores no pharmacological data; the probes store the
            grammar of the question, not the answer.
          </Block>
        </div>
      </section>
    </>
  );
}

function Block({ heading, children }) {
  return (
    <div>
      <h2 className="mb-3 text-xs uppercase tracking-[0.25em] text-primaryDark">
        {heading}
      </h2>
      <div>{children}</div>
    </div>
  );
}
