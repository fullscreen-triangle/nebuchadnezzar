// Cell spectral hologram math, after spectral-holograms.tex.
// Three-state superposition: ground IR + excited Raman + emission fluorescence,
// vibrational coupling matrix K_ij, Franck-Condon ladders, Stokes decomposition.
//
// Cells are modelled as weighted mixtures of representative biomolecules with
// characteristic vibrational modes. Spectra are synthesised from the mixture;
// the hologram is the time-resolved superposition.

// --- Cell type presets: dominant biomolecule mixture (weight, ground-state freqs cm^-1)
export const CELL_TYPES = {
  hepatocyte: {
    label: "Hepatocyte",
    molecules: [
      { name: "lipid bilayer", w: 0.30, omega: [1080, 1300, 1450, 1660, 2850, 2920, 3010] },
      { name: "albumin", w: 0.25, omega: [1240, 1340, 1450, 1550, 1660, 2940, 3300] },
      { name: "glycogen", w: 0.20, omega: [480, 850, 940, 1080, 1340, 2900] },
      { name: "CYP450", w: 0.15, omega: [350, 750, 1130, 1370, 1500, 1620, 2940] },
      { name: "water", w: 0.10, omega: [1640, 3400, 3600] },
    ],
    coherence: 0.72,
    partition_depth: 16,
  },
  neuron: {
    label: "Neuron",
    molecules: [
      { name: "neurolipid", w: 0.35, omega: [1080, 1290, 1440, 1660, 2850, 2920, 3000] },
      { name: "tubulin", w: 0.20, omega: [1240, 1340, 1450, 1550, 1660, 2920] },
      { name: "neurotransmitter pool", w: 0.15, omega: [620, 980, 1180, 1450, 1620, 3050] },
      { name: "ion channel set", w: 0.20, omega: [320, 720, 1100, 1300, 1530, 1640] },
      { name: "water/cytoplasm", w: 0.10, omega: [1640, 3400] },
    ],
    coherence: 0.81,
    partition_depth: 18,
  },
  cardiomyocyte: {
    label: "Cardiomyocyte",
    molecules: [
      { name: "actin/myosin", w: 0.40, omega: [930, 1240, 1340, 1450, 1660, 2920, 3290] },
      { name: "mitochondrial", w: 0.25, omega: [560, 750, 1130, 1370, 1580, 1620, 2940] },
      { name: "Ca-handling", w: 0.15, omega: [340, 720, 1080, 1280, 1480] },
      { name: "membrane", w: 0.15, omega: [1080, 1450, 1660, 2850, 2920] },
      { name: "water", w: 0.05, omega: [1640, 3400] },
    ],
    coherence: 0.76,
    partition_depth: 17,
  },
  cancer: {
    label: "Tumour cell (HeLa)",
    molecules: [
      { name: "DNA", w: 0.25, omega: [785, 1095, 1340, 1380, 1485, 1580] },
      { name: "RNA", w: 0.15, omega: [810, 1100, 1240, 1320, 1490, 1580] },
      { name: "altered lipids", w: 0.25, omega: [1080, 1290, 1450, 1660, 2850, 2940] },
      { name: "Warburg metabolites", w: 0.20, omega: [840, 980, 1130, 1280, 1450] },
      { name: "membrane", w: 0.15, omega: [1080, 1450, 1660, 2920] },
    ],
    coherence: 0.41, // diseased: low coherence
    partition_depth: 14,
  },
};

// --- Excited-state shifts: K_ij couples ground -> excited frequencies.
// In the spectral-hologram framework, excited-state vibrations are shifted
// from ground by mode-specific factors. We model with a deterministic seed.
function modeShift(omega, i) {
  // Mode-dependent shift: heavier modes shift more
  const norm = Math.min(omega / 4000, 1);
  return -25 - 12 * norm + 8 * Math.sin(i * 1.7);
}

export function buildSpectra(cell, sigma = 18) {
  // Combine all molecules into a single mode list (ground state).
  const ground = [];
  for (const mol of cell.molecules) {
    for (const w of mol.omega) {
      ground.push({ omega: w, intensity: mol.w / mol.omega.length });
    }
  }
  // Excited-state Raman: shifted ground modes with re-weighted intensities.
  const excited = ground.map((m, i) => ({
    omega: m.omega + modeShift(m.omega, i),
    intensity: m.intensity * (0.85 + 0.3 * Math.cos(i * 0.7)),
  }));
  // Fluorescence emission: broad envelope at the average Stokes-shifted frequency.
  const emCenter = 380; // visible-range peak in cm^-1 from 0–0
  return { ground, excited, emCenter, sigma };
}

// --- Sampled spectra on a frequency grid for plotting/superposition
export function sampleSpectrum(spectrum, grid, sigma = 18) {
  return grid.map((omega) => {
    let s = 0;
    for (const m of spectrum) {
      const d = (omega - m.omega) / sigma;
      s += m.intensity * Math.exp(-0.5 * d * d);
    }
    return s;
  });
}

// --- Hologram H(omega, t): three-state superposition with phase encoded by t.
// Returns 2D array values[T][W] of complex magnitudes |H| and phases.
export function buildHologram(cell, gridOmega, gridT, sigma = 18) {
  const { ground, excited } = buildSpectra(cell, sigma);
  const sG = sampleSpectrum(ground, gridOmega, sigma);
  const sE = sampleSpectrum(excited, gridOmega, sigma);
  // Emission: broad gaussian in middle of frequency range
  const mid = (gridOmega[0] + gridOmega[gridOmega.length - 1]) / 2;
  const sEm = gridOmega.map((w) => Math.exp(-Math.pow((w - mid) / 600, 2)) * 0.8);

  const T = gridT.length;
  const W = gridOmega.length;
  const mag = new Array(T);
  const phase = new Array(T);
  for (let ti = 0; ti < T; ti++) {
    const t = gridT[ti];
    // Three coefficients evolving in time:
    //   c_g(t) = sin^2(pi t),  c_e(t) = cos^2(pi t/2),  c_em(t) = sin(2 pi t)
    const cg = Math.pow(Math.sin(Math.PI * t), 2);
    const ce = Math.pow(Math.cos((Math.PI * t) / 2), 2);
    const cem = Math.sin(2 * Math.PI * t);
    const phig = 0;
    const phie = Math.PI * t;
    const phiem = (3 * Math.PI * t) / 2;
    mag[ti] = new Array(W);
    phase[ti] = new Array(W);
    for (let wi = 0; wi < W; wi++) {
      const re =
        cg * sG[wi] * Math.cos(phig) +
        ce * sE[wi] * Math.cos(phie) +
        cem * sEm[wi] * Math.cos(phiem);
      const im =
        cg * sG[wi] * Math.sin(phig) +
        ce * sE[wi] * Math.sin(phie) +
        cem * sEm[wi] * Math.sin(phiem);
      mag[ti][wi] = Math.sqrt(re * re + im * im);
      phase[ti][wi] = Math.atan2(im, re);
    }
  }
  return { mag, phase, gridOmega, gridT };
}

// --- Vibrational coupling matrix K_ij: pairwise, derived from how strongly
// modes shift between ground and excited.
export function couplingMatrix(cell) {
  const { ground, excited } = buildSpectra(cell);
  const n = Math.min(ground.length, 12); // top-12 modes only for legibility
  const K = [];
  for (let i = 0; i < n; i++) {
    K.push([]);
    for (let j = 0; j < n; j++) {
      if (i === j) {
        K[i][j] = 1;
      } else {
        const di = (excited[i].omega - ground[i].omega) / 100;
        const dj = (excited[j].omega - ground[j].omega) / 100;
        // Coupling = correlation of frequency shifts, gated by mode separation
        const sep = Math.abs(ground[i].omega - ground[j].omega) / 1000;
        K[i][j] = (di * dj) * Math.exp(-sep * sep);
        // Clip to [-1, 1]
        if (K[i][j] > 1) K[i][j] = 1;
        if (K[i][j] < -1) K[i][j] = -1;
      }
    }
  }
  const labels = ground.slice(0, n).map((g) => Math.round(g.omega));
  return { K, labels };
}

// --- Franck-Condon factor ladder: progression of FC factors for the dominant mode.
export function franckCondon(cell, dominantMode = 1450) {
  // Closest mode in the spectrum
  const { ground } = buildSpectra(cell);
  const target = ground.reduce((p, c) =>
    Math.abs(c.omega - dominantMode) < Math.abs(p.omega - dominantMode) ? c : p
  );
  // Huang-Rhys factor S derived from cell coherence (low coherence => more vibronic structure)
  const S = (1 - cell.coherence) * 1.5 + 0.2;
  // FC factors |<n|0>|^2 for n = 0..6
  const factors = [];
  let sFact = 1;
  for (let n = 0; n <= 6; n++) {
    if (n > 0) sFact *= n;
    factors.push((Math.exp(-S) * Math.pow(S, n)) / sFact);
  }
  return { factors, S, mode: Math.round(target.omega) };
}

// --- Stokes shift decomposition into vibrational + solvent components.
export function stokesShift(cell) {
  const { ground } = buildSpectra(cell);
  const meanShift = ground.reduce((s, m, i) => s + Math.abs(modeShift(m.omega, i)), 0) /
    ground.length;
  const vibrational = meanShift; // cm^-1
  const solvent = (1 - cell.coherence) * 800; // diseased cells have larger solvent reorg
  const total = vibrational + solvent;
  return { vibrational, solvent, total, lambdaReorg: total / 2 };
}

// --- Loop holonomy across cycles. Healthy cells have det near 1; diseased deviate.
export function holonomy(cell) {
  const cycles = ["TCA", "glycolysis", "ETC", "Ca²⁺", "redox", "nucleotide", "lipid", "MAPK"];
  const baseRng = mulberry32(cell.label.charCodeAt(0));
  return cycles.map((name, i) => {
    const noise = (baseRng() - 0.5) * 0.04;
    const drift = cell.coherence < 0.5 ? (baseRng() - 0.4) * 0.7 : 0;
    const det = 1.0 + noise + drift;
    return { name, det, healthy: Math.abs(det - 1) < 0.1 };
  });
}

// --- Sparse drug perturbation eta* needed to close holonomy.
// Enumerates 12 candidate edges; non-zero entries are the required interventions.
export function sparsePerturbation(cell, hol) {
  const edges = [
    "PDH", "GAPDH", "PFK", "G6P", "ATPase", "Complex I",
    "Complex III", "RyR2", "SERCA", "GSH-px", "PRPS1", "ACC",
  ];
  const eta = [];
  // Activate only edges associated with diseased cycles
  for (let i = 0; i < edges.length; i++) {
    const cycleIdx = i % hol.length;
    const c = hol[cycleIdx];
    let e = 0;
    if (!c.healthy) {
      // Sign and magnitude proportional to det(Hol) deviation
      e = Math.sign(c.det - 1) * Math.min(0.95, Math.abs(c.det - 1) * 1.5);
    }
    eta.push({ name: edges[i], value: e });
  }
  const l1 = eta.reduce((s, x) => s + Math.abs(x.value), 0);
  return { eta, l1 };
}

// Tiny PRNG for deterministic per-cell variation
function mulberry32(seed) {
  let a = seed | 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
