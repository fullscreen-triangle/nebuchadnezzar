// S-entropy coordinates from a vibrational frequency list.
// Reference: categorical-compound-database.tex, Definitions 3.1–3.3.

export const REF = {
  omegaMax: 4401, // cm^-1, H2 stretch
  omegaMin: 218, // cm^-1, CCl4 lowest
  bRotMin: 0.39, // cm^-1, large-moment-of-inertia diatomic
  delta: 0.05, // harmonic proximity tolerance
  qMax: 8,
};

// --- Knowledge entropy: normalised Shannon entropy of frequency distribution
export function knowledge(omega) {
  const N = omega.length;
  if (N === 0) return 0;
  if (N === 1) return Math.min(1, omega[0] / REF.omegaMax);
  const sum = omega.reduce((s, w) => s + w, 0);
  if (sum <= 0) return 0;
  let H = 0;
  for (const w of omega) {
    const p = w / sum;
    if (p > 0) H -= p * Math.log(p);
  }
  return H / Math.log(N);
}

// --- Temporal entropy: log-ratio of frequency span vs reference span
export function temporal(omega, bRot = null) {
  if (omega.length === 0) return 0;
  if (omega.length === 1) {
    const b = bRot ?? REF.bRotMin;
    return clamp01(
      Math.log(omega[0] / b) / Math.log(REF.omegaMax / REF.bRotMin)
    );
  }
  const wMax = Math.max(...omega);
  const wMin = Math.min(...omega);
  if (wMin <= 0) return 0;
  return clamp01(
    Math.log(wMax / wMin) / Math.log(REF.omegaMax / REF.omegaMin)
  );
}

// --- Evolution entropy: harmonic-edge density
export function evolution(omega, delta = REF.delta, qMax = REF.qMax) {
  const N = omega.length;
  if (N < 2) return 0;
  const nPairs = (N * (N - 1)) / 2;
  let nHarm = 0;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const ratio = Math.max(omega[i], omega[j]) / Math.min(omega[i], omega[j]);
      if (nearestRationalProximity(ratio, qMax) < delta) nHarm++;
    }
  }
  return nHarm / nPairs;
}

function nearestRationalProximity(x, qMax) {
  let best = Infinity;
  for (let q = 1; q <= qMax; q++) {
    for (let p = q; p <= qMax; p++) {
      const d = Math.abs(x - p / q);
      if (d < best) best = d;
    }
  }
  return best;
}

// --- Full coordinate triple
export function sEntropy(omega, bRot = null) {
  return {
    Sk: knowledge(omega),
    St: temporal(omega, bRot),
    Se: evolution(omega),
  };
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}
