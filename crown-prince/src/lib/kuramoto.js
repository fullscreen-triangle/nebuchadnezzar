// Kuramoto oscillator model for organelle phase coupling.
// Reference: swarm-federations.tex §3 (Theorems 3.1, 3.2).

const TWO_PI = 2 * Math.PI;

// RK4 step of Kuramoto mean-field equation:
//   φ̇ᵢ = ωᵢ + K·R·sin(ψ - φᵢ)
export function kuramotoStep(phases, omegas, K, dt) {
  const n = phases.length;
  const deriv = (ph) => {
    const { Rens, psi } = orderParameter(ph);
    return ph.map((p, i) => omegas[i] + K * Rens * Math.sin(psi - p));
  };
  const k1 = deriv(phases);
  const ph2 = phases.map((p, i) => p + k1[i] * dt / 2);
  const k2 = deriv(ph2);
  const ph3 = phases.map((p, i) => p + k2[i] * dt / 2);
  const k3 = deriv(ph3);
  const ph4 = phases.map((p, i) => p + k3[i] * dt);
  const k4 = deriv(ph4);
  return phases.map((p, i) => {
    const next = p + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * dt / 6;
    return ((next % TWO_PI) + TWO_PI) % TWO_PI;
  });
}

// Ensemble order parameter  R e^{iψ}
export function orderParameter(phases) {
  let sc = 0, ss = 0;
  for (const p of phases) { sc += Math.cos(p); ss += Math.sin(p); }
  const n = phases.length;
  return { Rens: Math.sqrt(sc * sc + ss * ss) / n, psi: Math.atan2(ss, sc) };
}

// Critical coupling  Kc = 2σω/π  (swarm-federations Theorem 3.1)
export function criticalCoupling(omegas) {
  const n = omegas.length;
  const mean = omegas.reduce((a, b) => a + b, 0) / n;
  const sigmaOmega = Math.sqrt(omegas.reduce((s, w) => s + (w - mean) ** 2, 0) / n);
  return { Kc: (2 * sigmaOmega) / Math.PI, sigmaOmega, mean };
}

// Mean-field fixed point  R* = sqrt(1 - Kc/K)  for K > Kc
export function meanFieldRstar(K, Kc) {
  return K <= Kc ? 0 : Math.sqrt(Math.max(0, 1 - Kc / K));
}

// Five-regime classification (swarm-federations Theorem 3.2)
export function classifyRegime(Rens) {
  if (Rens < 0.30) return { index: 1, label: "Turbulent",            cost: "O(R⁻²)",         color: "#EE6677" };
  if (Rens < 0.50) return { index: 2, label: "Aperture-Dominated",   cost: "O(n²/R)",         color: "#CCBB44" };
  if (Rens < 0.80) return { index: 3, label: "Hierarchical Cascade", cost: "O(n log n / R²)", color: "#AA3377" };
  if (Rens < 0.95) return { index: 4, label: "Coherent",             cost: "O(log n)",        color: "#4477AA" };
  return             { index: 5, label: "Phase-Locked",               cost: "0",               color: "#58E6D9" };
}

// Simulate Kuramoto dynamics. Returns Rens trajectory + final state.
export function runKuramoto(n, K, steps = 120, dt = 0.06, seed = 42) {
  let s = (seed >>> 0) || 1;
  const lcg = () => { s = (Math.imul(1664525, s) + 1013904223) | 0; return (s >>> 0) / 0xffffffff; };
  const gauss = () => Math.sqrt(-2 * Math.log(lcg() || 1e-10)) * Math.cos(TWO_PI * lcg());
  const omegas = Array.from({ length: n }, gauss);
  let phases = Array.from({ length: n }, () => lcg() * TWO_PI);
  const { Kc, sigmaOmega } = criticalCoupling(omegas);
  const trajectory = [];
  for (let i = 0; i < steps; i++) {
    trajectory.push(orderParameter(phases).Rens);
    phases = kuramotoStep(phases, omegas, K, dt);
  }
  const { Rens: finalRens } = orderParameter(phases);
  return { trajectory, phases, omegas, Kc, sigmaOmega, finalRens };
}

// T(m, D) = D·(D+1)^(m−1)  (swarm-federations Theorem 5.3)
export function trajectoryCount(m, D) {
  return D * Math.pow(D + 1, m - 1);
}
