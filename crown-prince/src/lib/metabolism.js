/**
 * Partitioned Metabolism Engine — math library.
 * All formulas from the paper; no fitted parameters.
 */

const r = (v, d) => Math.round(v * 10 ** d) / 10 ** d;

// ─── Capacitances (F) ────────────────────────────────────────────────────────
export const C_BRAIN = 1.0e-3;
export const C_MOTOR = 141.0e-6;
export const C_PERC  = 500.0e-6;

// ─── Information weights per level ───────────────────────────────────────────
export const ALPHA = [1.0, 0.9, 0.8, 0.7, 0.6];

// Output-to-input flux compression ratio per level (OxPhos efficiency etc.)
const COMPRESS = [0.95, 0.90, 0.85, 0.80, 0.75];

// Max possible I_tot (all 5 levels active) — used for normalisation
export const I_TOT_MAX = ALPHA.reduce(
  (s, a, i) => s + a * Math.log2(1 / COMPRESS[i]),
  0
);

// ─── BMR ─────────────────────────────────────────────────────────────────────
/** Mifflin–St Jeor formula → watts */
export function computeBMR(weight, height, age, sex) {
  const kcal = 10 * weight + 6.25 * height - 5 * age + (sex === "M" ? 5 : -161);
  return r((kcal * 4184) / 86400, 2);
}

// ─── Charge decomposition ────────────────────────────────────────────────────
/** Q = sqrt(2 C P) in mC/s (Δt = 1 s) */
function q(C, P) {
  return r(Math.sqrt(2 * C * Math.max(P, 0)) * 1000, 2);
}

/**
 * @param {{ bmr, rmssd, heartRate, cadence, stepLength, peakForce }} params
 * @returns {{ Qb, Qm, Qp, Qt, Qd, ratio, ratioTarget, fPerc, kappa, Pcog, Pbrain, Ploc }}
 */
export function computeCharges({ bmr, rmssd, heartRate, cadence, stepLength, peakForce }) {
  const Pbrain = 0.20 * bmr;
  const Pb     = 0.50 * Pbrain;
  const Pcog   = 0.50 * Pbrain;
  const Pd     = 0.95 * Pcog;

  const fCard    = heartRate / 60.0;
  const kappa    = fCard * (rmssd / 1000);       // rmssd in ms → s
  const kappaRef = 0.060;
  const fPerc    = Math.min(0.60, Math.max(0.20, r(0.40 * kappa / kappaRef, 3)));
  const Pp       = fPerc * Pcog;

  const Ploc = Math.min(300, 0.5 * peakForce * stepLength * (cadence / 60));

  const Qt = q(C_BRAIN, Pcog);
  const Qd = q(C_BRAIN, Pd);

  return {
    Qb:          q(C_BRAIN, Pb),
    Qm:          q(C_MOTOR, Ploc),
    Qp:          q(C_PERC,  Pp),
    Qt,
    Qd,
    ratio:       r(Qd / Qt, 4),
    ratioTarget: r(Math.sqrt(0.95), 4),
    fPerc,
    kappa:       r(kappa, 4),
    Pcog:        r(Pcog, 2),
    Pbrain:      r(Pbrain, 2),
    Ploc:        r(Ploc, 1),
  };
}

// ─── Hierarchical depth ───────────────────────────────────────────────────────
const THRESH = 0.10;

/**
 * @param {number[]} proxies – five values in [0, 1] for L1–L5
 */
export function computeDepth(proxies) {
  const active = proxies.map((v) => v > THRESH);
  const D      = r(active.filter(Boolean).length / 5, 2);
  const Itot   = r(
    proxies.reduce((s, v, i) => s + (active[i] ? ALPHA[i] * Math.log2(1 / COMPRESS[i]) : 0), 0),
    4
  );
  return { D, active, proxies, Itot };
}

// ─── Mirror law ───────────────────────────────────────────────────────────────
/**
 * @param {{ activityMetH, deepMin, remMin, sleepEff }} params
 */
export function computeMirror({ activityMetH, deepMin, remMin, sleepEff }) {
  const Eday   = Math.max(activityMetH, 0.01);
  const Cnight = 0.1 * (2.5 * (deepMin / 60) + 2.0 * (remMin / 60)) * sleepEff;
  const mu     = r(Cnight / Eday, 3);
  const stable = mu >= 0.8 && mu <= 1.2;
  return { mu, stable, error: r(stable ? 0 : Math.abs(mu - 1), 3) };
}

// ─── Drug efficacy ────────────────────────────────────────────────────────────
export function computeEfficacy(Dpre, Dpost) {
  if (Dpre >= 1.0) return 0;
  return r((Dpost - Dpre) / (1 - Dpre), 4);
}

// ─── Disease signatures ───────────────────────────────────────────────────────
export const DISEASE_SIGNATURES = [
  { name: "Healthy",            D: 1.0, R: 0.74, ItotN: 1.00, color: "#2ECC71" },
  { name: "Metabolic Syndrome", D: 0.4, R: 0.52, ItotN: 0.14, color: "#E74C3C" },
  { name: "Type 2 Diabetes",    D: 0.6, R: 0.60, ItotN: 0.40, color: "#F39C12" },
  { name: "Parkinson's",        D: 0.8, R: 0.74, ItotN: 0.60, color: "#9B59B6" },
  { name: "Alzheimer's",        D: 0.6, R: 0.60, ItotN: 0.45, color: "#E67E22" },
  { name: "Major Depression",   D: 0.8, R: 0.25, ItotN: 0.55, color: "#3498DB" },
];

export function prodromeMatch(D, R, ItotN) {
  return DISEASE_SIGNATURES.map((sig) => {
    const dist = Math.sqrt(
      (D - sig.D) ** 2 + (R - sig.R) ** 2 + (ItotN - sig.ItotN) ** 2
    );
    return { ...sig, dist: r(dist, 4) };
  }).sort((a, b) => a.dist - b.dist);
}

// ─── Reference interventions ──────────────────────────────────────────────────
export const INTERVENTIONS = [
  { name: "GLP-1",     Dpre: 0.4, Dpost: 0.40, color: "#9B59B6" },
  { name: "SGLT2i",    Dpre: 0.4, Dpost: 0.52, color: "#95A5A6" },
  { name: "Metformin", Dpre: 0.4, Dpost: 0.80, color: "#3498DB" },
  { name: "Lifestyle", Dpre: 0.4, Dpost: 0.72, color: "#F1C40F" },
  { name: "Exercise",  Dpre: 0.4, Dpost: 0.95, color: "#1ABC9C" },
  { name: "Fasting",   Dpre: 0.4, Dpost: 0.68, color: "#D35400" },
].map((d) => ({ ...d, eta: computeEfficacy(d.Dpre, d.Dpost) }));
