// (n, ℓ, m, s) partition field for volume holography.
// Channels encode:
//   n(r): refractive index proxy  (RI tomography; RI ∝ dry mass density)
//   ℓ(r): phase gradient magnitude (membrane curvature)
//   m(r): phase orientation        (structural anisotropy)
//   s(r): chirality                (Hessian off-diagonal sign)
//
// Reference: distributed-control-system.tex §3 (S-entropy normalisation [0,100]).

// Organelle biophysical parameters from published RI tomography.
// ri: refractive index offset from cytoplasm (1.36 baseline)
// grad: membrane gradient magnitude [0,1]
// radius: typical radius in µm
// chirality: ±1
export const ORGANELLES = {
  nucleus:       { ri:  0.040, grad: 0.85, radius: 5.2,   chirality:  1 },
  nucleolus:     { ri:  0.060, grad: 0.90, radius: 1.8,   chirality:  1 },
  mitochondria:  { ri:  0.025, grad: 0.60, radius: 0.9,   chirality: -1 },
  er_rough:      { ri:  0.015, grad: 0.50, radius: 0.4,   chirality:  1 },
  er_smooth:     { ri:  0.010, grad: 0.32, radius: 0.3,   chirality: -1 },
  golgi:         { ri:  0.020, grad: 0.70, radius: 1.5,   chirality:  1 },
  lysosome:      { ri:  0.030, grad: 0.75, radius: 0.45,  chirality: -1 },
  lipid_droplet: { ri: -0.015, grad: 0.50, radius: 0.7,   chirality:  1 },
  ribosome:      { ri:  0.050, grad: 0.18, radius: 0.015, chirality:  1 },
};

// Per-cell-type organelle count profile
export const CELL_CONFIGS = {
  hepatocyte:    { nucleus: 1, nucleolus: 1, mitochondria: 22, er_rough: 12, er_smooth:  8, golgi: 2, lysosome: 15, lipid_droplet: 18, ribosome: 60 },
  neuron:        { nucleus: 1, nucleolus: 1, mitochondria: 35, er_rough:  8, er_smooth:  4, golgi: 1, lysosome:  5, lipid_droplet:  2, ribosome: 80 },
  cardiomyocyte: { nucleus: 1, nucleolus: 1, mitochondria: 45, er_rough:  6, er_smooth: 10, golgi: 2, lysosome:  6, lipid_droplet:  4, ribosome: 40 },
  cancer:        { nucleus: 1, nucleolus: 3, mitochondria: 12, er_rough: 20, er_smooth:  6, golgi: 3, lysosome: 25, lipid_droplet:  8, ribosome: 100 },
  macrophage:    { nucleus: 1, nucleolus: 1, mitochondria: 18, er_rough:  6, er_smooth:  4, golgi: 2, lysosome: 60, lipid_droplet:  3, ribosome: 30 },
  beta_cell:     { nucleus: 1, nucleolus: 1, mitochondria: 28, er_rough: 20, er_smooth:  5, golgi: 3, lysosome:  8, lipid_droplet:  6, ribosome: 70 },
};

function makeLcg(seed) {
  let s = (seed >>> 0) || 1;
  return () => { s = (Math.imul(1664525, s) + 1013904223) | 0; return (s >>> 0) / 0xffffffff; };
}
function hashStr(str) {
  return str.split("").reduce((h, c) => (Math.imul(31, h) + c.charCodeAt(0)) | 0, 5381) >>> 0;
}

// Build the (n, ℓ, m, s) volume as Uint8Array[size³ × 4].
// Returns { data, size, placed, organelleCount }
export function buildPartitionField(cellType = "hepatocyte", size = 64) {
  const cfg = CELL_CONFIGS[cellType] || CELL_CONFIGS.hepatocyte;
  const rand = makeLcg(hashStr(cellType));
  const cx = size * 0.5, cy = size * 0.5, cz = size * 0.5;
  const cellR = size * 0.44;
  const scale = cellR / 8; // 8 µm cell radius → voxels

  const N = size * size * size;
  const data = new Uint8Array(N * 4);

  // ── Step 1: cytoplasm baseline (inside cell sphere) ──────────────────────
  const cellR2 = cellR * cellR;
  for (let iz = 0; iz < size; iz++) {
    for (let iy = 0; iy < size; iy++) {
      for (let ix = 0; ix < size; ix++) {
        const dx = ix - cx, dy = iy - cy, dz = iz - cz;
        if (dx * dx + dy * dy + dz * dz > cellR2) continue;
        const base = (iz * size * size + iy * size + ix) * 4;
        data[base]     = 72;   // n ≈ 0.28 (cytoplasm RI offset)
        data[base + 1] = 18;   // ℓ ≈ 0.07 (low interior gradient)
        data[base + 2] = Math.floor(((Math.atan2(dy, dx) / Math.PI + 1) * 0.5) * 255);
        data[base + 3] = 128;  // s = neutral
      }
    }
  }

  // ── Step 2: cell membrane gradient highlight ──────────────────────────────
  const memThick = Math.max(1, cellR * 0.06);
  for (let iz = 0; iz < size; iz++) {
    for (let iy = 0; iy < size; iy++) {
      for (let ix = 0; ix < size; ix++) {
        const dx = ix - cx, dy = iy - cy, dz = iz - cz;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (Math.abs(r - cellR) > memThick) continue;
        const base = (iz * size * size + iy * size + ix) * 4;
        if (data[base] === 0) continue;
        const edge = Math.exp(-0.5 * ((r - cellR) / (memThick * 0.4)) ** 2);
        data[base + 1] = Math.min(255, data[base + 1] + Math.floor(edge * 200));
      }
    }
  }

  // ── Step 3: place organelles (bounding-box rasterisation) ────────────────
  const placed = [];
  for (const [type, count] of Object.entries(cfg)) {
    const params = ORGANELLES[type];
    if (!params) continue;
    const r_vox = Math.max(1.5, params.radius * scale);
    const isNucleus = type === "nucleus" || type === "nucleolus";
    const nVal = Math.max(0, Math.min(255, Math.floor((0.5 + params.ri * 7) * 255)));
    const lVal = Math.floor(params.grad * 255);
    const sVal = params.chirality > 0 ? 200 : 56;

    for (let i = 0; i < count; i++) {
      let ox = 0, oy = 0, oz = 0, tries = 0;
      do {
        const bound = isNucleus ? cellR * 0.28 : cellR - r_vox * 2;
        ox = (rand() - 0.5) * 2 * bound;
        oy = (rand() - 0.5) * 2 * bound;
        oz = (rand() - 0.5) * 2 * bound;
        tries++;
      } while (Math.sqrt(ox * ox + oy * oy + oz * oz) > cellR - r_vox * 1.5 && tries < 80);
      if (tries >= 80) continue;

      placed.push({ x: cx + ox, y: cy + oy, z: cz + oz, r: r_vox, type, ...params });

      // Rasterise Gaussian kernel over bounding box
      const pad = r_vox * 2.8;
      const x0 = Math.max(0, Math.floor(cx + ox - pad)), x1 = Math.min(size - 1, Math.ceil(cx + ox + pad));
      const y0 = Math.max(0, Math.floor(cy + oy - pad)), y1 = Math.min(size - 1, Math.ceil(cy + oy + pad));
      const z0 = Math.max(0, Math.floor(cz + oz - pad)), z1 = Math.min(size - 1, Math.ceil(cz + oz + pad));
      const sig2 = (r_vox * 0.65) ** 2;

      for (let iz = z0; iz <= z1; iz++) {
        for (let iy = y0; iy <= y1; iy++) {
          for (let ix = x0; ix <= x1; ix++) {
            const ex = ix - (cx + ox), ey = iy - (cy + oy), ez = iz - (cz + oz);
            const d2 = ex * ex + ey * ey + ez * ez;
            const w = Math.exp(-0.5 * d2 / sig2);
            if (w < 0.04) continue;
            const base = (iz * size * size + iy * size + ix) * 4;
            if (data[base] === 0) continue; // outside cell
            data[base]     = Math.min(255, data[base]     + ((nVal - data[base])     * w) | 0);
            data[base + 1] = Math.min(255, data[base + 1] + ((lVal - data[base + 1]) * w) | 0);
            const phi = Math.floor(((Math.atan2(ey, ex) / Math.PI + 1) * 0.5) * 255);
            data[base + 2] = Math.min(255, data[base + 2] + ((phi - data[base + 2]) * w) | 0);
            data[base + 3] = Math.min(255, data[base + 3] + ((sVal - data[base + 3]) * w) | 0);
          }
        }
      }
    }
  }

  return { data, size, placed, organelleCount: placed.length };
}

// Average field channels over cell interior → S-entropy proxy coordinates
export function fieldToSEntropy(data, size) {
  const cx = size * 0.5, cellR2 = (size * 0.44) ** 2;
  let sn = 0, sl = 0, sm = 0, ss = 0, count = 0;
  for (let iz = 0; iz < size; iz++) {
    for (let iy = 0; iy < size; iy++) {
      for (let ix = 0; ix < size; ix++) {
        const dx = ix - cx, dy = iy - cx, dz = iz - cx;
        if (dx * dx + dy * dy + dz * dz > cellR2) continue;
        const base = (iz * size * size + iy * size + ix) * 4;
        if (data[base] === 0 && data[base + 1] === 0) continue;
        sn += data[base] / 255;
        sl += data[base + 1] / 255;
        sm += data[base + 2] / 255;
        ss += data[base + 3] / 255;
        count++;
      }
    }
  }
  const N = count || 1;
  return { Sk: sn / N, St: sl / N, Se: sm / N, Schir: ss / N };
}

// Per-organelle Kuramoto natural frequency derived from RI + gradient
// (distributed-control-system.tex §3.2: ΔP ↔ oscillator deviation)
export function organelleFrequencies(placed) {
  return placed.map((o) => 1.0 + (o.ri / 0.06) * 0.35 + (o.grad - 0.5) * 0.15);
}
