// CPU fallback for PartitionShader. Evaluates the same partition observation
// function A(u; M) on a 2D canvas. Same math, no GPU.
import { useEffect, useRef } from "react";

export default function PartitionCanvas2D({ omega, coords, omegaRef = 4401 }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const img = ctx.createImageData(W, H);
    const sigma = 0.012 * (1 - 0.5 * coords.Sk);
    const tEnv = 0.1 + 0.4 * coords.St;

    for (let py = 0; py < H; py++) {
      const v = py / (H - 1);
      const dv = (v - 0.5) / tEnv;
      const envelope = Math.exp(-0.5 * dv * dv);
      for (let px = 0; px < W; px++) {
        const u = px / (W - 1);
        // Mode sum
        let s = 0;
        for (let i = 0; i < omega.length; i++) {
          const mu = omega[i] / omegaRef;
          const d = (u - mu) / sigma;
          s += Math.exp(-0.5 * d * d);
        }
        // Depth fringes (no time term in static fallback)
        const depthMod = 1 + 0.3 * coords.Se * Math.sin(50 * coords.Se * u);
        let a = s * envelope * depthMod;
        if (a > 1.5) a = 1.5;
        if (a < 0) a = 0;

        // Map to teal-on-dark.
        const t = Math.min(1, a / 1.2);
        const r = Math.round(0.105 * 255 + (0.345 - 0.105) * 255 * t);
        const g = Math.round(0.105 * 255 + (0.902 - 0.105) * 255 * t);
        const b = Math.round(0.105 * 255 + (0.851 - 0.105) * 255 * t);

        const idx = (py * W + px) * 4;
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [omega, coords, omegaRef]);

  return (
    <div className="aspect-[2/1] w-full overflow-hidden rounded-md border border-light/10 bg-dark/40">
      <canvas ref={ref} style={{ width: "100%", height: "100%" }} />
    </div>
  );
}
