// Tiny radix-2 1D + 2D FFT for the diffraction-pattern panel.
// Inputs and outputs are arrays of complex numbers as {re, im}.

function bitReverse(n, bits) {
  let r = 0;
  for (let i = 0; i < bits; i++) {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  return r;
}

export function fft1d(input) {
  const N = input.length;
  const bits = Math.log2(N);
  if (!Number.isInteger(bits)) throw new Error("FFT length must be power of 2");
  const a = input.map((c) => ({ re: c.re, im: c.im }));
  // Bit-reverse permutation
  for (let i = 0; i < N; i++) {
    const j = bitReverse(i, bits);
    if (j > i) [a[i], a[j]] = [a[j], a[i]];
  }
  for (let size = 2; size <= N; size *= 2) {
    const half = size / 2;
    const tableStep = (-2 * Math.PI) / size;
    for (let i = 0; i < N; i += size) {
      for (let k = 0; k < half; k++) {
        const ang = tableStep * k;
        const tre = Math.cos(ang) * a[i + k + half].re - Math.sin(ang) * a[i + k + half].im;
        const tim = Math.cos(ang) * a[i + k + half].im + Math.sin(ang) * a[i + k + half].re;
        a[i + k + half].re = a[i + k].re - tre;
        a[i + k + half].im = a[i + k].im - tim;
        a[i + k].re += tre;
        a[i + k].im += tim;
      }
    }
  }
  return a;
}

export function fft2d(grid) {
  // grid: rows of complex; returns 2D complex
  const rows = grid.length;
  const cols = grid[0].length;
  const out = grid.map((row) => fft1d(row));
  // FFT columns
  for (let c = 0; c < cols; c++) {
    const col = out.map((row) => row[c]);
    const fcol = fft1d(col);
    for (let r = 0; r < rows; r++) out[r][c] = fcol[r];
  }
  return out;
}

export function fftShift2d(grid) {
  const rows = grid.length;
  const cols = grid[0].length;
  const hr = Math.floor(rows / 2);
  const hc = Math.floor(cols / 2);
  const out = Array.from({ length: rows }, () => new Array(cols));
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const nr = (r + hr) % rows;
      const nc = (c + hc) % cols;
      out[nr][nc] = grid[r][c];
    }
  }
  return out;
}

export function magnitude2d(grid) {
  return grid.map((row) => row.map((c) => Math.sqrt(c.re * c.re + c.im * c.im)));
}
