// Interleaved ternary encoding of (S_k, S_t, S_e) into a k-trit address.
// Reference: categorical-compound-database.tex, Section 4.2 + Algorithm 1.

// Encode a coordinate triple to k trits by interleaved refinement.
// trit position j refines dimension j mod 3:
//   0 -> S_k, 1 -> S_t, 2 -> S_e
export function encode({ Sk, St, Se }, k = 18) {
  const r = [Sk, St, Se];
  const trits = new Array(k);
  for (let j = 0; j < k; j++) {
    const dim = j % 3;
    const x = 3 * r[dim];
    let t = Math.floor(x);
    if (t > 2) t = 2;
    trits[j] = t;
    r[dim] = x - t;
  }
  return trits;
}

// Recover the cell-centre coordinate from a trit string.
export function decode(trits) {
  const r = [0, 0, 0];
  const depths = [0, 0, 0];
  for (let j = 0; j < trits.length; j++) {
    const dim = j % 3;
    depths[dim]++;
    r[dim] += (trits[j] + 0.5) / Math.pow(3, depths[dim]);
  }
  return { Sk: r[0], St: r[1], Se: r[2] };
}

// Length of the longest common prefix between two trit strings.
export function sharedPrefix(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return n;
}

// Maximum diagonal of a level-k cell.
export function cellDiameter(k) {
  return Math.sqrt(3) * Math.pow(3, -Math.floor(k / 3));
}

// Format a trit string as a single line, dimension-coloured.
export function formatTritLine(trits) {
  return trits.map((t, i) => `${["k", "t", "e"][i % 3]}${t}`).join(" ");
}
