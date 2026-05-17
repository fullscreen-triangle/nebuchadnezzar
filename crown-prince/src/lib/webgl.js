// Detect WebGL availability without mounting a Canvas component.
export function detectWebGL() {
  if (typeof window === "undefined") return { supported: false, version: 0 };
  try {
    const canvas = document.createElement("canvas");
    const gl2 =
      canvas.getContext("webgl2") || canvas.getContext("experimental-webgl2");
    if (gl2) return { supported: true, version: 2 };
    const gl1 =
      canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    if (gl1) return { supported: true, version: 1 };
    return { supported: false, version: 0 };
  } catch (e) {
    return { supported: false, version: 0 };
  }
}
