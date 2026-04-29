import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef, useState, useEffect } from "react";
import * as THREE from "three";
import { detectWebGL } from "@/lib/webgl";
import WebGLBoundary from "./WebGLBoundary";
import PartitionCanvas2D from "./PartitionCanvas2D";

// Fragment shader: the partition observation function from Model V
// (cheminformatics-model.tex Definition 6.2).
//
//   A(u; M) = E(u; St) * D(u; Se) * sum_i G(u, omega_i / omega_ref, sigma(Sk))
//
// The shader receives up to MAX_MODES vibrational modes packed into a uniform
// array, plus the (Sk, St, Se) triple. It writes the observed intensity at
// every pixel — i.e. it performs W*H simultaneous partition observations in
// one draw call.

const MAX_MODES = 32;

const FRAG = `
  precision highp float;
  varying vec2 vUv;
  uniform float uModes[${MAX_MODES}];
  uniform int uModeCount;
  uniform float uOmegaRef;
  uniform float uSk;
  uniform float uSt;
  uniform float uSe;
  uniform float uTime;

  float gaussian(float u, float mu, float sigma) {
    float d = (u - mu) / sigma;
    return exp(-0.5 * d * d);
  }

  void main() {
    float u = vUv.x;
    float v = vUv.y;

    // Cell width: narrows with high knowledge entropy.
    float sigma = 0.012 * (1.0 - 0.5 * uSk);

    // Mode sum.
    float modeSum = 0.0;
    for (int i = 0; i < ${MAX_MODES}; i++) {
      if (i >= uModeCount) break;
      float mu = uModes[i] / uOmegaRef;
      modeSum += gaussian(u, mu, sigma);
    }

    // Temporal envelope.
    float envelope = gaussian(v, 0.5, 0.1 + 0.4 * uSt);

    // Partition depth fringes.
    float depthMod = 1.0 + 0.3 * uSe * sin(50.0 * uSe * u + uTime * 0.5);

    float a = modeSum * envelope * depthMod;
    a = clamp(a, 0.0, 1.5);

    // Map intensity to teal-on-dark.
    vec3 dark = vec3(0.105, 0.105, 0.105);
    vec3 teal = vec3(0.345, 0.902, 0.851);
    vec3 col = mix(dark, teal, smoothstep(0.0, 1.2, a));

    gl_FragColor = vec4(col, 1.0);
  }
`;

const VERT = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

function ObservationPlane({ omega, coords, omegaRef = 4401 }) {
  const matRef = useRef();
  const padded = useMemo(() => {
    const arr = new Array(MAX_MODES).fill(0);
    for (let i = 0; i < Math.min(omega.length, MAX_MODES); i++) arr[i] = omega[i];
    return arr;
  }, [omega]);

  const uniforms = useMemo(
    () => ({
      uModes: { value: padded },
      uModeCount: { value: Math.min(omega.length, MAX_MODES) },
      uOmegaRef: { value: omegaRef },
      uSk: { value: coords.Sk },
      uSt: { value: coords.St },
      uSe: { value: coords.Se },
      uTime: { value: 0 },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  // Update on prop changes without re-creating uniforms.
  if (matRef.current) {
    matRef.current.uniforms.uModes.value = padded;
    matRef.current.uniforms.uModeCount.value = Math.min(omega.length, MAX_MODES);
    matRef.current.uniforms.uOmegaRef.value = omegaRef;
    matRef.current.uniforms.uSk.value = coords.Sk;
    matRef.current.uniforms.uSt.value = coords.St;
    matRef.current.uniforms.uSe.value = coords.Se;
  }

  useFrame(({ clock }) => {
    if (matRef.current) matRef.current.uniforms.uTime.value = clock.getElapsedTime();
  });

  return (
    <mesh>
      <planeGeometry args={[2, 2]} />
      <shaderMaterial
        ref={matRef}
        uniforms={uniforms}
        vertexShader={VERT}
        fragmentShader={FRAG}
      />
    </mesh>
  );
}

export default function PartitionShader({ omega, coords }) {
  const [gl, setGl] = useState({ supported: true, version: 2 });
  useEffect(() => setGl(detectWebGL()), []);

  const fallback = <PartitionCanvas2D omega={omega} coords={coords} />;

  if (!gl.supported) return fallback;

  return (
    <WebGLBoundary fallback={fallback}>
      <div className="aspect-[2/1] w-full overflow-hidden rounded-md border border-light/10 bg-dark/40">
        <Canvas
          orthographic
          camera={{ zoom: 100, position: [0, 0, 1] }}
          dpr={[1, 2]}
          gl={{
            antialias: false,
            alpha: false,
            powerPreference: "default",
            failIfMajorPerformanceCaveat: false,
            preserveDrawingBuffer: false,
          }}
          onCreated={({ gl }) => gl.setClearColor("#1b1b1b")}
        >
          <ObservationPlane omega={omega} coords={coords} />
        </Canvas>
      </div>
    </WebGLBoundary>
  );
}
