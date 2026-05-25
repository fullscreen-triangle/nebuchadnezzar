import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useMemo, useRef, useState, useEffect } from "react";
import * as THREE from "three";
import { detectWebGL } from "@/lib/webgl";
import WebGLBoundary, { GLFallback } from "./WebGLBoundary";

// ── GLSL 3 ES shaders ─────────────────────────────────────────────────────────
// Three.js prepends "#version 300 es" when glslVersion = THREE.GLSL3.
// Must use: out/in instead of varying, out vec4 fragColor instead of gl_FragColor.

const VERT = `
  out vec3 vLocalPos;
  void main() {
    vLocalPos = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const FRAG = `
  precision highp sampler3D;

  uniform sampler3D uVolume;
  uniform vec3 uCamLocalPos;
  uniform float uSteps;
  uniform float uAlpha;
  uniform int uChannel;

  in vec3 vLocalPos;
  out vec4 fragColor;

  // Slab ray-box intersection for unit box [-0.5, 0.5]^3
  vec2 boxHit(vec3 ro, vec3 rd) {
    vec3 m = 1.0 / rd;
    vec3 n = m * ro;
    vec3 k = abs(m) * 0.5;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    return vec2(max(max(t1.x, t1.y), t1.z),
                min(min(t2.x, t2.y), t2.z));
  }

  void main() {
    vec3 rd = normalize(vLocalPos - uCamLocalPos);
    vec2 t  = boxHit(uCamLocalPos, rd);
    if (t.y < t.x || t.y < 0.0) discard;

    float tStart = max(t.x, 0.0);
    float stepSz = (t.y - tStart) / uSteps;
    vec3  pos    = uCamLocalPos + tStart * rd;

    vec4 result = vec4(0.0);

    for (int i = 0; i < 256; i++) {
      if (float(i) >= uSteps) break;

      vec3 tc = pos + 0.5; // [-0.5,0.5] → [0,1]
      if (any(lessThan(tc, vec3(0.001))) ||
          any(greaterThan(tc, vec3(0.999)))) break;

      vec4 s = texture(uVolume, tc);

      if (s.r > 0.01 || s.g > 0.02) {
        float val;
        if      (uChannel == 1) val = s.r;
        else if (uChannel == 2) val = s.g;
        else if (uChannel == 3) val = s.b;
        else if (uChannel == 4) val = s.a;
        else val = s.r * 0.35 + s.g * 0.40 + s.b * 0.15 + s.a * 0.10;

        float alpha = clamp(val * uAlpha * 18.0 * stepSz, 0.0, 1.0);

        vec3 col;
        if (uChannel == 0) {
          col = vec3(0.12 + s.b * 0.78, 0.12 + s.r * 0.72, 0.20 + (1.0 - s.b) * 0.68);
        } else if (uChannel == 2) {
          col = mix(vec3(0.1, 0.1, 0.1), vec3(0.90, 0.73, 0.26), val);
        } else {
          col = mix(vec3(0.08, 0.55, 0.53), vec3(0.67, 0.20, 0.47), val);
        }

        result.rgb += (1.0 - result.a) * alpha * col;
        result.a   += (1.0 - result.a) * alpha;
        if (result.a > 0.99) break;
      }
      pos += rd * stepSz;
    }

    if (result.a < 0.005) discard;
    fragColor = result;
  }
`;

// 1×1×1 transparent placeholder — keeps uVolume non-null before real data arrives
function makeDummyTexture() {
  const t = new THREE.Data3DTexture(new Uint8Array(4), 1, 1, 1);
  t.format = THREE.RGBAFormat;
  t.type   = THREE.UnsignedByteType;
  t.needsUpdate = true;
  return t;
}

// ── Inner mesh component ──────────────────────────────────────────────────────
function RaymarchMesh({ volume, channel, alpha, steps }) {
  const meshRef  = useRef();
  const matRef   = useRef();
  const camLocal = useMemo(() => new THREE.Vector3(), []);
  const invMat   = useMemo(() => new THREE.Matrix4(), []);

  const texture = useMemo(() => {
    if (!volume) return null;
    const { data, size } = volume;
    const tex = new THREE.Data3DTexture(data, size, size, size);
    tex.format         = THREE.RGBAFormat;
    tex.type           = THREE.UnsignedByteType;
    tex.minFilter      = THREE.LinearFilter;
    tex.magFilter      = THREE.LinearFilter;
    tex.unpackAlignment = 1;
    tex.needsUpdate    = true;
    return tex;
  }, [volume]);

  // Uniforms created once; real texture is injected via useEffect after mount.
  const uniforms = useMemo(() => ({
    uVolume:      { value: makeDummyTexture() },
    uCamLocalPos: { value: new THREE.Vector3() },
    uSteps:       { value: steps },
    uAlpha:       { value: alpha },
    uChannel:     { value: channel },
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }), []);

  // Sync volume texture after commit (matRef.current is available by then)
  useEffect(() => {
    if (!matRef.current) return;
    matRef.current.uniforms.uVolume.value = texture ?? makeDummyTexture();
  }, [texture]);

  // Sync scalar uniforms
  useEffect(() => {
    if (!matRef.current) return;
    matRef.current.uniforms.uSteps.value   = steps;
    matRef.current.uniforms.uAlpha.value   = alpha;
    matRef.current.uniforms.uChannel.value = channel;
  }, [steps, alpha, channel]);

  // Per-frame: transform camera position into model space
  useFrame(({ camera }) => {
    if (!matRef.current || !meshRef.current) return;
    invMat.copy(meshRef.current.matrixWorld).invert();
    camera.getWorldPosition(camLocal);
    camLocal.applyMatrix4(invMat);
    matRef.current.uniforms.uCamLocalPos.value.copy(camLocal);
  });

  return (
    <mesh ref={meshRef}>
      <boxGeometry args={[1, 1, 1]} />
      <shaderMaterial
        ref={matRef}
        uniforms={uniforms}
        vertexShader={VERT}
        fragmentShader={FRAG}
        glslVersion={THREE.GLSL3}
        side={THREE.DoubleSide}
        transparent
        depthWrite={false}
      />
    </mesh>
  );
}

// ── Public component ──────────────────────────────────────────────────────────
export default function VolumeRaymarch({ volume, channel = 0, alpha = 1.2, steps = 96 }) {
  const [gl, setGl] = useState({ supported: true, version: 2 });
  useEffect(() => setGl(detectWebGL()), []);

  const fallback = (
    <GLFallback message="Volume rendering requires WebGL 2." aspect="square" />
  );

  if (!gl.supported || gl.version < 2) return fallback;

  return (
    <WebGLBoundary fallback={fallback}>
      <div className="aspect-square w-full overflow-hidden rounded-md border border-light/10 bg-dark/40">
        <Canvas
          camera={{ position: [1.6, 1.2, 1.8], fov: 40 }}
          dpr={[1, 1.5]}
          gl={{
            antialias: false,
            alpha: false,
            powerPreference: "default",
            failIfMajorPerformanceCaveat: false,
          }}
          onCreated={({ gl }) => gl.setClearColor("#111111")}
        >
          <RaymarchMesh volume={volume} channel={channel} alpha={alpha} steps={steps} />
          <OrbitControls
            enablePan={false}
            minDistance={1.2}
            maxDistance={4.0}
            autoRotate
            autoRotateSpeed={0.5}
          />
        </Canvas>
      </div>
    </WebGLBoundary>
  );
}
