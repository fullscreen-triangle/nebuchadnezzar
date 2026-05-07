/**
 * SignatureCube — 3-D disease-signature space (D, R, I_norm) via R3F.
 * Follows the SEntropyCube architecture exactly.
 * Axes: X = hierarchical depth D, Y = Kuramoto R, Z = I_tot normalised.
 */
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useMemo, useRef, useEffect, useState } from "react";
import { DISEASE_SIGNATURES } from "@/lib/metabolism";
import { detectWebGL } from "@/lib/webgl";
import WebGLBoundary from "./WebGLBoundary";
import SignatureSVG from "./SignatureSVG";

function CubeFrame() {
  const edges = useMemo(() => {
    const c = [
      [0,0,0],[1,0,0],[1,1,0],[0,1,0],
      [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ];
    const idx = [
      [0,1],[1,2],[2,3],[3,0],
      [4,5],[5,6],[6,7],[7,4],
      [0,4],[1,5],[2,6],[3,7],
    ];
    const pts = [];
    for (const [a, b] of idx) pts.push(...c[a], ...c[b]);
    return new Float32Array(pts);
  }, []);

  return (
    <lineSegments>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={edges.length / 3}
          array={edges}
          itemSize={3}
        />
      </bufferGeometry>
      <lineBasicMaterial color="#3a3a3a" />
    </lineSegments>
  );
}

function AxisTick({ position, color }) {
  return (
    <mesh position={position}>
      <boxGeometry args={[0.018, 0.018, 0.018]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

function DiseasePoints() {
  return (
    <group>
      {DISEASE_SIGNATURES.map((sig) => (
        <mesh key={sig.name} position={[sig.D, sig.R, sig.ItotN]}>
          <sphereGeometry args={[0.030, 14, 14]} />
          <meshBasicMaterial color={sig.color} transparent opacity={0.80} />
        </mesh>
      ))}
    </group>
  );
}

function UserPoint({ query }) {
  const ref = useRef();
  useFrame(({ clock }) => {
    if (ref.current) {
      const s = 1 + 0.22 * Math.sin(clock.getElapsedTime() * 3.2);
      ref.current.scale.set(s, s, s);
    }
  });
  if (!query) return null;
  return (
    <mesh ref={ref} position={[query.D, query.R, query.ItotN]}>
      <sphereGeometry args={[0.036, 20, 20]} />
      <meshBasicMaterial color="#58E6D9" />
    </mesh>
  );
}

export default function SignatureCube({ query }) {
  const [gl, setGl] = useState({ supported: true });
  useEffect(() => setGl(detectWebGL()), []);

  if (!gl.supported) return <SignatureSVG query={query} />;

  return (
    <WebGLBoundary fallback={<SignatureSVG query={query} />}>
      <div className="aspect-square w-full overflow-hidden rounded-md border border-light/10 bg-dark/40">
        <Canvas
          camera={{ position: [1.7, 1.5, 1.9], fov: 44 }}
          dpr={[1, 2]}
          gl={{
            antialias: true,
            alpha: false,
            powerPreference: "default",
            failIfMajorPerformanceCaveat: false,
            preserveDrawingBuffer: false,
          }}
          onCreated={({ gl }) => gl.setClearColor("#1b1b1b")}
        >
          <group position={[-0.5, -0.5, -0.5]}>
            <CubeFrame />
            <DiseasePoints />
            <UserPoint query={query} />
            {/* Axis tip markers */}
            <AxisTick position={[1, 0, 0]} color="#E74C3C" />   {/* D */}
            <AxisTick position={[0, 1, 0]} color="#2ECC71" />   {/* R */}
            <AxisTick position={[0, 0, 1]} color="#4477AA" />   {/* I */}
          </group>
          <OrbitControls
            enablePan={false}
            minDistance={1.6}
            maxDistance={4.2}
            autoRotate
            autoRotateSpeed={0.5}
          />
        </Canvas>
      </div>
    </WebGLBoundary>
  );
}
