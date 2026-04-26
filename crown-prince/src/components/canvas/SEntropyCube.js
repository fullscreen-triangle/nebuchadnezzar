import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useMemo, useRef } from "react";
import { COMPOUNDS, TYPE_COLOR } from "@/lib/compounds";
import { sEntropy } from "@/lib/sentropy";

// Pre-compute reference cloud once.
const REFERENCE_POINTS = COMPOUNDS.map((c) => {
  const s = sEntropy(c.omega, c.bRot);
  return { name: c.name, type: c.type, pos: [s.Sk, s.St, s.Se] };
});

function CubeFrame() {
  const edges = useMemo(() => {
    const corners = [
      [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
      [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ];
    const idx = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    const points = [];
    for (const [a, b] of idx) {
      points.push(...corners[a], ...corners[b]);
    }
    return new Float32Array(points);
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
      <lineBasicMaterial color="#4a4a4a" />
    </lineSegments>
  );
}

function ReferenceCloud() {
  return (
    <group>
      {REFERENCE_POINTS.map((p) => (
        <mesh key={p.name} position={p.pos}>
          <sphereGeometry args={[0.012, 12, 12]} />
          <meshBasicMaterial color={TYPE_COLOR[p.type]} transparent opacity={0.7} />
        </mesh>
      ))}
    </group>
  );
}

function QueryPoint({ coords }) {
  const ref = useRef();
  useFrame(({ clock }) => {
    if (ref.current) {
      const s = 1 + 0.2 * Math.sin(clock.getElapsedTime() * 3);
      ref.current.scale.set(s, s, s);
    }
  });
  if (!coords) return null;
  return (
    <mesh ref={ref} position={[coords.Sk, coords.St, coords.Se]}>
      <sphereGeometry args={[0.025, 24, 24]} />
      <meshBasicMaterial color="#58E6D9" />
    </mesh>
  );
}

function AxisLabel({ position, text, color = "#888" }) {
  // Note: drei's Text would be cleaner, but we keep deps minimal.
  // Use a simple coloured tick instead.
  return (
    <mesh position={position}>
      <boxGeometry args={[0.02, 0.02, 0.02]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

export default function SEntropyCube({ query }) {
  return (
    <div className="aspect-square w-full overflow-hidden rounded-md border border-light/10 bg-dark/40">
      <Canvas
        camera={{ position: [1.6, 1.4, 1.8], fov: 45 }}
        dpr={[1, 2]}
      >
        <group position={[-0.5, -0.5, -0.5]}>
          <CubeFrame />
          <ReferenceCloud />
          <QueryPoint coords={query} />
          <AxisLabel position={[1, 0, 0]} color="#EE6677" />
          <AxisLabel position={[0, 1, 0]} color="#228833" />
          <AxisLabel position={[0, 0, 1]} color="#4477AA" />
        </group>
        <OrbitControls
          enablePan={false}
          minDistance={1.4}
          maxDistance={4}
          autoRotate
          autoRotateSpeed={0.6}
        />
      </Canvas>
    </div>
  );
}
