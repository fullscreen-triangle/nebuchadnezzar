import { Suspense, useRef, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import { useGLTF, useAnimations, OrbitControls } from "@react-three/drei";
import WebGLBoundary from "./WebGLBoundary";

function Model() {
  const group = useRef();
  const { scene, animations } = useGLTF(
    "/nebuchadnezzar_cylinder_cuneiform_1096.glb"
  );
  const { actions } = useAnimations(animations, group);

  useEffect(() => {
    Object.values(actions).forEach((action) => {
      if (action) action.reset().play();
    });
  }, [actions]);

  return (
    <group ref={group}>
      <primitive object={scene} />
    </group>
  );
}

useGLTF.preload("/nebuchadnezzar_cylinder_cuneiform_1096.glb");

export default function ModelViewer() {
  return (
    <WebGLBoundary fallback={<div className="h-full w-full bg-dark" />}>
    <Canvas
      camera={{ position: [0, 1, 4], fov: 42 }}
      dpr={[1, 1.5]}
      gl={{
        antialias: true,
        alpha: true,
        powerPreference: "default",
        failIfMajorPerformanceCaveat: false,
      }}
    >
      <ambientLight intensity={0.35} />
      <directionalLight position={[4, 8, 4]}  intensity={1.6} color="#ffffff" />
      <directionalLight position={[-3, 2, -3]} intensity={0.5} color="#58E6D9" />
      <Suspense fallback={null}>
        <Model />
      </Suspense>
      <OrbitControls
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.7}
        minDistance={1.5}
        maxDistance={12}
        enableDamping
        dampingFactor={0.06}
      />
    </Canvas>
    </WebGLBoundary>
  );
}
