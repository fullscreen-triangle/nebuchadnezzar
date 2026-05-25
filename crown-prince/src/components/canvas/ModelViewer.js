import { Suspense, useRef, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import { useGLTF, useAnimations, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import WebGLBoundary from "./WebGLBoundary";

function Model() {
  const group = useRef();
  const { scene, animations } = useGLTF(
    "/nebuchadnezzar_cylinder_cuneiform_1096.glb"
  );
  const { actions } = useAnimations(animations, group);

  useEffect(() => {
    // Fit the model to a 2-unit bounding sphere regardless of original scale/origin
    const box = new THREE.Box3().setFromObject(scene);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = maxDim > 0 ? 2.0 / maxDim : 1;
    // Translate center to origin, then apply uniform scale
    scene.position.copy(center).negate().multiplyScalar(scale);
    scene.scale.setScalar(scale);

    Object.values(actions).forEach((a) => { if (a) a.reset().play(); });
  }, [scene, actions]);

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
        camera={{ position: [0, 1.2, 3.5], fov: 42 }}
        dpr={[1, 1.5]}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: "default",
          failIfMajorPerformanceCaveat: false,
        }}
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[4, 8, 4]}  intensity={1.8} color="#ffffff" />
        <directionalLight position={[-3, 2, -3]} intensity={0.6} color="#58E6D9" />
        <Suspense fallback={null}>
          <Model />
        </Suspense>
        <OrbitControls
          enablePan={false}
          autoRotate
          autoRotateSpeed={0.7}
          minDistance={1.5}
          maxDistance={10}
          enableDamping
          dampingFactor={0.06}
        />
      </Canvas>
    </WebGLBoundary>
  );
}
