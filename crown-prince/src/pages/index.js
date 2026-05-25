import Head from "next/head";
import dynamic from "next/dynamic";

const ModelViewer = dynamic(
  () => import("@/components/canvas/ModelViewer"),
  { ssr: false, loading: () => null }
);

export default function Home() {
  return (
    <>
      <Head>
        <title>Crown Prince</title>
      </Head>
      <div className="relative flex-1 bg-dark">
        <ModelViewer />
        <div className="pointer-events-none absolute inset-x-0 bottom-10 flex flex-col items-center gap-1">
          <p className="font-mono text-[10px] uppercase tracking-[0.45em] text-light/25">
            Crown Prince
          </p>
        </div>
      </div>
    </>
  );
}
