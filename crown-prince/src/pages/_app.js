import "@/styles/globals.css";
import "katex/dist/katex.min.css";
import { Inter, JetBrains_Mono } from "next/font/google";
import Head from "next/head";
import Shell from "@/components/Shell";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });
const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export default function App({ Component, pageProps }) {
  return (
    <>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta
          name="description"
          content="Geometric pharmacology and chemistry tools, in the browser."
        />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <div
        className={`${inter.variable} ${jetbrains.variable} font-sans min-h-screen bg-dark text-light`}
        style={{ fontFamily: "var(--font-sans)" }}
      >
        <Shell>
          <Component {...pageProps} />
        </Shell>
      </div>
    </>
  );
}
