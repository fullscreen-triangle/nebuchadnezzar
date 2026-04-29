import { Component } from "react";

class ErrorBoundary extends Component {
  state = { hasError: false };
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  componentDidCatch(err) {
    // eslint-disable-next-line no-console
    console.warn("WebGL canvas failed:", err.message);
  }
  render() {
    if (this.state.hasError) return this.props.fallback;
    return this.props.children;
  }
}

export default function WebGLBoundary({ children, fallback }) {
  return <ErrorBoundary fallback={fallback}>{children}</ErrorBoundary>;
}

export function GLFallback({ message = "WebGL unavailable.", aspect = "square" }) {
  const aspectClass =
    aspect === "wide" ? "aspect-[2/1]" : aspect === "square" ? "aspect-square" : "";
  return (
    <div
      className={`${aspectClass} flex w-full flex-col items-center justify-center gap-2 rounded-md border border-light/15 bg-dark/40 p-4 text-center`}
    >
      <p className="text-[10px] uppercase tracking-[0.25em] text-light/40">
        Canvas2D fallback
      </p>
      <p className="text-xs text-light/55">{message}</p>
      <p className="mt-2 max-w-xs text-[10px] leading-relaxed text-light/40">
        Enable hardware acceleration or check{" "}
        <span className="font-mono accent">chrome://gpu</span> for details. The
        partition function is being evaluated on CPU instead.
      </p>
    </div>
  );
}
