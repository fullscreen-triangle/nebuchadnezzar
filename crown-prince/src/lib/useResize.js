import { useEffect, useRef, useState } from "react";

export function useResize() {
  const ref = useRef(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!ref.current) return;
    const obs = new ResizeObserver((entries) => {
      const e = entries[0];
      if (!e) return;
      setSize({
        width: Math.round(e.contentRect.width),
        height: Math.round(e.contentRect.height),
      });
    });
    obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);

  return [ref, size];
}
