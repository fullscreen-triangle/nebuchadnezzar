// Placeholder for the cuneiform glyph. Replace with the IGI/ME mark.
// The wrapper provides a consistent 32x32 box and primaryDark stroke colour.
export default function Mark({ size = 32 }) {
  return (
    <span
      className="inline-flex items-center justify-center font-mono text-primaryDark"
      style={{ width: size, height: size, fontSize: size * 0.7 }}
      aria-label="Crown Prince"
    >
      𒅆
    </span>
  );
}
