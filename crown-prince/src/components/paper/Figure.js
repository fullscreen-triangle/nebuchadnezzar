export function Figure({ src, caption, num }) {
  return (
    <figure className="my-8">
      <div className="overflow-hidden rounded-md border border-light/10 bg-light/[0.02]">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={caption || ""} className="w-full" />
      </div>
      {caption && (
        <figcaption className="mt-2 text-xs leading-relaxed text-light/60">
          <span className="font-mono text-primaryDark">Fig {num}.</span> {caption}
        </figcaption>
      )}
    </figure>
  );
}

export function Chart({ caption, num, children }) {
  return (
    <figure className="my-8">
      <div className="overflow-hidden rounded-md border border-light/10 bg-light/[0.02] p-4">
        {children}
      </div>
      {caption && (
        <figcaption className="mt-2 text-xs leading-relaxed text-light/60">
          <span className="font-mono text-primaryDark">Chart {num}.</span> {caption}
        </figcaption>
      )}
    </figure>
  );
}
