import Link from "next/link";

export default function NotFound() {
  return (
    <section className="mx-auto flex min-h-[60vh] max-w-3xl flex-col items-start justify-center px-6 md:px-10">
      <p className="text-[11px] uppercase tracking-[0.3em] text-primaryDark">404</p>
      <h1 className="mt-3 text-3xl font-semibold tracking-tight">
        That cell is empty.
      </h1>
      <p className="mt-3 max-w-lg text-sm text-light/60">
        The trie has no entry at this address. Try a different prefix.
      </p>
      <Link href="/" className="btn mt-6">
        Back to root
      </Link>
    </section>
  );
}
