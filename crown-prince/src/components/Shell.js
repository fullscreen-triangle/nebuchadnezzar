import Link from "next/link";
import { useRouter } from "next/router";
import Mark from "./Mark";

const NAV = [
  { href: "/", label: "Home" },
  { href: "/tools", label: "Tools" },
  { href: "/papers", label: "Papers" },
  { href: "/framework", label: "Framework" },
];

function NavLink({ href, label }) {
  const router = useRouter();
  const active =
    href === "/" ? router.pathname === "/" : router.pathname.startsWith(href);
  return (
    <Link
      href={href}
      className={`relative px-3 py-1 text-xs uppercase tracking-[0.2em] transition ${
        active ? "text-primaryDark" : "text-light/60 hover:text-light"
      }`}
    >
      {label}
      {active && (
        <span className="absolute -bottom-1 left-3 right-3 h-px bg-primaryDark" />
      )}
    </Link>
  );
}

export default function Shell({ children }) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="flex items-center justify-between border-b border-light/10 px-6 py-3 md:px-10">
        <Link href="/" className="flex items-center gap-3">
          <Mark size={28} />
          <span className="text-xs uppercase tracking-[0.3em] text-light/70">
            Crown Prince
          </span>
        </Link>
        <nav className="flex items-center gap-1 md:gap-2">
          {NAV.map((n) => (
            <NavLink key={n.href} {...n} />
          ))}
        </nav>
      </header>
      <main className="flex flex-1 flex-col">{children}</main>
      <footer className="flex items-center justify-between border-t border-light/10 px-6 py-3 text-[10px] uppercase tracking-[0.25em] text-light/40 md:px-10">
        <span>Bounded Phase Space Law &middot; empty dictionary</span>
        <span className="font-mono">v0.1</span>
      </footer>
    </div>
  );
}
