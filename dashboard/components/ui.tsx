import type { ReactNode } from "react";

export function Card({
  title,
  subtitle,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-edge bg-surface/60 p-4 sm:p-5 ${className}`}
    >
      {title && (
        <header className="mb-3">
          <h3 className="text-sm font-semibold text-bright">{title}</h3>
          {subtitle && <p className="mt-1 text-xs leading-relaxed text-muted">{subtitle}</p>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "good" | "bad";
}) {
  const toneClass =
    tone === "good" ? "text-ok" : tone === "bad" ? "text-fraud" : "text-bright";
  return (
    <div className="rounded-lg border border-edge bg-surface/40 px-3 py-3">
      <div className="text-[11px] uppercase tracking-wide text-muted">{label}</div>
      <div className={`tabular mt-1 text-xl font-semibold sm:text-2xl ${toneClass}`}>
        {value}
      </div>
      {hint && <div className="mt-0.5 text-[11px] text-muted">{hint}</div>}
    </div>
  );
}

export function Callout({
  tone = "info",
  children,
}: {
  tone?: "info" | "warn";
  children: ReactNode;
}) {
  const border = tone === "warn" ? "border-l-warn" : "border-l-good";
  return (
    <div
      className={`rounded-r-lg border-l-2 bg-surface-alt/40 px-4 py-3 text-xs leading-relaxed text-body ${border}`}
    >
      {children}
    </div>
  );
}

export const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
export const pp = (value: number, digits = 2) =>
  `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)} pp`;
export const usd = (value: number) =>
  `$${Math.round(value).toLocaleString("en-US")}`;
