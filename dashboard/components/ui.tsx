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
    // min-w-0 wajib: tanpa itu grid item tidak boleh menyusut di bawah lebar
    // kontennya, sehingga label panjang mendorong kartu keluar dari grid.
    <div className="min-w-0 overflow-hidden rounded-lg border border-edge bg-surface/40 px-3 py-3">
      <div className="truncate text-[11px] uppercase tracking-wide text-muted" title={label}>
        {label}
      </div>
      <div
        className={`tabular mt-1 truncate text-xl font-semibold sm:text-2xl ${toneClass}`}
        title={value}
      >
        {value}
      </div>
      {hint && <div className="mt-0.5 truncate text-[11px] text-muted">{hint}</div>}
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

/** Muat JSON dari public/data dengan pesan galat yang bisa ditindaklanjuti.
 *  Tanpa ini, kegagalan fetch membuat komponen tergantung di "memuat…" selamanya
 *  tanpa petunjuk apa pun. */
export async function fetchJson<T>(file: string): Promise<T> {
  const url = `/data/${file}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`gagal memuat ${url} (HTTP ${response.status})`);
  }
  return (await response.json()) as T;
}

export function LoadError({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-fraud/40 bg-surface/60 px-4 py-6 text-center">
      <p className="text-sm font-medium text-fraud">Data tidak dapat dimuat</p>
      <p className="mt-1.5 text-xs text-muted">{message}</p>
      <p className="mt-3 text-xs text-muted">
        Jalankan{" "}
        <code className="rounded bg-surface-alt px-1.5 py-0.5">
          python -m src.viz.export_dashboard_data
        </code>{" "}
        dari root repo untuk membuat ulang berkas data.
      </p>
    </div>
  );
}

export const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
export const pp = (value: number, digits = 2) =>
  `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)} pp`;
export const usd = (value: number) =>
  `$${Math.round(value).toLocaleString("en-US")}`;
