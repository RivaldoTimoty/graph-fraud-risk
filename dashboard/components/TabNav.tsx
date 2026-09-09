"use client";

export type TabId = "models" | "network" | "scores";

const TABS: { id: TabId; label: string; short: string }[] = [
  { id: "models", label: "Perbandingan model", short: "Model" },
  { id: "network", label: "Jaringan fraud", short: "Jaringan" },
  { id: "scores", label: "Eksplorasi skor", short: "Skor" },
];

export default function TabNav({
  active,
  onChange,
}: {
  active: TabId;
  onChange: (id: TabId) => void;
}) {
  return (
    <nav
      className="flex gap-1 overflow-x-auto rounded-lg border border-edge bg-surface/50 p-1"
      role="tablist"
    >
      {TABS.map((tab) => {
        const selected = tab.id === active;
        return (
          <button
            key={tab.id}
            role="tab"
            aria-selected={selected}
            onClick={() => onChange(tab.id)}
            className={`flex-1 whitespace-nowrap rounded-md px-3 py-2 text-sm transition-colors ${
              selected
                ? "bg-surface-alt font-medium text-bright"
                : "text-muted hover:text-body"
            }`}
          >
            <span className="hidden sm:inline">{tab.label}</span>
            <span className="sm:hidden">{tab.short}</span>
          </button>
        );
      })}
    </nav>
  );
}
