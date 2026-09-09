"use client";

import { useEffect, useMemo, useState } from "react";
import Plot from "./Plot";
import { Callout, Card, Stat, pct } from "./ui";
import { NORD, plotConfig, plotLayout } from "@/lib/theme";
import type { Subgraph } from "@/lib/types";

export default function NetworkExplorer() {
  const [subgraphs, setSubgraphs] = useState<Subgraph[] | null>(null);
  const [selected, setSelected] = useState(0);

  useEffect(() => {
    fetch("data/subgraphs.json").then((r) => r.json()).then(setSubgraphs);
  }, []);

  const traces = useMemo(() => {
    if (!subgraphs?.length) return null;
    const graph = subgraphs[selected];
    const position = new Map(graph.nodes.map((n) => [n.id, n]));

    // Edge digambar sebagai satu trace dengan null sebagai pemisah segmen —
    // jauh lebih ringan daripada satu trace per edge.
    const edgeX: (number | null)[] = [];
    const edgeY: (number | null)[] = [];
    for (const edge of graph.edges) {
      const a = position.get(edge.source);
      const b = position.get(edge.target);
      if (!a || !b) continue;
      edgeX.push(a.x, b.x, null);
      edgeY.push(a.y, b.y, null);
    }

    const transactions = graph.nodes.filter((n) => n.kind === "transaction");
    const attributes = graph.nodes.filter((n) => n.kind === "attribute");
    const fraud = transactions.filter((n) => n.fraud === 1);
    const clean = transactions.filter((n) => n.fraud !== 1);

    const txTrace = (nodes: typeof transactions, name: string, color: string) => ({
      x: nodes.map((n) => n.x),
      y: nodes.map((n) => n.y),
      type: "scatter" as const,
      mode: "markers" as const,
      name,
      marker: { color, size: 11, line: { color: NORD.bg, width: 1.5 } },
      text: nodes.map((n) => `$${n.amount?.toFixed(2) ?? "?"}`),
      hovertemplate: `${name}<br>nominal %{text}<extra></extra>`,
    });

    return [
      {
        x: edgeX,
        y: edgeY,
        type: "scatter" as const,
        mode: "lines" as const,
        line: { color: NORD.border, width: 1 },
        hoverinfo: "skip" as const,
        showlegend: false,
      },
      txTrace(clean, "transaksi sah", NORD.good),
      txTrace(fraud, "fraud", NORD.fraud),
      {
        x: attributes.map((n) => n.x),
        y: attributes.map((n) => n.y),
        type: "scatter" as const,
        mode: "markers" as const,
        name: "atribut bersama",
        marker: {
          color: NORD.yellow,
          size: 16,
          symbol: "diamond",
          line: { color: NORD.bg, width: 1.5 },
        },
        text: attributes.map((n) => n.label ?? n.id),
        hovertemplate: "%{text}<br>terhubung ke %{customdata} transaksi<extra></extra>",
        customdata: attributes.map((n) => n.degree),
      },
    ];
  }, [subgraphs, selected]);

  if (!subgraphs) return <p className="py-12 text-center text-sm text-muted">memuat…</p>;

  const graph = subgraphs[selected];
  const hideAxis = {
    showgrid: false,
    zeroline: false,
    showticklabels: false,
    linecolor: "rgba(0,0,0,0)",
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Transaksi" value={String(graph.n_transactions)} hint="dalam klaster ini" />
        <Stat label="Fraud" value={String(graph.n_fraud)} tone={graph.n_fraud > 0 ? "bad" : "good"} />
        <Stat
          label="Fraud rate klaster"
          value={pct(graph.fraud_rate, 0)}
          hint="rata-rata populasi 3,5%"
          tone={graph.fraud_rate > 0.5 ? "bad" : "default"}
        />
        <Stat label="Atribut penghubung" value={graph.anchor_column} />
      </div>

      <Card
        title="Pilih klaster"
        subtitle="Diurutkan dari fraud rate tertinggi. Klaster dibentuk oleh transaksi yang berbagi kartu, alamat, atau device yang sama."
      >
        <div className="flex flex-wrap gap-1.5">
          {subgraphs.map((g, i) => (
            <button
              key={g.id}
              onClick={() => setSelected(i)}
              className={`tabular rounded-md border px-2.5 py-1.5 text-xs transition-colors ${
                i === selected
                  ? "border-edge bg-surface-alt font-medium text-bright"
                  : "border-edge/50 text-muted hover:text-body"
              }`}
              title={g.id}
            >
              <span
                className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle"
                style={{
                  background:
                    g.fraud_rate > 0.5 ? NORD.fraud : g.fraud_rate > 0 ? NORD.yellow : NORD.green,
                }}
              />
              {g.n_transactions} tx · {pct(g.fraud_rate, 0)}
            </button>
          ))}
        </div>
      </Card>

      <Card
        title={graph.id}
        subtitle="Lingkaran = transaksi (merah bila fraud), berlian kuning = atribut yang dibagi bersama. Garis berarti transaksi memiliki atribut itu."
      >
        {traces && (
          <Plot
            data={traces}
            layout={plotLayout({
              height: 520,
              xaxis: hideAxis,
              yaxis: hideAxis,
              margin: { l: 10, r: 10, t: 10, b: 40 },
              legend: { orientation: "h", y: -0.05 },
              hovermode: "closest",
            })}
            config={plotConfig}
            style={{ width: "100%" }}
          />
        )}
      </Card>

      <Callout>
        <strong className="text-bright">Yang membuat klaster ini menarik.</strong> Transaksi yang
        berbagi kartu atau device membentuk kelompok, dan pada sebagian kelompok hampir seluruh
        anggotanya fraud — pola yang mendorong dibangunnya fitur graph. Tetapi ablation menunjukkan
        pola ini <em>tidak</em> bertahan ke periode berikutnya: klaster baru terus bermunculan, dan
        dua pertiga transaksi di periode uji tidak punya tetangga berlabel apa pun. Visualisasi yang
        meyakinkan tidak sama dengan fitur yang berguna.
      </Callout>
    </div>
  );
}
