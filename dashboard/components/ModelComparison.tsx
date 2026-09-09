"use client";

import { useEffect, useState } from "react";
import Plot from "./Plot";
import { Callout, Card, LoadError, Stat, fetchJson, pct, pp } from "./ui";
import { NORD, plotConfig, plotLayout } from "@/lib/theme";
import {
  METRIC_LABELS,
  MODEL_LABELS,
  type GainCurve,
  type MetricKey,
  type MetricsPayload,
} from "@/lib/types";

const METRIC_ORDER: MetricKey[] = [
  "auc",
  "ks",
  "pr_auc",
  "recall_at_1pct",
  "recall_at_5pct",
  "recall_at_10pct",
];
const MODEL_ORDER = ["M1_baseline_noC", "M2_graph_noC", "M3_baseline_full", "M4_graph_full"];

export default function ModelComparison() {
  const [metrics, setMetrics] = useState<MetricsPayload | null>(null);
  const [gains, setGains] = useState<GainCurve | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchJson<MetricsPayload>("metrics.json"), fetchJson<GainCurve>("gain_curve.json")])
      .then(([m, g]) => {
        setMetrics(m);
        setGains(g);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <LoadError message={error} />;
  if (!metrics) return <p className="py-12 text-center text-sm text-muted">memuat…</p>;

  const byName = Object.fromEntries(metrics.models.map((m) => [m.name, m]));
  const best = byName["M3_baseline_full"];
  const markers = [0.01, 0.05, 0.1];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Model final (M3)" value={best.test.auc.toFixed(4)} hint="AUC test out-of-time" />
        <Stat label="KS" value={best.test.ks.toFixed(4)} hint="standar credit scoring" />
        <Stat
          label="Lift graph (di atas M3)"
          value={pp(metrics.lifts.lift_b2.auc)}
          hint="AUC - negatif"
          tone="bad"
        />
        <Stat
          label="Graph struktural saja"
          value="+2,30 pp"
          hint="validation, belum diuji OOT"
          tone="good"
        />
      </div>

      <Card
        title="Hasil test out-of-time"
        subtitle="Hari 151–181, 89.326 transaksi. Test dibuka satu kali untuk keempat model setelah konfigurasi dikunci."
      >
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead>
              <tr className="border-b border-edge text-xs uppercase tracking-wide text-muted">
                <th className="py-2 text-left font-medium">Model</th>
                <th className="py-2 text-right font-medium">Fitur</th>
                {METRIC_ORDER.map((k) => (
                  <th key={k} className="py-2 text-right font-medium">
                    {METRIC_LABELS[k]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="tabular">
              {MODEL_ORDER.map((name) => {
                const model = byName[name];
                const isBest = name === "M3_baseline_full";
                return (
                  <tr
                    key={name}
                    className={`border-b border-edge/50 ${isBest ? "bg-surface-alt/40" : ""}`}
                  >
                    <td className={`py-2.5 pr-3 ${isBest ? "font-medium text-bright" : ""}`}>
                      {MODEL_LABELS[name]}
                    </td>
                    <td className="py-2.5 text-right text-muted">{model.n_features}</td>
                    {METRIC_ORDER.map((k) => {
                      const value = model.test[k];
                      const top = Math.max(...MODEL_ORDER.map((n) => byName[n].test[k]));
                      return (
                        <td
                          key={k}
                          className={`py-2.5 text-right ${
                            value === top ? "font-semibold text-ok" : ""
                          }`}
                        >
                          {k.startsWith("recall") ? pct(value) : value.toFixed(4)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card
          title="Gain chart - M3"
          subtitle="Berapa fraud tertangkap pada tiap kapasitas review."
        >
          {gains && (
            <Plot
              data={[
                {
                  x: gains.review_rate,
                  y: gains.recall,
                  type: "scatter",
                  mode: "lines",
                  name: "model",
                  line: { color: NORD.fraud, width: 2.5 },
                  hovertemplate: "review %{x:.0%} → recall %{y:.1%}<extra></extra>",
                },
                {
                  x: [0, 1],
                  y: [0, 1],
                  type: "scatter",
                  mode: "lines",
                  name: "acak",
                  line: { color: NORD.muted, width: 1, dash: "dash" },
                  hoverinfo: "skip",
                },
                {
                  x: markers,
                  y: markers.map((k) => {
                    const i = gains.review_rate.reduce(
                      (bestIdx, v, idx) =>
                        Math.abs(v - k) < Math.abs(gains.review_rate[bestIdx] - k) ? idx : bestIdx,
                      0,
                    );
                    return gains.recall[i];
                  }),
                  type: "scatter",
                  mode: "markers",
                  name: "1% / 5% / 10%",
                  marker: { color: NORD.yellow, size: 9 },
                  hovertemplate: "review %{x:.0%} → recall %{y:.1%}<extra></extra>",
                },
              ]}
              layout={plotLayout({
                height: 340,
                xaxis: { title: { text: "review rate" }, tickformat: ".0%", gridcolor: NORD.border },
                yaxis: { title: { text: "recall" }, tickformat: ".0%", gridcolor: NORD.border },
                legend: { orientation: "h", y: -0.22 },
              })}
              config={plotConfig}
              style={{ width: "100%" }}
            />
          )}
        </Card>

        <Card
          title="Dekomposisi fitur graph"
          subtitle={metrics.decomposition.note}
        >
          <Plot
            data={[
              {
                x: metrics.decomposition.rows.map((r) => r.name),
                y: metrics.decomposition.rows.map((r) => r.auc),
                type: "bar",
                marker: {
                  color: metrics.decomposition.rows.map((r) =>
                    r.name.includes("Level 1-2") ? NORD.green : NORD.good,
                  ),
                },
                hovertemplate: "%{x}<br>AUC %{y:.4f}<extra></extra>",
              },
            ]}
            layout={plotLayout({
              height: 340,
              yaxis: { range: [0.89, 0.935], title: { text: "AUC validation" }, gridcolor: NORD.border },
              xaxis: { tickangle: -18, gridcolor: NORD.border },
            })}
            config={plotConfig}
            style={{ width: "100%" }}
          />
        </Card>
      </div>

      <Callout tone="warn">
        <strong className="text-bright">Kenapa graph justru menurunkan performa.</strong> Fitur graph
        struktural (degree, PageRank, ukuran komponen) memberi{" "}
        <span className="text-ok">+2,30 pp AUC</span> di validation. Fitur berbasis label
        (neighbor fraud rate, UID fraud rate) merusak generalisasi dan menyeret yang baik turun
        bersamanya - penyebabnya bukan kebocoran label, melainkan runtuhnya kekuatan bukti:{" "}
        <code className="text-warn">uid_labeled_count</code> turun 52% dari train ke test, dan 67,3%
        baris test tidak punya tetangga berlabel sama sekali. Kombinasi terbaik (baseline + graph
        struktural) belum diuji out-of-time karena test set sudah dikunci.
      </Callout>
    </div>
  );
}
