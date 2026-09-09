"use client";

import { useEffect, useMemo, useState } from "react";
import Plot from "./Plot";
import { Callout, Card, LoadError, Stat, fetchJson, pct, usd } from "./ui";
import { NORD, plotConfig, plotLayout } from "@/lib/theme";
import type { CostPayload, ScoreBand, ScoreSample } from "@/lib/types";

/** Non-fraud disubsample saat precompute, jadi setiap baris non-fraud mewakili
 *  `good_weight` transaksi asli. Semua hitungan populasi harus memakai bobot ini. */
function simulate(sample: ScoreSample, costFp: number, threshold: number) {
  const w = sample.sampling.good_weight;
  let flaggedGood = 0;
  let flaggedFraud = 0;
  let missedValue = 0;
  let totalFraud = 0;
  let totalPopulation = 0;

  for (let i = 0; i < sample.score.length; i++) {
    const isFraud = sample.label[i] === 1;
    const weight = isFraud ? 1 : w;
    totalPopulation += weight;
    // Skor rendah = risiko tinggi, jadi yang ditinjau adalah skor DI BAWAH ambang.
    const flagged = sample.score[i] <= threshold;
    if (isFraud) {
      totalFraud += 1;
      if (flagged) flaggedFraud += 1;
      else missedValue += sample.amount[i];
    } else if (flagged) {
      flaggedGood += weight;
    }
  }

  const reviewed = flaggedGood + flaggedFraud;
  return {
    reviewRate: reviewed / totalPopulation,
    recall: totalFraud ? flaggedFraud / totalFraud : 0,
    precision: reviewed ? flaggedFraud / reviewed : 0,
    fnCost: missedValue,
    fpCost: flaggedGood * costFp,
    totalCost: missedValue + flaggedGood * costFp,
  };
}

export default function ScoreExplorer() {
  const [sample, setSample] = useState<ScoreSample | null>(null);
  const [bands, setBands] = useState<ScoreBand[] | null>(null);
  const [cost, setCost] = useState<CostPayload | null>(null);
  const [threshold, setThreshold] = useState(569);
  const [costFp, setCostFp] = useState(5);

  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetchJson<ScoreSample>("score_sample.json"),
      fetchJson<ScoreBand[]>("score_bands.json"),
      fetchJson<CostPayload>("cost_sensitivity.json"),
    ])
      .then(([s, b, c]) => {
        setSample(s);
        setBands(b);
        setCost(c);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const result = useMemo(
    () => (sample ? simulate(sample, costFp, threshold) : null),
    [sample, costFp, threshold],
  );

  const histogram = useMemo(() => {
    if (!sample) return null;
    const fraud: number[] = [];
    const clean: number[] = [];
    for (let i = 0; i < sample.score.length; i++) {
      (sample.label[i] === 1 ? fraud : clean).push(sample.score[i]);
    }
    return { fraud, clean };
  }, [sample]);

  if (error) return <LoadError message={error} />;
  if (!sample || !bands || !cost || !result || !histogram) {
    return <p className="py-12 text-center text-sm text-muted">memuat…</p>;
  }

  const savings = cost.baseline_cost - result.totalCost;
  const perMonth = (savings * 30) / cost.days_in_period;
  const optimal = cost.scenarios.find((s) => s.cost_fp === costFp);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Transaksi ditinjau" value={pct(result.reviewRate)} hint="dari total populasi" />
        <Stat label="Fraud tertangkap" value={pct(result.recall)} tone="good" />
        <Stat label="Presisi review" value={pct(result.precision)} hint="benar fraud per review" />
        <Stat
          label="Penghematan/bulan"
          value={usd(perMonth)}
          hint={`vs ${usd(cost.baseline_cost)} tanpa model`}
          tone={savings > 0 ? "good" : "bad"}
        />
      </div>

      <Card
        title="Simulasi ambang skor"
        subtitle="Geser untuk melihat trade-off antara kapasitas review dan fraud yang tertangkap."
      >
        <div className="space-y-5">
          <div>
            <div className="mb-2 flex items-baseline justify-between">
              <label htmlFor="threshold" className="text-xs text-muted">
                Tinjau transaksi berskor di bawah
              </label>
              <span className="tabular text-lg font-semibold text-bright">{threshold}</span>
            </div>
            <input
              id="threshold"
              type="range"
              min={300}
              max={780}
              step={1}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-full"
            />
            <div className="mt-1 flex justify-between text-[11px] text-muted">
              <span>300 · paling berisiko</span>
              <span>780 · paling aman</span>
            </div>
          </div>

          <div>
            <div className="mb-2 text-xs text-muted">Biaya review manual per transaksi</div>
            <div className="flex flex-wrap gap-2">
              {cost.scenarios.map((s) => (
                <button
                  key={s.cost_fp}
                  onClick={() => setCostFp(s.cost_fp)}
                  className={`rounded-md border px-3 py-1.5 text-sm transition-colors ${
                    s.cost_fp === costFp
                      ? "border-edge bg-surface-alt font-medium text-bright"
                      : "border-edge/50 text-muted hover:text-body"
                  }`}
                >
                  ${s.cost_fp}
                </button>
              ))}
            </div>
          </div>

          {optimal && (
            <div className="flex flex-wrap items-center gap-2 rounded-lg border border-edge/60 bg-surface-alt/30 px-3 py-2.5 text-xs">
              <span className="text-muted">
                Titik biaya-minimum pada ${costFp}: review {pct(optimal.review_rate)}, tangkap{" "}
                {pct(optimal.recall)}, hemat {usd(optimal.savings_per_month)}/bulan
              </span>
              <button
                onClick={() => {
                  // Cari skor yang menghasilkan review rate paling dekat dengan optimum.
                  let bestScore = threshold;
                  let bestGap = Infinity;
                  for (let s = 300; s <= 780; s += 2) {
                    const gap = Math.abs(
                      simulate(sample, costFp, s).reviewRate - optimal.review_rate,
                    );
                    if (gap < bestGap) {
                      bestGap = gap;
                      bestScore = s;
                    }
                  }
                  setThreshold(bestScore);
                }}
                className="rounded border border-edge px-2 py-1 text-[11px] text-body hover:bg-surface-alt"
              >
                pakai titik ini
              </button>
            </div>
          )}
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Distribusi skor" subtitle="Garis kuning = ambang saat ini.">
          <Plot
            data={[
              {
                x: histogram.clean,
                type: "histogram",
                name: "sah",
                marker: { color: NORD.good },
                opacity: 0.75,
                histnorm: "probability density",
                nbinsx: 55,
              },
              {
                x: histogram.fraud,
                type: "histogram",
                name: "fraud",
                marker: { color: NORD.fraud },
                opacity: 0.75,
                histnorm: "probability density",
                nbinsx: 55,
              },
            ]}
            layout={plotLayout({
              height: 320,
              barmode: "overlay",
              xaxis: { title: { text: "skor" }, gridcolor: NORD.border },
              yaxis: { title: { text: "densitas" }, gridcolor: NORD.border },
              legend: { orientation: "h", y: -0.24 },
              shapes: [
                {
                  type: "line",
                  x0: threshold,
                  x1: threshold,
                  yref: "paper",
                  y0: 0,
                  y1: 1,
                  line: { color: NORD.yellow, width: 2, dash: "dash" },
                },
              ],
            })}
            config={plotConfig}
            style={{ width: "100%" }}
          />
        </Card>

        <Card title="Fraud rate per band skor" subtitle="Sepuluh band, masing-masing 10% populasi.">
          <Plot
            data={[
              {
                x: bands.map((b) => `${b.score_min.toFixed(0)}`),
                y: bands.map((b) => b.fraud_rate),
                type: "bar",
                marker: {
                  color: bands.map((b) =>
                    b.fraud_rate > 0.1 ? NORD.fraud : b.fraud_rate > 0.02 ? NORD.yellow : NORD.green,
                  ),
                },
                hovertemplate: "skor dari %{x}<br>fraud rate %{y:.2%}<extra></extra>",
              },
            ]}
            layout={plotLayout({
              height: 320,
              xaxis: { title: { text: "batas bawah skor band" }, gridcolor: NORD.border },
              yaxis: { title: { text: "fraud rate" }, tickformat: ".0%", gridcolor: NORD.border },
            })}
            config={plotConfig}
            style={{ width: "100%" }}
          />
        </Card>
      </div>

      <Callout>
        <strong className="text-bright">Kenapa tidak ada satu ambang &ldquo;terbaik&rdquo;.</strong>{" "}
        Titik biaya-minimum bergeser lima kali lipat - dari meninjau 35,7% transaksi (biaya review
        $2) menjadi 7,0% ($25). Angka biaya review hanya bisa ditentukan dari data operasional
        internal, sehingga yang diserahkan adalah alat untuk memilih titik operasi, bukan satu
        rekomendasi tunggal. Perlu dicatat: pada band paling berisiko sekalipun, sekitar tiga
        perempat transaksi tetap sah - sistem ini penentu prioritas review, bukan alat penolakan
        otomatis.
      </Callout>
    </div>
  );
}
