"use client";

import { useState } from "react";
import ModelComparison from "@/components/ModelComparison";
import NetworkExplorer from "@/components/NetworkExplorer";
import ScoreExplorer from "@/components/ScoreExplorer";
import TabNav, { type TabId } from "@/components/TabNav";

export default function Home() {
  const [tab, setTab] = useState<TabId>("models");

  return (
    <div className="min-h-screen bg-bg">
      <header className="border-b border-edge bg-surface/30">
        <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 sm:py-12">
          <p className="text-xs uppercase tracking-widest text-muted">
            IEEE-CIS · 590.540 transaksi · validasi out-of-time
          </p>
          <h1 className="mt-3 text-2xl font-semibold leading-tight text-bright sm:text-4xl">
            Fitur graph menaikkan AUC di validation,{" "}
            <span className="text-fraud">menurunkannya di test</span>
          </h1>
          <p className="mt-4 max-w-3xl text-sm leading-relaxed text-body sm:text-base">
            Ablation jujur atas nilai tambah graph features pada deteksi fraud. Fitur graph berbasis
            label menurunkan AUC 1,91 pp di periode uji, sementara fitur graph struktural justru
            memberi +2,30 pp. Perbedaan itu hanya terlihat karena splitnya temporal - dengan random
            split, keduanya akan tampak menang.
          </p>
          <div className="mt-6 flex flex-wrap gap-x-6 gap-y-2 text-xs text-muted">
            <span>
              Model final <span className="tabular text-bright">AUC 0,9007</span>
            </span>
            <span>
              KS <span className="tabular text-bright">0,6471</span>
            </span>
            <span>
              69,4% fraud pada <span className="tabular text-bright">10% review</span>
            </span>
            <span>
              Test dibuka <span className="text-bright">satu kali</span>
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6 sm:py-8">
        <TabNav active={tab} onChange={setTab} />
        <div className="mt-5">
          {tab === "models" && <ModelComparison />}
          {tab === "network" && <NetworkExplorer />}
          {tab === "scores" && <ScoreExplorer />}
        </div>
      </main>

      <footer className="mt-8 border-t border-edge py-6">
        <div className="mx-auto max-w-6xl px-4 text-xs leading-relaxed text-muted sm:px-6">
          Seluruh angka berasal dari test set out-of-time (hari 151–181) yang dibuka satu kali
          setelah konfigurasi dikunci. Detail metodologi dan catatan keputusan ada di repositori.
        </div>
      </footer>
    </div>
  );
}
