"use client";

import dynamic from "next/dynamic";

/** Plotly tidak mendukung server-side rendering, jadi harus dimuat dinamis
 *  dengan ssr:false. Placeholder menjaga tinggi supaya layout tidak melompat. */
const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[320px] items-center justify-center text-sm text-muted">
      memuat chart…
    </div>
  ),
});

export default Plot;
