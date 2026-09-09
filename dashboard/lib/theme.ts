/** Palet Nord — sama dengan yang dipakai figur matplotlib di reports/figures/,
 *  supaya dashboard dan laporan terlihat satu sistem. */
export const NORD = {
  bg: "#242933",
  surface: "#2e3440",
  surfaceAlt: "#3b4252",
  border: "#434c5e",
  muted: "#7b88a1",
  text: "#e5e9f0",
  textBright: "#eceff4",
  fraud: "#bf616a",
  good: "#5e81ac",
  green: "#a3be8c",
  yellow: "#ebcb8b",
  purple: "#b48ead",
  cyan: "#88c0d0",
} as const;

/** Layout Plotly bertema gelap, dipakai semua chart. */
export const plotLayout = (overrides: Record<string, unknown> = {}) => ({
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: NORD.text, family: "var(--font-geist-sans), system-ui, sans-serif", size: 12 },
  margin: { l: 56, r: 20, t: 30, b: 48 },
  xaxis: { gridcolor: NORD.border, zerolinecolor: NORD.border, linecolor: NORD.border },
  yaxis: { gridcolor: NORD.border, zerolinecolor: NORD.border, linecolor: NORD.border },
  legend: { bgcolor: "rgba(0,0,0,0)", font: { size: 11 } },
  hoverlabel: { bgcolor: NORD.surfaceAlt, bordercolor: NORD.border },
  ...overrides,
});

export const plotConfig = {
  displayModeBar: false,
  responsive: true,
} as const;
