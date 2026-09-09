export type MetricKey =
  | "auc"
  | "ks"
  | "pr_auc"
  | "recall_at_1pct"
  | "recall_at_5pct"
  | "recall_at_10pct";

export type MetricSet = Record<MetricKey, number>;

export interface ModelResult {
  name: string;
  n_features: number;
  has_graph: boolean;
  has_c: boolean;
  val: MetricSet;
  test: MetricSet;
}

export interface MetricsPayload {
  models: ModelResult[];
  lifts: {
    lift_b1: MetricSet;
    lift_b2: MetricSet;
    value_of_c: MetricSet;
    overlap: MetricSet;
  };
  decomposition: {
    note: string;
    rows: { name: string; n_features: number; auc: number; ks: number }[];
  };
}

export interface GainCurve {
  review_rate: number[];
  recall: number[];
  lift: number[];
  precision: number[];
}

export interface ScoreSample {
  sampling: {
    n_total_test: number;
    n_sampled: number;
    n_fraud_sampled: number;
    /** Bobot ekstrapolasi: non-fraud disubsample, fraud diambil seluruhnya. */
    good_weight: number;
    note: string;
  };
  score: number[];
  prob: number[];
  label: number[];
  amount: number[];
}

export interface CostScenario {
  cost_fp: number;
  optimal_threshold: number;
  review_rate: number;
  recall: number;
  total_cost: number;
  savings_per_month: number;
  pct_cost_reduction: number;
}

export interface CostPayload {
  baseline_cost: number;
  days_in_period: number;
  scenarios: CostScenario[];
}

export interface SubgraphNode {
  id: string;
  x: number;
  y: number;
  kind: "transaction" | "attribute";
  degree: number;
  fraud?: number;
  amount?: number;
  label?: string;
}

export interface Subgraph {
  id: string;
  anchor_column: string;
  n_transactions: number;
  n_fraud: number;
  fraud_rate: number;
  nodes: SubgraphNode[];
  edges: { source: string; target: string }[];
}

export interface ScoreBand {
  band: number;
  n: number;
  n_fraud: number;
  fraud_rate: number;
  score_min: number;
  score_max: number;
  cum_pct_fraud: number;
}

export const MODEL_LABELS: Record<string, string> = {
  M1_baseline_noC: "M1 · baseline tanpa C",
  M2_graph_noC: "M2 · tanpa C + graph",
  M3_baseline_full: "M3 · baseline (final)",
  M4_graph_full: "M4 · baseline + graph",
};

export const METRIC_LABELS: Record<MetricKey, string> = {
  auc: "AUC",
  ks: "KS",
  pr_auc: "PR-AUC",
  recall_at_1pct: "recall@1%",
  recall_at_5pct: "recall@5%",
  recall_at_10pct: "recall@10%",
};
