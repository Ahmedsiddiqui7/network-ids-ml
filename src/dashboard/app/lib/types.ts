// Mirrors src/api/schemas.py's PredictionResponse (Cycle 4) and the
// extended ModelService.info() payload (Cycle 5). Kept in sync by hand --
// there are only two call sites (app/api/replay, app/api/model-info) --
// rather than generating a client from the OpenAPI schema, which would be
// overkill for a two-endpoint dashboard.

export interface PredictionResponse {
  prediction: string;
  is_malicious: boolean;
  confidence: number;
  risk_score: number;
  model_version: string;
  timestamp: string;
}

export interface ReplayRow extends PredictionResponse {
  id: string;
  true_label: string;
  correct: boolean;
}

export interface CurvePoints {
  fpr?: number[];
  tpr?: number[];
  precision?: number[];
  recall?: number[];
}

export interface ModelInfo {
  model_version: string;
  feature_count: number;
  commit_hash: string;
  test_metrics: {
    macro_recall_stable_classes: number;
    macro_f1: number;
    macro_pr_auc: number;
    aggregate_benign_fpr: number;
  };
  operating_threshold: {
    fpr_budget: number;
    threshold: number;
    val_recall_at_threshold: number;
    val_fpr_at_threshold: number;
  };
  confusion_matrix: number[][];
  confusion_matrix_labels: string[];
  roc_curve: { fpr: number[]; tpr: number[] };
  pr_curve: { precision: number[]; recall: number[] };
}
