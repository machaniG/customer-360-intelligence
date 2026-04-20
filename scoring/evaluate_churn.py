# run from repo root with: python models/churn/evaluate_scores.py
import argparse
import sys
from pathlib import Path

# ensure repo root is on PYTHONPATH when script is run directly
def _add_repo_root_to_path():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

_add_repo_root_to_path()

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

from models.churn.config import DATA_PATH, OUTPUT_PATH
from labeling import build_churn_labels


def top_k_metrics(df, score_col="churn_risk", label_col="churn_label", pct=0.10):
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    k = max(1, int(len(df) * pct))
    top = df.iloc[:k]
    overall_rate = df[label_col].mean()
    top_rate = top[label_col].mean()
    lift = top_rate / overall_rate if overall_rate > 0 else np.nan
    capture = top[label_col].sum() / df[label_col].sum() if df[label_col].sum() > 0 else np.nan
    return {"pct": pct, "k": k, "top_rate": top_rate, "overall_rate": overall_rate, "lift": lift, "capture": capture}


def evaluate_scores(scores_path, tx_path, score_date, label_end_date, horizon_months=1, top_pcts=(0.1, 0.2, 0.05), output_csv=None, use_llm=False, llm_provider="groq", api_key=None):
    scores = pd.read_csv(scores_path)
    tx = pd.read_csv(tx_path, parse_dates=["InvoiceDate"])

    try:
        labels_df = build_churn_labels(tx, as_of_date=score_date, horizon_months=horizon_months)
    except Exception as e:
        raise ValueError(
            f"Failed to build churn labels: {e}. "
            "Try using an earlier score_date or smaller horizon_months."
        )

    eval_df = scores.merge(labels_df, on="Customer ID", how="left")
    eval_df = eval_df.dropna(subset=["churn_label"])
    eval_df["churn_label"] = eval_df["churn_label"].astype(int)

    y_true = eval_df["churn_label"].values
    y_score = eval_df["churn_risk"].values

    print("n_customers:", len(eval_df))
    print("Churn rate (observed):", y_true.mean())

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Brier score: {brier:.4f}")

    thresh = np.percentile(y_score, 80)
    y_pred = (y_score >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print(f"Top-20% threshold = {thresh:.4f}: precision={precision:.3f}, recall={recall:.3f}, TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Collect top-k results
    top_k_results = []
    for pct in top_pcts:
        m = top_k_metrics(eval_df, pct=pct)
        print(f"Top {int(pct*100)}% (k={m['k']}) top_rate={m['top_rate']:.3f}, overall={m['overall_rate']:.3f}, lift={m['lift']:.2f}, capture={m['capture']:.2f}")
        top_k_results.append((pct, m['lift'], m['capture']))

    if output_csv:
        eval_df.to_csv(output_csv, index=False)
        print(f"Evaluation table saved to: {output_csv}")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate churn prediction scores")
    parser.add_argument("--scores_path", type=str, default=OUTPUT_PATH, help="Input churn scores CSV path")
    parser.add_argument("--tx_path", type=str, default=DATA_PATH, help="Input transaction CSV path")
    parser.add_argument("--score_date", type=str, default="2010-09-30", help="Score cutoff date")
    parser.add_argument("--label_end_date", type=str, default="2010-09-30", help="Label window end date (not currently used directly)")
    parser.add_argument("--horizon_months", type=int, default=1, help="Horizon in months for churn label")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional path to save the evaluation table")
    args = parser.parse_args()

    evaluate_scores(
        scores_path=args.scores_path,
        tx_path=args.tx_path,
        score_date=args.score_date,
        label_end_date=args.label_end_date,
        horizon_months=args.horizon_months,
        output_csv=args.output_csv
    )

# The script performs the following steps:  
# 1. Loads churn scores and transaction data from specified CSV paths.
# 2. Builds churn labels using the transaction data and specified score date and horizon.
# 3. Merges scores with labels and evaluates the predictions using various metrics (ROC AUC, PR AUC, Brier score) and confusion matrix at the top 20% threshold.
# 4. Calculates top-k metrics for specified top percentages.
# 5. Optionally saves the evaluation table to a CSV file.   
"""Terminal usage now
From repo root:

source env/bin/activate
python models/churn/evaluate_scores.py
with params:
python -m scoring.evaluate_churn --score_date 2010-09-30 --horizon_months 1
"""
