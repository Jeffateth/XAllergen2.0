#!/usr/bin/env python3
"""Rank pretrained InterPLM features by allergen-vs-non-allergen separation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SUMMARY_KEYS = {
    "max": "feature_max",
    "mean": "feature_mean",
    "frac_active": "feature_frac_active",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-npz", required=True, help="Output from extract_interplm_features.py")
    parser.add_argument(
        "--summary",
        default="max",
        choices=sorted(SUMMARY_KEYS),
        help="Protein-level summary to score.",
    )
    parser.add_argument("--output-csv", required=True, help="Ranked per-feature statistics")
    parser.add_argument(
        "--top-k-variance",
        type=int,
        default=512,
        help="Number of highest-variance features to include in the sparse logistic baseline.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out fraction for the sparse baseline.")
    parser.add_argument("--random-seed", type=int, default=7, help="Random seed for train/test split.")
    return parser.parse_args()


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.allclose(scores, scores[0]):
        return 0.5
    return float(roc_auc_score(y_true, scores))


def main() -> None:
    args = parse_args()
    bundle = np.load(Path(args.features_npz), allow_pickle=True)

    y = bundle["label"].astype(np.float32)
    if np.isnan(y).any():
        raise SystemExit("Feature bundle has missing labels. Add a `label` column to the input CSV before evaluation.")
    y = y.astype(int)
    if set(np.unique(y)) - {0, 1}:
        raise SystemExit("Labels must be binary 0/1.")

    X = bundle[SUMMARY_KEYS[args.summary]].astype(np.float32)
    feature_ids = bundle["feature_id"]

    pos_mask = y == 1
    neg_mask = y == 0
    pos_mean = X[pos_mask].mean(axis=0)
    neg_mean = X[neg_mask].mean(axis=0)
    delta = pos_mean - neg_mean

    pooled_std = np.sqrt((X[pos_mask].var(axis=0) + X[neg_mask].var(axis=0)) / 2.0 + 1e-8)
    effect_size = delta / pooled_std

    aurocs = np.array([safe_auroc(y, X[:, idx]) for idx in range(X.shape[1])], dtype=np.float32)
    allergen_enriched = np.where(delta >= 0.0, 1, 0)
    directional_auroc = np.where(allergen_enriched == 1, aurocs, 1.0 - aurocs)

    results = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "summary": args.summary,
            "allergen_mean": pos_mean,
            "non_allergen_mean": neg_mean,
            "delta_mean": delta,
            "effect_size": effect_size,
            "auroc": aurocs,
            "directional_auroc": directional_auroc,
            "allergen_enriched": allergen_enriched,
        }
    ).sort_values(["directional_auroc", "effect_size"], ascending=False)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    variances = X.var(axis=0)
    top_idx = np.argsort(variances)[::-1][: min(args.top_k_variance, X.shape[1])]
    X_top = X[:, top_idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X_top,
        y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=4000,
        random_state=args.random_seed,
    )
    clf.fit(X_train, y_train)
    test_proba = clf.predict_proba(X_test)[:, 1]
    baseline_auroc = roc_auc_score(y_test, test_proba)
    nonzero = int(np.count_nonzero(clf.coef_))

    print(f"Saved ranked feature statistics to {output_path}")
    print(f"Top feature: {results.iloc[0]['feature_id']}")
    print(f"Sparse logistic baseline AUROC ({args.summary} summaries): {baseline_auroc:.4f}")
    print(f"Non-zero coefficients in sparse baseline: {nonzero}")


if __name__ == "__main__":
    main()
