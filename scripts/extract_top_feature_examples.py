#!/usr/bin/env python3
"""Extract top-activating proteins for selected InterPLM features.

This is a lightweight interpretation helper:
given a feature ranking CSV, the source protein CSV, and the saved feature bundle,
it reports the proteins with the highest activation for chosen feature IDs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SUMMARY_KEYS = {
    "max": "feature_max",
    "mean": "feature_mean",
    "frac_active": "feature_frac_active",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-npz", required=True, help="Feature bundle from extract_interplm_features.py")
    parser.add_argument("--source-csv", required=True, help="Source protein CSV used to create the bundle")
    parser.add_argument("--ranking-csv", required=True, help="Feature ranking CSV from evaluate_feature_associations.py")
    parser.add_argument(
        "--feature-ids",
        nargs="*",
        default=None,
        help="Explicit feature IDs to inspect, e.g. f_6168 f_4670",
    )
    parser.add_argument(
        "--top-n-ranked",
        type=int,
        default=0,
        help="If > 0, also inspect the top N ranked features from the ranking CSV.",
    )
    parser.add_argument("--summary", default="max", choices=sorted(SUMMARY_KEYS), help="Which summary to inspect.")
    parser.add_argument("--top-k-proteins", type=int, default=10, help="How many proteins to show per feature.")
    parser.add_argument("--output-csv", required=True, help="Where to write the extracted examples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle = np.load(Path(args.features_npz), allow_pickle=True)
    ranking = pd.read_csv(args.ranking_csv)
    source = pd.read_csv(args.source_csv)

    feature_matrix = bundle[SUMMARY_KEYS[args.summary]].astype(np.float32)
    protein_ids = pd.Series(bundle["protein_id"], name="protein_id").astype(str)
    feature_ids = pd.Series(bundle["feature_id"], name="feature_id").astype(str)

    selected_feature_ids: list[str] = []
    if args.top_n_ranked > 0:
        selected_feature_ids.extend(ranking["feature_id"].head(args.top_n_ranked).astype(str).tolist())
    if args.feature_ids:
        selected_feature_ids.extend(args.feature_ids)
    selected_feature_ids = list(dict.fromkeys(selected_feature_ids))

    if not selected_feature_ids:
        raise SystemExit("No features selected. Use --feature-ids and/or --top-n-ranked.")

    protein_index = {pid: idx for idx, pid in enumerate(protein_ids)}
    source = source.rename(columns={"id": "protein_id"}).copy()
    source["protein_id"] = source["protein_id"].astype(str)
    source["seq_len"] = source["sequence"].astype(str).str.len()
    source = source[source["protein_id"].isin(protein_index)].copy()

    rows = []
    for feature_id in selected_feature_ids:
        matches = np.where(feature_ids.values == feature_id)[0]
        if len(matches) == 0:
            continue
        feat_idx = int(matches[0])
        scores = feature_matrix[:, feat_idx]
        top_indices = np.argsort(scores)[::-1][: args.top_k_proteins]

        feature_meta = ranking[ranking["feature_id"] == feature_id].iloc[0]
        for rank, protein_pos in enumerate(top_indices, start=1):
            protein_id = str(protein_ids.iloc[protein_pos])
            src_row = source[source["protein_id"] == protein_id].iloc[0]
            rows.append(
                {
                    "feature_id": feature_id,
                    "feature_rank_in_ranking": int(ranking.index[ranking["feature_id"] == feature_id][0]) + 1,
                    "directional_auroc": float(feature_meta["directional_auroc"]),
                    "allergen_enriched": int(feature_meta["allergen_enriched"]),
                    "protein_rank_for_feature": rank,
                    "protein_id": protein_id,
                    "label": int(src_row["label"]),
                    "seq_len": int(src_row["seq_len"]),
                    "feature_score": float(scores[protein_pos]),
                    "sequence": str(src_row["sequence"]),
                }
            )

    out = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved feature examples to {output_path}")
    print(f"Features covered: {out['feature_id'].nunique() if not out.empty else 0}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()
