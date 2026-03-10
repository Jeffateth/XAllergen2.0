#!/usr/bin/env python3
"""Prepare an AlgPred2 CSV for InterPLM experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", help="AlgPred2 training CSV")
    parser.add_argument("--test-csv", help="AlgPred2 test CSV")
    parser.add_argument("--output-csv", required=True, help="Prepared output CSV")
    parser.add_argument(
        "--use-splits",
        nargs="+",
        choices=["train", "test"],
        default=["train", "test"],
        help="Which provided splits to include.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1022,
        help="Keep only sequences up to this length to fit ESM-2 cleanly.",
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Downsample all classes to the smallest class size after filtering.",
    )
    parser.add_argument(
        "--per-class-limit",
        type=int,
        default=None,
        help="Optional cap per class after filtering/balancing.",
    )
    parser.add_argument("--random-seed", type=int, default=7, help="Sampling seed.")
    return parser.parse_args()


def load_split(path: str, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "sequence", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
    df = df.rename(columns={"id": "protein_id"}).copy()
    df["split"] = split_name
    df["sequence"] = df["sequence"].astype(str).str.strip()
    df["seq_len"] = df["sequence"].str.len()
    return df


def balanced_cap(df: pd.DataFrame, per_class_limit: int, random_seed: int) -> pd.DataFrame:
    parts = []
    for label_value, group in df.groupby("label", sort=True):
        if len(group) < per_class_limit:
            raise SystemExit(
                f"Requested --per-class-limit={per_class_limit}, but label {label_value} only has {len(group)} rows."
            )
        parts.append(group.sample(n=per_class_limit, random_state=random_seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)


def balance_to_smallest_class(df: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    class_counts = df["label"].value_counts()
    if class_counts.empty:
        return df
    target = int(class_counts.min())
    return balanced_cap(df, target, random_seed)


def main() -> None:
    args = parse_args()

    provided = []
    if "train" in args.use_splits:
        if not args.train_csv:
            raise SystemExit("--train-csv is required when --use-splits includes train")
        provided.append(load_split(args.train_csv, "train"))
    if "test" in args.use_splits:
        if not args.test_csv:
            raise SystemExit("--test-csv is required when --use-splits includes test")
        provided.append(load_split(args.test_csv, "test"))
    if not provided:
        raise SystemExit("No input splits selected.")

    merged = pd.concat(provided, ignore_index=True)
    merged = merged.drop_duplicates(subset=["protein_id"]).copy()

    before_filter = len(merged)
    merged = merged[merged["seq_len"] <= args.max_seq_len].copy()
    removed = before_filter - len(merged)

    if args.balance_classes:
        merged = balance_to_smallest_class(merged, args.random_seed)

    if args.per_class_limit is not None:
        merged = balanced_cap(merged, args.per_class_limit, args.random_seed)

    output_cols = ["protein_id", "sequence", "label", "split", "seq_len"]
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, columns=output_cols)

    print(f"Saved prepared dataset to {output_path}")
    print(f"Rows kept: {len(merged)}")
    print(f"Rows removed for length > {args.max_seq_len}: {removed}")
    print(f"Label counts: {merged['label'].value_counts().sort_index().to_dict()}")
    print(f"Split counts: {merged['split'].value_counts().to_dict()}")
    print(f"Sequence length range: {merged['seq_len'].min()}-{merged['seq_len'].max()}")


if __name__ == "__main__":
    main()
