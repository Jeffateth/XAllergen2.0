#!/usr/bin/env python3
"""Create a homology-aware `subtrain/subval` split with CD-HIT.

Why this script exists
----------------------
For protein tasks, a naive random split is often too optimistic because closely related
proteins can land in both training and validation sets. The downstream model then sees
very similar sequences during development and evaluation.

AlgPred 2.0 addressed this by clustering proteins at 40% sequence identity and splitting
whole clusters rather than individual sequences. This script reproduces that idea for the
current ESM-2 + InterPLM workflow.

High-level workflow
-------------------
1. Read the original training CSV (`id, sequence, label`).
2. Filter to ESM-compatible sequence length (`<= 1022 aa` by default).
3. Write the filtered proteins to FASTA for external clustering.
4. Run CD-HIT at the chosen identity threshold (default 40%).
5. Parse the `.clstr` output to recover cluster membership.
6. Assign whole clusters to either `subtrain` or `subval`.
7. Optionally audit filtered test-vs-train leakage using `cd-hit-2d`.
8. Save split CSVs and a compact summary file.

Expected outputs
----------------
- `train_filtered_1022.csv`
- `train_filtered_1022.fasta`
- `train_filtered_1022_cdhit40.fasta`
- `train_filtered_1022_cdhit40.fasta.clstr`
- `train_subtrain_1022_h40.csv`
- `train_subval_1022_h40.csv`
- `homology_split_summary.csv`
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Define the CLI used to run the full homology-aware split workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, help="Original AlgPred2 training CSV.")
    parser.add_argument("--out-dir", required=True, help="Directory for filtered FASTA, clusters, and split CSVs.")
    parser.add_argument("--test-csv", help="Optional untouched test CSV for leakage audit only.")
    parser.add_argument("--max-seq-len", type=int, default=1022, help="Keep only sequences with length <= this value.")
    parser.add_argument("--identity", type=float, default=0.4, help="CD-HIT identity threshold.")
    parser.add_argument("--word-size", type=int, default=2, help="CD-HIT word size for the chosen identity.")
    parser.add_argument("--subval-fraction", type=float, default=0.2, help="Target fraction assigned to subval.")
    parser.add_argument("--skip-cdhit", action="store_true", help="Only prepare filtered inputs; do not run CD-HIT.")
    return parser.parse_args()


def load_and_filter_csv(path: str, max_seq_len: int) -> pd.DataFrame:
    """Load the AlgPred2-style CSV and keep only ESM-compatible proteins.

    The expected input columns are:
    - `id`: sequence identifier
    - `sequence`: amino acid sequence
    - `label`: binary class label (0/1)

    We rename `id -> protein_id` to make the downstream tables more explicit, and add a
    `seq_len` column because the sequence length is a first-class property in this workflow.
    """
    df = pd.read_csv(path)
    required = {"id", "sequence", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
    df = df.rename(columns={"id": "protein_id"}).copy()
    df["label"] = df["label"].astype(int)
    df["sequence"] = df["sequence"].astype(str).str.strip()
    df["seq_len"] = df["sequence"].str.len()
    return df[df["seq_len"] <= max_seq_len].copy()


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    """Write a simple FASTA file for CD-HIT.

    CD-HIT works on FASTA input, so the filtered CSV is converted into:

        >protein_id
        SEQUENCE

    one protein at a time.
    """
    with path.open("w") as handle:
        for row in df.itertuples(index=False):
            handle.write(f">{row.protein_id}\n{row.sequence}\n")


def run_checked(cmd: list[str], cwd: Path) -> None:
    """Run an external command and fail loudly if it does not complete successfully."""
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def run_cdhit(input_fasta: Path, output_fasta: Path, identity: float, word_size: int) -> Path:
    """Cluster the filtered training proteins with CD-HIT.

    Important parameters:
    - `-c identity`: target identity threshold. We default to 0.4 to match AlgPred 2.0.
    - `-n word_size`: CD-HIT k-mer size; for 40% identity, 2 is the standard choice.
    - `-d 0`: keep full sequence identifiers in the output.
    - `-M 0`: no explicit memory limit.
    - `-T 0`: allow CD-HIT to decide threading; ignored if OpenMP is disabled.

    Returns the path to the `.clstr` file, which is the cluster-membership file we parse.
    """
    input_fasta = input_fasta.resolve()
    output_fasta = output_fasta.resolve()
    exe = shutil.which("cd-hit")
    if not exe:
        raise SystemExit("`cd-hit` is not available on PATH.")
    cmd = [
        exe,
        "-i",
        str(input_fasta),
        "-o",
        str(output_fasta),
        "-c",
        str(identity),
        "-n",
        str(word_size),
        "-d",
        "0",
        "-M",
        "0",
        "-T",
        "0",
    ]
    run_checked(cmd, cwd=input_fasta.parent)
    clstr = Path(str(output_fasta) + ".clstr")
    if not clstr.exists():
        raise SystemExit(f"Expected cluster file not found: {clstr}")
    return clstr


def run_cdhit_2d(query_fasta: Path, reference_fasta: Path, output_prefix: Path, identity: float, word_size: int) -> Path:
    """Audit potential leakage from test proteins into training proteins.

    `cd-hit-2d` compares one FASTA (`query_fasta`) against another (`reference_fasta`).
    Here, we use it only as a diagnostic check:
    - query = filtered test proteins
    - reference = filtered training proteins

    If many test proteins cluster against train proteins at 40% identity, then the original
    train/test separation may still contain family-level overlap after our length filtering.
    """
    query_fasta = query_fasta.resolve()
    reference_fasta = reference_fasta.resolve()
    output_prefix = output_prefix.resolve()
    exe = shutil.which("cd-hit-2d")
    if not exe:
        raise SystemExit("`cd-hit-2d` is not available on PATH.")
    cmd = [
        exe,
        "-i",
        str(query_fasta),
        "-i2",
        str(reference_fasta),
        "-o",
        str(output_prefix),
        "-c",
        str(identity),
        "-n",
        str(word_size),
        "-d",
        "0",
        "-M",
        "0",
        "-T",
        "0",
    ]
    run_checked(cmd, cwd=output_prefix.parent)
    return Path(str(output_prefix) + ".clstr")


def parse_clstr(path: Path) -> list[list[str]]:
    """Parse a CD-HIT `.clstr` file into a Python list of clusters.

    Each cluster is returned as a list of `protein_id` strings.

    Example CD-HIT fragment:
        >Cluster 0
        0   123aa, >P_1... *
        1   120aa, >P_2... at 89%

    becomes:
        [["P_1", "P_2"], ...]
    """
    clusters: list[list[str]] = []
    current: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">Cluster "):
                if current:
                    clusters.append(current)
                current = []
                continue
            if ">" not in line:
                continue
            protein_id = line.split(">", 1)[1].split("...", 1)[0]
            current.append(protein_id)
    if current:
        clusters.append(current)
    return clusters


def split_clusters(df: pd.DataFrame, clusters: list[list[str]], subval_fraction: float) -> tuple[set[str], set[str]]:
    """Assign whole clusters to `subtrain` or `subval`.

    This is the key homology-aware step.

    We do *not* split individual proteins at random. Instead:
    - each cluster is treated as indivisible
    - clusters are sorted by size, largest first
    - clusters are greedily added to `subval` only when doing so moves us closer to the
      target class counts for the validation split

    This is a simple deterministic heuristic, not an exact optimizer. The goal is practical:
    keep homology boundaries intact while roughly matching the requested validation fraction
    and preserving class balance as well as possible.
    """
    label_lookup = dict(zip(df["protein_id"], df["label"]))
    cluster_items = []
    for cluster in clusters:
        counts = Counter(label_lookup[protein_id] for protein_id in cluster)
        cluster_items.append((cluster, len(cluster), counts.get(0, 0), counts.get(1, 0)))

    cluster_items.sort(key=lambda item: item[1], reverse=True)

    target_0 = int(round((df["label"] == 0).sum() * subval_fraction))
    target_1 = int(round((df["label"] == 1).sum() * subval_fraction))
    subval_ids: set[str] = set()
    subtrain_ids: set[str] = set()
    val_0 = 0
    val_1 = 0

    for cluster, _, c0, c1 in cluster_items:
        # Compare two options for the current cluster:
        # 1. put it into subval
        # 2. leave it in subtrain
        #
        # The smaller score means "closer to the desired number of class-0/class-1
        # proteins in subval".
        to_val_score = abs(target_0 - (val_0 + c0)) + abs(target_1 - (val_1 + c1))
        stay_train_score = abs(target_0 - val_0) + abs(target_1 - val_1)
        if to_val_score <= stay_train_score:
            subval_ids.update(cluster)
            val_0 += c0
            val_1 += c1
        else:
            subtrain_ids.update(cluster)

    # Safety net: if any protein somehow did not get assigned, force it into subtrain.
    all_ids = set(df["protein_id"])
    missing = all_ids - (subtrain_ids | subval_ids)
    subtrain_ids.update(missing)
    return subtrain_ids, subval_ids


def write_split_csv(df: pd.DataFrame, protein_ids: set[str], split_name: str, path: Path) -> pd.DataFrame:
    """Write one split (`subtrain` or `subval`) back to CSV."""
    out = df[df["protein_id"].isin(protein_ids)].copy()
    out["split"] = split_name
    out.to_csv(path, index=False, columns=["protein_id", "sequence", "label", "seq_len", "split"])
    return out


def write_summary(path: Path, rows: list[tuple[str, str]]) -> None:
    """Write a compact machine-readable summary of what the script produced."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def main() -> None:
    """Run the full split pipeline from filtered training CSV to split outputs."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: load the original training CSV and remove proteins too long for the
    # downstream ESM-2 setup.
    train_df = load_and_filter_csv(args.train_csv, args.max_seq_len)
    filtered_csv = out_dir / "train_filtered_1022.csv"
    filtered_csv.write_text(train_df.to_csv(index=False, columns=["protein_id", "sequence", "label", "seq_len"]))

    # Step 2: export filtered proteins to FASTA because CD-HIT works on FASTA, not CSV.
    train_fasta = out_dir / "train_filtered_1022.fasta"
    write_fasta(train_df, train_fasta)

    # These summary rows are always written, even if we stop after preparing inputs.
    summary = [
        ("filtered_train_rows", str(len(train_df))),
        ("filtered_train_label_0", str(int((train_df["label"] == 0).sum()))),
        ("filtered_train_label_1", str(int((train_df["label"] == 1).sum()))),
        ("max_seq_len", str(args.max_seq_len)),
        ("identity_threshold", str(args.identity)),
        ("subval_fraction", str(args.subval_fraction)),
    ]

    if args.skip_cdhit:
        # Useful when the user wants the filtered FASTA prepared first, but has not yet
        # installed CD-HIT.
        summary.append(("status", "prepared_only"))
        write_summary(out_dir / "homology_split_summary.csv", summary)
        print(f"Prepared filtered training inputs in {out_dir}")
        return

    # Step 3: run CD-HIT clustering on the filtered training proteins.
    clustered_fasta = out_dir / "train_filtered_1022_cdhit40.fasta"
    clstr_path = run_cdhit(train_fasta, clustered_fasta, args.identity, args.word_size)

    # Step 4: recover cluster membership from the `.clstr` file.
    clusters = parse_clstr(clstr_path)

    # Step 5: assign clusters to `subtrain` or `subval`.
    subtrain_ids, subval_ids = split_clusters(train_df, clusters, args.subval_fraction)

    # Step 6: materialize the two splits as CSV files for downstream feature extraction.
    subtrain_df = write_split_csv(train_df, subtrain_ids, "subtrain", out_dir / "train_subtrain_1022_h40.csv")
    subval_df = write_split_csv(train_df, subval_ids, "subval", out_dir / "train_subval_1022_h40.csv")

    summary.extend(
        [
            ("n_clusters", str(len(clusters))),
            ("subtrain_rows", str(len(subtrain_df))),
            ("subtrain_label_0", str(int((subtrain_df["label"] == 0).sum()))),
            ("subtrain_label_1", str(int((subtrain_df["label"] == 1).sum()))),
            ("subval_rows", str(len(subval_df))),
            ("subval_label_0", str(int((subval_df["label"] == 0).sum()))),
            ("subval_label_1", str(int((subval_df["label"] == 1).sum()))),
        ]
    )

    if args.test_csv:
        # Optional Step 7: prepare filtered test FASTA and audit train/test overlap using
        # `cd-hit-2d`. This does not change the split; it is only a diagnostic.
        test_df = load_and_filter_csv(args.test_csv, args.max_seq_len)
        test_fasta = out_dir / "test_filtered_1022.fasta"
        write_fasta(test_df, test_fasta)
        audit_path = run_cdhit_2d(test_fasta, train_fasta, out_dir / "test_vs_train_audit", args.identity, args.word_size)
        summary.append(("test_vs_train_audit_clstr", str(audit_path)))

    # Step 8: save the summary last, after all outputs are known.
    write_summary(out_dir / "homology_split_summary.csv", summary)

    print(f"Filtered training CSV: {filtered_csv}")
    print(f"Subtrain CSV: {out_dir / 'train_subtrain_1022_h40.csv'}")
    print(f"Subval CSV: {out_dir / 'train_subval_1022_h40.csv'}")
    print(f"Summary: {out_dir / 'homology_split_summary.csv'}")


if __name__ == "__main__":
    main()
