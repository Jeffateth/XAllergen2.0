"""Utilities for exporting DeepAlgPro sequences and parsing NetSurfP RSA output."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_OUTPUT_JSON",
    "DEFAULT_TEST_OUTPUT_JSON",
    "DEFAULT_NETSURFP_MAX_SEQUENCES",
    "DEFAULT_TEST_CSV",
    "DEFAULT_TRAIN_OUTPUT_JSON",
    "DEFAULT_TRAIN_CSV",
    "DEFAULT_TRAIN_SS3_JSON",
    "DEFAULT_TEST_SS3_JSON",
    "compute_rsa_ss3_structured_correlation",
    "export_deepalgpro_for_rsa",
    "extract_ss3_structured_lookup",
    "load_dataset",
    "load_expected_sequences",
    "parse_netsurfp_rsa",
    "parse_split_netsurfp_rsa",
]

REQUIRED_COLUMNS = ("sequence_id", "sequence", "label")
CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_TRAIN_CSV = Path("data/deepalgpro_train_cleaned.csv")
DEFAULT_TEST_CSV = Path("data/deepalgpro_test_cleaned.csv")
DEFAULT_OUTPUT_DIR = Path("data/rsa")
DEFAULT_NETSURFP_MAX_SEQUENCES = 5000
DEFAULT_TRAIN_FASTA_BASENAME = "deepalgpro_train"
DEFAULT_TEST_FASTA_NAME = "deepalgpro_test_for_netsurfp.fasta"
DEFAULT_OUTPUT_JSON = Path("data/rsa/deepalgpro_rsa.json")
DEFAULT_TRAIN_OUTPUT_JSON = Path("data/rsa/deepalgpro_train_rsa.json")
DEFAULT_TEST_OUTPUT_JSON = Path("data/rsa/deepalgpro_test_rsa.json")
TABLE_SUFFIXES = {".csv", ".tsv", ".txt"}
ID_COLUMN_CANDIDATES = ("sequence_id", "id", "name", "seq_name", "identifier")
RSA_COLUMN_CANDIDATES = (
    "rsa",
    "rel_sasa",
    "relative_surface_accessibility",
    "relative_solvent_accessibility",
)
RESIDUE_COLUMN_CANDIDATES = ("residue", "aa", "seq", "sequence")
Q3_COLUMN_CANDIDATES = ("q3",)
DEFAULT_TRAIN_SS3_JSON = Path("data/ss3/deepalgpro_train_ss3_structured.json.gz")
DEFAULT_TEST_SS3_JSON = Path("data/ss3/deepalgpro_test_ss3_structured.json.gz")


def _normalize_sequence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped IDs and upper-cased sequences."""
    normalized = df.copy()
    normalized["sequence_id"] = normalized["sequence_id"].astype(str).str.strip()
    normalized["sequence"] = normalized["sequence"].astype(str).str.strip().str.upper()
    return normalized


def _validate_non_empty_fields(df: pd.DataFrame, split_name: str) -> None:
    """Validate that IDs and sequences are present after normalization."""
    if df["sequence_id"].eq("").any():
        raise ValueError(f"{split_name} CSV contains empty sequence_id values")
    if df["sequence"].eq("").any():
        raise ValueError(f"{split_name} CSV contains empty sequence values")


def _validate_canonical_sequences(df: pd.DataFrame, split_name: str) -> None:
    """Validate that every sequence contains only canonical amino acids."""
    invalid_rows = []
    for row in df.itertuples(index=False):
        invalid_residues = sorted(set(row.sequence) - CANONICAL_AMINO_ACIDS)
        if invalid_residues:
            invalid_rows.append((row.sequence_id, "".join(invalid_residues)))
    if invalid_rows:
        preview = ", ".join(
            f"{sequence_id}:[{invalid}]"
            for sequence_id, invalid in invalid_rows[:10]
        )
        raise ValueError(
            f"{split_name} CSV contains non-canonical amino acids: {preview}"
        )


def _load_required_columns(csv_path: Path, split_name: str) -> pd.DataFrame:
    """Load a DeepAlgPro CSV and ensure the expected columns exist."""
    df = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"{split_name} CSV is missing required columns: {', '.join(missing)}"
        )
    return df.loc[:, REQUIRED_COLUMNS].copy()


def load_dataset(csv_path: Path, split_name: str) -> pd.DataFrame:
    """Load, normalize, and validate one DeepAlgPro split."""
    df = _load_required_columns(csv_path, split_name)
    df = _normalize_sequence_columns(df)
    df["split"] = split_name
    _validate_non_empty_fields(df, split_name)
    _validate_canonical_sequences(df, split_name)
    return df


def summarize_lengths(lengths: pd.Series) -> str:
    """Return a compact descriptive summary for sequence lengths."""
    return (
        f"min={int(lengths.min())}, "
        f"median={float(lengths.median()):.1f}, "
        f"mean={float(lengths.mean()):.1f}, "
        f"max={int(lengths.max())}"
    )


def _write_fasta(output_fasta: Path, split_df: pd.DataFrame) -> None:
    """Write one FASTA file for a dataframe of sequences."""
    with output_fasta.open("w", encoding="utf-8") as handle:
        for row in split_df.itertuples(index=False):
            handle.write(f">{row.sequence_id}\n{row.sequence}\n")


def export_deepalgpro_for_rsa(
    train_csv: Path = DEFAULT_TRAIN_CSV,
    test_csv: Path = DEFAULT_TEST_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    train_chunk_size: int = DEFAULT_NETSURFP_MAX_SEQUENCES,
) -> dict[str, object]:
    """Export validated train/test sequences to NetSurfP-ready FASTA files."""
    train_df = load_dataset(train_csv, "train")
    test_df = load_dataset(test_csv, "test")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    duplicate_id_mask = combined_df["sequence_id"].duplicated(keep=False)
    duplicate_sequence_mask = combined_df["sequence"].duplicated(keep=False)

    duplicate_id_count = int(duplicate_id_mask.sum())
    duplicate_sequence_count = int(duplicate_sequence_mask.sum())

    if duplicate_id_count:
        duplicate_ids = (
            combined_df.loc[duplicate_id_mask, "sequence_id"].drop_duplicates().tolist()
        )
        preview = ", ".join(duplicate_ids[:10])
        raise ValueError(
            f"Found {duplicate_id_count} duplicated sequence_id rows across train/test. "
            f"Examples: {preview}"
        )

    lengths = combined_df["sequence"].str.len()
    combined_df = combined_df.assign(sequence_length=lengths)

    if duplicate_sequence_count:
        duplicate_sequences = (
            combined_df.loc[
                duplicate_sequence_mask, ["sequence_id", "sequence_length", "split"]
            ].sort_values(["sequence_length", "sequence_id"])
        )
        print("Exact duplicate sequences detected (not removed):")
        print(duplicate_sequences.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    test_output_fasta = output_dir / DEFAULT_TEST_FASTA_NAME
    train_output_fastas: list[Path] = []

    if train_chunk_size <= 0:
        raise ValueError("train_chunk_size must be a positive integer")

    for chunk_index, start in enumerate(range(0, len(train_df), train_chunk_size), start=1):
        stop = start + train_chunk_size
        train_chunk_df = train_df.iloc[start:stop]
        train_output_fasta = (
            output_dir / f"{DEFAULT_TRAIN_FASTA_BASENAME}_part{chunk_index}_for_netsurfp.fasta"
        )
        _write_fasta(train_output_fasta, train_chunk_df)
        train_output_fastas.append(train_output_fasta)

    _write_fasta(test_output_fasta, test_df)

    return {
        "train_sequences": len(train_df),
        "test_sequences": len(test_df),
        "total_sequences": len(combined_df),
        "duplicate_sequence_ids": duplicate_id_count,
        "exact_duplicate_sequences": duplicate_sequence_count,
        "sequence_length_summary": summarize_lengths(lengths),
        "output_dir": output_dir,
        "train_chunk_size": train_chunk_size,
        "train_output_fastas": train_output_fastas,
        "test_output_fasta": test_output_fasta,
    }


def load_expected_sequences(train_csv: Path, test_csv: Path) -> pd.DataFrame:
    """Load expected IDs and lengths for validating NetSurfP outputs."""
    frames = []
    for split_name, csv_path in (("train", train_csv), ("test", test_csv)):
        df = _load_required_columns(csv_path, split_name)
        df = _normalize_sequence_columns(df)
        _validate_non_empty_fields(df, split_name)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    duplicate_id_mask = combined["sequence_id"].duplicated(keep=False)
    if duplicate_id_mask.any():
        duplicate_ids = combined.loc[duplicate_id_mask, "sequence_id"].drop_duplicates()
        preview = ", ".join(duplicate_ids.head(10))
        raise ValueError(f"Duplicate sequence_id values in input CSVs: {preview}")

    combined["sequence_length"] = combined["sequence"].str.len()
    return combined


def iter_input_tables(netsurfp_input: Path) -> Iterable[Path]:
    """Yield NetSurfP table files from a file path or directory tree."""
    if netsurfp_input.is_file():
        yield netsurfp_input
        return

    if not netsurfp_input.is_dir():
        raise FileNotFoundError(f"NetSurfP input not found: {netsurfp_input}")

    aggregate_csvs = sorted(
        path
        for path in netsurfp_input.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv" and not path.name.startswith(".")
    )
    if aggregate_csvs:
        for path in aggregate_csvs:
            yield path
        return

    for path in sorted(netsurfp_input.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() in TABLE_SUFFIXES
            and not path.name.startswith(".")
        ):
            yield path


def infer_separator(path: Path) -> str:
    """Infer the delimiter from the file suffix."""
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the first matching column name, case-insensitively."""
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def parse_table(path: Path) -> dict[str, list[float]]:
    """Parse one NetSurfP residue-level table into an ID-to-RSA mapping."""
    df = pd.read_csv(path, sep=infer_separator(path))
    if df.empty:
        return {}

    df.columns = [str(column).strip() for column in df.columns]

    id_column = find_column(df.columns, ID_COLUMN_CANDIDATES)
    rsa_column = find_column(df.columns, RSA_COLUMN_CANDIDATES)
    residue_column = find_column(df.columns, RESIDUE_COLUMN_CANDIDATES)

    if id_column is None or rsa_column is None:
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            f"Could not infer NetSurfP columns in {path}. "
            f"Need an ID column from {ID_COLUMN_CANDIDATES} and an RSA column from "
            f"{RSA_COLUMN_CANDIDATES}. Available columns: {available}"
        )

    subset_columns = [id_column, rsa_column]
    if residue_column is not None:
        subset_columns.append(residue_column)
    table = df[subset_columns].copy()
    table[id_column] = table[id_column].astype(str).str.strip().str.removeprefix(">")
    table[rsa_column] = pd.to_numeric(table[rsa_column], errors="coerce")

    invalid_rsa_mask = table[rsa_column].isna()
    if invalid_rsa_mask.any():
        invalid_ids = table.loc[invalid_rsa_mask, id_column].drop_duplicates().tolist()
        preview = ", ".join(invalid_ids[:10])
        raise ValueError(f"Found non-numeric RSA values in {path}: {preview}")

    rsa_by_id: dict[str, list[float]] = {}
    for sequence_id, group in table.groupby(id_column, sort=False):
        if sequence_id == "":
            raise ValueError(f"Found empty sequence_id in {path}")

        values = group[rsa_column].astype(float).tolist()
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"RSA values out of range [0, 1] for {sequence_id} in {path}")

        if residue_column is not None:
            residues = group[residue_column].astype(str).str.strip()
            if residues.eq("").any():
                raise ValueError(f"Found empty residue values for {sequence_id} in {path}")

        if sequence_id in rsa_by_id:
            raise ValueError(f"Duplicate sequence_id {sequence_id} within {path}")
        rsa_by_id[sequence_id] = values

    return rsa_by_id


def parse_netsurfp_rsa(
    netsurfp_input: Path,
    train_csv: Path = DEFAULT_TRAIN_CSV,
    test_csv: Path = DEFAULT_TEST_CSV,
    output_json: Path = DEFAULT_OUTPUT_JSON,
) -> dict[str, object]:
    """Validate and convert NetSurfP RSA output into the training JSON format."""
    expected_df = load_expected_sequences(train_csv, test_csv)
    expected_lengths = dict(
        zip(expected_df["sequence_id"], expected_df["sequence_length"], strict=True)
    )

    rsa_by_id: dict[str, list[float]] = {}
    input_paths = list(iter_input_tables(netsurfp_input))
    if not input_paths:
        raise FileNotFoundError(f"No NetSurfP tables found in {netsurfp_input}")

    for path in input_paths:
        parsed = parse_table(path)
        overlapping_ids = set(rsa_by_id).intersection(parsed)
        if overlapping_ids:
            preview = ", ".join(sorted(overlapping_ids)[:10])
            raise ValueError(f"Duplicate sequence IDs across NetSurfP outputs: {preview}")
        rsa_by_id.update(parsed)

    expected_ids = set(expected_lengths)
    observed_ids = set(rsa_by_id)
    missing_ids = sorted(expected_ids - observed_ids)
    extra_ids = sorted(observed_ids - expected_ids)
    length_mismatches = sorted(
        sequence_id
        for sequence_id, values in rsa_by_id.items()
        if sequence_id in expected_lengths and len(values) != expected_lengths[sequence_id]
    )

    print(f"Expected sequences: {len(expected_ids)}")
    print(f"Parsed RSA entries: {len(rsa_by_id)}")
    print(f"Missing RSA entries: {len(missing_ids)}")
    print(f"Extra RSA entries: {len(extra_ids)}")
    print(f"Length mismatches: {len(length_mismatches)}")

    if missing_ids:
        print("Missing IDs:", ", ".join(missing_ids[:20]))
    if extra_ids:
        print("Extra IDs:", ", ".join(extra_ids[:20]))
    if length_mismatches:
        print("Length-mismatched IDs:", ", ".join(length_mismatches[:20]))

    if missing_ids or extra_ids or length_mismatches:
        raise ValueError("NetSurfP RSA validation failed; see reported issues above.")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(rsa_by_id, handle)

    return {
        "expected_sequences": len(expected_ids),
        "parsed_rsa_entries": len(rsa_by_id),
        "missing_rsa_entries": len(missing_ids),
        "extra_rsa_entries": len(extra_ids),
        "length_mismatches": len(length_mismatches),
        "output_json": output_json,
    }


def parse_split_netsurfp_rsa(
    train_netsurfp_csv: Path,
    test_netsurfp_csv: Path,
    train_csv: Path = DEFAULT_TRAIN_CSV,
    test_csv: Path = DEFAULT_TEST_CSV,
    train_output_json: Path = DEFAULT_TRAIN_OUTPUT_JSON,
    test_output_json: Path = DEFAULT_TEST_OUTPUT_JSON,
) -> dict[str, object]:
    """Write split-specific RSA JSON files without merging train and test."""
    train_expected = load_dataset(train_csv, "train")
    test_expected = load_dataset(test_csv, "test")

    train_ids = set(train_expected["sequence_id"])
    test_ids = set(test_expected["sequence_id"])
    if train_ids.intersection(test_ids):
        preview = ", ".join(sorted(train_ids.intersection(test_ids))[:10])
        raise ValueError(
            "Train/test sequence_id overlap detected while preparing split RSA files: "
            f"{preview}"
        )

    train_rsa = parse_table(train_netsurfp_csv)
    test_rsa = parse_table(test_netsurfp_csv)

    train_missing = sorted(train_ids - set(train_rsa))
    train_extra = sorted(set(train_rsa) - train_ids)
    test_missing = sorted(test_ids - set(test_rsa))
    test_extra = sorted(set(test_rsa) - test_ids)

    train_length_mismatches = sorted(
        sequence_id
        for sequence_id, values in train_rsa.items()
        if sequence_id in train_ids
        and len(values)
        != int(
            train_expected.loc[
                train_expected["sequence_id"].eq(sequence_id), "sequence"
            ].str.len().iloc[0]
        )
    )
    test_length_mismatches = sorted(
        sequence_id
        for sequence_id, values in test_rsa.items()
        if sequence_id in test_ids
        and len(values)
        != int(
            test_expected.loc[
                test_expected["sequence_id"].eq(sequence_id), "sequence"
            ].str.len().iloc[0]
        )
    )

    if train_missing or train_extra or train_length_mismatches:
        raise ValueError(
            "Train NetSurfP RSA validation failed: "
            f"missing={len(train_missing)}, extra={len(train_extra)}, "
            f"length_mismatches={len(train_length_mismatches)}"
        )
    if test_missing or test_extra or test_length_mismatches:
        raise ValueError(
            "Test NetSurfP RSA validation failed: "
            f"missing={len(test_missing)}, extra={len(test_extra)}, "
            f"length_mismatches={len(test_length_mismatches)}"
        )

    train_output_json.parent.mkdir(parents=True, exist_ok=True)
    test_output_json.parent.mkdir(parents=True, exist_ok=True)
    with train_output_json.open("w", encoding="utf-8") as handle:
        json.dump(train_rsa, handle)
    with test_output_json.open("w", encoding="utf-8") as handle:
        json.dump(test_rsa, handle)

    return {
        "train_sequences": len(train_ids),
        "test_sequences": len(test_ids),
        "train_output_json": train_output_json,
        "test_output_json": test_output_json,
    }


def extract_ss3_structured_lookup(
    raw_netsurfp_path: Path,
    frame: pd.DataFrame,
    add_special_tokens: bool = False,  # noqa: ARG001 — reserved for API symmetry with RSA loaders
    output_json: Path = DEFAULT_TRAIN_SS3_JSON,
) -> dict[str, list[float]]:
    """Parse per-residue SS3-structured indicator from a NetSurfP CSV and save as JSON.gz.

    Converts the `q3` column (H/E/C) to a binary float: 1.0 for helix (H) or strand (E),
    0.0 for coil/loop (C). With loss term (1 - f_i), this penalises attention on coil
    residues and rewards attention on structured regions.
    `add_special_tokens` is accepted for API symmetry but not applied to the saved file;
    alignment is handled at load time by `load_precomputed_rsa_mapping`.
    """
    df = pd.read_csv(raw_netsurfp_path)
    df.columns = [str(c).strip() for c in df.columns]

    id_col = find_column(df.columns, ID_COLUMN_CANDIDATES)
    q3_col = find_column(df.columns, Q3_COLUMN_CANDIDATES)

    if id_col is None or q3_col is None:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Could not find required columns in {raw_netsurfp_path}. "
            f"Need an ID column from {ID_COLUMN_CANDIDATES} and a Q3 column from "
            f"{Q3_COLUMN_CANDIDATES}. Available: {available}"
        )

    df[id_col] = df[id_col].astype(str).str.strip().str.removeprefix(">")

    frame = frame.copy()
    frame["sequence_id"] = frame["sequence_id"].astype(str).str.strip()
    frame["sequence"] = frame["sequence"].astype(str).str.strip().str.upper()
    expected_lengths: dict[str, int] = {
        row.sequence_id: len(row.sequence)
        for row in frame[["sequence_id", "sequence"]].itertuples(index=False)
    }

    lookup: dict[str, list[float]] = {}
    for seq_id, group in df.groupby(id_col, sort=False):
        seq_id = str(seq_id).strip()
        if seq_id not in expected_lengths:
            continue
        lookup[seq_id] = [
            1.0 if str(v).strip().upper() in {"H", "E"} else 0.0
            for v in group[q3_col].tolist()
        ]

    missing = sorted(set(expected_lengths) - set(lookup))
    length_mismatches = sorted(
        seq_id for seq_id, vals in lookup.items()
        if len(vals) != expected_lengths.get(seq_id, -1)
    )

    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing ss3_structured entries for {len(missing)} sequences: {preview}")
    if length_mismatches:
        preview = ", ".join(length_mismatches[:10])
        raise ValueError(
            f"ss3_structured residue-count mismatch for {len(length_mismatches)} sequences: {preview}"
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_json, "wt", encoding="utf-8") as handle:
        json.dump(lookup, handle)

    return lookup


def compute_rsa_ss3_structured_correlation(
    rsa_lookup: dict[str, list[float]],
    ss3_structured_lookup: dict[str, list[float]],
) -> float:
    """Pearson r between RSA values and SS3-structured indicator, pooled across shared sequences.

    Call this before running the SS3-structured sweep to assess whether the feature adds
    independent signal over RSA. Low |r| suggests the two constraints are complementary;
    high |r| suggests SS3-structured is largely redundant with RSA.
    """
    import numpy as np

    common_ids = sorted(set(rsa_lookup) & set(ss3_structured_lookup))
    if not common_ids:
        raise ValueError("No shared sequence IDs between RSA and SS3-structured lookups.")

    rsa_vals: list[float] = []
    ss3_vals: list[float] = []
    for seq_id in common_ids:
        rsa_vals.extend(rsa_lookup[seq_id])
        ss3_vals.extend(ss3_structured_lookup[seq_id])

    r = float(np.corrcoef(
        np.asarray(rsa_vals, dtype=np.float64),
        np.asarray(ss3_vals, dtype=np.float64),
    )[0, 1])
    print(
        f"RSA vs SS3-structured Pearson r = {r:.4f} "
        f"(n = {len(rsa_vals):,} residues, {len(common_ids):,} sequences)"
    )
    return r
