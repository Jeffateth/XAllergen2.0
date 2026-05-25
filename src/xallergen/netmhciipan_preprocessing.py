"""Utilities for preparing DeepAlgPro FASTA splits for NetMHCIIpan 4.1 EL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .rsa_preprocessing import DEFAULT_TEST_CSV, DEFAULT_TRAIN_CSV, load_dataset, summarize_lengths


__all__ = [
    "DEFAULT_FASTA_DIR",
    "DEFAULT_MANIFEST_CSV",
    "DEFAULT_MERGED_RESULTS_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RAW_RESULTS_DIR",
    "DEFAULT_TEST_CSV",
    "DEFAULT_TRAIN_CSV",
    "DEFAULT_MIN_SEQUENCE_LENGTH",
    "export_deepalgpro_for_netmhciipan41el",
    "merge_netmhciipan41el_outputs",
]

DEFAULT_OUTPUT_DIR = Path("data/netmhciipan41_el")
DEFAULT_FASTA_DIR = DEFAULT_OUTPUT_DIR / "fasta"
DEFAULT_RAW_RESULTS_DIR = DEFAULT_OUTPUT_DIR / "raw_results"
DEFAULT_MERGED_RESULTS_DIR = DEFAULT_OUTPUT_DIR / "merged_results"
DEFAULT_MANIFEST_CSV = DEFAULT_OUTPUT_DIR / "deepalgpro_netmhciipan41el_manifest.csv"
DEFAULT_EXPORT_SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "deepalgpro_netmhciipan41el_export_summary.json"
DEFAULT_MERGE_SUMMARY_JSON = DEFAULT_MERGED_RESULTS_DIR / "deepalgpro_netmhciipan41el_merge_summary.json"

DEFAULT_MIN_SEQUENCE_LENGTH = 15
SHORT_BUCKET = "shorter_than_15"
LONG_BUCKET = "length_15_or_more"
TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".xls"}
ID_COLUMN_CANDIDATES = (
    "sequence_id",
    "protein_id",
    "protein id",
    "id",
    "name",
    "identity",
    "seq_name",
    "identifier",
)


def _write_fasta(output_fasta: Path, split_df: pd.DataFrame) -> None:
    """Write one FASTA file for a dataframe of sequences."""
    with output_fasta.open("w", encoding="utf-8") as handle:
        for row in split_df.itertuples(index=False):
            handle.write(f">{row.sequence_id}\n{row.sequence}\n")


def _bucket_name(sequence_length: int, min_sequence_length: int) -> str:
    return SHORT_BUCKET if sequence_length < min_sequence_length else LONG_BUCKET


def _build_fasta_name(split_name: str, bucket_name: str, min_sequence_length: int) -> str:
    return (
        f"deepalgpro_{split_name}_{bucket_name}_for_netmhciipan41el_epitope_len_{min_sequence_length}.fasta"
    )


def _build_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "output_dir": output_dir,
        "fasta_dir": output_dir / "fasta",
        "raw_results_dir": output_dir / "raw_results",
        "merged_results_dir": output_dir / "merged_results",
        "manifest_csv": output_dir / DEFAULT_MANIFEST_CSV.name,
        "export_summary_json": output_dir / DEFAULT_EXPORT_SUMMARY_JSON.name,
        "merge_summary_json": output_dir / "merged_results" / DEFAULT_MERGE_SUMMARY_JSON.name,
    }


def _validate_non_overlapping_ids(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_ids = set(train_df["sequence_id"])
    test_ids = set(test_df["sequence_id"])
    overlap = sorted(train_ids.intersection(test_ids))
    if overlap:
        preview = ", ".join(overlap[:10])
        raise ValueError(f"Train/test sequence_id overlap detected: {preview}")


def _filter_allergenic_only(split_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only allergenic proteins for NetMHCIIpan export."""
    return split_df.loc[split_df["label"].eq(1)].reset_index(drop=True)


def _build_manifest_frame(
    split_name: str,
    split_df: pd.DataFrame,
    min_sequence_length: int,
) -> pd.DataFrame:
    frame = split_df.copy()
    frame["sequence_length"] = frame["sequence"].str.len().astype(int)
    frame["submission_bucket"] = frame["sequence_length"].map(
        lambda value: _bucket_name(int(value), min_sequence_length)
    )
    frame["eligible_for_len15_prediction"] = frame["submission_bucket"].eq(LONG_BUCKET)
    frame["split"] = split_name
    frame["fasta_filename"] = frame["submission_bucket"].map(
        lambda bucket: _build_fasta_name(split_name, bucket, min_sequence_length)
    )
    frame["source_row_index"] = range(len(frame))
    return frame.loc[
        :,
        [
            "sequence_id",
            "split",
            "label",
            "sequence_length",
            "submission_bucket",
            "eligible_for_len15_prediction",
            "fasta_filename",
            "source_row_index",
        ],
    ].copy()


def export_deepalgpro_for_netmhciipan41el(
    train_csv: Path = DEFAULT_TRAIN_CSV,
    test_csv: Path = DEFAULT_TEST_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    min_sequence_length: int = DEFAULT_MIN_SEQUENCE_LENGTH,
) -> dict[str, object]:
    """Export allergenic DeepAlgPro FASTA files split by train/test and sequence-length bucket."""
    if min_sequence_length <= 0:
        raise ValueError("min_sequence_length must be a positive integer")

    output_paths = _build_output_paths(output_dir)
    fasta_dir = output_paths["fasta_dir"]
    raw_results_dir = output_paths["raw_results_dir"]
    merged_results_dir = output_paths["merged_results_dir"]
    fasta_dir.mkdir(parents=True, exist_ok=True)
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    merged_results_dir.mkdir(parents=True, exist_ok=True)

    train_df = _filter_allergenic_only(load_dataset(train_csv, "train"))
    test_df = _filter_allergenic_only(load_dataset(test_csv, "test"))
    _validate_non_overlapping_ids(train_df, test_df)

    manifest_frames = []
    split_summaries: dict[str, dict[str, object]] = {}
    for split_name, split_df in (("train", train_df), ("test", test_df)):
        manifest_frame = _build_manifest_frame(split_name, split_df, min_sequence_length)
        manifest_frames.append(manifest_frame)

        split_summary = {
            "total_sequences": int(len(split_df)),
            "sequence_length_summary": summarize_lengths(split_df["sequence"].str.len()),
        }
        split_summaries[split_name] = split_summary

        for bucket_name in (SHORT_BUCKET, LONG_BUCKET):
            bucket_mask = manifest_frame["submission_bucket"].eq(bucket_name).to_numpy()
            bucket_df = split_df.loc[bucket_mask].reset_index(drop=True)
            fasta_name = _build_fasta_name(split_name, bucket_name, min_sequence_length)
            output_fasta = fasta_dir / fasta_name
            _write_fasta(output_fasta, bucket_df)
            split_summary[f"{bucket_name}_sequences"] = int(len(bucket_df))
            split_summary[f"{bucket_name}_fasta"] = str(output_fasta)

    manifest_df = pd.concat(manifest_frames, ignore_index=True)
    manifest_df.to_csv(output_paths["manifest_csv"], index=False)

    summary = {
        "min_sequence_length": int(min_sequence_length),
        "label_filter": "allergenic_only",
        "train": split_summaries["train"],
        "test": split_summaries["test"],
        "manifest_csv": str(output_paths["manifest_csv"]),
        "fasta_dir": str(fasta_dir),
        "raw_results_dir": str(raw_results_dir),
        "merged_results_dir": str(merged_results_dir),
    }
    with output_paths["export_summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _iter_input_tables(result_input: Path) -> Iterable[Path]:
    if result_input.is_file():
        yield result_input
        return

    if not result_input.is_dir():
        raise FileNotFoundError(f"Prediction input not found: {result_input}")

    for path in sorted(result_input.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() in TABLE_SUFFIXES
            and not path.name.startswith(".")
        ):
            yield path


def _infer_separator(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".txt", ".xls"} else ","


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lowered = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _load_result_frame(result_input: Path, id_column: str | None = None) -> tuple[pd.DataFrame, str]:
    frames = []
    inferred_id_column: str | None = id_column

    input_paths = list(_iter_input_tables(result_input))
    if not input_paths:
        raise FileNotFoundError(f"No prediction tables found in {result_input}")

    for path in input_paths:
        frame = pd.read_csv(path, sep=_infer_separator(path), comment="#")
        if frame.empty:
            continue
        frame.columns = [str(column).strip() for column in frame.columns]
        current_id_column = id_column or _find_column(frame.columns, ID_COLUMN_CANDIDATES)
        if current_id_column is None:
            available = ", ".join(frame.columns)
            raise ValueError(
                f"Could not infer a sequence ID column in {path}. "
                f"Tried {ID_COLUMN_CANDIDATES}. Available columns: {available}"
            )
        if inferred_id_column is None:
            inferred_id_column = current_id_column
        frame[current_id_column] = frame[current_id_column].astype(str).str.strip().str.removeprefix(">")
        frame["_source_file"] = str(path)
        frame["_source_row_index"] = range(len(frame))
        frames.append(frame)

    if not frames or inferred_id_column is None:
        raise ValueError(f"No non-empty prediction tables found in {result_input}")

    combined = pd.concat(frames, ignore_index=True)
    if combined[inferred_id_column].eq("").any():
        raise ValueError(f"Empty sequence IDs found in prediction tables under {result_input}")
    return combined, inferred_id_column


def _load_manifest(manifest_csv: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    required_columns = {
        "sequence_id",
        "split",
        "submission_bucket",
        "source_row_index",
    }
    missing = sorted(required_columns - set(manifest.columns))
    if missing:
        raise ValueError(
            f"Manifest {manifest_csv} is missing required columns: {', '.join(missing)}"
        )
    manifest["sequence_id"] = manifest["sequence_id"].astype(str).str.strip()
    manifest["split"] = manifest["split"].astype(str).str.strip()
    manifest["submission_bucket"] = manifest["submission_bucket"].astype(str).str.strip()
    manifest["source_row_index"] = manifest["source_row_index"].astype(int)
    return manifest


def _merge_bucket_results(
    result_input: Path,
    expected_ids: set[str],
    sequence_order: dict[str, int],
    id_column: str | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame, resolved_id_column = _load_result_frame(result_input, id_column=id_column)
    observed_ids = set(frame[resolved_id_column])
    missing_ids = sorted(expected_ids - observed_ids)
    extra_ids = sorted(observed_ids - expected_ids)
    if missing_ids or extra_ids:
        raise ValueError(
            f"Prediction coverage validation failed for {result_input}: "
            f"missing={len(missing_ids)}, extra={len(extra_ids)}"
        )

    frame["_sequence_sort_index"] = frame[resolved_id_column].map(sequence_order)
    frame = frame.sort_values(
        ["_sequence_sort_index", "_source_file", "_source_row_index"],
        kind="stable",
    ).reset_index(drop=True)
    return frame, {
        "result_input": str(result_input),
        "rows": int(len(frame)),
        "sequence_ids": int(len(observed_ids)),
        "id_column": resolved_id_column,
    }


def merge_netmhciipan41el_outputs(
    train_ge15_input: Path,
    train_lt15_input: Path,
    test_ge15_input: Path,
    test_lt15_input: Path,
    manifest_csv: Path = DEFAULT_MANIFEST_CSV,
    output_dir: Path = DEFAULT_MERGED_RESULTS_DIR,
    id_column: str | None = None,
) -> dict[str, object]:
    """Merge separate NetMHCIIpan result tables back into train/test-wide outputs."""
    manifest = _load_manifest(manifest_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "manifest_csv": str(manifest_csv),
        "merged_results_dir": str(output_dir),
    }

    for split_name, ge15_input, lt15_input in (
        ("train", train_ge15_input, train_lt15_input),
        ("test", test_ge15_input, test_lt15_input),
    ):
        split_manifest = manifest.loc[manifest["split"].eq(split_name)].copy()
        split_manifest = split_manifest.sort_values("source_row_index", kind="stable").reset_index(drop=True)
        sequence_order = {
            row.sequence_id: int(row.source_row_index)
            for row in split_manifest[["sequence_id", "source_row_index"]].itertuples(index=False)
        }

        ge15_ids = set(
            split_manifest.loc[split_manifest["submission_bucket"].eq(LONG_BUCKET), "sequence_id"]
        )
        lt15_ids = set(
            split_manifest.loc[split_manifest["submission_bucket"].eq(SHORT_BUCKET), "sequence_id"]
        )

        ge15_frame, ge15_summary = _merge_bucket_results(
            ge15_input,
            expected_ids=ge15_ids,
            sequence_order=sequence_order,
            id_column=id_column,
        )
        lt15_frame, lt15_summary = _merge_bucket_results(
            lt15_input,
            expected_ids=lt15_ids,
            sequence_order=sequence_order,
            id_column=id_column,
        )

        merged = pd.concat([ge15_frame, lt15_frame], ignore_index=True)
        merged = merged.sort_values(
            ["_sequence_sort_index", "_source_file", "_source_row_index"],
            kind="stable",
        ).reset_index(drop=True)
        merged = merged.drop(columns=["_sequence_sort_index", "_source_file", "_source_row_index"])

        output_path = output_dir / f"deepalgpro_{split_name}_netmhciipan41el_merged.tsv"
        merged.to_csv(output_path, sep="\t", index=False)

        summary[split_name] = {
            "merged_output_path": str(output_path),
            "merged_rows": int(len(merged)),
            "expected_sequence_ids": int(len(split_manifest)),
            "ge15": ge15_summary,
            "lt15": lt15_summary,
        }

    merge_summary_json = output_dir / DEFAULT_MERGE_SUMMARY_JSON.name
    with merge_summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export",
        help="Export DeepAlgPro FASTA files for NetMHCIIpan 4.1 EL submission.",
    )
    export_parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    export_parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    export_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    export_parser.add_argument(
        "--min-sequence-length",
        type=int,
        default=DEFAULT_MIN_SEQUENCE_LENGTH,
        help="Split threshold for the chosen epitope length.",
    )

    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge separate train/test NetMHCIIpan outputs back into split-wide tables.",
    )
    merge_parser.add_argument("--train-ge15-input", type=Path, required=True)
    merge_parser.add_argument("--train-lt15-input", type=Path, required=True)
    merge_parser.add_argument("--test-ge15-input", type=Path, required=True)
    merge_parser.add_argument("--test-lt15-input", type=Path, required=True)
    merge_parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    merge_parser.add_argument("--output-dir", type=Path, default=DEFAULT_MERGED_RESULTS_DIR)
    merge_parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Override the sequence ID column if the downloaded tables use a custom header.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "export":
        summary = export_deepalgpro_for_netmhciipan41el(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            output_dir=args.output_dir,
            min_sequence_length=args.min_sequence_length,
        )
    else:
        summary = merge_netmhciipan41el_outputs(
            train_ge15_input=args.train_ge15_input,
            train_lt15_input=args.train_lt15_input,
            test_ge15_input=args.test_ge15_input,
            test_lt15_input=args.test_lt15_input,
            manifest_csv=args.manifest_csv,
            output_dir=args.output_dir,
            id_column=args.id_column,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
