"""Run allergen-only DeepAlgPro MHC-II epitope prediction with epitopepredict."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd

from .rsa_preprocessing import DEFAULT_TEST_CSV, DEFAULT_TRAIN_CSV, load_dataset, summarize_lengths


__all__ = [
    "DEFAULT_ALLELES",
    "DEFAULT_BINDER_CUTOFF",
    "DEFAULT_EPITOPE_LENGTH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PROMISCUOUS_N",
    "run_epitopepredict_tcell_workflow",
]

DEFAULT_OUTPUT_DIR = Path("data/epitopepredict_tcell")
DEFAULT_PREDICTIONS_DIRNAME = "predictions"
DEFAULT_SUMMARY_JSON = "deepalgpro_epitopepredict_tcell_summary.json"
DEFAULT_MANIFEST_CSV = "deepalgpro_epitopepredict_tcell_manifest.csv"

DEFAULT_EPITOPE_LENGTH = 15
DEFAULT_BINDER_CUTOFF = 0.95
DEFAULT_PROMISCUOUS_N = 2
DEFAULT_THREADS = min(8, os.cpu_count() or 1)
DEFAULT_ALLELES = [
    "HLA-DRB1*01:01",
    "HLA-DRB1*03:01",
    "HLA-DRB1*04:01",
    "HLA-DRB1*07:01",
    "HLA-DRB1*11:01",
    "HLA-DRB1*13:01",
    "HLA-DRB1*15:01",
]


def _ensure_epitopepredict_home(repo_home: Path) -> None:
    """Redirect epitopepredict's config writes into the repository."""
    os.environ["HOME"] = str(repo_home)


def _load_epitopepredict():
    """Import epitopepredict lazily after HOME is set."""
    from epitopepredict import base

    return base


def _predict_chunk(task: tuple[list[dict[str, str]], list[str], int, str]) -> pd.DataFrame:
    """Run one epitopepredict chunk in a worker process."""
    records_payload, alleles, epitope_length, repo_home = task
    _ensure_epitopepredict_home(Path(repo_home))
    base = _load_epitopepredict()
    predictor = base.get_predictor("tepitope")
    records = pd.DataFrame.from_records(records_payload)
    return predictor.predict_sequences(
        records,
        alleles=alleles,
        key="locus_tag",
        seqkey="translation",
        length=epitope_length,
        overlap=1,
        threads=1,
    )


def _filter_allergenic_only(split_df: pd.DataFrame) -> pd.DataFrame:
    return split_df.loc[split_df["label"].eq(1)].reset_index(drop=True)


def _to_epitopepredict_records(split_df: pd.DataFrame) -> pd.DataFrame:
    records = split_df.loc[:, ["sequence_id", "sequence"]].copy()
    records = records.rename(columns={"sequence_id": "locus_tag", "sequence": "translation"})
    return records


def _build_manifest_frame(split_name: str, split_df: pd.DataFrame) -> pd.DataFrame:
    frame = split_df.copy()
    frame["split"] = split_name
    frame["sequence_length"] = frame["sequence"].str.len().astype(int)
    frame["source_row_index"] = range(len(frame))
    return frame.loc[
        :,
        ["sequence_id", "split", "label", "sequence_length", "source_row_index"],
    ].copy()


def _predict_one_split(
    split_name: str,
    split_df: pd.DataFrame,
    output_dir: Path,
    alleles: list[str],
    epitope_length: int,
    binder_cutoff: float,
    promiscuous_n: int,
    threads: int,
) -> dict[str, object]:
    records = _to_epitopepredict_records(split_df)
    repo_home = output_dir.resolve().parent.parent
    base = _load_epitopepredict()
    predictor = base.get_predictor("tepitope")

    if threads == 1:
        raw_predictions = predictor.predict_sequences(
            records,
            alleles=alleles,
            key="locus_tag",
            seqkey="translation",
            length=epitope_length,
            overlap=1,
            threads=1,
        )
    else:
        chunk_size = math.ceil(len(records) / threads)
        tasks = []
        for start in range(0, len(records), chunk_size):
            stop = start + chunk_size
            chunk = records.iloc[start:stop].to_dict(orient="records")
            tasks.append((chunk, alleles, epitope_length, str(repo_home)))
        with mp.get_context("spawn").Pool(processes=threads) as pool:
            chunk_results = pool.map(_predict_chunk, tasks)
        raw_predictions = pd.concat(chunk_results, ignore_index=True)
    if raw_predictions is None:
        raise RuntimeError(f"epitopepredict returned no predictions for split {split_name}")

    raw_predictions = raw_predictions.reset_index(drop=True)
    raw_predictions_path = output_dir / f"deepalgpro_{split_name}_tepitope_raw_predictions.csv.gz"
    raw_predictions.to_csv(raw_predictions_path, index=False, compression="gzip")

    binders = predictor.get_binders(cutoff=binder_cutoff, cutoff_method="default")
    if binders is None:
        binders = pd.DataFrame(columns=list(raw_predictions.columns))
    else:
        binders = binders.reset_index(drop=True)
    binders_path = output_dir / f"deepalgpro_{split_name}_tepitope_binders.csv.gz"
    binders.to_csv(binders_path, index=False, compression="gzip")

    promiscuous = predictor.promiscuous_binders(
        binders=binders,
        cutoff=binder_cutoff,
        cutoff_method="default",
        n=promiscuous_n,
        unique_core=True,
    )
    if promiscuous is None:
        promiscuous = pd.DataFrame()
    else:
        promiscuous = promiscuous.reset_index(drop=True)
    promiscuous_path = output_dir / f"deepalgpro_{split_name}_tepitope_promiscuous_binders.csv.gz"
    promiscuous.to_csv(promiscuous_path, index=False, compression="gzip")

    return {
        "total_sequences": int(len(split_df)),
        "sequence_length_summary": summarize_lengths(split_df["sequence"].str.len()),
        "raw_prediction_rows": int(len(raw_predictions)),
        "binder_rows": int(len(binders)),
        "promiscuous_binder_rows": int(len(promiscuous)),
        "raw_predictions_path": str(raw_predictions_path),
        "binders_path": str(binders_path),
        "promiscuous_binders_path": str(promiscuous_path),
    }


def run_epitopepredict_tcell_workflow(
    train_csv: Path = DEFAULT_TRAIN_CSV,
    test_csv: Path = DEFAULT_TEST_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    alleles: list[str] | None = None,
    epitope_length: int = DEFAULT_EPITOPE_LENGTH,
    binder_cutoff: float = DEFAULT_BINDER_CUTOFF,
    promiscuous_n: int = DEFAULT_PROMISCUOUS_N,
    threads: int = DEFAULT_THREADS,
) -> dict[str, object]:
    """Run allergen-only MHC-II T-cell epitope prediction with epitopepredict."""
    train_csv = train_csv.resolve()
    test_csv = test_csv.resolve()
    output_dir = output_dir.resolve()

    if epitope_length <= 0:
        raise ValueError("epitope_length must be a positive integer")
    if not 0 < binder_cutoff <= 1:
        raise ValueError("binder_cutoff must be in the interval (0, 1]")
    if promiscuous_n <= 0:
        raise ValueError("promiscuous_n must be a positive integer")
    if threads <= 0:
        raise ValueError("threads must be a positive integer")

    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_epitopepredict_home(output_dir.parent.parent)

    selected_alleles = list(DEFAULT_ALLELES if alleles is None else alleles)
    train_df = _filter_allergenic_only(load_dataset(train_csv, "train"))
    test_df = _filter_allergenic_only(load_dataset(test_csv, "test"))

    manifest_df = pd.concat(
        [
            _build_manifest_frame("train", train_df),
            _build_manifest_frame("test", test_df),
        ],
        ignore_index=True,
    )
    manifest_path = output_dir / DEFAULT_MANIFEST_CSV
    manifest_df.to_csv(manifest_path, index=False)

    split_summaries = {}
    for split_name, split_df in (("train", train_df), ("test", test_df)):
        split_summaries[split_name] = _predict_one_split(
            split_name=split_name,
            split_df=split_df,
            output_dir=output_dir,
            alleles=selected_alleles,
            epitope_length=epitope_length,
            binder_cutoff=binder_cutoff,
            promiscuous_n=promiscuous_n,
            threads=threads,
        )

    summary = {
        "predictor": "epitopepredict_tepitope",
        "label_filter": "allergenic_only",
        "epitope_length": int(epitope_length),
        "binder_cutoff": float(binder_cutoff),
        "promiscuous_n": int(promiscuous_n),
        "threads": int(threads),
        "alleles": selected_alleles,
        "manifest_csv": str(manifest_path),
        "train": split_summaries["train"],
        "test": split_summaries["test"],
    }
    summary_path = output_dir / DEFAULT_SUMMARY_JSON
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_json"] = str(summary_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allele",
        dest="alleles",
        action="append",
        default=None,
        help="Add one HLA-DR allele. Repeat this flag to build a custom panel.",
    )
    parser.add_argument("--epitope-length", type=int, default=DEFAULT_EPITOPE_LENGTH)
    parser.add_argument("--binder-cutoff", type=float, default=DEFAULT_BINDER_CUTOFF)
    parser.add_argument("--promiscuous-n", type=int, default=DEFAULT_PROMISCUOUS_N)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    summary = run_epitopepredict_tcell_workflow(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        alleles=args.alleles,
        epitope_length=args.epitope_length,
        binder_cutoff=args.binder_cutoff,
        promiscuous_n=args.promiscuous_n,
        threads=args.threads,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
