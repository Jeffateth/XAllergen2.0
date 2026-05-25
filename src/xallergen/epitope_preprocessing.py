"""Convert epitopepredict T-cell prediction outputs to per-residue feature lookup dicts.

Per-residue epitope score  e_i ∈ [0, 1]  used in the attention regularization loss:

    L = λ_cls · L_cls  +  λ_reg · 1[y=1] · (1/L) · Σ_i α_i (1 - e_i)

Two modes
---------
score
    Max-pool raw TEpitope log-odds scores across all overlapping 15-mer peptides
    and all HLA-DR alleles.  Negative scores are clipped to 0 (below-threshold
    binders treated the same as uncovered positions).  The resulting vector is
    min-max normalized *per protein* to [0, 1].

binder
    Binary.  A residue is marked 1 if it is covered by at least one peptide
    whose ``rank`` column satisfies  rank <= rank_threshold  (for any allele).
    Default threshold: 10  (top-10 ranked binders per allele).

Column conventions (raw_predictions / binders files)
-----------------------------------------------------
allele   : HLA allele string
name     : protein sequence_id  (same as in train/test CSV)
peptide  : 15-mer sequence
pos      : 0-based start position of the peptide in the protein sequence
rank     : integer rank within the allele (lower = stronger binder)
score    : raw TEPITOPE log-odds score (higher = stronger binder; range ≈ -10..+4)

Non-allergen proteins are assigned ``None`` in the returned lookup dict so that
the existing ``has_rSASA`` masking in ``collate_batch`` naturally excludes them
from the regularization loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

EpitopeMode = Literal["score", "binder"]

PEPTIDE_LENGTH: int = 15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_seq_maps(frame: pd.DataFrame) -> tuple[dict[str, int], dict[str, str]]:
    """Return (sequence_id → length, sequence_id → sequence) from a DataFrame."""
    len_map: dict[str, int] = {}
    seq_map: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        sid = str(row.sequence_id).strip()
        seq = str(row.sequence).strip().upper()
        len_map[sid] = len(seq)
        seq_map[sid] = seq
    return len_map, seq_map


def _max_pool_score(
    group: pd.DataFrame,
    L: int,
    peptide_length: int,
) -> np.ndarray:
    """Return a per-residue max-pooled score vector (not yet normalized)."""
    vec = np.full(L, -np.inf, dtype=np.float32)
    starts  = group["pos"].to_numpy(dtype=np.int32)
    scores  = group["score"].to_numpy(dtype=np.float32)
    offsets = np.arange(peptide_length, dtype=np.int32)

    for j in range(len(starts)):
        s = int(starts[j])
        e = min(s + peptide_length, L)
        if s >= L or s < 0:
            continue
        end_off = e - s
        positions = (s + offsets[:end_off]).clip(0, L - 1)
        np.maximum.at(vec, positions, scores[j])

    # Residues with no prediction → treat as 0 (below-threshold)
    vec = np.where(np.isfinite(vec), vec, 0.0)
    # Clip negatives: below-threshold binders treated like non-covered positions
    vec = np.maximum(vec, 0.0)
    # Per-protein min-max normalize to [0, 1]
    max_val = float(vec.max())
    if max_val > 0.0:
        vec = vec / max_val
    return vec


def _binary_binder(
    group: pd.DataFrame,
    L: int,
    peptide_length: int,
    rank_threshold: float,
) -> np.ndarray:
    """Return a binary per-residue vector: 1 where a top-ranked binder covers the residue."""
    vec     = np.zeros(L, dtype=np.float32)
    mask    = group["rank"].to_numpy(dtype=np.float32) <= rank_threshold
    starts  = group["pos"].to_numpy(dtype=np.int32)[mask]
    offsets = np.arange(peptide_length, dtype=np.int32)

    for s in starts:
        s = int(s)
        e = min(s + peptide_length, L)
        if s >= L or s < 0:
            continue
        end_off = e - s
        positions = (s + offsets[:end_off]).clip(0, L - 1)
        vec[positions] = 1.0
    return vec


def _pad_special_tokens(vec: np.ndarray) -> np.ndarray:
    """Prepend and append a zero for BOS/EOS tokens (add_special_tokens=True)."""
    return np.concatenate([[0.0], vec, [0.0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_epitope_residue_lookup(
    predictions_path: Path,
    frame: pd.DataFrame,
    add_special_tokens: bool = False,
    mode: EpitopeMode = "score",
    rank_threshold: float = 10.0,
    peptide_length: int = PEPTIDE_LENGTH,
) -> dict[str, torch.Tensor | None]:
    """Build a per-residue epitope feature lookup dict from an epitopepredict output CSV.

    Parameters
    ----------
    predictions_path:
        Path to ``*_tepitope_raw_predictions.csv.gz`` (for ``mode="score"``) or
        ``*_tepitope_binders.csv.gz`` (for ``mode="binder"``).
    frame:
        Full protein DataFrame (train or test, including non-allergens).
        Non-allergens receive ``None`` and are excluded from the loss.
    add_special_tokens:
        If True, prepend and append a zero for BOS/EOS tokens.
        Must match the value used in the tokenizer call.  All existing sweeps
        use ``False``.
    mode:
        ``"score"``: continuous, max-pooled and per-protein normalized.
        ``"binder"``: binary; 1 where covered by a peptide with rank ≤ threshold.
    rank_threshold:
        Used only for ``mode="binder"``.  Residues covered by a peptide whose
        ``rank`` column is ≤ this value (for any allele) are marked 1.
    peptide_length:
        Expected peptide length (default 15).

    Returns
    -------
    dict mapping sequence_id → Tensor of shape (L,) for allergens that have
    predictions, and ``None`` for non-allergens and unpredicted allergens.
    """
    len_map, _ = _build_seq_maps(frame)

    # Initialise all proteins to None (non-allergens stay None)
    lookup: dict[str, torch.Tensor | None] = {sid: None for sid in len_map}

    pred_df = pd.read_csv(predictions_path, dtype={"pos": "int32", "score": "float32", "rank": "float32"})

    for name, group in pred_df.groupby("name", sort=False):
        sid = str(name).strip()
        if sid not in len_map:
            continue
        L = len_map[sid]
        if L < peptide_length:
            # Protein shorter than peptide length — no predictions expected; leave as None
            continue

        if mode == "score":
            vec = _max_pool_score(group, L, peptide_length)
        else:
            vec = _binary_binder(group, L, peptide_length, rank_threshold)

        if add_special_tokens:
            vec = _pad_special_tokens(vec)

        expected_len = L + (2 if add_special_tokens else 0)
        if vec.shape[0] != expected_len:
            raise ValueError(
                f"Epitope vector length mismatch for {sid}: "
                f"got {vec.shape[0]}, expected {expected_len}"
            )

        lookup[sid] = torch.tensor(vec, dtype=torch.float32)

    return lookup


def load_epitope_lookup_dicts(
    train_predictions_path: Path,
    test_predictions_path: Path,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    add_special_tokens: bool = False,
    mode: EpitopeMode = "score",
    rank_threshold: float = 10.0,
    peptide_length: int = PEPTIDE_LENGTH,
) -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor | None], dict[str, Any]]:
    """Load train and test epitope lookup dicts (mirrors ``load_rsa_lookup_dicts``).

    Returns
    -------
    train_lookup, test_lookup, summary_dict
        summary_dict has ``"train"`` and ``"test"`` keys with coverage statistics.
    """
    train_lookup = build_epitope_residue_lookup(
        train_predictions_path, train_frame,
        add_special_tokens=add_special_tokens,
        mode=mode, rank_threshold=rank_threshold, peptide_length=peptide_length,
    )
    test_lookup = build_epitope_residue_lookup(
        test_predictions_path, test_frame,
        add_special_tokens=add_special_tokens,
        mode=mode, rank_threshold=rank_threshold, peptide_length=peptide_length,
    )

    def _summarize(lookup: dict, frame: pd.DataFrame, split: str) -> dict[str, Any]:
        n_total     = len(frame)
        n_allergen  = int((frame["label"] == 1).sum())
        n_with_feat = sum(1 for v in lookup.values() if v is not None)
        return {
            "split":              split,
            "n_total_proteins":   n_total,
            "n_allergens":        n_allergen,
            "n_with_epitope_vec": n_with_feat,
            "coverage_allergens": round(n_with_feat / n_allergen, 4) if n_allergen else 0.0,
            "mode":               mode,
            "predictions_path":   str(train_predictions_path if split == "train" else test_predictions_path),
        }

    summary = {
        "train": _summarize(train_lookup, train_frame, "train"),
        "test":  _summarize(test_lookup,  test_frame,  "test"),
    }
    return train_lookup, test_lookup, summary


def inspect_epitope_inputs(
    train_predictions_path: Path,
    test_predictions_path: Path,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    mode: EpitopeMode = "score",
    rank_threshold: float = 10.0,
) -> pd.DataFrame:
    """Return a two-row summary DataFrame (train / test) for notebook display."""
    rows = []
    for path, frame, split in [
        (train_predictions_path, train_frame, "train"),
        (test_predictions_path,  test_frame,  "test"),
    ]:
        pred_df = pd.read_csv(path, dtype={"score": "float32", "rank": "float32"})
        n_allergen = int((frame["label"] == 1).sum())
        n_covered  = pred_df["name"].nunique()
        rows.append({
            "path":              str(path),
            "split":             split,
            "n_prediction_rows": len(pred_df),
            "n_unique_proteins": n_covered,
            "n_allergens_frame": n_allergen,
            "coverage":          round(n_covered / n_allergen, 4) if n_allergen else 0.0,
            "score_min":         float(pred_df["score"].min()),
            "score_max":         float(pred_df["score"].max()),
            "rank_min":          float(pred_df["rank"].min()),
            "rank_max":          float(pred_df["rank"].max()),
            "mode":              mode,
            "rank_threshold":    rank_threshold if mode == "binder" else None,
        })
    return pd.DataFrame(rows)
