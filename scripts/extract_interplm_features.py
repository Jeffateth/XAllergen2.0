#!/usr/bin/env python3
"""Extract protein-level feature summaries from a pretrained InterPLM SAE.

Input CSV schema:
    protein_id,sequence,label

`label` is optional for extraction. When present, expected values are 0/1.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


MODEL_NAME_MAP = {
    "esm2-8m": "esm2_t6_8M_UR50D",
    "esm2-650m": "esm2_t33_650M_UR50D",
}

VALID_LAYERS = {
    "esm2-8m": {1, 2, 3, 4, 5, 6},
    "esm2-650m": {1, 9, 18, 24, 30, 33},
}

LOCAL_INTERPLM_SRC = Path(__file__).resolve().parents[1] / "external" / "InterPLM"
LOCAL_INTERPLM_DATA = Path(__file__).resolve().parents[1] / "outputs" / "interplm_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="CSV with protein_id,sequence,label")
    parser.add_argument("--output-npz", required=True, help="Path to output .npz feature bundle")
    parser.add_argument(
        "--plm-model",
        default="esm2-8m",
        choices=sorted(MODEL_NAME_MAP),
        help="Pretrained ESM-2 backbone matched to the InterPLM SAE.",
    )
    parser.add_argument(
        "--plm-layer",
        type=int,
        default=4,
        help="ESM-2 layer to extract from. Must match an available InterPLM checkpoint.",
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.5,
        help="Threshold used when computing per-feature active residue fraction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of proteins for a smoke test.",
    )
    return parser.parse_args()


def bootstrap_local_interplm_source() -> None:
    """Prefer the vendored InterPLM source tree over the broken installed wheel."""
    os.environ.setdefault("INTERPLM_DATA", str(LOCAL_INTERPLM_DATA))
    LOCAL_INTERPLM_DATA.mkdir(parents=True, exist_ok=True)
    if LOCAL_INTERPLM_SRC.exists():
        sys.path.insert(0, str(LOCAL_INTERPLM_SRC))


def load_interplm():
    bootstrap_local_interplm_source()
    try:
        from interplm.embedders.esm import ESM
        from interplm.sae.inference import load_sae_from_hf
    except ImportError as exc:
        raise SystemExit(
            "Could not import InterPLM. Expected vendored source at:\n"
            f"  {LOCAL_INTERPLM_SRC}\n"
            "If that folder is missing, clone the repo there or install InterPLM manually."
        ) from exc
    return ESM, load_sae_from_hf


def read_input_table(path: Path, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"protein_id", "sequence"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV is missing required columns: {sorted(missing)}")
    if limit is not None:
        df = df.head(limit).copy()
    df["sequence"] = df["sequence"].astype(str).str.strip()
    df = df[df["sequence"] != ""].copy()
    return df


def summarize_feature_matrix(feature_matrix: torch.Tensor, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feature_matrix.ndim != 2:
        raise ValueError(f"Expected a 2D [residue, feature] tensor, got shape {tuple(feature_matrix.shape)}")
    feature_matrix = feature_matrix.detach().cpu()
    max_act = feature_matrix.max(dim=0).values.numpy()
    mean_act = feature_matrix.mean(dim=0).numpy()
    frac_active = (feature_matrix > threshold).float().mean(dim=0).numpy()
    return max_act, mean_act, frac_active


def main() -> None:
    args = parse_args()
    if args.plm_layer not in VALID_LAYERS[args.plm_model]:
        valid = sorted(VALID_LAYERS[args.plm_model])
        raise SystemExit(f"Invalid layer {args.plm_layer} for {args.plm_model}. Valid layers: {valid}")

    ESM, load_sae_from_hf = load_interplm()
    df = read_input_table(Path(args.input_csv), args.limit)

    sae = load_sae_from_hf(plm_model=args.plm_model, plm_layer=args.plm_layer)
    sae.eval()
    embedder = ESM(model_name=MODEL_NAME_MAP[args.plm_model])

    protein_ids: list[str] = []
    labels: list[float] = []
    lengths: list[int] = []
    max_rows: list[np.ndarray] = []
    mean_rows: list[np.ndarray] = []
    frac_rows: list[np.ndarray] = []

    label_present = "label" in df.columns

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Extracting features"):
        sequence = row.sequence
        embedding_dict = embedder.extract_embeddings_with_boundaries(
            sequences=[sequence],
            layer=args.plm_layer,
            batch_size=1,
        )
        embeddings = embedding_dict["embeddings"]
        with torch.no_grad():
            features = sae.encode(embeddings)
        if not torch.is_tensor(features):
            features = torch.as_tensor(features)
        max_act, mean_act, frac_active = summarize_feature_matrix(features, args.activation_threshold)

        protein_ids.append(str(row.protein_id))
        lengths.append(len(sequence))
        labels.append(float(row.label) if label_present else np.nan)
        max_rows.append(max_act)
        mean_rows.append(mean_act)
        frac_rows.append(frac_active)

    max_matrix = np.stack(max_rows, axis=0)
    mean_matrix = np.stack(mean_rows, axis=0)
    frac_matrix = np.stack(frac_rows, axis=0)
    feature_ids = np.array([f"f_{idx}" for idx in range(max_matrix.shape[1])], dtype=object)

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        protein_id=np.array(protein_ids, dtype=object),
        label=np.array(labels, dtype=np.float32),
        length=np.array(lengths, dtype=np.int32),
        feature_id=feature_ids,
        feature_max=max_matrix.astype(np.float32),
        feature_mean=mean_matrix.astype(np.float32),
        feature_frac_active=frac_matrix.astype(np.float32),
        plm_model=np.array(args.plm_model, dtype=object),
        plm_layer=np.array(args.plm_layer, dtype=np.int32),
        activation_threshold=np.array(args.activation_threshold, dtype=np.float32),
    )

    print(f"Saved feature bundle to {output_path}")
    print(f"Proteins: {len(protein_ids)}")
    print(f"Features: {max_matrix.shape[1]}")


if __name__ == "__main__":
    main()
