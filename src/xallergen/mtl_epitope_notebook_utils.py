from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .baseline_notebook_utils import (
    ESM_MODEL_NAME,
    HF_MODEL_NAME,
    MAX_SEQ_LEN,
    RANDOM_STATE,
    THRESHOLD,
    compute_attention_weights,
    compute_gradient_x_input_scores,
    compute_integrated_gradients,
    compute_residue_probabilities,
    compute_smoothgrad_ig_scores,
    load_baseline_checkpoint,
    load_mtl_checkpoint,
    mean_metric_dicts,
    normalize_scores,
    parse_epitope_label,
    serialize_score_array,
    tokenize_sequence,
)


@dataclass(frozen=True)
class MTLDataPaths:
    positive_train_csv: Path
    positive_test_csv: Path
    negative_train_csv: Path
    negative_test_csv: Path


@dataclass(frozen=True)
class MTLOutputPaths:
    baseline_checkpoint_path: Path
    checkpoint_path: Path
    metrics_path: Path
    probe_rows_path: Path
    baseline_probe_rows_path: Path
    combined_probe_rows_path: Path
    probe_summary_path: Path
    compare_summary_path: Path
    combined_violins_png: Path
    combined_auroc_density_png: Path
    combined_auprc_density_png: Path
    baseline_summary_csv: Path
    mtl_family_label: str = "MTL (05)"
    baseline_family_label: str = "Baseline (04)"


@dataclass(frozen=True)
class MTLHyperparameters:
    classification_batch_size: int = 24
    epochs: int = 30
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_cls: float = 1.0
    lambda_epi: float = 0.5
    epitope_hidden_dim: int = 128
    val_fraction: float = 0.1
    use_protein_pos_weight: bool = False
    protein_imbalance_tolerance: float = 0.1
    n_random_draws: int = 100
    ig_internal_batch_size: int = 10


class MixedAllergenEpitopeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        seq_len = int(row["seq_len"])
        residue_labels = np.asarray(row["epitope_label"], dtype=np.float32)
        if residue_labels.shape[0] != seq_len:
            raise ValueError(
                f"Residue label length mismatch for {row['sequence_id']}: "
                f"labels={residue_labels.shape[0]}, seq_len={seq_len}"
            )
        return {
            "sequence_id": str(row["sequence_id"]),
            "sequence": row["sequence"],
            "protein_label": float(row["protein_label"]),
            "residue_labels": residue_labels,
            "seq_len": seq_len,
            "has_epitope_supervision": int(row["has_epitope_supervision"]),
            "data_source": row["data_source"],
        }


def annotate_epitopes(frame: pd.DataFrame) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["epitope_label"] = [
        parse_epitope_label(seq, start, end)
        for seq, start, end in zip(
            annotated["sequence"],
            annotated["epitope_start"],
            annotated["epitope_end"],
        )
    ]
    annotated["seq_len"] = annotated["sequence"].str.len().astype(int)
    annotated["n_epitope_residues"] = annotated["epitope_label"].map(lambda arr: int(arr.sum()))
    annotated["epitope_density"] = annotated["n_epitope_residues"] / annotated["seq_len"]
    return annotated


def filter_max_len(frame: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
    keep = frame[sequence_col].astype(str).str.len() <= MAX_SEQ_LEN
    return frame.loc[keep].reset_index(drop=True)


def audit_frame(csv_path: Path, frame_name: str, sequence_col: str = "sequence") -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(csv_path)
    filtered = filter_max_len(raw, sequence_col=sequence_col)
    audit = {
        "frame_name": frame_name,
        "csv_path": str(csv_path),
        "raw_rows": len(raw),
        "kept_rows": len(filtered),
        "dropped_rows": len(raw) - len(filtered),
    }
    return filtered, audit


def print_audit_block(audit_rows: list[dict[str, Any]]) -> None:
    print("Data audit:")
    for row in audit_rows:
        print(
            f"  {row['frame_name']}: raw_rows={row['raw_rows']}, "
            f"kept_rows={row['kept_rows']}, dropped_over_max_len={row['dropped_rows']}"
        )


def require_columns(frame: pd.DataFrame, required: list[str], frame_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {frame_name}: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )


def get_sequence_id_column(frame: pd.DataFrame, preferred: list[str], frame_name: str) -> str:
    for column in preferred:
        if column in frame.columns:
            return column
    raise ValueError(
        f"Could not find a sequence identifier column for {frame_name}. "
        f"Tried {preferred}. Available columns: {list(frame.columns)}"
    )


def prepare_positive_frame(csv_path: Path, split_name: str, frame_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, audit = audit_frame(csv_path, frame_name)
    require_columns(frame, ["sequence", "epitope_start", "epitope_end"], csv_path.name)
    sequence_id_col = get_sequence_id_column(frame, ["accession", "sequence_id"], csv_path.name)
    frame = annotate_epitopes(frame)
    frame = frame.copy()
    frame["sequence_id"] = frame[sequence_id_col].astype(str)
    frame["protein_label"] = 1.0
    frame["has_epitope_supervision"] = 1
    frame["split_name"] = split_name
    frame["data_source"] = "positive"
    return frame, audit


def prepare_negative_frame(csv_path: Path, split_name: str, frame_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, audit = audit_frame(csv_path, frame_name)
    require_columns(frame, ["sequence"], csv_path.name)
    sequence_id_col = get_sequence_id_column(frame, ["entry", "sequence_id", "accession"], csv_path.name)
    frame = frame.copy()
    frame["sequence_id"] = frame[sequence_id_col].astype(str)
    frame["seq_len"] = frame["sequence"].str.len().astype(int)
    frame["epitope_label"] = frame["seq_len"].map(lambda seq_len: np.zeros(seq_len, dtype=np.float32))
    frame["n_epitope_residues"] = 0
    frame["epitope_density"] = 0.0
    frame["protein_label"] = 0.0
    frame["has_epitope_supervision"] = 0
    frame["split_name"] = split_name
    frame["data_source"] = "negative"
    return frame, audit


def build_mixed_frame(positive_frame: pd.DataFrame, negative_frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "sequence_id",
        "sequence",
        "protein_label",
        "epitope_label",
        "seq_len",
        "has_epitope_supervision",
        "n_epitope_residues",
        "epitope_density",
        "split_name",
        "data_source",
    ]
    mixed = pd.concat(
        [positive_frame[columns], negative_frame[columns]],
        ignore_index=True,
    )
    return mixed.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def prepare_mtl_splits(
    data_paths: MTLDataPaths,
    val_fraction: float,
) -> dict[str, Any]:
    positive_train_full_df, positive_train_audit = prepare_positive_frame(
        data_paths.positive_train_csv, "splitA", "positive_train_full"
    )
    positive_test_df, positive_test_audit = prepare_positive_frame(
        data_paths.positive_test_csv, "splitB", "positive_test"
    )
    negative_train_full_df, negative_train_audit = prepare_negative_frame(
        data_paths.negative_train_csv, "splitA", "negative_train_full"
    )
    negative_test_df, negative_test_audit = prepare_negative_frame(
        data_paths.negative_test_csv, "splitB", "negative_test"
    )

    positive_train_df, positive_val_df = train_test_split(
        positive_train_full_df,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
    )
    negative_train_df, negative_val_df = train_test_split(
        negative_train_full_df,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
    )

    train_mixed_df = build_mixed_frame(positive_train_df, negative_train_df)
    val_mixed_df = build_mixed_frame(positive_val_df, negative_val_df)
    test_mixed_df = build_mixed_frame(positive_test_df, negative_test_df)
    epitope_probe_df = positive_test_df.copy().reset_index(drop=True)

    return {
        "audit_rows": [
            positive_train_audit,
            positive_test_audit,
            negative_train_audit,
            negative_test_audit,
        ],
        "positive_train_full_df": positive_train_full_df,
        "positive_train_df": positive_train_df.reset_index(drop=True),
        "positive_val_df": positive_val_df.reset_index(drop=True),
        "positive_test_df": positive_test_df.reset_index(drop=True),
        "negative_train_full_df": negative_train_full_df,
        "negative_train_df": negative_train_df.reset_index(drop=True),
        "negative_val_df": negative_val_df.reset_index(drop=True),
        "negative_test_df": negative_test_df.reset_index(drop=True),
        "train_mixed_df": train_mixed_df,
        "val_mixed_df": val_mixed_df,
        "test_mixed_df": test_mixed_df,
        "epitope_probe_df": epitope_probe_df,
    }


def summarize_split_bundle(bundle: dict[str, Any]) -> None:
    print_audit_block(bundle["audit_rows"])
    print(
        "Post-filter split inputs:",
        f"positive_train_full={len(bundle['positive_train_full_df'])}",
        f"positive_test={len(bundle['positive_test_df'])}",
        f"negative_train_full={len(bundle['negative_train_full_df'])}",
        f"negative_test={len(bundle['negative_test_df'])}",
    )
    print(
        "Mixed train/val/test:",
        len(bundle["train_mixed_df"]),
        len(bundle["val_mixed_df"]),
        len(bundle["test_mixed_df"]),
    )
    print(
        "Positive train/val/test:",
        len(bundle["positive_train_df"]),
        len(bundle["positive_val_df"]),
        len(bundle["positive_test_df"]),
    )
    print(
        "Negative train/val/test:",
        len(bundle["negative_train_df"]),
        len(bundle["negative_val_df"]),
        len(bundle["negative_test_df"]),
    )
    print(
        "Positive train density mean:",
        round(float(bundle["positive_train_df"]["epitope_density"].mean()), 4),
    )
    print(
        "Positive test density mean:",
        round(float(bundle["positive_test_df"]["epitope_density"].mean()), 4),
    )


def collate_mixed_batch(batch: list[dict[str, Any]], tokenizer) -> dict[str, Any]:
    sequences = [item["sequence"] for item in batch]
    encodings = tokenizer(
        sequences,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    max_len = encodings["input_ids"].shape[1]
    residue_labels = torch.zeros(len(batch), max_len, dtype=torch.float32)
    residue_loss_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for idx, item in enumerate(batch):
        seq_len = min(item["seq_len"], max_len)
        residue_labels[idx, :seq_len] = torch.tensor(item["residue_labels"][:seq_len], dtype=torch.float32)
        if item["has_epitope_supervision"]:
            residue_loss_mask[idx, :seq_len] = True

    return {
        "sequence_id": [item["sequence_id"] for item in batch],
        "sequence": sequences,
        "protein_label": torch.tensor([item["protein_label"] for item in batch], dtype=torch.float32),
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "residue_labels": residue_labels,
        "residue_loss_mask": residue_loss_mask,
        "has_epitope_supervision": torch.tensor(
            [item["has_epitope_supervision"] for item in batch],
            dtype=torch.float32,
        ),
        "data_source": [item["data_source"] for item in batch],
    }


def move_mixed_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = dict(batch)
    for key in [
        "protein_label",
        "input_ids",
        "attention_mask",
        "residue_labels",
        "residue_loss_mask",
        "has_epitope_supervision",
    ]:
        moved[key] = batch[key].to(device)
    return moved


def build_dataloaders(
    train_mixed_df: pd.DataFrame,
    val_mixed_df: pd.DataFrame,
    test_mixed_df: pd.DataFrame,
    tokenizer,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_mixed_batch(batch, tokenizer)

    train_loader = DataLoader(
        MixedAllergenEpitopeDataset(train_mixed_df),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        MixedAllergenEpitopeDataset(val_mixed_df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    test_loader = DataLoader(
        MixedAllergenEpitopeDataset(test_mixed_df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    return train_loader, val_loader, test_loader


def compute_loss_weights(
    positive_train_df: pd.DataFrame,
    negative_train_df: pd.DataFrame,
    device: str,
    use_protein_pos_weight: bool,
    protein_imbalance_tolerance: float,
) -> dict[str, Any]:
    protein_pos_weight_value = len(negative_train_df) / max(len(positive_train_df), 1)
    use_protein_weight = use_protein_pos_weight and abs(protein_pos_weight_value - 1.0) > protein_imbalance_tolerance
    protein_pos_weight = (
        torch.tensor(protein_pos_weight_value, dtype=torch.float32, device=device)
        if use_protein_weight
        else None
    )

    total_epitope_residues = float(positive_train_df["n_epitope_residues"].sum())
    total_non_epitope_residues = float(
        (positive_train_df["seq_len"] - positive_train_df["n_epitope_residues"]).sum()
    )
    pos_weight_epi_value = total_non_epitope_residues / max(total_epitope_residues, 1.0)
    residue_pos_weight = torch.tensor(pos_weight_epi_value, dtype=torch.float32, device=device)

    return {
        "protein_pos_weight_value": protein_pos_weight_value,
        "use_protein_pos_weight": use_protein_weight,
        "protein_pos_weight": protein_pos_weight,
        "total_epitope_residues": total_epitope_residues,
        "total_non_epitope_residues": total_non_epitope_residues,
        "residue_pos_weight": residue_pos_weight,
        "residue_pos_weight_value": pos_weight_epi_value,
    }


def print_training_balance_summary(
    positive_train_df: pd.DataFrame,
    negative_train_df: pd.DataFrame,
    weight_info: dict[str, Any],
    model: nn.Module,
    trainable_params: list[torch.nn.Parameter],
    lambda_cls: float,
    lambda_epi: float,
) -> None:
    trainable_parameter_count = int(sum(param.numel() for param in trainable_params))
    print(f"Training protein positives: {len(positive_train_df)}")
    print(f"Training protein negatives: {len(negative_train_df)}")
    print(f"Protein pos_weight candidate: {weight_info['protein_pos_weight_value']:.3f}")
    print(f"Using protein pos_weight: {weight_info['use_protein_pos_weight']}")
    print(f"Training epitope residues (positive set only): {weight_info['total_epitope_residues']:.0f}")
    print(
        f"Training non-epitope residues (positive set only): "
        f"{weight_info['total_non_epitope_residues']:.0f}"
    )
    print(
        "pos_weight_epi = total_non_epitope_residues / total_epitope_residues = "
        f"{weight_info['total_non_epitope_residues']:.0f} / {weight_info['total_epitope_residues']:.0f} = "
        f"{weight_info['residue_pos_weight_value']:.3f}"
    )
    print(f"Trainable parameter tensors: {len(trainable_params)}")
    print(f"Total trainable parameter count: {trainable_parameter_count}")
    print(f"Backbone hidden size: {model.backbone.config.hidden_size}")
    print(f"Lambda cls: {lambda_cls}")
    print(f"Lambda epi: {lambda_epi}")


def compute_protein_loss(
    logits: torch.Tensor,
    protein_labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {"reduction": "mean"}
    if pos_weight is not None:
        kwargs["pos_weight"] = pos_weight
    return F.binary_cross_entropy_with_logits(logits, protein_labels, **kwargs)


def compute_masked_residue_loss(
    residue_logits: torch.Tensor,
    residue_labels: torch.Tensor,
    residue_loss_mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    valid_mask = residue_loss_mask.bool()
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return residue_logits.sum() * 0.0, 0

    valid_logits = residue_logits[valid_mask]
    valid_labels = residue_labels[valid_mask]
    loss = F.binary_cross_entropy_with_logits(
        valid_logits,
        valid_labels,
        reduction="mean",
        pos_weight=pos_weight,
    )
    return loss, valid_count


def _tensor_min_max(tensor: torch.Tensor) -> tuple[float, float]:
    detached = tensor.detach()
    return float(detached.min().cpu().item()), float(detached.max().cpu().item())


def _raise_if_non_finite_losses(
    stage: str,
    batch: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    cls_loss: torch.Tensor,
    epi_loss: torch.Tensor,
    total_loss: torch.Tensor,
) -> None:
    non_finite_losses = [
        name
        for name, value in {"cls_loss": cls_loss, "epi_loss": epi_loss, "total_loss": total_loss}.items()
        if not torch.isfinite(value).all()
    ]
    if not non_finite_losses:
        return

    batch_size = int(batch["protein_label"].shape[0])
    valid_positions = int(batch["residue_loss_mask"].bool().sum().item())
    logits_min, logits_max = _tensor_min_max(outputs["logits"])
    residue_logits_min, residue_logits_max = _tensor_min_max(outputs["residue_logits"])
    sequence_ids = list(batch.get("sequence_id", []))
    if sequence_ids:
        print(f"[{stage}] Non-finite loss batch sequence_ids: {sequence_ids}")

    raise ValueError(
        f"[{stage}] Non-finite loss detected in {', '.join(non_finite_losses)} | "
        f"cls_loss={float(cls_loss.detach().cpu().item())} | "
        f"epi_loss={float(epi_loss.detach().cpu().item())} | "
        f"total_loss={float(total_loss.detach().cpu().item())} | "
        f"batch_size={batch_size} | "
        f"valid_residue_positions={valid_positions} | "
        f"logits_min={logits_min} | logits_max={logits_max} | "
        f"residue_logits_min={residue_logits_min} | residue_logits_max={residue_logits_max} | "
        f"sequence_ids={sequence_ids}"
    )


def train_one_epoch_mtl(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    protein_pos_weight: torch.Tensor | None,
    residue_pos_weight: torch.Tensor,
    lambda_cls: float,
    lambda_epi: float,
    trainable_params: list[torch.nn.Parameter],
) -> dict[str, float]:
    model.train()
    total_cls_numerator = 0.0
    total_cls_examples = 0
    total_epi_numerator = 0.0
    total_epi_positions = 0
    total_total_loss = 0.0
    total_steps = 0

    for batch in loader:
        batch = move_mixed_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch["input_ids"], batch["attention_mask"])
        cls_loss = compute_protein_loss(outputs["logits"], batch["protein_label"], protein_pos_weight)
        epi_loss, epi_positions = compute_masked_residue_loss(
            outputs["residue_logits"],
            batch["residue_labels"],
            batch["residue_loss_mask"],
            residue_pos_weight,
        )
        total_loss = lambda_cls * cls_loss + lambda_epi * epi_loss
        _raise_if_non_finite_losses("train", batch, outputs, cls_loss, epi_loss, total_loss)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        batch_size = int(batch["protein_label"].shape[0])
        total_cls_numerator += float(cls_loss.item()) * batch_size
        total_cls_examples += batch_size
        if epi_positions > 0:
            total_epi_numerator += float(epi_loss.item()) * epi_positions
            total_epi_positions += epi_positions
        total_total_loss += float(total_loss.item())
        total_steps += 1

    train_cls_loss = total_cls_numerator / max(total_cls_examples, 1)
    train_epi_loss = total_epi_numerator / max(total_epi_positions, 1)
    train_total_loss = total_total_loss / max(total_steps, 1)
    return {
        "train_total_loss": train_total_loss,
        "train_cls_loss": train_cls_loss,
        "train_epi_loss": train_epi_loss,
        "train_weighted_cls": lambda_cls * train_cls_loss,
        "train_weighted_epi": lambda_epi * train_epi_loss,
    }


@torch.no_grad()
def evaluate_mtl(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    protein_pos_weight: torch.Tensor | None,
    residue_pos_weight: torch.Tensor,
    lambda_cls: float,
    lambda_epi: float,
) -> dict[str, float]:
    model.eval()
    total_cls_numerator = 0.0
    total_cls_examples = 0
    total_epi_numerator = 0.0
    total_epi_positions = 0
    total_total_loss = 0.0
    total_steps = 0

    for batch in loader:
        batch = move_mixed_batch_to_device(batch, device)
        outputs = model(batch["input_ids"], batch["attention_mask"])
        cls_loss = compute_protein_loss(outputs["logits"], batch["protein_label"], protein_pos_weight)
        epi_loss, epi_positions = compute_masked_residue_loss(
            outputs["residue_logits"],
            batch["residue_labels"],
            batch["residue_loss_mask"],
            residue_pos_weight,
        )
        total_loss = lambda_cls * cls_loss + lambda_epi * epi_loss
        _raise_if_non_finite_losses("eval", batch, outputs, cls_loss, epi_loss, total_loss)

        batch_size = int(batch["protein_label"].shape[0])
        total_cls_numerator += float(cls_loss.item()) * batch_size
        total_cls_examples += batch_size
        if epi_positions > 0:
            total_epi_numerator += float(epi_loss.item()) * epi_positions
            total_epi_positions += epi_positions
        total_total_loss += float(total_loss.item())
        total_steps += 1

    cls_loss_value = total_cls_numerator / max(total_cls_examples, 1)
    epi_loss_value = total_epi_numerator / max(total_epi_positions, 1)
    total_loss_value = total_total_loss / max(total_steps, 1)
    return {
        "total_loss": total_loss_value,
        "cls_loss": cls_loss_value,
        "epi_loss": epi_loss_value,
        "weighted_cls": lambda_cls * cls_loss_value,
        "weighted_epi": lambda_epi * epi_loss_value,
    }


@torch.no_grad()
def predict_mtl(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model.eval()
    protein_rows = []
    residue_predictions = []
    flat_residue_labels = []
    flat_residue_scores = []

    for batch in loader:
        batch = move_mixed_batch_to_device(batch, device)
        outputs = model(batch["input_ids"], batch["attention_mask"])
        protein_probs = torch.sigmoid(outputs["logits"]).detach().cpu()
        residue_probs = torch.sigmoid(outputs["residue_logits"]).detach().cpu()
        attention_mask = batch["attention_mask"].detach().cpu()
        residue_labels = batch["residue_labels"].detach().cpu()
        has_supervision = batch["has_epitope_supervision"].detach().cpu()
        protein_labels = batch["protein_label"].detach().cpu()

        for idx, sequence_id in enumerate(batch["sequence_id"]):
            prob = float(protein_probs[idx].item())
            protein_rows.append(
                {
                    "sequence_id": sequence_id,
                    "sequence": batch["sequence"][idx],
                    "label": int(protein_labels[idx].item()),
                    "pred_prob": prob,
                    "pred_label": int(prob >= THRESHOLD),
                    "logit": float(outputs["logits"][idx].detach().cpu().item()),
                }
            )

            if int(has_supervision[idx].item()) == 1:
                seq_len = int(attention_mask[idx].sum().item())
                labels = residue_labels[idx, :seq_len].numpy().astype(np.float32)
                scores = residue_probs[idx, :seq_len].numpy().astype(np.float32)
                residue_predictions.append(
                    {
                        "sequence_id": sequence_id,
                        "residue_labels": labels,
                        "residue_scores": scores,
                    }
                )
                flat_residue_labels.append(labels)
                flat_residue_scores.append(scores)

    payload = {
        "residue_predictions": residue_predictions,
        "residue_labels_flat": np.concatenate(flat_residue_labels)
        if flat_residue_labels
        else np.array([], dtype=np.float32),
        "residue_scores_flat": np.concatenate(flat_residue_scores)
        if flat_residue_scores
        else np.array([], dtype=np.float32),
    }
    return pd.DataFrame(protein_rows), payload


def compute_classification_metrics(pred_df: pd.DataFrame) -> dict[str, Any]:
    y_true = pred_df["label"].to_numpy()
    y_prob = pred_df["pred_prob"].to_numpy()
    y_pred = pred_df["pred_label"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": THRESHOLD,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else math.nan,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def compute_flattened_residue_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    n_valid = int(labels.shape[0])
    n_positive = int(labels.sum())

    if n_valid == 0 or n_positive == 0 or n_positive == n_valid:
        return {
            "n_valid_residues": n_valid,
            "n_positive_residues": n_positive,
            "auroc": math.nan,
            "auprc": math.nan,
            "precision_at_k": math.nan,
        }

    k = max(n_positive, 1)
    top_k = np.argsort(scores)[-k:]
    return {
        "n_valid_residues": n_valid,
        "n_positive_residues": n_positive,
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "precision_at_k": float(labels[top_k].mean()),
    }


def train_mtl_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    protein_pos_weight: torch.Tensor | None,
    residue_pos_weight: torch.Tensor,
    lambda_cls: float,
    lambda_epi: float,
    epochs: int,
    patience: int,
    trainable_params: list[torch.nn.Parameter],
    checkpoint_path: Path,
    baseline_checkpoint_path: Path,
    esm_model_name: str = ESM_MODEL_NAME,
    architecture_hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    best_val_total = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False

    for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
        train_stats = train_one_epoch_mtl(
            model,
            train_loader,
            optimizer,
            device,
            protein_pos_weight=protein_pos_weight,
            residue_pos_weight=residue_pos_weight,
            lambda_cls=lambda_cls,
            lambda_epi=lambda_epi,
            trainable_params=trainable_params,
        )
        val_stats = evaluate_mtl(
            model,
            val_loader,
            device,
            protein_pos_weight=protein_pos_weight,
            residue_pos_weight=residue_pos_weight,
            lambda_cls=lambda_cls,
            lambda_epi=lambda_epi,
        )

        row = {
            "epoch": epoch,
            "train_total_loss": float(train_stats["train_total_loss"]),
            "train_cls_loss": float(train_stats["train_cls_loss"]),
            "train_epi_loss": float(train_stats["train_epi_loss"]),
            "train_weighted_cls": float(train_stats["train_weighted_cls"]),
            "train_weighted_epi": float(train_stats["train_weighted_epi"]),
            "val_total_loss": float(val_stats["total_loss"]),
            "val_cls_loss": float(val_stats["cls_loss"]),
            "val_epi_loss": float(val_stats["epi_loss"]),
            "val_weighted_cls": float(val_stats["weighted_cls"]),
            "val_weighted_epi": float(val_stats["weighted_epi"]),
        }
        history.append(row)

        if val_stats["total_loss"] < best_val_total:
            best_val_total = float(val_stats["total_loss"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "esm_model_name": esm_model_name,
                    "baseline_checkpoint_path": str(baseline_checkpoint_path),
                    "architecture_hyperparameters": architecture_hyperparameters or {},
                    "training_history": history,
                    "best_epoch": best_epoch,
                    "lambda_cls": lambda_cls,
                    "lambda_epi": lambda_epi,
                    "protein_pos_weight": None if protein_pos_weight is None else float(protein_pos_weight.item()),
                    "residue_pos_weight": float(residue_pos_weight.item()),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train_total={train_stats['train_total_loss']:.5f} | "
            f"train_cls={train_stats['train_cls_loss']:.5f} | "
            f"train_epi={train_stats['train_epi_loss']:.5f} | "
            f"train_lambda_cls={train_stats['train_weighted_cls']:.5f} | "
            f"train_lambda_epi={train_stats['train_weighted_epi']:.5f} | "
            f"val_total={val_stats['total_loss']:.5f} | "
            f"val_cls={val_stats['cls_loss']:.5f} | "
            f"val_epi={val_stats['epi_loss']:.5f} | "
            f"val_lambda_cls={val_stats['weighted_cls']:.5f} | "
            f"val_lambda_epi={val_stats['weighted_epi']:.5f} | "
            f"best={best_epoch}"
        )

        if epochs_without_improvement >= patience:
            early_stopped = True
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"Best validation objective: {best_val_total:.5f} at epoch {best_epoch}")
    print(f"Early stopping triggered: {early_stopped}")
    print(f"Checkpoint saved to: {checkpoint_path}")

    return {
        "history": history,
        "best_val_total": best_val_total,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "checkpoint_path": checkpoint_path,
    }


def evaluate_saved_mtl_checkpoint(
    checkpoint_path: Path,
    device: str,
    val_loader: DataLoader,
    test_loader: DataLoader,
    protein_pos_weight: torch.Tensor | None,
    residue_pos_weight: torch.Tensor,
    lambda_cls: float,
    lambda_epi: float,
    baseline_checkpoint_path: Path,
    metrics_path: Path,
    architecture_hyperparameters: dict[str, Any],
    training_hparams: MTLHyperparameters,
    weight_info: dict[str, Any],
    hidden_dim: int,
    dropout: float,
    esm_model_name: str = ESM_MODEL_NAME,
    model_name: str = HF_MODEL_NAME,
    early_stopped: bool = False,
) -> dict[str, Any]:
    model, checkpoint = load_mtl_checkpoint(
        checkpoint_path,
        device,
        model_name=model_name,
        hidden_dim=hidden_dim,
        dropout=dropout,
        epitope_hidden_dim=architecture_hyperparameters["epitope_hidden_dim"],
    )

    history_df = pd.DataFrame(checkpoint["training_history"])
    best_epoch = int(
        checkpoint.get("best_epoch", int(history_df.loc[history_df["val_total_loss"].idxmin(), "epoch"]))
    )

    val_stats = evaluate_mtl(
        model,
        val_loader,
        device,
        protein_pos_weight=protein_pos_weight,
        residue_pos_weight=residue_pos_weight,
        lambda_cls=lambda_cls,
        lambda_epi=lambda_epi,
    )
    test_stats = evaluate_mtl(
        model,
        test_loader,
        device,
        protein_pos_weight=protein_pos_weight,
        residue_pos_weight=residue_pos_weight,
        lambda_cls=lambda_cls,
        lambda_epi=lambda_epi,
    )

    val_predictions_df, val_residue_payload = predict_mtl(model, val_loader, device)
    test_predictions_df, test_residue_payload = predict_mtl(model, test_loader, device)

    val_classification_metrics = compute_classification_metrics(val_predictions_df)
    val_residue_metrics = compute_flattened_residue_metrics(
        val_residue_payload["residue_labels_flat"],
        val_residue_payload["residue_scores_flat"],
    )
    test_metrics = compute_classification_metrics(test_predictions_df)
    test_metrics["test_total_loss"] = float(test_stats["total_loss"])
    test_metrics["test_cls_loss"] = float(test_stats["cls_loss"])
    test_metrics["test_epi_loss"] = float(test_stats["epi_loss"])
    test_metrics["test_weighted_cls"] = float(test_stats["weighted_cls"])
    test_metrics["test_weighted_epi"] = float(test_stats["weighted_epi"])
    test_metrics["best_epoch"] = best_epoch
    test_metrics["n_test_sequences"] = int(len(test_predictions_df))

    test_residue_metrics = compute_flattened_residue_metrics(
        test_residue_payload["residue_labels_flat"],
        test_residue_payload["residue_scores_flat"],
    )

    metrics_payload = {
        "esm_model_name": esm_model_name,
        "baseline_checkpoint_path": str(baseline_checkpoint_path),
        "architecture_hyperparameters": architecture_hyperparameters,
        "training": {
            "batch_size": training_hparams.classification_batch_size,
            "epochs_requested": training_hparams.epochs,
            "early_stopping_patience": training_hparams.patience,
            "optimizer": "AdamW",
            "lr": training_hparams.learning_rate,
            "weight_decay": training_hparams.weight_decay,
            "lambda_cls": training_hparams.lambda_cls,
            "lambda_epi": training_hparams.lambda_epi,
            "use_protein_pos_weight": weight_info["use_protein_pos_weight"],
            "protein_pos_weight": None
            if protein_pos_weight is None
            else float(protein_pos_weight.item()),
            "residue_pos_weight": float(residue_pos_weight.item()),
            "residue_pos_weight_formula": "total_non_epitope_residues / total_epitope_residues",
            "total_epitope_residues_train": float(weight_info["total_epitope_residues"]),
            "total_non_epitope_residues_train": float(weight_info["total_non_epitope_residues"]),
            "best_epoch": best_epoch,
            "early_stopped": bool(early_stopped),
        },
        "validation_losses": {
            "total_loss": float(val_stats["total_loss"]),
            "cls_loss": float(val_stats["cls_loss"]),
            "epi_loss": float(val_stats["epi_loss"]),
            "weighted_cls": float(val_stats["weighted_cls"]),
            "weighted_epi": float(val_stats["weighted_epi"]),
        },
        "validation_classification_metrics": val_classification_metrics,
        "validation_residue_metrics": val_residue_metrics,
        "test_metrics": test_metrics,
        "test_residue_metrics": test_residue_metrics,
    }

    ensure_output_parent(metrics_path)
    with metrics_path.open("w") as handle:
        json.dump(metrics_payload, handle, indent=2)

    print("Validation classification metrics:")
    print(json.dumps(val_classification_metrics, indent=2))
    print("Validation residue metrics:")
    print(json.dumps(val_residue_metrics, indent=2))
    print("Test classification metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("Test residue metrics:")
    print(json.dumps(test_residue_metrics, indent=2))
    print(f"Saved metrics to: {metrics_path}")

    return {
        "model": model,
        "checkpoint": checkpoint,
        "history_df": history_df,
        "best_epoch": best_epoch,
        "val_stats": val_stats,
        "test_stats": test_stats,
        "val_predictions_df": val_predictions_df,
        "test_predictions_df": test_predictions_df,
        "val_residue_payload": val_residue_payload,
        "test_residue_payload": test_residue_payload,
        "metrics_payload": metrics_payload,
    }


def compute_probe_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    seq_len = len(labels)
    positives = int(labels.sum())

    if seq_len == 0 or positives == 0 or positives == seq_len:
        return {"auroc": np.nan, "auprc": np.nan, "precision_at_k": np.nan}

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    k = max(positives, 1)
    top_k = np.argsort(scores)[-k:]
    precision_at_k = float(labels[top_k].mean())

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "precision_at_k": precision_at_k,
    }


def compute_occlusion_scores_mtl(
    model,
    tokenizer,
    sequence: str,
    device: str,
) -> np.ndarray:
    """
    Single-residue occlusion for any model with the shared classifier interface.
    Replaces residue i with mask token, records delta_p on the
    classification head output only (not the epitope head).
    Returns float32 array of shape (L,).
    """
    assert tokenizer.mask_token is not None, (
        "Tokenizer has no mask token. Rebuild with add_special_tokens=True."
    )

    model.eval()

    def _forward(seq: str) -> float:
        enc = tokenizer(
            seq,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        enc = {key: value.to(device) for key, value in enc.items()}
        with torch.no_grad():
            outputs = model(enc["input_ids"], enc["attention_mask"])
        return float(torch.sigmoid(outputs["logits"]).item())

    p_base = _forward(sequence)
    residues = list(sequence)
    delta_p = np.zeros(len(residues), dtype=np.float32)

    for idx in range(len(residues)):
        masked = residues.copy()
        masked[idx] = tokenizer.mask_token
        delta_p[idx] = p_base - _forward("".join(masked))

    return delta_p


def validate_probe_scores(
    accession: str,
    method: str,
    model_family: str,
    labels: np.ndarray,
    scores: np.ndarray,
) -> None:
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != labels.shape:
        raise ValueError(
            f"{model_family} {method} for {accession} returned shape {scores.shape}, "
            f"expected {labels.shape}."
        )
    if scores.size > 0 and np.isnan(scores).all():
        raise ValueError(f"{model_family} {method} for {accession} returned only NaN scores.")


def scramble_labels(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly permute residue labels within a protein.
    Preserves sequence length and number of positive residues.
    """
    return rng.permutation(np.asarray(labels, dtype=np.float32))


def validate_probe_metrics(labels: np.ndarray, metrics: dict[str, float]) -> None:
    labels = np.asarray(labels, dtype=np.float32)
    degenerate = labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size
    if not degenerate:
        for metric_name, metric_value in metrics.items():
            if pd.isna(metric_value):
                raise ValueError(f"{metric_name} is NaN for non-degenerate labels.")


def build_probe_row(
    base: dict[str, Any],
    model_family: str,
    method: str,
    labels: np.ndarray,
    scores: np.ndarray,
    serialize_scores: bool = False,
    score_column: str | None = None,
    label_variant: str = "original",
) -> dict[str, Any]:
    validate_probe_scores(str(base["accession"]), method, model_family, labels, scores)
    metrics = compute_probe_metrics(labels, scores)
    validate_probe_metrics(labels, metrics)
    row = {
        **base,
        "model_family": model_family,
        "method": method,
        "label_variant": label_variant,
        **metrics,
    }
    if serialize_scores:
        row["scores_json"] = serialize_score_array(scores)
        if score_column is not None:
            row[score_column] = row["scores_json"]
    return row


def build_probe_rows_with_label_scrambling(
    base: dict[str, Any],
    model_family: str,
    method: str,
    labels: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    serialize_scores: bool = False,
    score_column: str | None = None,
) -> list[dict[str, Any]]:
    scrambled_labels = scramble_labels(labels, rng)
    return [
        build_probe_row(
            base,
            model_family,
            method,
            labels,
            scores,
            serialize_scores=serialize_scores,
            score_column=score_column,
            label_variant="original",
        ),
        build_probe_row(
            base,
            model_family,
            method,
            scrambled_labels,
            scores,
            serialize_scores=serialize_scores,
            score_column=score_column,
            label_variant="scrambled",
        ),
    ]


def ensure_label_variant_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "label_variant" not in frame.columns:
        frame["label_variant"] = "original"
    frame["label_variant"] = frame["label_variant"].fillna("original").astype(str)
    return frame


def validate_unique_probe_rows(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    frame = ensure_label_variant_column(frame)
    duplicated = frame.duplicated(["accession", "method", "model_family", "label_variant"], keep=False)
    if duplicated.any():
        examples = frame.loc[
            duplicated, ["accession", "method", "model_family", "label_variant"]
        ].drop_duplicates().head(10)
        raise ValueError(
            "Duplicate probe rows found for accession/method/model_family/label_variant:\n"
            f"{examples.to_string(index=False)}"
        )


def run_probe_suite(
    model,
    tokenizer,
    epitope_probe_df: pd.DataFrame,
    baseline_checkpoint_path: Path,
    device: str,
    hidden_dim: int,
    dropout: float,
    output_paths: MTLOutputPaths,
    ig_steps: int,
    n_random_draws: int,
    ig_internal_batch_size: int,
    model_name: str = HF_MODEL_NAME,
    resume: bool = True,
    save_every: int = 1,
    precomputed_baseline_probe_df: pd.DataFrame | None = None,
    smoothgrad_ig_samples: int = 10,
    smoothgrad_ig_noise_std: float = 0.05,
) -> dict[str, pd.DataFrame]:
    ig_probe_device = device
    rng = np.random.default_rng(RANDOM_STATE)
    expected_mtl_methods = {
        "residue_head",
        "attention_weights",
        "integrated_gradients",
        "gradient_x_input",
        "smoothgrad_ig",
        "occlusion",
        "random_mean",
    }
    expected_baseline_methods = {
        "attention_weights",
        "integrated_gradients",
        "gradient_x_input",
        "smoothgrad_ig",
        "occlusion",
        "random_mean",
    }
    probe_rows = []
    baseline_probe_rows = []

    ig_model = model.to(ig_probe_device)
    ig_model.eval()
    baseline_model = None
    baseline_ig_model = None
    use_precomputed_baseline = False
    if precomputed_baseline_probe_df is not None:
        precomputed_baseline_probe_df = ensure_label_variant_column(precomputed_baseline_probe_df)
        precomputed_methods = set(precomputed_baseline_probe_df["method"].astype(str))
        precomputed_label_variants = set(precomputed_baseline_probe_df["label_variant"].astype(str))
        missing_methods = expected_baseline_methods - precomputed_methods
        missing_label_variants = {"original", "scrambled"} - precomputed_label_variants
        use_precomputed_baseline = not missing_methods and not missing_label_variants
        if missing_methods or missing_label_variants:
            print(
                "Precomputed baseline probe rows are missing methods or label variants "
                f"(methods={sorted(missing_methods)}, label_variants={sorted(missing_label_variants)}); "
                "recomputing baseline probes."
            )

    if not use_precomputed_baseline:
        baseline_model, _ = load_baseline_checkpoint(
            baseline_checkpoint_path,
            ig_probe_device,
            model_name=model_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        baseline_ig_model = baseline_model
        baseline_ig_model.eval()
    else:
        accession_column = "accession"
        current_accessions = set(epitope_probe_df[accession_column].astype(str))
        baseline_probe_rows = (
            precomputed_baseline_probe_df.loc[
                precomputed_baseline_probe_df[accession_column].astype(str).isin(current_accessions)
            ]
            .copy()
            .to_dict("records")
        )

    print(f"Integrated Gradients device: {ig_probe_device}")
    print(f"IG_STEPS: {ig_steps}")
    print(f"IG internal_batch_size: {ig_internal_batch_size}")
    print(f"SmoothGrad-IG samples: {smoothgrad_ig_samples}")
    print(f"SmoothGrad-IG noise_std: {smoothgrad_ig_noise_std}")
    for output_path in [
        output_paths.probe_rows_path,
        output_paths.baseline_probe_rows_path,
        output_paths.combined_probe_rows_path,
    ]:
        ensure_output_parent(output_path)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    processed_since_save = 0
    for _, row in tqdm(epitope_probe_df.iterrows(), total=len(epitope_probe_df), desc="Probing splitB"):
        accession = str(row["accession"])

        sequence = row["sequence"]
        epitope_labels = row["epitope_label"]
        base = {
            "accession": accession,
            "seq_len": int(row["seq_len"]),
            "epitope_density": float(row["epitope_density"]),
            "n_epitope_residues": int(row["n_epitope_residues"]),
        }

        residue_scores = compute_residue_probabilities(ig_model, tokenizer, sequence, ig_probe_device)
        probe_rows.extend(
            build_probe_rows_with_label_scrambling(
                base, output_paths.mtl_family_label, "residue_head", epitope_labels, residue_scores, rng
            )
        )

        attention_scores = compute_attention_weights(ig_model, tokenizer, sequence, ig_probe_device)
        probe_rows.extend(
            build_probe_rows_with_label_scrambling(
                base, output_paths.mtl_family_label, "attention_weights", epitope_labels, attention_scores, rng
            )
        )

        if not use_precomputed_baseline:
            baseline_attention_scores = compute_attention_weights(
                baseline_model, tokenizer, sequence, ig_probe_device
            )
            baseline_probe_rows.extend(
                build_probe_rows_with_label_scrambling(
                    base,
                    output_paths.baseline_family_label,
                    "attention_weights",
                    epitope_labels,
                    baseline_attention_scores,
                    rng,
                )
            )
        else:
            baseline_attention_scores = None

        ig_scores = compute_integrated_gradients(
            ig_model,
            tokenizer,
            sequence,
            ig_probe_device,
            steps=ig_steps,
            normalize=False,
            internal_batch_size=ig_internal_batch_size,
        )
        probe_rows.extend(
            build_probe_rows_with_label_scrambling(
                base,
                output_paths.mtl_family_label,
                "integrated_gradients",
                epitope_labels,
                ig_scores,
                rng,
                serialize_scores=True,
                score_column="ig_scores_json",
            )
        )

        gradient_x_input_scores = compute_gradient_x_input_scores(
            ig_model,
            tokenizer,
            sequence,
            ig_probe_device,
        )
        probe_rows.extend(
            build_probe_rows_with_label_scrambling(
                base,
                output_paths.mtl_family_label,
                "gradient_x_input",
                epitope_labels,
                gradient_x_input_scores,
                rng,
                serialize_scores=True,
                score_column="gradient_x_input_scores_json",
            )
        )

        smoothgrad_ig_scores = compute_smoothgrad_ig_scores(
            ig_model,
            tokenizer,
            sequence,
            ig_probe_device,
            steps=ig_steps,
            n_samples=smoothgrad_ig_samples,
            noise_std=smoothgrad_ig_noise_std,
            internal_batch_size=ig_internal_batch_size,
        )
        probe_rows.extend(
            build_probe_rows_with_label_scrambling(
                base,
                output_paths.mtl_family_label,
                "smoothgrad_ig",
                epitope_labels,
                smoothgrad_ig_scores,
                rng,
                serialize_scores=True,
                score_column="smoothgrad_ig_scores_json",
            )
        )

        occlusion_scores = normalize_scores(
            compute_occlusion_scores_mtl(ig_model, tokenizer, sequence, ig_probe_device)
        )
        probe_rows.extend(
            build_probe_rows_with_label_scrambling(
                base,
                output_paths.mtl_family_label,
                "occlusion",
                epitope_labels,
                occlusion_scores,
                rng,
                serialize_scores=True,
                score_column="occlusion_scores_json",
            )
        )

        if not use_precomputed_baseline:
            baseline_ig_scores = compute_integrated_gradients(
                baseline_ig_model,
                tokenizer,
                sequence,
                ig_probe_device,
                steps=ig_steps,
                normalize=False,
                internal_batch_size=ig_internal_batch_size,
            )
            baseline_probe_rows.extend(
                build_probe_rows_with_label_scrambling(
                    base,
                    output_paths.baseline_family_label,
                    "integrated_gradients",
                    epitope_labels,
                    baseline_ig_scores,
                    rng,
                    serialize_scores=True,
                    score_column="ig_scores_json",
                )
            )

            baseline_gradient_x_input_scores = compute_gradient_x_input_scores(
                baseline_ig_model,
                tokenizer,
                sequence,
                ig_probe_device,
            )
            baseline_probe_rows.extend(
                build_probe_rows_with_label_scrambling(
                    base,
                    output_paths.baseline_family_label,
                    "gradient_x_input",
                    epitope_labels,
                    baseline_gradient_x_input_scores,
                    rng,
                    serialize_scores=True,
                    score_column="gradient_x_input_scores_json",
                )
            )

            baseline_smoothgrad_ig_scores = compute_smoothgrad_ig_scores(
                baseline_ig_model,
                tokenizer,
                sequence,
                ig_probe_device,
                steps=ig_steps,
                n_samples=smoothgrad_ig_samples,
                noise_std=smoothgrad_ig_noise_std,
                internal_batch_size=ig_internal_batch_size,
            )
            baseline_probe_rows.extend(
                build_probe_rows_with_label_scrambling(
                    base,
                    output_paths.baseline_family_label,
                    "smoothgrad_ig",
                    epitope_labels,
                    baseline_smoothgrad_ig_scores,
                    rng,
                    serialize_scores=True,
                    score_column="smoothgrad_ig_scores_json",
                )
            )

            baseline_occlusion_scores = normalize_scores(
                compute_occlusion_scores_mtl(baseline_ig_model, tokenizer, sequence, ig_probe_device)
            )
            baseline_probe_rows.extend(
                build_probe_rows_with_label_scrambling(
                    base,
                    output_paths.baseline_family_label,
                    "occlusion",
                    epitope_labels,
                    baseline_occlusion_scores,
                    rng,
                    serialize_scores=True,
                    score_column="occlusion_scores_json",
                )
            )
        else:
            baseline_ig_scores = None
            baseline_gradient_x_input_scores = None
            baseline_smoothgrad_ig_scores = None
            baseline_occlusion_scores = None

        random_score_draws = [
            rng.uniform(0.0, 1.0, size=len(epitope_labels))
            for _ in range(n_random_draws)
        ]
        random_metrics = [
            compute_probe_metrics(epitope_labels, random_scores)
            for random_scores in random_score_draws
        ]
        random_scrambled_metrics = [
            compute_probe_metrics(scramble_labels(epitope_labels, rng), random_scores)
            for random_scores in random_score_draws
        ]
        random_summary = mean_metric_dicts(random_metrics)
        random_scrambled_summary = mean_metric_dicts(random_scrambled_metrics)
        validate_probe_metrics(epitope_labels, random_summary)
        validate_probe_metrics(epitope_labels, random_scrambled_summary)
        probe_rows.append(
            {
                **base,
                "model_family": output_paths.mtl_family_label,
                "method": "random_mean",
                "label_variant": "original",
                **random_summary,
            }
        )
        probe_rows.append(
            {
                **base,
                "model_family": output_paths.mtl_family_label,
                "method": "random_mean",
                "label_variant": "scrambled",
                **random_scrambled_summary,
            }
        )
        if not use_precomputed_baseline:
            baseline_probe_rows.append(
                {
                    **base,
                    "model_family": output_paths.baseline_family_label,
                    "method": "random_mean",
                    "label_variant": "original",
                    **random_summary,
                }
            )
            baseline_probe_rows.append(
                {
                    **base,
                    "model_family": output_paths.baseline_family_label,
                    "method": "random_mean",
                    "label_variant": "scrambled",
                    **random_scrambled_summary,
                }
            )

        del (
            residue_scores,
            attention_scores,
            baseline_attention_scores,
            ig_scores,
            gradient_x_input_scores,
            smoothgrad_ig_scores,
            occlusion_scores,
            baseline_ig_scores,
            baseline_gradient_x_input_scores,
            baseline_smoothgrad_ig_scores,
            baseline_occlusion_scores,
            random_score_draws,
            random_metrics,
            random_scrambled_metrics,
            random_summary,
            random_scrambled_summary,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        processed_since_save += 1
        if save_every > 0 and processed_since_save >= save_every:
            probe_df = ensure_label_variant_column(pd.DataFrame(probe_rows))
            baseline_probe_df = ensure_label_variant_column(pd.DataFrame(baseline_probe_rows))
            combined_probe_df = ensure_label_variant_column(pd.concat([baseline_probe_df, probe_df], ignore_index=True))
            validate_unique_probe_rows(probe_df)
            validate_unique_probe_rows(baseline_probe_df)
            validate_unique_probe_rows(combined_probe_df)
            probe_df.to_csv(output_paths.probe_rows_path, index=False)
            baseline_probe_df.to_csv(output_paths.baseline_probe_rows_path, index=False)
            combined_probe_df.to_csv(output_paths.combined_probe_rows_path, index=False)
            processed_since_save = 0

    del ig_model
    if baseline_ig_model is not None:
        del baseline_ig_model
    if baseline_model is not None:
        del baseline_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    probe_df = ensure_label_variant_column(pd.DataFrame(probe_rows))
    baseline_probe_df = ensure_label_variant_column(pd.DataFrame(baseline_probe_rows))
    combined_probe_df = ensure_label_variant_column(pd.concat([baseline_probe_df, probe_df], ignore_index=True))
    validate_unique_probe_rows(probe_df)
    validate_unique_probe_rows(baseline_probe_df)
    validate_unique_probe_rows(combined_probe_df)

    probe_df.to_csv(output_paths.probe_rows_path, index=False)
    baseline_probe_df.to_csv(output_paths.baseline_probe_rows_path, index=False)
    combined_probe_df.to_csv(output_paths.combined_probe_rows_path, index=False)
    print(f"Saved MTL row-wise probe metrics to: {output_paths.probe_rows_path}")
    print(f"Saved baseline row-wise probe metrics to: {output_paths.baseline_probe_rows_path}")
    print(f"Saved combined row-wise probe metrics to: {output_paths.combined_probe_rows_path}")

    return {
        "probe_df": probe_df,
        "baseline_probe_df": baseline_probe_df,
        "combined_probe_df": combined_probe_df,
    }


def bootstrap_mean_ci(
    values: pd.Series | np.ndarray | list[float],
    n_bootstrap: int = 2000,
    ci: float = 95.0,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    clean = pd.Series(values, dtype=float).dropna().to_numpy()
    if clean.size == 0:
        return math.nan, math.nan, math.nan

    mean_value = float(clean.mean())
    if clean.size == 1:
        return mean_value, mean_value, mean_value

    rng = np.random.default_rng(random_state)
    bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample = rng.choice(clean, size=clean.size, replace=True)
        bootstrap_means[idx] = sample.mean()

    alpha = (100.0 - ci) / 2.0
    ci_low, ci_high = np.percentile(bootstrap_means, [alpha, 100.0 - alpha])
    return mean_value, float(ci_low), float(ci_high)


MAIN_LOCALIZATION_METHODS = ["random_mean", "integrated_gradients", "occlusion", "residue_head"]
SUPPLEMENTARY_SIGNAL_METHODS = [
    "random_mean",
    "attention_weights",
    "integrated_gradients",
    "gradient_x_input",
    "smoothgrad_ig",
    "occlusion",
    "residue_head",
]


METHOD_CATEGORY = {
    "random_mean": "Null baseline",
    "integrated_gradients": "Post-hoc attribution",
    "gradient_x_input": "Post-hoc attribution",
    "smoothgrad_ig": "Post-hoc attribution",
    "occlusion": "Perturbation sensitivity",
    "attention_weights": "Model-internal signal",
    "residue_head": "Supervised residue predictor",
}


def classification_results_dir(results_dir: Path) -> Path:
    return Path(results_dir) / "classification"


def probe_rows_dir(results_dir: Path) -> Path:
    return Path(results_dir) / "probing" / "rows"


def probe_summaries_dir(results_dir: Path) -> Path:
    return Path(results_dir) / "probing" / "summaries"


def diagnostic_figures_dir(results_dir: Path) -> Path:
    return Path(results_dir) / "figures" / "diagnostics"


def ensure_output_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_publication_results_tree(results_dir: Path) -> dict[str, Path]:
    paths = {
        "classification": classification_results_dir(results_dir),
        "probe_rows": probe_rows_dir(results_dir),
        "probe_summaries": probe_summaries_dir(results_dir),
        "main_figures": Path(results_dir) / "figures" / "main",
        "supplementary_figures": Path(results_dir) / "figures" / "supplementary",
        "diagnostic_figures": diagnostic_figures_dir(results_dir),
        "insilico_mutagenesis": Path(results_dir) / "insilico_mutagenesis",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def original_label_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_label_variant_column(frame)
    return frame.loc[frame["label_variant"] == "original"].copy()


def summarize_probe_methods(frame: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    frame = ensure_label_variant_column(frame)
    summary_rows = []
    group_cols = ["method", "label_variant"]
    if "model_family" in frame.columns:
        group_cols = ["model_family", "method", "label_variant"]

    for group_key, method_df in frame[frame["method"].isin(methods)].groupby(group_cols, dropna=False):
        if "model_family" in group_cols:
            model_family, method, label_variant = group_key
        else:
            model_family = None
            method, label_variant = group_key
        auroc_mean, auroc_ci_low, auroc_ci_high = bootstrap_mean_ci(method_df["auroc"])
        auprc_mean, auprc_ci_low, auprc_ci_high = bootstrap_mean_ci(method_df["auprc"])
        precision_mean, precision_ci_low, precision_ci_high = bootstrap_mean_ci(method_df["precision_at_k"])
        row = {
            "method": method,
            "label_variant": label_variant,
            "method_category": METHOD_CATEGORY.get(method, "Uncategorized"),
            "auroc_mean": round(auroc_mean, 4),
            "auroc_ci_low": round(auroc_ci_low, 4),
            "auroc_ci_high": round(auroc_ci_high, 4),
            "auprc_mean": round(auprc_mean, 4),
            "auprc_ci_low": round(auprc_ci_low, 4),
            "auprc_ci_high": round(auprc_ci_high, 4),
            "precision_at_k_mean": round(precision_mean, 4),
            "precision_at_k_ci_low": round(precision_ci_low, 4),
            "precision_at_k_ci_high": round(precision_ci_high, 4),
            "n_proteins": int(method_df["accession"].nunique()) if "accession" in method_df else int(len(method_df)),
        }
        if model_family is not None:
            row = {"model_family": model_family, **row}
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return summary_df
    method_rank = {method: idx for idx, method in enumerate(methods)}
    summary_df["_method_rank"] = summary_df["method"].map(method_rank)
    summary_df["_label_rank"] = summary_df["label_variant"].map({"original": 0, "scrambled": 1}).fillna(2)
    sort_cols = ["_method_rank"]
    if "model_family" in summary_df.columns:
        family_rank = {family: idx for idx, family in enumerate(DEFAULT_FAMILY_ORDER)}
        summary_df["_family_rank"] = summary_df["model_family"].map(family_rank).fillna(len(family_rank))
        sort_cols = ["_family_rank", "_method_rank", "_label_rank"]
    else:
        sort_cols = ["_method_rank", "_label_rank"]
    summary_df = summary_df.sort_values(sort_cols).drop(columns=[col for col in ["_family_rank", "_method_rank", "_label_rank"] if col in summary_df])
    return summary_df.reset_index(drop=True)


def save_localization_summary_csvs(
    combined_probe_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    original_df = original_label_rows(combined_probe_df)
    main_summary_df = summarize_probe_methods(original_df, MAIN_LOCALIZATION_METHODS)
    supplementary_summary_df = summarize_probe_methods(original_df, SUPPLEMENTARY_SIGNAL_METHODS)
    scrambling_summary_df = summarize_probe_methods(combined_probe_df, SUPPLEMENTARY_SIGNAL_METHODS)
    main_summary_path = output_dir / "main_localization_summary.csv"
    supplementary_summary_path = output_dir / "supplementary_all_signals_summary.csv"
    scrambling_summary_path = output_dir / "probe_summary_with_scrambling.csv"
    main_summary_df.to_csv(main_summary_path, index=False)
    supplementary_summary_df.to_csv(supplementary_summary_path, index=False)
    scrambling_summary_df.to_csv(scrambling_summary_path, index=False)
    print(f"Saved main localization summary to: {main_summary_path}")
    print(f"Saved supplementary all-signals summary to: {supplementary_summary_path}")
    print(f"Saved label-scrambling summary to: {scrambling_summary_path}")
    return {
        "main_summary_df": main_summary_df,
        "supplementary_summary_df": supplementary_summary_df,
        "scrambling_summary_df": scrambling_summary_df,
    }


PROBE_FAMILY_LABEL_OVERRIDES = {
    "baseline": "Baseline (04)",
    "frozen": "MTL (05 frozen)",
    "top1_unfrozen": "MTL (06 top1_unfrozen)",
}


def infer_probe_variant_from_checkpoint_name(checkpoint_name: str) -> tuple[str | None, str | None]:
    if checkpoint_name == "baseline_frozen_esm2.pt":
        return "baseline", PROBE_FAMILY_LABEL_OVERRIDES["baseline"]
    if checkpoint_name == "mtl_frozen_esm2_epitope.pt":
        return "frozen", PROBE_FAMILY_LABEL_OVERRIDES["frozen"]
    if checkpoint_name.startswith("mtl_") and checkpoint_name.endswith("_esm2_epitope.pt"):
        variant = checkpoint_name[len("mtl_") : -len("_esm2_epitope.pt")]
        if variant:
            return variant, PROBE_FAMILY_LABEL_OVERRIDES.get(variant, f"MTL ({variant})")
    return None, None


def probe_rows_path_for_variant(results_dir: Path, variant: str) -> Path:
    rows_dir = probe_rows_dir(results_dir)
    if variant == "baseline":
        return rows_dir / "baseline_probing_rows.csv"
    if variant == "frozen":
        return rows_dir / "mtl_probing_rows.csv"
    return rows_dir / f"mtl_{variant}_probing_rows.csv"


def legacy_probe_rows_path_for_variant(results_dir: Path, variant: str) -> Path:
    if variant == "baseline":
        return Path(results_dir) / "baseline_probing_rows.csv"
    if variant == "frozen":
        return Path(results_dir) / "mtl_probing_rows.csv"
    return Path(results_dir) / f"mtl_{variant}_probing_rows.csv"


def discover_probe_row_artifacts(models_dir: Path, results_dir: Path) -> pd.DataFrame:
    records = []
    for checkpoint_path in sorted(models_dir.glob("*.pt")):
        variant, default_label = infer_probe_variant_from_checkpoint_name(checkpoint_path.name)
        if variant is None:
            continue
        probe_rows_path = probe_rows_path_for_variant(results_dir, variant)
        if not probe_rows_path.exists():
            legacy_path = legacy_probe_rows_path_for_variant(results_dir, variant)
            if legacy_path.exists():
                probe_rows_path = legacy_path
        records.append(
            {
                "checkpoint_name": checkpoint_path.name,
                "variant": variant,
                "model_family": default_label,
                "checkpoint_path": checkpoint_path,
                "probe_rows_path": probe_rows_path,
                "probe_rows_exists": probe_rows_path.exists(),
            }
        )
    return pd.DataFrame(records)


def load_available_probe_rows(discovery_df: pd.DataFrame) -> pd.DataFrame:
    available_df = discovery_df.loc[discovery_df["probe_rows_exists"]].copy()
    if available_df.empty:
        raise FileNotFoundError("No matching probe-row CSVs were found. Generate probe artifacts first.")

    frames = []
    for row in available_df.itertuples(index=False):
        header = pd.read_csv(row.probe_rows_path, nrows=0)
        columns = [
            "accession",
            "seq_len",
            "epitope_density",
            "n_epitope_residues",
            "model_family",
            "method",
            "label_variant",
            "auroc",
            "auprc",
            "precision_at_k",
        ]
        frame = ensure_label_variant_column(
            pd.read_csv(row.probe_rows_path, usecols=[column for column in columns if column in header.columns])
        )
        family_label = PROBE_FAMILY_LABEL_OVERRIDES.get(row.variant, row.model_family)
        frame = frame.copy()
        frame["model_family"] = family_label
        frame["source_probe_rows_path"] = str(row.probe_rows_path)
        frames.append(frame)

    combined_df = ensure_label_variant_column(pd.concat(frames, ignore_index=True))
    validate_unique_probe_rows(combined_df)
    return combined_df


def save_combined_probe_tables(
    combined_probe_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame | Path]:
    ensure_publication_results_tree(output_dir)
    combined_probe_df = ensure_label_variant_column(combined_probe_df)
    rows_path = probe_rows_dir(output_dir) / "all_models_probing_rows.csv"
    summary_path = probe_summaries_dir(output_dir) / "all_models_probing_summary.csv"
    combined_probe_df.to_csv(rows_path, index=False)
    summary_df = summarize_probe_methods(combined_probe_df, SUPPLEMENTARY_SIGNAL_METHODS)
    summary_df.to_csv(summary_path, index=False)
    summary_payload = save_localization_summary_csvs(combined_probe_df, probe_summaries_dir(output_dir))
    return {
        "rows_path": rows_path,
        "summary_path": summary_path,
        "summary_df": summary_df,
        **summary_payload,
    }


def render_probe_figures_from_rows(
    combined_probe_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    combined_probe_df = ensure_label_variant_column(combined_probe_df)
    ensure_publication_results_tree(output_dir)
    main_dir = output_dir / "figures" / "main"
    supplementary_dir = output_dir / "figures" / "supplementary"
    main_dir.mkdir(parents=True, exist_ok=True)
    supplementary_dir.mkdir(parents=True, exist_ok=True)

    main_base = main_dir / "main_localization.png"
    supplementary_base = supplementary_dir / "supplementary_all_signals.png"
    scrambling_base = supplementary_dir / "label_scrambling_sanity_check.png"
    plot_main_localization_figure(combined_probe_df, main_base)
    plot_supplementary_all_signals_figure(combined_probe_df, supplementary_base)
    plot_label_scrambling_sanity_check(combined_probe_df, scrambling_base)

    return {
        "main_dir": main_dir,
        "supplementary_dir": supplementary_dir,
        "main_base": main_base,
        "supplementary_base": supplementary_base,
        "scrambling_base": scrambling_base,
    }


def replot_probe_figures_from_csv(
    rows_csv: Path,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    rows_csv = Path(rows_csv)
    if output_dir is None:
        output_dir = rows_csv.parents[2] if "probing" in rows_csv.parts else rows_csv.parent
    header = pd.read_csv(rows_csv, nrows=0)
    plotting_columns = [
        "accession",
        "seq_len",
        "epitope_density",
        "n_epitope_residues",
        "model_family",
        "method",
        "label_variant",
        "auroc",
        "auprc",
        "precision_at_k",
    ]
    usecols = [column for column in plotting_columns if column in header.columns]
    combined_probe_df = ensure_label_variant_column(pd.read_csv(rows_csv, usecols=usecols))
    return render_probe_figures_from_rows(combined_probe_df, Path(output_dir))


def summarize_probe_outputs(
    probe_df: pd.DataFrame,
    baseline_probe_df: pd.DataFrame,
    output_paths: MTLOutputPaths,
) -> dict[str, pd.DataFrame]:
    summary_df = summarize_probe_methods(
        probe_df,
        SUPPLEMENTARY_SIGNAL_METHODS,
    )
    baseline_summary_df = summarize_probe_methods(
        baseline_probe_df,
        [method for method in SUPPLEMENTARY_SIGNAL_METHODS if method != "residue_head"],
    )
    ensure_output_parent(output_paths.probe_summary_path)
    ensure_output_parent(output_paths.compare_summary_path)
    summary_df.to_csv(output_paths.probe_summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved summary to: {output_paths.probe_summary_path}")

    save_localization_summary_csvs(
        pd.concat([baseline_probe_df, probe_df], ignore_index=True),
        output_paths.probe_summary_path.parent,
    )

    comparable_methods = [
        "attention_weights",
        "integrated_gradients",
        "gradient_x_input",
        "smoothgrad_ig",
        "occlusion",
        "random_mean",
    ]
    baseline_comparison_df = baseline_summary_df.loc[
        baseline_summary_df["label_variant"].eq("original")
        & baseline_summary_df["method"].isin(comparable_methods)
    ]
    mtl_comparison_df = summary_df.loc[
        summary_df["label_variant"].eq("original")
        & summary_df["method"].isin(comparable_methods)
    ]
    comparison_df = baseline_comparison_df.merge(
        mtl_comparison_df,
        on="method",
        suffixes=("_baseline", "_mtl"),
    )
    for metric in ["auroc_mean", "auprc_mean", "precision_at_k_mean"]:
        comparison_df[f"delta_{metric}"] = (
            comparison_df[f"{metric}_mtl"] - comparison_df[f"{metric}_baseline"]
        ).round(4)
    comparison_df.to_csv(output_paths.compare_summary_path, index=False)
    print("\nBaseline vs MTL comparison")
    print(comparison_df.to_string(index=False))
    print(f"Saved comparison to: {output_paths.compare_summary_path}")

    return {
        "summary_df": summary_df,
        "baseline_summary_df": baseline_summary_df,
        "comparison_df": comparison_df,
    }


PALETTE = {
    "random_mean": "#55A868",
    "attention_weights": "#4C72B0",
    "integrated_gradients": "#DD8452",
    "gradient_x_input": "#64B5CD",
    "smoothgrad_ig": "#C44E52",
    "occlusion": "#8172B3",
    "residue_head": "#937860",
}
METHOD_XLABELS = {
    "random_mean": "Random",
    "integrated_gradients": "Integrated\nGradients",
    "gradient_x_input": "Gradient ×\nInput",
    "smoothgrad_ig": "SmoothGrad-\nIG",
    "occlusion": "Occlusion",
    "attention_weights": "Attention\nPooling",
    "residue_head": "MTL\nResidue Head",
}
DEFAULT_FAMILY_ORDER = ["Baseline (04)", "MTL (05)", "MTL (05 frozen)", "MTL (06 top1_unfrozen)"]
DEFAULT_FAMILY_LINESTYLE = {
    "Baseline (04)": "--",
    "MTL (05)": "-",
    "MTL (05 frozen)": "-",
    "MTL (06 top1_unfrozen)": "-.",
}
DEFAULT_FAMILY_MARKER = {
    "Baseline (04)": "o",
    "MTL (05)": "^",
    "MTL (05 frozen)": "^",
    "MTL (06 top1_unfrozen)": "s",
}
PAPER_FIGSIZE = (12, 8)
PAPER_WIDE_FIGSIZE = (15, 8.5)
PAPER_TITLE_FONTSIZE = 24
PAPER_LABEL_FONTSIZE = 22
PAPER_TICK_FONTSIZE = 18
PAPER_LEGEND_FONTSIZE = 16
PAPER_LEGEND_TITLE_FONTSIZE = 17
PAPER_ANNOTATION_FONTSIZE = 16


def get_family_order(combined_probe_df: pd.DataFrame) -> list[str]:
    families = list(pd.unique(combined_probe_df["model_family"]))
    ordered = [family for family in DEFAULT_FAMILY_ORDER if family in families]
    remaining = sorted(family for family in families if family not in ordered)
    return ordered + remaining


def get_family_styles(
    family_order: list[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    fallback_linestyles: list[Any] = ["-", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    fallback_markers = ["D", "P", "X", "v", ">"]
    family_linestyle = {}
    family_marker = {}
    fallback_idx = 0
    for family in family_order:
        if family in DEFAULT_FAMILY_LINESTYLE:
            family_linestyle[family] = DEFAULT_FAMILY_LINESTYLE[family]
        else:
            family_linestyle[family] = fallback_linestyles[fallback_idx % len(fallback_linestyles)]
        if family in DEFAULT_FAMILY_MARKER:
            family_marker[family] = DEFAULT_FAMILY_MARKER[family]
        else:
            family_marker[family] = fallback_markers[fallback_idx % len(fallback_markers)]
        if family not in DEFAULT_FAMILY_LINESTYLE or family not in DEFAULT_FAMILY_MARKER:
            fallback_idx += 1
    return family_linestyle, family_marker


def describe_linestyle(linestyle: Any) -> str:
    if linestyle == "--":
        return "dashed"
    if linestyle == "-.":
        return "dash-dot"
    if linestyle == ":":
        return "dotted"
    if linestyle == "-":
        return "solid"
    return "custom dash"


def describe_marker(marker: str) -> str:
    marker_names = {
        "o": "circle",
        "^": "triangle",
        "s": "square",
        "D": "diamond",
        "P": "plus",
        "X": "x",
        "v": "down-triangle",
        ">": "right-triangle",
    }
    return marker_names.get(marker, marker)


def metric_out_path(out_path: Path, metric_col: str) -> Path:
    out_path = Path(out_path)
    return out_path.with_name(f"{out_path.stem}_{metric_col}{out_path.suffix}")


def apply_paper_axis_style(
    ax,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    ax.set_title(title, fontsize=PAPER_TITLE_FONTSIZE, pad=18)
    ax.set_xlabel(xlabel, fontsize=PAPER_LABEL_FONTSIZE, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=PAPER_LABEL_FONTSIZE, labelpad=12)
    ax.tick_params(axis="both", labelsize=PAPER_TICK_FONTSIZE)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)


def set_method_xticklabels(ax, methods: list[str]) -> None:
    ax.set_xticks(
        range(len(methods)),
        [METHOD_XLABELS.get(method, method) for method in methods],
        fontsize=PAPER_TICK_FONTSIZE,
    )


def style_legend(ax, title: str | None = None) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    if title is not None:
        legend.set_title(title)
    legend.get_title().set_fontsize(PAPER_LEGEND_TITLE_FONTSIZE)
    for text in legend.get_texts():
        text.set_fontsize(PAPER_LEGEND_FONTSIZE)


def _save_png_and_pdf(fig, out_path: Path) -> None:
    out_path = Path(out_path)
    png_path = out_path.with_suffix(".png")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    try:
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved plot to: {png_path}")
        print(f"Saved plot to: {pdf_path}")
    except Exception as exc:
        print(f"Saved plot to: {png_path}")
        print(f"Could not save PDF to {pdf_path}: {exc}")


def print_localization_caption_drafts() -> None:
    print("\nMain figure caption draft:")
    print(
        "We compare a null random baseline, post-hoc protein-level explanations "
        "(Integrated Gradients and occlusion sensitivity), and a supervised MTL "
        "residue-level epitope head. Metrics are computed per protein against IEDB "
        "epitope annotations and averaged across proteins. Attention-pooling weights "
        "and additional gradient variants are reported in Supplementary Fig. X because "
        "they represent model-internal signals or robustness checks rather than the "
        "core attribution comparison."
    )
    print("\nSupplementary caption draft:")
    print(
        "Additional residue-level signals include attention-pooling weights, Gradient "
        "× Input, and SmoothGrad-IG. Similar near-random performance across multiple "
        "attribution families would indicate that poor epitope localization is not "
        "specific to a single attribution algorithm."
    )


def _plot_localization_figure(
    combined_probe_df: pd.DataFrame,
    out_path: Path,
    methods: list[str],
    title: str,
    figsize: tuple[float, float],
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = original_label_rows(combined_probe_df)
    plot_df = plot_df[plot_df["method"].isin(methods)].copy()
    if plot_df.empty:
        raise ValueError("No rows available for requested localization methods.")

    family_order = get_family_order(plot_df)
    metrics_config = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("precision_at_k", "Precision@k"),
    ]
    palette = {method: PALETTE[method] for method in methods if method in PALETTE}

    for metric_col, metric_label in metrics_config:
        metric_df = plot_df.dropna(subset=[metric_col]).copy()
        fig, ax = plt.subplots(figsize=figsize)
        sns.violinplot(
            data=metric_df,
            x="method",
            y=metric_col,
            order=methods,
            hue="method",
            palette=palette,
            inner=None,
            cut=0,
            linewidth=1.0,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=metric_df,
            x="method",
            y=metric_col,
            order=methods,
            hue="model_family",
            hue_order=family_order,
            dodge=True,
            alpha=0.45,
            size=3.5,
            jitter=0.18,
            ax=ax,
        )
        random_values = metric_df.loc[metric_df["method"] == "random_mean", metric_col]
        if not random_values.empty:
            ax.axhline(
                float(random_values.mean()),
                color="dimgray",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
            )
        apply_paper_axis_style(ax, f"{title}: {metric_label}", "", metric_label)
        set_method_xticklabels(ax, methods)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), title="Model", loc="best")
        style_legend(ax, title="Model")
        fig.tight_layout()
        _save_png_and_pdf(fig, metric_out_path(Path(out_path), metric_col))
        plt.close(fig)
    print_localization_caption_drafts()


def warn_label_scrambling_sanity_check(
    probe_df: pd.DataFrame,
    tolerance: float = 0.05,
) -> None:
    frame = ensure_label_variant_column(probe_df)
    scrambled_df = frame.loc[frame["label_variant"] == "scrambled"].copy()
    if scrambled_df.empty:
        print("Label scrambling sanity check warning: no scrambled-label rows found.")
        return

    group_cols = ["method"]
    if "model_family" in scrambled_df.columns:
        group_cols = ["model_family", "method"]

    for group_key, group_df in scrambled_df.groupby(group_cols, dropna=False):
        if isinstance(group_key, tuple):
            label = " / ".join(str(value) for value in group_key)
        else:
            label = str(group_key)
        mean_auroc = float(group_df["auroc"].dropna().mean())
        mean_auprc = float(group_df["auprc"].dropna().mean())
        expected_auprc = float(group_df["epitope_density"].dropna().mean())
        auroc_failed = not math.isnan(mean_auroc) and abs(mean_auroc - 0.5) >= tolerance
        auprc_failed = (
            not math.isnan(mean_auprc)
            and not math.isnan(expected_auprc)
            and abs(mean_auprc - expected_auprc) >= tolerance
        )
        if auroc_failed or auprc_failed:
            print(
                "Sanity check failed: metrics not collapsing under label scrambling "
                f"for {label}. AUROC={mean_auroc:.3f} (expected 0.500), "
                f"AUPRC={mean_auprc:.3f} (expected {expected_auprc:.3f})."
            )


def plot_label_scrambling_sanity_check(probe_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = ensure_label_variant_column(probe_df)
    methods = [method for method in SUPPLEMENTARY_SIGNAL_METHODS if method in set(frame["method"])]
    plot_df = frame.loc[
        frame["method"].isin(methods)
        & frame["label_variant"].isin(["original", "scrambled"])
    ].copy()
    if plot_df.empty or "scrambled" not in set(plot_df["label_variant"]):
        print("Label scrambling sanity check skipped: no scrambled-label rows found.")
        return

    metrics_config = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("precision_at_k", "Precision@k"),
    ]
    label_palette = {"original": "#4C72B0", "scrambled": "#999999"}

    for metric_col, metric_label in metrics_config:
        metric_df = plot_df.dropna(subset=[metric_col]).copy()
        fig, ax = plt.subplots(figsize=PAPER_WIDE_FIGSIZE)
        sns.violinplot(
            data=metric_df,
            x="method",
            y=metric_col,
            hue="label_variant",
            order=methods,
            hue_order=["original", "scrambled"],
            palette=label_palette,
            inner=None,
            cut=0,
            linewidth=1.0,
            dodge=True,
            ax=ax,
        )
        sns.stripplot(
            data=metric_df,
            x="method",
            y=metric_col,
            hue="label_variant",
            order=methods,
            hue_order=["original", "scrambled"],
            palette=label_palette,
            dodge=True,
            alpha=0.35,
            size=3.0,
            jitter=0.18,
            ax=ax,
        )
        apply_paper_axis_style(ax, f"Sanity Check: Label Scrambling ({metric_label})", "", metric_label)
        set_method_xticklabels(ax, methods)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), title="Labels", loc="best")
        style_legend(ax, title="Labels")
        fig.tight_layout()
        _save_png_and_pdf(fig, metric_out_path(Path(out_path), metric_col))
        plt.close(fig)
    warn_label_scrambling_sanity_check(frame)
    print(
        "Label scrambling sanity check:\n"
        "All methods should collapse to chance-level performance if the evaluation is well-calibrated.\n"
        "If not, metric bias or leakage may be present."
    )


def plot_main_localization_figure(combined_probe_df: pd.DataFrame, out_path: Path) -> None:
    _plot_localization_figure(
        combined_probe_df,
        out_path,
        MAIN_LOCALIZATION_METHODS,
        "Residue-Level Localization vs. IEDB Epitopes",
        figsize=PAPER_FIGSIZE,
    )


def plot_supplementary_all_signals_figure(combined_probe_df: pd.DataFrame, out_path: Path) -> None:
    _plot_localization_figure(
        combined_probe_df,
        out_path,
        SUPPLEMENTARY_SIGNAL_METHODS,
        "All Residue-Level Signals vs. IEDB Epitopes",
        figsize=PAPER_WIDE_FIGSIZE,
    )


def plot_probe_violins(combined_probe_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    violin_order = ["attention_weights", "integrated_gradients", "occlusion", "random_mean", "residue_head"]
    violin_df = original_label_rows(combined_probe_df)
    violin_df = violin_df[violin_df["method"].isin(violin_order)].copy()
    family_order = get_family_order(violin_df)
    family_linestyle, _ = get_family_styles(family_order)
    metrics_config = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("precision_at_k", "Precision@k"),
    ]

    for col, label in metrics_config:
        plot_data = violin_df.dropna(subset=[col]).copy()
        fig, ax = plt.subplots(figsize=PAPER_WIDE_FIGSIZE)

        sns.violinplot(
            data=plot_data,
            x="method",
            y=col,
            hue="model_family",
            order=violin_order,
            hue_order=family_order,
            inner=None,
            cut=0,
            dodge=True,
            ax=ax,
        )
        sns.stripplot(
            data=plot_data,
            x="method",
            y=col,
            hue="model_family",
            order=violin_order,
            hue_order=family_order,
            dodge=True,
            alpha=0.35,
            size=3.5,
            jitter=True,
            ax=ax,
        )

        for family in family_order:
            family_random = plot_data[
                (plot_data["method"] == "random_mean") & (plot_data["model_family"] == family)
            ][col]
            if not family_random.empty:
                ax.axhline(
                    float(family_random.mean()),
                    color="gray",
                    linestyle=family_linestyle[family],
                    linewidth=1.2,
                    alpha=0.9,
                )

        overall_mean, overall_ci_low, overall_ci_high = bootstrap_mean_ci(plot_data[col])
        apply_paper_axis_style(
            ax,
            f"Residue Attribution Faithfulness: {label}\nmean [95% CI]: {overall_mean:.3f} [{overall_ci_low:.3f}, {overall_ci_high:.3f}]",
            "Method",
            label,
        )
        set_method_xticklabels(ax, violin_order)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), title="Model family", loc="best")
        style_legend(ax, title="Model family")

        fig.tight_layout()
        _save_png_and_pdf(fig, metric_out_path(Path(out_path), col))
        plt.close(fig)


def plot_probe_paired_deltas(
    combined_probe_df: pd.DataFrame,
    out_path: Path,
    metric: str = "auprc",
    baseline_family: str = "Baseline (04)",
    compare_family: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    metric_labels = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "precision_at_k": "Precision@k",
    }
    if metric not in metric_labels:
        raise ValueError(f"Unsupported metric for paired delta plot: {metric}")

    if compare_family is None:
        family_order = get_family_order(combined_probe_df)
        compare_family = next((family for family in family_order if family != baseline_family), None)
        if compare_family is None:
            raise ValueError("No non-baseline model_family available for paired delta plotting.")

    combined_probe_df = original_label_rows(combined_probe_df)
    plot_df = combined_probe_df[
        combined_probe_df["model_family"].isin([baseline_family, compare_family])
    ].copy()
    plot_df = plot_df.loc[plot_df["method"] != "random_mean"].copy()
    plot_df = plot_df.dropna(subset=[metric]).copy()

    methods_in_both = []
    preferred_order = ["attention_weights", "integrated_gradients", "occlusion", "residue_head"]
    baseline_methods = set(plot_df.loc[plot_df["model_family"] == baseline_family, "method"])
    compare_methods = set(plot_df.loc[plot_df["model_family"] == compare_family, "method"])
    for method in preferred_order:
        if method in baseline_methods and method in compare_methods:
            methods_in_both.append(method)

    delta_frames = []
    for method in methods_in_both:
        baseline_method_df = plot_df.loc[
            (plot_df["model_family"] == baseline_family) & (plot_df["method"] == method),
            ["accession", metric],
        ].rename(columns={metric: "baseline_value"})
        compare_method_df = plot_df.loc[
            (plot_df["model_family"] == compare_family) & (plot_df["method"] == method),
            ["accession", metric],
        ].rename(columns={metric: "compare_value"})
        merged = baseline_method_df.merge(compare_method_df, on="accession", how="inner")
        if merged.empty:
            continue
        merged["method"] = method
        merged["delta"] = merged["compare_value"] - merged["baseline_value"]
        merged["baseline_family"] = baseline_family
        merged["compare_family"] = compare_family
        delta_frames.append(
            merged[
                [
                    "accession",
                    "method",
                    "delta",
                    "baseline_value",
                    "compare_value",
                    "baseline_family",
                    "compare_family",
                ]
            ]
        )

    if not delta_frames:
        raise ValueError("No paired accession-level comparisons available for paired delta plotting.")

    delta_df = pd.concat(delta_frames, ignore_index=True)
    if delta_df.empty:
        raise ValueError("No paired accession-level comparisons available for paired delta plotting.")

    method_order = [method for method in preferred_order if method in set(delta_df["method"])]
    palette = {method: PALETTE[method] for method in method_order}

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    sns.violinplot(
        data=delta_df,
        x="method",
        y="delta",
        order=method_order,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1.0,
        ax=ax,
    )
    sns.boxplot(
        data=delta_df,
        x="method",
        y="delta",
        order=method_order,
        width=0.18,
        showcaps=True,
        boxprops={"facecolor": "white", "alpha": 0.9, "zorder": 3},
        whiskerprops={"linewidth": 1.1},
        medianprops={"color": "black", "linewidth": 1.4},
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=delta_df,
        x="method",
        y="delta",
        order=method_order,
        hue="method",
        palette=palette,
        alpha=0.4,
        size=4,
        jitter=0.18,
        dodge=False,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.9)
    apply_paper_axis_style(
        ax,
        f"Per-protein Δ{metric_labels[metric]} vs Baseline\n{compare_family} - {baseline_family}",
        "Method",
        f"Δ{metric_labels[metric]}",
    )
    set_method_xticklabels(ax, method_order)

    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min if y_max > y_min else 1.0
    for idx, method in enumerate(method_order):
        method_df = delta_df.loc[delta_df["method"] == method]
        if method_df.empty:
            continue
        ax.text(
            idx,
            method_df["delta"].max() + 0.04 * y_span,
            f"median={method_df['delta'].median():.3f}\nn={len(method_df)}",
            ha="center",
            va="bottom",
            fontsize=PAPER_ANNOTATION_FONTSIZE,
        )

    fig.tight_layout()
    _save_png_and_pdf(fig, Path(out_path))
    plt.close(fig)


def plot_probe_binned_density_trends(
    combined_probe_df: pd.DataFrame,
    out_path: Path,
    metric: str = "auprc",
    n_bins: int = 6,
    include_methods: list[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    metric_labels = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "precision_at_k": "Precision@k",
    }
    if metric not in metric_labels:
        raise ValueError(f"Unsupported metric for binned density trend plot: {metric}")

    combined_probe_df = original_label_rows(combined_probe_df)
    if include_methods is None:
        include_methods = ["attention_weights", "integrated_gradients", "occlusion", "residue_head", "random_mean"]
    include_methods = [method for method in include_methods if method in set(combined_probe_df["method"])]

    plot_df = combined_probe_df.loc[combined_probe_df["method"].isin(include_methods)].copy()
    plot_df = plot_df.dropna(subset=[metric, "epitope_density"]).copy()
    plot_df["density_bin"] = pd.qcut(plot_df["epitope_density"], q=n_bins, duplicates="drop")

    if plot_df["density_bin"].nunique() < 3:
        raise ValueError("Fewer than 3 unique epitope-density bins remain after qcut; cannot plot binned density trends.")

    plot_df["bin_midpoint"] = plot_df["density_bin"].map(lambda interval: float((interval.left + interval.right) / 2))
    grouped = (
        plot_df.groupby(["model_family", "method", "density_bin"], observed=True)
        .agg(
            mean_metric=(metric, "mean"),
            std_metric=(metric, "std"),
            count=(metric, "count"),
            bin_midpoint=("bin_midpoint", "first"),
        )
        .reset_index()
    )
    grouped["sem_metric"] = np.where(
        grouped["count"] > 1,
        grouped["std_metric"] / np.sqrt(grouped["count"]),
        np.nan,
    )

    family_order = get_family_order(grouped)
    family_linestyle, family_marker = get_family_styles(family_order)
    method_order = [method for method in include_methods if method in set(grouped["method"])]

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    for family in family_order:
        for method in method_order:
            subset = grouped.loc[
                (grouped["model_family"] == family) & (grouped["method"] == method)
            ].sort_values("bin_midpoint")
            if subset.empty:
                continue
            ax.errorbar(
                subset["bin_midpoint"],
                subset["mean_metric"],
                yerr=subset["sem_metric"],
                color=PALETTE[method],
                linestyle=family_linestyle[family],
                marker=family_marker[family],
                linewidth=2.0,
                markersize=6,
                capsize=3,
                alpha=0.9,
            )

    apply_paper_axis_style(
        ax,
        f"Binned {metric_labels[metric]} vs. Epitope Density",
        "Epitope Density (quantile-bin midpoint)",
        metric_labels[metric],
    )

    method_handles = [
        Line2D([0], [0], color=PALETTE[method], linewidth=2.5, label=METHOD_XLABELS.get(method, method).replace("\n", " "))
        for method in method_order
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            color="dimgray",
            linewidth=2.6,
            linestyle=family_linestyle[family],
            marker=family_marker[family],
            markersize=7,
            label=(
                f"{family}: "
                f"{describe_linestyle(family_linestyle[family])} + "
                f"{describe_marker(family_marker[family])}"
            ),
        )
        for family in family_order
    ]

    method_legend = ax.legend(
        handles=method_handles,
        title="Method / Color",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
    )
    style_legend(ax, title="Method / Color")
    ax.add_artist(method_legend)
    ax.legend(
        handles=family_handles,
        title="Model family / Line + marker",
        loc="upper left",
        bbox_to_anchor=(0.40, 0.99),
    )
    style_legend(ax, title="Model family / Line + marker")
    fig.tight_layout()
    _save_png_and_pdf(fig, Path(out_path))
    plt.close(fig)


def plot_probe_density_trends(combined_probe_df: pd.DataFrame, auroc_out_path: Path, auprc_out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from statsmodels.nonparametric.smoothers_lowess import lowess

    combined_probe_df = original_label_rows(combined_probe_df)
    scatter_methods = ["attention_weights", "integrated_gradients", "occlusion", "random_mean", "residue_head"]
    scatter_df = combined_probe_df[combined_probe_df["method"].isin(scatter_methods)].copy()
    family_order = get_family_order(scatter_df)
    family_linestyle, family_marker = get_family_styles(family_order)

    for metric_col, metric_label, out_path in [
        ("auroc", "AUROC", auroc_out_path),
        ("auprc", "AUPRC", auprc_out_path),
    ]:
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

        for method in scatter_methods:
            for family in family_order:
                mdf = scatter_df[
                    (scatter_df["method"] == method) & (scatter_df["model_family"] == family)
                ].dropna(subset=[metric_col, "epitope_density"])
                if mdf.empty:
                    continue

                color = PALETTE[method]
                ax.scatter(
                    mdf["epitope_density"],
                    mdf[metric_col],
                    color=color,
                    alpha=0.38,
                    s=30,
                    marker=family_marker[family],
                    label=f"{METHOD_XLABELS[method].replace(chr(10), ' ')} - {family}",
                )
                if len(mdf) >= 5:
                    smoothed = lowess(
                        mdf[metric_col].values,
                        mdf["epitope_density"].values,
                        frac=0.5,
                        return_sorted=True,
                    )
                    ax.plot(
                        smoothed[:, 0],
                        smoothed[:, 1],
                        color=color,
                        linewidth=2.0,
                        linestyle=family_linestyle[family],
                    )

        apply_paper_axis_style(
            ax,
            f"{metric_label} vs. Epitope Density",
            "Epitope Density (fraction of residues)",
            metric_label,
        )

        method_handles = [
            Line2D([0], [0], color=PALETTE[method], linewidth=2.5, label=METHOD_XLABELS[method].replace("\n", " "))
            for method in scatter_methods
        ]
        family_handles = [
            Line2D(
                [0],
                [0],
                color="dimgray",
                linewidth=2.6,
                linestyle=family_linestyle[family],
                marker=family_marker[family],
                markersize=7,
                label=(
                    f"{family}: "
                    f"{describe_linestyle(family_linestyle[family])} + "
                    f"{describe_marker(family_marker[family])}"
                ),
            )
            for family in family_order
        ]

        method_legend = ax.legend(
            handles=method_handles,
            title="Method / Color",
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
        )
        style_legend(ax, title="Method / Color")
        ax.add_artist(method_legend)
        ax.legend(
            handles=family_handles,
            title="Model / Style\n(line + marker)",
            loc="upper left",
            bbox_to_anchor=(0.42, 0.99),
        )
        style_legend(ax, title="Model / Style\n(line + marker)")
        fig.tight_layout()
        _save_png_and_pdf(fig, Path(out_path))
        plt.close(fig)
