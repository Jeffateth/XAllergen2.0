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

from baseline_notebook_utils import (
    ESM_MODEL_NAME,
    HF_MODEL_NAME,
    MAX_SEQ_LEN,
    RANDOM_STATE,
    THRESHOLD,
    compute_attention_weights,
    compute_integrated_gradients,
    compute_residue_probabilities,
    load_baseline_checkpoint,
    load_mtl_checkpoint,
    mean_metric_dicts,
    normalize_scores,
    parse_epitope_label,
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
    Single-residue occlusion for MTL models.
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
) -> dict[str, pd.DataFrame]:
    ig_probe_device = device
    rng = np.random.default_rng(RANDOM_STATE)
    expected_mtl_methods = {
        "residue_head",
        "attention_weights",
        "integrated_gradients",
        "occlusion",
        "random_mean",
    }
    expected_baseline_methods = {"attention_weights", "integrated_gradients", "random_mean"}
    probe_rows = []
    baseline_probe_rows = []

    ig_model = model.to(ig_probe_device)
    ig_model.eval()
    baseline_model, _ = load_baseline_checkpoint(
        baseline_checkpoint_path,
        ig_probe_device,
        model_name=model_name,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    baseline_ig_model = baseline_model
    baseline_ig_model.eval()

    print(f"Integrated Gradients device: {ig_probe_device}")
    print(f"IG_STEPS: {ig_steps}")
    print(f"IG internal_batch_size: {ig_internal_batch_size}")

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
        probe_rows.append(
            {
                **base,
                "model_family": output_paths.mtl_family_label,
                "method": "residue_head",
                **compute_probe_metrics(epitope_labels, residue_scores),
            }
        )

        attention_scores = compute_attention_weights(ig_model, tokenizer, sequence, ig_probe_device)
        probe_rows.append(
            {
                **base,
                "model_family": output_paths.mtl_family_label,
                "method": "attention_weights",
                **compute_probe_metrics(epitope_labels, attention_scores),
            }
        )

        baseline_attention_scores = compute_attention_weights(
            baseline_model, tokenizer, sequence, ig_probe_device
        )
        baseline_probe_rows.append(
            {
                **base,
                "model_family": output_paths.baseline_family_label,
                "method": "attention_weights",
                **compute_probe_metrics(epitope_labels, baseline_attention_scores),
            }
        )

        ig_scores = compute_integrated_gradients(
            ig_model,
            tokenizer,
            sequence,
            ig_probe_device,
            steps=ig_steps,
            normalize=False,
            internal_batch_size=ig_internal_batch_size,
        )
        probe_rows.append(
            {
                **base,
                "model_family": output_paths.mtl_family_label,
                "method": "integrated_gradients",
                **compute_probe_metrics(epitope_labels, ig_scores),
            }
        )

        occlusion_scores = normalize_scores(
            compute_occlusion_scores_mtl(ig_model, tokenizer, sequence, ig_probe_device)
        )
        probe_rows.append(
            {
                **base,
                "model_family": output_paths.mtl_family_label,
                "method": "occlusion",
                **compute_probe_metrics(epitope_labels, occlusion_scores),
            }
        )

        baseline_ig_scores = compute_integrated_gradients(
            baseline_ig_model,
            tokenizer,
            sequence,
            ig_probe_device,
            steps=ig_steps,
            normalize=False,
            internal_batch_size=ig_internal_batch_size,
        )
        baseline_probe_rows.append(
            {
                **base,
                "model_family": output_paths.baseline_family_label,
                "method": "integrated_gradients",
                **compute_probe_metrics(epitope_labels, baseline_ig_scores),
            }
        )

        random_metrics = [
            compute_probe_metrics(epitope_labels, rng.uniform(0.0, 1.0, size=len(epitope_labels)))
            for _ in range(n_random_draws)
        ]
        random_summary = mean_metric_dicts(random_metrics)
        probe_rows.append(
            {
                **base,
                "model_family": output_paths.mtl_family_label,
                "method": "random_mean",
                **random_summary,
            }
        )
        baseline_probe_rows.append(
            {
                **base,
                "model_family": output_paths.baseline_family_label,
                "method": "random_mean",
                **random_summary,
            }
        )

        del (
            residue_scores,
            attention_scores,
            baseline_attention_scores,
            ig_scores,
            occlusion_scores,
            baseline_ig_scores,
            random_metrics,
            random_summary,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        processed_since_save += 1
        if save_every > 0 and processed_since_save >= save_every:
            probe_df = pd.DataFrame(probe_rows)
            baseline_probe_df = pd.DataFrame(baseline_probe_rows)
            combined_probe_df = pd.concat([baseline_probe_df, probe_df], ignore_index=True)
            probe_df.to_csv(output_paths.probe_rows_path, index=False)
            baseline_probe_df.to_csv(output_paths.baseline_probe_rows_path, index=False)
            combined_probe_df.to_csv(output_paths.combined_probe_rows_path, index=False)
            processed_since_save = 0

    del ig_model
    del baseline_ig_model
    del baseline_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    probe_df = pd.DataFrame(probe_rows)
    baseline_probe_df = pd.DataFrame(baseline_probe_rows)
    combined_probe_df = pd.concat([baseline_probe_df, probe_df], ignore_index=True)

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


def summarize_probe_methods(frame: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    summary_rows = []
    for method in methods:
        method_df = frame[frame["method"] == method]
        auroc_mean, auroc_ci_low, auroc_ci_high = bootstrap_mean_ci(method_df["auroc"])
        auprc_mean, auprc_ci_low, auprc_ci_high = bootstrap_mean_ci(method_df["auprc"])
        precision_mean, precision_ci_low, precision_ci_high = bootstrap_mean_ci(method_df["precision_at_k"])
        summary_rows.append(
            {
                "method": method,
                "auroc_mean": round(auroc_mean, 4),
                "auroc_ci_low": round(auroc_ci_low, 4),
                "auroc_ci_high": round(auroc_ci_high, 4),
                "auprc_mean": round(auprc_mean, 4),
                "auprc_ci_low": round(auprc_ci_low, 4),
                "auprc_ci_high": round(auprc_ci_high, 4),
                "precision_at_k_mean": round(precision_mean, 4),
                "precision_at_k_ci_low": round(precision_ci_low, 4),
                "precision_at_k_ci_high": round(precision_ci_high, 4),
                "n_proteins": int(len(method_df)),
            }
        )
    return pd.DataFrame(summary_rows)


def summarize_probe_outputs(
    probe_df: pd.DataFrame,
    baseline_probe_df: pd.DataFrame,
    output_paths: MTLOutputPaths,
) -> dict[str, pd.DataFrame]:
    summary_df = summarize_probe_methods(
        probe_df,
        ["residue_head", "attention_weights", "integrated_gradients", "occlusion", "random_mean"],
    )
    baseline_summary_df = summarize_probe_methods(
        baseline_probe_df,
        ["attention_weights", "integrated_gradients", "random_mean"],
    )
    summary_df.to_csv(output_paths.probe_summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved summary to: {output_paths.probe_summary_path}")

    comparable_methods = ["attention_weights", "integrated_gradients", "random_mean"]
    comparison_df = baseline_summary_df.merge(
        summary_df[summary_df["method"].isin(comparable_methods)],
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
    "attention_weights": "#4C72B0",
    "integrated_gradients": "#DD8452",
    "occlusion": "#8172B3",
    "random_mean": "#55A868",
    "residue_head": "#C44E52",
}
METHOD_XLABELS = {
    "attention_weights": "Attention\nWeights",
    "integrated_gradients": "Integrated\nGradients",
    "occlusion": "Occlusion",
    "random_mean": "Random\nMean",
    "residue_head": "Residue\nHead (MTL)",
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


def plot_probe_violins(combined_probe_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    violin_order = ["attention_weights", "integrated_gradients", "occlusion", "random_mean", "residue_head"]
    violin_df = combined_probe_df[combined_probe_df["method"].isin(violin_order)].copy()
    family_order = get_family_order(violin_df)
    family_linestyle, _ = get_family_styles(family_order)
    metrics_config = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("precision_at_k", "Precision@k"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (col, label) in zip(axes, metrics_config):
        plot_data = violin_df.dropna(subset=[col]).copy()

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
        ax.set_title(
            f"{label}\nmean [95% bootstrap CI]: {overall_mean:.3f} [{overall_ci_low:.3f}, {overall_ci_high:.3f}]",
            fontsize=11,
        )
        ax.set_xlabel("Method")
        ax.set_ylabel(label)
        ax.set_xticklabels([METHOD_XLABELS[m] for m in violin_order], fontsize=9)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if ax is axes[0]:
            ax.legend(unique.values(), unique.keys(), title="Model family", fontsize=8)
        else:
            ax.legend().remove()

    plt.suptitle("Residue Attribution Faithfulness vs. IEDB Epitopes", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to: {out_path}")


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

    fig, ax = plt.subplots(figsize=(10, 6))
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
    ax.set_xlabel("Method")
    ax.set_ylabel(f"Δ{metric_labels[metric]}")
    ax.set_xticklabels([METHOD_XLABELS.get(method, method) for method in method_order], fontsize=9)
    ax.set_title(
        f"Per-protein Δ{metric_labels[metric]} vs Baseline\n{compare_family} - {baseline_family}",
        fontsize=13,
    )

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
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to: {out_path}")


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

    fig, ax = plt.subplots(figsize=(10, 6))
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

    ax.set_title(f"Binned {metric_labels[metric]} vs. Epitope Density", fontsize=13)
    ax.set_xlabel("Epitope Density (quantile-bin midpoint)", fontsize=12)
    ax.set_ylabel(metric_labels[metric], fontsize=12)

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
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
    )
    ax.add_artist(method_legend)
    ax.legend(
        handles=family_handles,
        title="Model family / Line + marker",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0.40, 0.99),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to: {out_path}")


def plot_probe_density_trends(combined_probe_df: pd.DataFrame, auroc_out_path: Path, auprc_out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from statsmodels.nonparametric.smoothers_lowess import lowess

    scatter_methods = ["attention_weights", "integrated_gradients", "occlusion", "random_mean", "residue_head"]
    scatter_df = combined_probe_df[combined_probe_df["method"].isin(scatter_methods)].copy()
    family_order = get_family_order(scatter_df)
    family_linestyle, family_marker = get_family_styles(family_order)

    for metric_col, metric_label, out_path in [
        ("auroc", "AUROC", auroc_out_path),
        ("auprc", "AUPRC", auprc_out_path),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))

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

        ax.set_xlabel("Epitope Density (fraction of residues)", fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f"{metric_label} vs. Epitope Density", fontsize=13)

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
            fontsize=8,
            title_fontsize=9,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
        )
        ax.add_artist(method_legend)
        ax.legend(
            handles=family_handles,
            title="Model / Style\n(line + marker)",
            fontsize=8,
            title_fontsize=9,
            loc="upper left",
            bbox_to_anchor=(0.42, 0.99),
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved plot to: {out_path}")
