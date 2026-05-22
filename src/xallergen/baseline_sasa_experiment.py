from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .baseline_notebook_utils import (
    DROPOUT,
    ESM_MODEL_NAME,
    HF_MODEL_NAME,
    HIDDEN_DIM,
    RANDOM_STATE,
    THRESHOLD,
    FrozenESMAllergenClassifier,
    build_tokenizer,
    inspect_precomputed_rsa_file,
    load_baseline_checkpoint,
    load_precomputed_rsa_mapping,
    seed_everything,
    prepare_baseline_probe_frame,
    run_baseline_probe_suite,
)
from .mtl_epitope_notebook_utils import summarize_probe_methods


@dataclass(frozen=True)
class SASAExperimentConfig:
    batch_size: int = 24
    epochs: int = 30
    patience: int = 5
    min_delta: float = 1e-3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_cls: float = 1.0
    lambda_values: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 5.0)
    add_special_tokens: bool = False
    hidden_dim: int = HIDDEN_DIM
    dropout: float = DROPOUT
    threshold: float = THRESHOLD
    ig_steps: int = 50


class ProteinDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        rsa_lookup: dict[str, torch.Tensor | None],
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.rsa_lookup = rsa_lookup

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        sequence_id = str(row["sequence_id"]).strip()
        rsa_tensor = self.rsa_lookup.get(sequence_id)
        if rsa_tensor is not None:
            rsa_tensor = rsa_tensor.detach().clone()
            rsa_tensor.requires_grad_(False)
        return {
            "sequence_id": sequence_id,
            "sequence": str(row["sequence"]).strip().upper(),
            "label": int(row["label"]),
            "rSASA": rsa_tensor,
        }


def collate_batch(
    batch: list[dict[str, Any]],
    tokenizer,
    add_special_tokens: bool,
) -> dict[str, Any]:
    sequences = [item["sequence"] for item in batch]
    encodings = tokenizer(
        sequences,
        add_special_tokens=add_special_tokens,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    batch_size, max_tokens = encodings["input_ids"].shape
    rsa_tensor = torch.zeros((batch_size, max_tokens), dtype=torch.float32)
    has_rsa = torch.zeros(batch_size, dtype=torch.bool)

    for idx, item in enumerate(batch):
        values = item["rSASA"]
        if values is None:
            continue
        if values.shape[0] != int(encodings["attention_mask"][idx].sum().item()):
            raise ValueError(
                f"rSASA/token length mismatch for {item['sequence_id']}: "
                f"rsa={values.shape[0]} tokens={int(encodings['attention_mask'][idx].sum().item())}"
            )
        rsa_tensor[idx, : values.shape[0]] = values
        has_rsa[idx] = True

    return {
        "sequence_id": [item["sequence_id"] for item in batch],
        "sequence": sequences,
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "rSASA": rsa_tensor,
        "has_rSASA": has_rsa,
    }


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("input_ids", "attention_mask", "label", "rSASA", "has_rSASA"):
        moved[key] = batch[key].to(device)
    return moved


def build_dataloader(
    frame: pd.DataFrame,
    rsa_lookup: dict[str, torch.Tensor | None],
    tokenizer,
    batch_size: int,
    shuffle: bool,
    add_special_tokens: bool,
) -> DataLoader:
    dataset = ProteinDataset(frame, rsa_lookup)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, add_special_tokens),
    )


def inspect_rsa_inputs(
    train_rsa_path: Path,
    test_rsa_path: Path,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> pd.DataFrame:
    records = [
        inspect_precomputed_rsa_file(
            train_rsa_path,
            expected_ids=train_frame["sequence_id"].astype(str).str.strip().tolist(),
        ),
        inspect_precomputed_rsa_file(
            test_rsa_path,
            expected_ids=test_frame["sequence_id"].astype(str).str.strip().tolist(),
        ),
    ]
    return pd.DataFrame(records)


def load_rsa_lookup_dicts(
    train_rsa_path: Path,
    test_rsa_path: Path,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    add_special_tokens: bool,
) -> tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor | None], dict[str, Any]]:
    train_lookup, train_summary = load_precomputed_rsa_mapping(
        train_rsa_path,
        expected_frame=train_frame,
        add_special_tokens=add_special_tokens,
    )
    test_lookup, test_summary = load_precomputed_rsa_mapping(
        test_rsa_path,
        expected_frame=test_frame,
        add_special_tokens=add_special_tokens,
    )
    return train_lookup, test_lookup, {"train": train_summary, "test": test_summary}


def compute_sasa_loss(
    attention_weights: torch.Tensor,
    rsa_values: torch.Tensor,
    attention_mask: torch.Tensor,
    has_rsa: torch.Tensor,
) -> torch.Tensor | None:
    available = has_rsa.bool()
    if not bool(available.any()):
        return None
    valid_alpha = attention_weights[available]
    valid_rsa = rsa_values[available]
    valid_mask = attention_mask[available].to(dtype=valid_alpha.dtype)
    per_position = valid_alpha * (1.0 - valid_rsa) * valid_mask
    denom = valid_mask.sum(dim=-1).clamp_min(1.0)
    per_example = per_position.sum(dim=-1) / denom
    return per_example.mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    lambda_cls: float,
    lambda_sasa: float,
) -> dict[str, float | None]:
    model.train()
    total_cls_loss = 0.0
    total_sasa_loss = 0.0
    total_loss = 0.0
    total_examples = 0
    total_sasa_examples = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["input_ids"], batch["attention_mask"])
        cls_loss = criterion(outputs["logits"], batch["label"])
        loss = lambda_cls * cls_loss
        sasa_loss = None
        if lambda_sasa > 0.0:
            sasa_loss = compute_sasa_loss(
                outputs["attention_weights"],
                batch["rSASA"],
                batch["attention_mask"],
                batch["has_rSASA"],
            )
            if sasa_loss is not None:
                loss = loss + lambda_sasa * sasa_loss

        loss.backward()
        optimizer.step()

        batch_size = batch["label"].shape[0]
        total_examples += batch_size
        total_cls_loss += float(cls_loss.item()) * batch_size
        total_loss += float(loss.item()) * batch_size
        if sasa_loss is not None:
            sasa_examples = int(batch["has_rSASA"].sum().item())
            total_sasa_loss += float(sasa_loss.item()) * sasa_examples
            total_sasa_examples += sasa_examples

    return {
        "cls_loss": total_cls_loss / max(total_examples, 1),
        "sasa_loss": (
            total_sasa_loss / total_sasa_examples if total_sasa_examples > 0 else None
        ),
        "total_loss": total_loss / max(total_examples, 1),
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module | None = None,
    lambda_cls: float = 1.0,
    lambda_sasa: float = 0.0,
    threshold: float = THRESHOLD,
) -> tuple[dict[str, float | None] | None, pd.DataFrame]:
    model.eval()
    total_cls_loss = 0.0
    total_sasa_loss = 0.0
    total_loss = 0.0
    total_examples = 0
    total_sasa_examples = 0
    rows = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["input_ids"], batch["attention_mask"])
        logits = outputs["logits"]
        probs = torch.sigmoid(logits)

        if criterion is not None:
            cls_loss = criterion(logits, batch["label"])
            total = lambda_cls * cls_loss
            sasa_loss = None
            if lambda_sasa > 0.0:
                sasa_loss = compute_sasa_loss(
                    outputs["attention_weights"],
                    batch["rSASA"],
                    batch["attention_mask"],
                    batch["has_rSASA"],
                )
                if sasa_loss is not None:
                    total = total + lambda_sasa * sasa_loss
            batch_size = batch["label"].shape[0]
            total_examples += batch_size
            total_cls_loss += float(cls_loss.item()) * batch_size
            total_loss += float(total.item()) * batch_size
            if sasa_loss is not None:
                sasa_examples = int(batch["has_rSASA"].sum().item())
                total_sasa_loss += float(sasa_loss.item()) * sasa_examples
                total_sasa_examples += sasa_examples

        for idx in range(len(batch["sequence_id"])):
            rows.append(
                {
                    "sequence_id": batch["sequence_id"][idx],
                    "sequence": batch["sequence"][idx],
                    "label": int(batch["label"][idx].item()),
                    "logit": float(logits[idx].item()),
                    "pred_prob": float(probs[idx].item()),
                    "pred_label": int(probs[idx].item() >= threshold),
                }
            )

    if criterion is None:
        loss_summary = None
    else:
        loss_summary = {
            "cls_loss": total_cls_loss / max(total_examples, 1),
            "sasa_loss": (
                total_sasa_loss / total_sasa_examples if total_sasa_examples > 0 else None
            ),
            "total_loss": total_loss / max(total_examples, 1),
        }
    return loss_summary, pd.DataFrame(rows)


def compute_metrics(pred_df: pd.DataFrame, threshold: float = THRESHOLD) -> dict[str, Any]:
    y_true = pred_df["label"].to_numpy()
    y_prob = pred_df["pred_prob"].to_numpy()
    y_pred = pred_df["pred_label"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
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


def format_lambda_suffix(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "neg").replace(".", "p")


def train_single_lambda_run(
    lambda_sasa: float,
    config: SASAExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    checkpoint_path: Path,
    model_name: str = HF_MODEL_NAME,
) -> dict[str, Any]:
    seed_everything(RANDOM_STATE)
    model = FrozenESMAllergenClassifier(
        model_name,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    assert not any(param.requires_grad for param in model.backbone.parameters())
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_labels = train_loader.dataset.frame["label"].value_counts()
    n_neg = int(train_labels.get(0, 0))
    n_pos = int(train_labels.get(1, 0))
    protein_pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=protein_pos_weight)

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    epoch_bar = tqdm(range(1, config.epochs + 1), desc=f"lambda_sasa={lambda_sasa:g}", unit="epoch")
    for epoch in epoch_bar:
        train_summary = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            lambda_cls=config.lambda_cls,
            lambda_sasa=lambda_sasa,
        )
        val_summary, val_pred_df = predict(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            lambda_cls=config.lambda_cls,
            lambda_sasa=lambda_sasa,
            threshold=config.threshold,
        )
        assert val_summary is not None
        val_metrics = compute_metrics(val_pred_df, threshold=config.threshold)
        history.append(
            {
                "epoch": epoch,
                "train_cls_loss": train_summary["cls_loss"],
                "train_sasa_loss": train_summary["sasa_loss"],
                "train_total_loss": train_summary["total_loss"],
                "val_cls_loss": val_summary["cls_loss"],
                "val_sasa_loss": val_summary["sasa_loss"],
                "val_total_loss": val_summary["total_loss"],
                "val_auroc": val_metrics["auroc"],
            }
        )

        if val_summary["total_loss"] < best_val_loss - config.min_delta:
            best_val_loss = float(val_summary["total_loss"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "esm_model_name": ESM_MODEL_NAME,
                    "architecture_hyperparameters": {
                        "hidden_dim": config.hidden_dim,
                        "dropout": config.dropout,
                    },
                    "training_history": history,
                    "lambda_cls": config.lambda_cls,
                    "lambda_sasa": lambda_sasa,
                    "tokenizer_add_special_tokens": config.add_special_tokens,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        train_sasa_text = "None" if train_summary["sasa_loss"] is None else f"{train_summary['sasa_loss']:.5f}"
        val_sasa_text = "None" if val_summary["sasa_loss"] is None else f"{val_summary['sasa_loss']:.5f}"
        print(
            f"Epoch {epoch:>3}/{config.epochs} | "
            f"train_cls={train_summary['cls_loss']:.5f} | "
            f"train_sasa={train_sasa_text} | "
            f"train_total={train_summary['total_loss']:.5f} | "
            f"val_cls={val_summary['cls_loss']:.5f} | "
            f"val_sasa={val_sasa_text} | "
            f"val_total={val_summary['total_loss']:.5f} | "
            f"val_auroc={val_metrics['auroc']:.5f} | "
            f"best={best_epoch}"
        )
        epoch_bar.set_postfix(
            train_total=f"{train_summary['total_loss']:.5f}",
            val_total=f"{val_summary['total_loss']:.5f}",
            val_auroc=f"{val_metrics['auroc']:.5f}",
            best=best_epoch,
        )

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    return {
        "checkpoint_path": checkpoint_path,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_total_loss": best_val_loss,
        "protein_pos_weight": protein_pos_weight.item(),
    }


def evaluate_residue_localization(
    checkpoint_path: Path,
    device: str,
    positives_csv: Path,
    probe_rows_path: Path,
    model_name: str = HF_MODEL_NAME,
    threshold: float = THRESHOLD,
) -> dict[str, Any]:
    tokenizer = build_tokenizer(model_name)
    model, _ = load_baseline_checkpoint(
        checkpoint_path,
        device,
        model_name=model_name,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )
    probe_frame = prepare_baseline_probe_frame(positives_csv)
    probe_df = run_baseline_probe_suite(
        model=model,
        tokenizer=tokenizer,
        eval_df=probe_frame,
        device=device,
        enabled_methods={"attention_weights"},
    )
    probe_rows_path.parent.mkdir(parents=True, exist_ok=True)
    probe_df.to_csv(probe_rows_path, index=False)
    summary_df = summarize_probe_methods(probe_df, ["attention_weights"])
    if summary_df.empty:
        raise RuntimeError("No residue-level summary rows were produced.")
    summary_row = summary_df.iloc[0].to_dict()
    return {
        "probe_df": probe_df,
        "summary_df": summary_df,
        "summary_row": summary_row,
    }


def run_lambda_sasa_sweep(
    config: SASAExperimentConfig,
    train_split_df: pd.DataFrame,
    val_split_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_rsa_lookup: dict[str, torch.Tensor | None],
    test_rsa_lookup: dict[str, torch.Tensor | None],
    positives_splitb_csv: Path,
    model_dir: Path,
    results_dir: Path,
    device: str,
    model_name: str = HF_MODEL_NAME,
) -> pd.DataFrame:
    tokenizer = build_tokenizer(model_name)
    train_loader = build_dataloader(
        train_split_df,
        train_rsa_lookup,
        tokenizer,
        batch_size=config.batch_size,
        shuffle=True,
        add_special_tokens=config.add_special_tokens,
    )
    val_loader = build_dataloader(
        val_split_df,
        train_rsa_lookup,
        tokenizer,
        batch_size=config.batch_size,
        shuffle=False,
        add_special_tokens=config.add_special_tokens,
    )
    test_loader = build_dataloader(
        test_df,
        test_rsa_lookup,
        tokenizer,
        batch_size=config.batch_size,
        shuffle=False,
        add_special_tokens=config.add_special_tokens,
    )

    summary_rows = []
    classification_dir = results_dir / "classification"
    probe_rows_dir = results_dir / "probing" / "rows"
    classification_dir.mkdir(parents=True, exist_ok=True)
    probe_rows_dir.mkdir(parents=True, exist_ok=True)

    for lambda_sasa in config.lambda_values:
        suffix = format_lambda_suffix(lambda_sasa)
        checkpoint_path = model_dir / f"baseline_frozen_esm2_lambda_sasa_{suffix}.pt"
        metrics_path = classification_dir / f"baseline_lambda_sasa_{suffix}_metrics.json"
        probe_rows_path = probe_rows_dir / f"baseline_lambda_sasa_{suffix}_probing_rows.csv"

        training_payload = train_single_lambda_run(
            lambda_sasa=lambda_sasa,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
        )

        model, checkpoint = load_baseline_checkpoint(
            checkpoint_path,
            device,
            model_name=model_name,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([training_payload["protein_pos_weight"]], dtype=torch.float32).to(device)
        )
        val_summary, val_predictions_df = predict(
            model,
            val_loader,
            device,
            criterion=criterion,
            lambda_cls=config.lambda_cls,
            lambda_sasa=lambda_sasa,
            threshold=config.threshold,
        )
        test_summary, test_predictions_df = predict(
            model,
            test_loader,
            device,
            criterion=criterion,
            lambda_cls=config.lambda_cls,
            lambda_sasa=lambda_sasa,
            threshold=config.threshold,
        )
        assert val_summary is not None and test_summary is not None
        val_metrics = compute_metrics(val_predictions_df, threshold=config.threshold)
        test_metrics = compute_metrics(test_predictions_df, threshold=config.threshold)
        residue_payload = evaluate_residue_localization(
            checkpoint_path=checkpoint_path,
            device=device,
            positives_csv=positives_splitb_csv,
            probe_rows_path=probe_rows_path,
            model_name=model_name,
            threshold=config.threshold,
        )
        residue_row = residue_payload["summary_row"]

        metrics_payload = {
            "lambda_cls": config.lambda_cls,
            "lambda_sasa": lambda_sasa,
            "tokenizer_add_special_tokens": config.add_special_tokens,
            "best_epoch": training_payload["best_epoch"],
            "training_history": checkpoint["training_history"],
            "validation_metrics": val_metrics,
            "validation_loss": val_summary,
            "test_metrics": test_metrics,
            "test_loss": test_summary,
            "residue_summary": residue_row,
            "probe_rows_path": str(probe_rows_path),
        }
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, indent=2)

        summary_rows.append(
            {
                "lambda_sasa": lambda_sasa,
                "best_epoch": training_payload["best_epoch"],
                "val_auroc": val_metrics["auroc"],
                "test_auroc": test_metrics["auroc"],
                "test_f1": test_metrics["f1"],
                "test_mcc": test_metrics["mcc"],
                "residue_auroc": residue_row["auroc_mean"],
                "residue_auprc": residue_row["auprc_mean"],
                "residue_precision_at_k": residue_row["precision_at_k_mean"],
                "checkpoint_path": str(checkpoint_path),
                "metrics_path": str(metrics_path),
                "probe_rows_path": str(probe_rows_path),
            }
        )

    return pd.DataFrame(summary_rows)
