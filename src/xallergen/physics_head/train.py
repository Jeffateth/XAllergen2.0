"""Training script for the physics-informed ESM-2 allergenicity classifier.

Called from notebook 19 via ``from xallergen.physics_head.train import main``.
All hyperparameters match the baseline (notebook 03) so results are comparable.

Data loading strategy (Colab-compatible)
-----------------------------------------
Physics features are loaded in order of preference:
1. Compact JSON.gz files (RSA, disorder, phi) — small enough to commit to git and
   download on Colab.  Located under data/rsa/.
2. Full NetSurfP CSV (netsurfp_csv_path) — local fallback, not committed (551 MB).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ── Repo-root discovery ──────────────────────────────────────────────────────
def _find_repo_root(start: Path) -> Path:
    for p in [start.resolve(), *start.resolve().parents]:
        if (p / "data").exists() and (p / "src").exists():
            return p
    raise FileNotFoundError("Cannot locate repository root from " + str(start))


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from xallergen.baseline_notebook_utils import (
    DROPOUT,
    HF_MODEL_NAME,
    RANDOM_STATE,
    build_tokenizer,
    load_baseline_checkpoint,
    seed_everything,
)
from xallergen.physics_head.features import PhysicsScaler, build_physics_vector
from xallergen.physics_head.model import FrozenESM2WithPhysics

# ── Defaults ─────────────────────────────────────────────────────────────────
_DEFAULT_BASELINE = (
    _REPO_ROOT / "results" / "rsa_regularization" / "lambda_0" / "baseline_frozen_esm2.pt"
)
_DEFAULT_TRAIN_NETSURFP = _REPO_ROOT / "data" / "rsa" / "deepalgpro_train_netsurfp.csv"
_DEFAULT_TRAIN_CSV = _REPO_ROOT / "data" / "deepalgpro_train_cleaned.csv"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results" / "physics_head"
_RSA_DIR = _REPO_ROOT / "data" / "rsa"


# ── Data loading ─────────────────────────────────────────────────────────────

def _read_json_gz(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _load_physics_from_json(
    rsa_path: Path,
    disorder_path: Path,
    phi_path: Path,
    sequence_ids: set[str],
) -> dict[str, dict]:
    """Load per-residue features from compact JSON.gz files (Colab-compatible).

    psi is set to zeros because no channel in build_physics_vector uses it.
    """
    rsa_d      = _read_json_gz(rsa_path)
    disorder_d = _read_json_gz(disorder_path)
    phi_d      = _read_json_gz(phi_path)
    result: dict[str, dict] = {}
    for sid in sequence_ids:
        if sid in rsa_d and sid in disorder_d and sid in phi_d:
            L = len(rsa_d[sid])
            result[sid] = {
                "rsa":      np.array(rsa_d[sid],      dtype=np.float32),
                "disorder": np.array(disorder_d[sid], dtype=np.float32),
                "phi":      np.array(phi_d[sid],      dtype=np.float32),
                "psi":      np.zeros(L,               dtype=np.float32),
            }
    return result


def _load_netsurfp(csv_path: Path, sequence_ids: set[str]) -> dict[str, dict]:
    """Load from full NetSurfP CSV (local fallback, not committed to git)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    id_col = next(c for c in df.columns if c.strip().lower() in ("id", "sequence_id"))
    df[id_col] = df[id_col].astype(str).str.strip().str.removeprefix(">")
    n_col = next(c for c in df.columns if c.strip().lower() == "n")

    result: dict[str, dict] = {}
    for seq_id, grp in df.groupby(id_col, sort=False):
        seq_id = str(seq_id).strip()
        if seq_id not in sequence_ids:
            continue
        grp = grp.sort_values(n_col)
        result[seq_id] = {
            "rsa":      grp["rsa"].to_numpy(dtype=np.float32),
            "phi":      grp["phi"].to_numpy(dtype=np.float32),
            "psi":      grp["psi"].to_numpy(dtype=np.float32),
            "disorder": grp["disorder"].to_numpy(dtype=np.float32),
        }
    return result


def load_netsurfp_features(
    sequence_ids: set[str],
    rsa_json:      Optional[Path] = None,
    disorder_json: Optional[Path] = None,
    phi_json:      Optional[Path] = None,
    netsurfp_csv:  Optional[Path] = None,
) -> dict[str, dict]:
    """Load physics features, preferring JSON files (Colab) over CSV (local).

    Raises FileNotFoundError if neither source is available.
    """
    rsa_j      = Path(rsa_json      or _RSA_DIR / "deepalgpro_train_rsa.json.gz")
    disorder_j = Path(disorder_json or _RSA_DIR / "deepalgpro_train_disorder.json.gz")
    phi_j      = Path(phi_json      or _RSA_DIR / "deepalgpro_train_phi.json.gz")
    csv_p      = Path(netsurfp_csv  or _DEFAULT_TRAIN_NETSURFP)

    if rsa_j.exists() and disorder_j.exists() and phi_j.exists():
        print(f"Loading features from JSON files (Colab-compatible path)")
        return _load_physics_from_json(rsa_j, disorder_j, phi_j, sequence_ids)
    if csv_p.exists():
        print(f"Loading features from CSV (local fallback): {csv_p}")
        return _load_netsurfp(csv_p, sequence_ids)
    raise FileNotFoundError(
        "Physics features not found. Expected either:\n"
        f"  JSON files: {rsa_j}, {disorder_j}, {phi_j}\n"
        f"  or CSV: {csv_p}"
    )


def build_physics_map(
    frame: pd.DataFrame,
    netsurfp: dict[str, dict],
    scaler: Optional[PhysicsScaler] = None,
) -> dict[str, np.ndarray]:
    """Return {seq_id: (L, 10) scaled physics vector} for sequences that have NetSurfP data."""
    physics_map: dict[str, np.ndarray] = {}
    for row in frame.itertuples(index=False):
        sid = str(row.sequence_id)
        if sid not in netsurfp:
            continue
        ns = netsurfp[sid]
        seq = str(row.sequence).upper()
        if len(seq) != len(ns["rsa"]):
            continue
        vec = build_physics_vector(seq, ns["rsa"], ns["disorder"], ns["phi"], ns["psi"])
        if scaler is not None:
            vec = scaler.transform(vec)
        physics_map[sid] = vec
    return physics_map


# ── Dataset and collation ────────────────────────────────────────────────────

class PhysicsDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, physics_map: dict[str, np.ndarray]) -> None:
        # Keep only sequences that have physics features.
        mask = frame["sequence_id"].astype(str).isin(physics_map)
        self.frame = frame[mask].reset_index(drop=True)
        self.physics_map = physics_map

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        return {
            "sequence_id": str(row["sequence_id"]),
            "sequence": str(row["sequence"]),
            "label": int(row["label"]),
            "physics": self.physics_map[str(row["sequence_id"])],  # (L, 10)
        }


def _make_collate(tokenizer):
    def collate_fn(batch: list[dict]) -> dict:
        sequences = [item["sequence"] for item in batch]
        enc = tokenizer(
            sequences,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        B = len(sequences)
        max_L = enc["input_ids"].shape[1]
        phys_tensor = torch.zeros(B, max_L, 10, dtype=torch.float32)
        for i, item in enumerate(batch):
            p = item["physics"]  # (L, 10)
            L = p.shape[0]
            phys_tensor[i, :L] = torch.from_numpy(p)
        return {
            "sequence_id": [item["sequence_id"] for item in batch],
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "physics_features": phys_tensor,
        }
    return collate_fn


# ── Training loop ────────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        physics = batch["physics_features"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids, attention_mask, physics)
        loss = criterion(out["logits"], labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.shape[0]
        n += labels.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_logits, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        physics = batch["physics_features"].to(device)
        labels = batch["label"].to(device)
        out = model(input_ids, attention_mask, physics)
        loss = criterion(out["logits"], labels)
        total_loss += float(loss.item()) * labels.shape[0]
        n += labels.shape[0]
        all_logits.append(out["logits"].cpu())
        all_labels.append(labels.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    from sklearn.metrics import f1_score, roc_auc_score
    probs = torch.sigmoid(logits).numpy()
    lbl = labels.numpy()
    auroc = float(roc_auc_score(lbl, probs)) if len(np.unique(lbl)) == 2 else float("nan")
    preds = (probs >= 0.5).astype(int)
    f1 = float(f1_score(lbl, preds, zero_division=0))
    return total_loss / max(n, 1), auroc, f1


# ── Main entry point ─────────────────────────────────────────────────────────

def main(
    baseline_checkpoint_path: Optional[Path] = None,
    netsurfp_csv_path: Optional[Path] = None,
    rsa_json_path: Optional[Path] = None,
    disorder_json_path: Optional[Path] = None,
    phi_json_path: Optional[Path] = None,
    train_csv_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    epochs: int = 30,
    patience: int = 5,
    min_delta: float = 1e-3,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 24,
    val_fraction: float = 0.1,
    seed: int = RANDOM_STATE,
    device: Optional[str] = None,
) -> dict:
    """Train FrozenESM2WithPhysics and save checkpoint + history.

    Returns the training history list (one dict per epoch).
    Physics features are loaded from compact JSON.gz files when available
    (Colab-compatible), falling back to the full NetSurfP CSV locally.
    """
    seed_everything(seed)

    # ── paths ─────────────────────────────────────────────────────────────
    baseline_ckpt = Path(baseline_checkpoint_path or _DEFAULT_BASELINE)
    train_csv = Path(train_csv_path or _DEFAULT_TRAIN_CSV)
    out_dir = Path(output_dir or _DEFAULT_OUTPUT_DIR)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Device: {device}")

    # ── data ──────────────────────────────────────────────────────────────
    train_df = pd.read_csv(train_csv)
    train_df["sequence_id"] = train_df["sequence_id"].astype(str)
    train_df["sequence"] = train_df["sequence"].astype(str).str.strip().str.upper()
    train_df["label"] = train_df["label"].astype(int)

    train_split, val_split = train_test_split(
        train_df, test_size=val_fraction, random_state=seed, stratify=train_df["label"]
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    print(f"Train: {len(train_split)}, Val: {len(val_split)}")

    # ── physics features + scaler ─────────────────────────────────────────
    all_ids = set(train_df["sequence_id"].astype(str))
    netsurfp = load_netsurfp_features(
        all_ids,
        rsa_json=rsa_json_path,
        disorder_json=disorder_json_path,
        phi_json=phi_json_path,
        netsurfp_csv=netsurfp_csv_path,
    )
    print(f"Physics features loaded for {len(netsurfp)}/{len(all_ids)} sequences")

    # Fit scaler on training residues only
    raw_vecs = []
    for row in train_split.itertuples(index=False):
        sid = str(row.sequence_id)
        if sid in netsurfp:
            ns = netsurfp[sid]
            seq = str(row.sequence).upper()
            if len(seq) == len(ns["rsa"]):
                raw_vecs.append(
                    build_physics_vector(seq, ns["rsa"], ns["disorder"], ns["phi"], ns["psi"])
                )
    all_residues = np.concatenate(raw_vecs, axis=0)  # (N_total, 10)
    print(f"Fitting scaler on {all_residues.shape[0]:,} training residues")

    scaler = PhysicsScaler().fit(all_residues)
    scaler_path = out_dir / "physics_scaler.json"
    scaler.save(scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Build scaled physics maps for train and val
    train_phys = build_physics_map(train_split, netsurfp, scaler)
    val_phys = build_physics_map(val_split, netsurfp, scaler)
    print(f"Physics vectors: train={len(train_phys)}, val={len(val_phys)}")

    # ── model ─────────────────────────────────────────────────────────────
    baseline, _ = load_baseline_checkpoint(baseline_ckpt, device)
    model = FrozenESM2WithPhysics(baseline).to(device)
    del baseline

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: total={n_total:,}, trainable={n_trainable:,}")

    # ── dataloaders ────────────────────────────────────────────────────────
    tokenizer = build_tokenizer(HF_MODEL_NAME)
    collate_fn = _make_collate(tokenizer)

    train_loader = DataLoader(
        PhysicsDataset(train_split, train_phys),
        batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        PhysicsDataset(val_split, val_phys),
        batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    # ── optimizer and loss ────────────────────────────────────────────────
    counts = train_split["label"].value_counts()
    pos_weight = torch.tensor(
        [float(counts.get(0, 1)) / float(counts.get(1, 1))],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

    # ── training loop with early stopping ────────────────────────────────
    history: list[dict] = []
    best_val_loss = float("inf")
    no_improve = 0
    best_ckpt_path = ckpt_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auroc, val_f1 = _eval_epoch(model, val_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "val_f1": val_f1,
        }
        history.append(record)
        print(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_auroc={val_auroc:.4f} val_f1={val_f1:.4f}"
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_auroc": val_auroc,
                    "architecture_hyperparameters": {
                        "hidden_dim": model.classifier[0].out_features,
                        "dropout": DROPOUT,
                        "physics_dim": 10,
                    },
                    "training_history": history,
                },
                best_ckpt_path,
            )
            print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # ── save history ──────────────────────────────────────────────────────
    history_path = out_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Training history saved to {history_path}")

    return history


if __name__ == "__main__":
    main()
