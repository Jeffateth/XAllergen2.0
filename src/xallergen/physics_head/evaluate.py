"""Evaluation script for the physics-informed ESM-2 allergenicity classifier.

Evaluates on:
  1. DeepAlgPro test set  — classification metrics (AUROC, F1, MCC, accuracy)
  2. IEDB splitB proteins  — pooling attention AUROC against epitope masks

Called from notebook 19 via ``from xallergen.physics_head.evaluate import main``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader

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
    HF_MODEL_NAME,
    RANDOM_STATE,
    build_tokenizer,
    parse_epitope_label,
    seed_everything,
)
from xallergen.physics_head.features import PhysicsScaler, build_physics_vector
from xallergen.physics_head.model import FrozenESM2WithPhysics, get_weight_summary
from xallergen.physics_head.train import (
    PhysicsDataset,
    _load_netsurfp,
    _load_physics_from_json,
    load_netsurfp_features,
    build_physics_map,
    _make_collate,
)

_DEFAULT_CKPT        = _REPO_ROOT / "results" / "physics_head" / "checkpoints" / "best.pt"
_DEFAULT_SCALER      = _REPO_ROOT / "results" / "physics_head" / "physics_scaler.json"
_DEFAULT_TEST_NETSURFP = _REPO_ROOT / "data" / "rsa" / "deepalgpro_test_netsurfp.csv"
_DEFAULT_IEDB_NETSURFP = _REPO_ROOT / "data" / "ss3" / "iedb_positive_sequences_predictions.csv"
_DEFAULT_TEST_CSV    = _REPO_ROOT / "data" / "deepalgpro_test_cleaned.csv"
_DEFAULT_SPLITB_CSV  = _REPO_ROOT / "data" / "positives_splitB.csv"
_DEFAULT_OUTPUT_DIR  = _REPO_ROOT / "results" / "physics_head"
_RSA_DIR             = _REPO_ROOT / "data" / "rsa"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_iedb_netsurfp(csv_path: Path, sequence_ids: set[str]) -> dict[str, dict]:
    """Load NetSurfP predictions from the IEDB predictions CSV."""
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
            "rsa": grp["rsa"].to_numpy(dtype=np.float32),
            "phi": grp["phi"].to_numpy(dtype=np.float32),
            "psi": grp["psi"].to_numpy(dtype=np.float32),
            "disorder": grp["disorder"].to_numpy(dtype=np.float32),
        }
    return result


# ── Classification evaluation ────────────────────────────────────────────────

@torch.no_grad()
def _evaluate_classification(model, loader, device) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        physics = batch["physics_features"].to(device)
        labels = batch["label"]
        out = model(input_ids, attention_mask, physics)
        all_logits.append(out["logits"].cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "accuracy": float(accuracy_score(labels, preds)),
    }


# ── Epitope alignment (attention AUROC on splitB) ────────────────────────────

@torch.no_grad()
def _evaluate_epitope_alignment(
    model,
    tokenizer,
    splitb_df: pd.DataFrame,
    physics_map: dict[str, np.ndarray],
    device: str,
) -> pd.DataFrame:
    """Compute per-protein attention-weight AUROC against IEDB epitope masks."""
    from sklearn.metrics import roc_auc_score as _auroc

    model.eval()
    rows = []
    for row in splitb_df.itertuples(index=False):
        acc = str(row.accession)
        seq = str(row.sequence).upper()
        label_vec = parse_epitope_label(seq, row.epitope_start, row.epitope_end)
        if label_vec.sum() == 0 or label_vec.sum() == len(seq):
            continue
        if acc not in physics_map:
            continue
        phys = torch.from_numpy(physics_map[acc]).unsqueeze(0).to(device)  # (1, L, 10)
        enc = tokenizer(seq, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = model(input_ids, attention_mask, phys)
        alpha = out["attention_weights"].squeeze(0).cpu().numpy()
        valid_len = int(attention_mask.sum().item())
        alpha = alpha[:valid_len]
        auroc = float(_auroc(label_vec, alpha))
        rows.append({"accession": acc, "auroc": auroc})
    return pd.DataFrame(rows)


# ── Main entry point ─────────────────────────────────────────────────────────

def main(
    checkpoint_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    # JSON files (Colab-compatible, preferred)
    rsa_json_path: Optional[Path] = None,
    disorder_json_path: Optional[Path] = None,
    phi_json_path: Optional[Path] = None,
    # CSV fallback (local only)
    test_netsurfp_path: Optional[Path] = None,
    iedb_netsurfp_path: Optional[Path] = None,
    test_csv_path: Optional[Path] = None,
    splitb_csv_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
    seed: int = RANDOM_STATE,
    batch_size: int = 24,
) -> dict:
    """Evaluate physics head on test set and splitB epitope alignment.

    Returns a metrics dict that is also written to results/physics_head/metrics.json.
    """
    seed_everything(seed)

    ckpt_path = Path(checkpoint_path or _DEFAULT_CKPT)
    scaler_file = Path(scaler_path or _DEFAULT_SCALER)
    test_netsurfp = Path(test_netsurfp_path or _DEFAULT_TEST_NETSURFP)
    iedb_netsurfp = Path(iedb_netsurfp_path or _DEFAULT_IEDB_NETSURFP)
    test_csv = Path(test_csv_path or _DEFAULT_TEST_CSV)
    splitb_csv = Path(splitb_csv_path or _DEFAULT_SPLITB_CSV)
    out_dir = Path(output_dir or _DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Device: {device}")

    # ── Load scaler and checkpoint ────────────────────────────────────────
    scaler = PhysicsScaler.load(scaler_file)
    print(f"Loaded scaler from {scaler_file}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("architecture_hyperparameters", {})

    from xallergen.baseline_notebook_utils import load_baseline_checkpoint
    baseline_ckpt_path = (
        _REPO_ROOT / "results" / "rsa_regularization" / "lambda_0" / "baseline_frozen_esm2.pt"
    )
    baseline, _ = load_baseline_checkpoint(baseline_ckpt_path, device)
    model = FrozenESM2WithPhysics(
        baseline,
        physics_dim=arch.get("physics_dim", 10),
        dropout=arch.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    del baseline
    print(f"Loaded physics head checkpoint from {ckpt_path} (epoch {ckpt.get('epoch')})")

    tokenizer = build_tokenizer(HF_MODEL_NAME)

    # ── Classification on test set ────────────────────────────────────────
    test_df = pd.read_csv(test_csv)
    test_df["sequence_id"] = test_df["sequence_id"].astype(str)
    test_df["sequence"] = test_df["sequence"].astype(str).str.strip().str.upper()
    test_df["label"] = test_df["label"].astype(int)

    test_ids = set(test_df["sequence_id"])
    # Prefer compact JSON files; fall back to full CSV
    rsa_j      = Path(rsa_json_path      or _RSA_DIR / "deepalgpro_test_rsa.json.gz")
    disorder_j = Path(disorder_json_path or _RSA_DIR / "deepalgpro_test_disorder.json.gz")
    phi_j      = Path(phi_json_path      or _RSA_DIR / "deepalgpro_test_phi.json.gz")
    if rsa_j.exists() and disorder_j.exists() and phi_j.exists():
        from xallergen.physics_head.train import _load_physics_from_json
        test_netsurfp_data = _load_physics_from_json(rsa_j, disorder_j, phi_j, test_ids)
    else:
        test_netsurfp_data = _load_netsurfp(test_netsurfp, test_ids)
    test_phys = build_physics_map(test_df, test_netsurfp_data, scaler)
    print(f"Test physics vectors: {len(test_phys)}/{len(test_df)}")

    test_loader = DataLoader(
        PhysicsDataset(test_df, test_phys),
        batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=_make_collate(tokenizer),
    )
    cls_metrics = _evaluate_classification(model, test_loader, device)
    print("Classification metrics:")
    for k, v in cls_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Epitope alignment on splitB ───────────────────────────────────────
    splitb_df = pd.read_csv(splitb_csv)
    splitb_df["accession"] = splitb_df["accession"].astype(str).str.strip()
    splitb_df["sequence"] = splitb_df["sequence"].astype(str).str.strip().str.upper()

    splitb_ids = set(splitb_df["accession"])
    iedb_netsurfp_data = _load_iedb_netsurfp(iedb_netsurfp, splitb_ids)
    splitb_phys: dict[str, np.ndarray] = {}
    for row in splitb_df.itertuples(index=False):
        acc = str(row.accession)
        seq = str(row.sequence).upper()
        if acc not in iedb_netsurfp_data:
            continue
        ns = iedb_netsurfp_data[acc]
        if len(seq) != len(ns["rsa"]):
            continue
        vec = build_physics_vector(seq, ns["rsa"], ns["disorder"], ns["phi"], ns["psi"])
        splitb_phys[acc] = scaler.transform(vec)
    print(f"SplitB physics vectors: {len(splitb_phys)}/{len(splitb_df)}")

    alignment_df = _evaluate_epitope_alignment(model, tokenizer, splitb_df, splitb_phys, device)
    print(f"Epitope alignment computed for {len(alignment_df)} splitB proteins")

    mean_auroc = float(alignment_df["auroc"].mean())
    mean_coil_auroc = 1.0 - mean_auroc
    if len(alignment_df) >= 2:
        _, wilcoxon_p = wilcoxon(
            alignment_df["auroc"].values,
            1.0 - alignment_df["auroc"].values,
            alternative="greater",
        )
    else:
        wilcoxon_p = float("nan")

    alignment_metrics = {
        "n_proteins": int(len(alignment_df)),
        "mean_attention_auroc": mean_auroc,
        "mean_non_structured_auroc": mean_coil_auroc,
        "wilcoxon_p": float(wilcoxon_p),
    }
    print("Epitope alignment metrics:")
    for k, v in alignment_metrics.items():
        print(f"  {k}: {v}")

    # ── Learned weight summary ────────────────────────────────────────────
    weight_df = get_weight_summary(model)
    weight_path = out_dir / "weight_summary.csv"
    weight_df.to_csv(weight_path, index=False)
    print(f"Weight summary saved to {weight_path}")

    # ── Save all metrics ──────────────────────────────────────────────────
    metrics = {
        "classification": cls_metrics,
        "epitope_alignment": alignment_metrics,
        "per_protein_alignment": alignment_df.to_dict(orient="records"),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")

    return metrics


if __name__ == "__main__":
    main()
