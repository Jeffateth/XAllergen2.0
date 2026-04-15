from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, EsmModel


RANDOM_STATE = 42
ESM_MODEL_NAME = "esm2_t6_8M_UR50D"
HF_MODEL_NAME = f"facebook/{ESM_MODEL_NAME}"
HIDDEN_DIM = 128
DROPOUT = 0.3
THRESHOLD = 0.5
IG_STEPS = 50
MAX_SEQ_LEN = 1022


def configure_matplotlib_cache(cwd: Path) -> None:
    mplconfigdir = cwd.resolve() / ".matplotlib"
    mplconfigdir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfigdir))


def resolve_hf_model_source(model_name: str = HF_MODEL_NAME) -> str:
    override = os.environ.get("XALLERGEN_HF_MODEL_DIR")
    if override:
        return override

    repo_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        for snapshot in sorted(snapshots_dir.iterdir(), reverse=True):
            if snapshot.is_dir() and any(
                list(snapshot.glob(pat))
                for pat in ("*.safetensors", "*.bin")
            ):
                return str(snapshot)

    return model_name


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root. Make sure VSCode is opened from inside the XAllergen2.0 folder."
    )


def seed_everything(seed: int = RANDOM_STATE) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_runtime_context(device: str, project_root: Path) -> None:
    run_target = os.environ.get("XALLERGEN_RUN_TARGET", "local")
    print(f"RUN_TARGET: {run_target}")
    print(f"Device: {device}")
    print(f"Project root: {project_root}")
    if device == "cuda":
        print("GPU configuration:")
        print(f"  backend: CUDA")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"  Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    elif device == "mps":
        print("GPU configuration:")
        print("  backend: Apple Metal Performance Shaders (MPS)")
        print(f"  built with MPS: {torch.backends.mps.is_built()}")
        print(f"  MPS available: {torch.backends.mps.is_available()}")
    else:
        print("GPU configuration:")
        print("  No MPS accelerator detected. Running on CPU.")


def build_tokenizer(model_name: str = HF_MODEL_NAME):
    return AutoTokenizer.from_pretrained(resolve_hf_model_source(model_name))


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(
        self, residue_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Expand mask to match batch dimension (handles IG's 50-step batching)
        mask = attention_mask.bool().expand(residue_embeddings.shape[0], -1)
        scores = self.score(residue_embeddings).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * residue_embeddings, dim=1)
        return pooled, weights


class FrozenESMAllergenClassifier(nn.Module):
    def __init__(self, model_name: str, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.backbone = EsmModel.from_pretrained(resolve_hf_model_source(model_name))
        for param in self.backbone.parameters():
            param.requires_grad = False
        embed_dim = self.backbone.config.hidden_size
        self.attention_pool = AttentionPooling(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward_from_residue_embeddings(
        self, residue_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict:
        pooled, attention_weights = self.attention_pool(residue_embeddings, attention_mask)
        logits = self.classifier(pooled).squeeze(-1)
        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "residue_embeddings": residue_embeddings,
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        with torch.no_grad():
            residue_embeddings = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        return self.forward_from_residue_embeddings(residue_embeddings, attention_mask)

    def forward_from_inputs_embeds(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict:
        # ESM-2 token_dropout uses input_ids to locate mask tokens; when we
        # pass inputs_embeds directly (Captum IG path), input_ids is None so
        # (None == mask_token_id) → Python False → False.unsqueeze(-1) →
        # AttributeError. Token dropout is irrelevant here (we already have
        # the embeddings), so disable it for this call.
        token_dropout_backup = self.backbone.embeddings.token_dropout
        self.backbone.embeddings.token_dropout = False
        try:
            residue_embeddings = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            ).last_hidden_state
        finally:
            self.backbone.embeddings.token_dropout = token_dropout_backup
        return self.forward_from_residue_embeddings(residue_embeddings, attention_mask)


class FrozenESMAllergenMTLClassifier(FrozenESMAllergenClassifier):
    def __init__(
        self,
        model_name: str,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        epitope_hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__(model_name, hidden_dim=hidden_dim, dropout=dropout)
        embed_dim = self.backbone.config.hidden_size
        self.epitope_head = nn.Sequential(
            nn.Linear(embed_dim, epitope_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(epitope_hidden_dim, 1),
        )

    def forward_from_residue_embeddings(
        self, residue_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict:
        outputs = super().forward_from_residue_embeddings(residue_embeddings, attention_mask)
        residue_logits = self.epitope_head(residue_embeddings).squeeze(-1)
        residue_mask = attention_mask.bool().expand(residue_logits.shape[0], -1)
        residue_logits = residue_logits.masked_fill(~residue_mask, torch.finfo(residue_logits.dtype).min)
        outputs["residue_logits"] = residue_logits
        outputs["residue_probs"] = torch.sigmoid(residue_logits) * residue_mask
        return outputs


def tokenize_sequence(tokenizer, sequence: str, device: str) -> dict:
    encodings = tokenizer(
        sequence,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )
    return {
        "input_ids": encodings["input_ids"].to(device),
        "attention_mask": encodings["attention_mask"].to(device),
    }


def load_baseline_checkpoint(
    checkpoint_path: Path,
    device: str,
    model_name: str = HF_MODEL_NAME,
    hidden_dim: int = HIDDEN_DIM,
    dropout: float = DROPOUT,
) -> tuple[FrozenESMAllergenClassifier, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get(
        "architecture_hyperparameters",
        {"hidden_dim": hidden_dim, "dropout": dropout},
    )
    model = FrozenESMAllergenClassifier(
        model_name,
        hidden_dim=architecture.get("hidden_dim", hidden_dim),
        dropout=architecture.get("dropout", dropout),
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatible.missing_keys:
        non_positional = [k for k in incompatible.missing_keys if "position_embeddings" not in k]
        if non_positional:
            raise RuntimeError(
                "Unexpected missing keys in checkpoint:\n"
                + "\n".join(f"  {k}" for k in non_positional)
            )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected extra keys in checkpoint:\n"
            + "\n".join(f"  {k}" for k in incompatible.unexpected_keys)
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, checkpoint


def load_mtl_checkpoint(
    checkpoint_path: Path,
    device: str,
    model_name: str = HF_MODEL_NAME,
    hidden_dim: int = HIDDEN_DIM,
    dropout: float = DROPOUT,
    epitope_hidden_dim: int = HIDDEN_DIM,
) -> tuple[FrozenESMAllergenMTLClassifier, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get(
        "architecture_hyperparameters",
        {
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "epitope_hidden_dim": epitope_hidden_dim,
        },
    )
    model = FrozenESMAllergenMTLClassifier(
        model_name,
        hidden_dim=architecture.get("hidden_dim", hidden_dim),
        dropout=architecture.get("dropout", dropout),
        epitope_hidden_dim=architecture.get("epitope_hidden_dim", epitope_hidden_dim),
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatible.missing_keys:
        non_positional = [k for k in incompatible.missing_keys if "position_embeddings" not in k]
        if non_positional:
            raise RuntimeError(
                "Unexpected missing keys in checkpoint:\n"
                + "\n".join(f"  {k}" for k in non_positional)
            )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected extra keys in checkpoint:\n"
            + "\n".join(f"  {k}" for k in incompatible.unexpected_keys)
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, checkpoint


def parse_epitope_label(sequence: str, epitope_start: str, epitope_end: str) -> np.ndarray:
    labels = np.zeros(len(sequence), dtype=np.float32)
    starts = [int(s) for s in str(epitope_start).split(";") if str(s).strip()]
    ends = [int(e) for e in str(epitope_end).split(";") if str(e).strip()]
    if len(starts) != len(ends):
        raise ValueError(
            f"Mismatched interval counts: {len(starts)} starts vs {len(ends)} ends"
        )
    for start, end in zip(starts, ends):
        left = max(start - 1, 0)
        right = min(end, len(sequence))
        if left < right:
            labels[left:right] = 1.0
    return labels


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.maximum(scores, 0.0)
    total = scores.sum()
    return scores / total if total > 0 else scores


def compute_attention_weights(model, tokenizer, sequence: str, device: str) -> np.ndarray:
    encodings = tokenize_sequence(tokenizer, sequence, device)
    with torch.no_grad():
        outputs = model(encodings["input_ids"], encodings["attention_mask"])
    weights = outputs["attention_weights"].squeeze(0).detach().cpu().numpy()
    valid_length = int(encodings["attention_mask"].sum().item())
    return weights[:valid_length]


def compute_residue_probabilities(model, tokenizer, sequence: str, device: str) -> np.ndarray:
    encodings = tokenize_sequence(tokenizer, sequence, device)
    with torch.no_grad():
        outputs = model(encodings["input_ids"], encodings["attention_mask"])
    if "residue_probs" not in outputs:
        raise ValueError("Model does not expose residue_probs. Use FrozenESMAllergenMTLClassifier.")
    residue_probs = outputs["residue_probs"].squeeze(0).detach().cpu().numpy()
    valid_length = int(encodings["attention_mask"].sum().item())
    return residue_probs[:valid_length]


def compute_integrated_gradients(
    model,
    tokenizer,
    sequence: str,
    device: str,
    steps: int = IG_STEPS,
    normalize: bool = False,
) -> np.ndarray:
    from captum.attr import IntegratedGradients

    encodings = tokenize_sequence(tokenizer, sequence, device)
    attention_mask = encodings["attention_mask"]  # shape: (1, seq_len), dtype: long

    input_embeds = model.backbone.get_input_embeddings()(encodings["input_ids"]).detach()
    baseline = torch.zeros_like(input_embeds)

    def ig_forward(inputs_embeds: torch.Tensor) -> torch.Tensor:
        return model.forward_from_inputs_embeds(inputs_embeds, attention_mask)["logits"]

    attributions = IntegratedGradients(ig_forward).attribute(
        inputs=input_embeds,
        baselines=baseline,
        n_steps=steps,
    )
    importance = attributions.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
    valid_length = int(attention_mask.sum().item())
    importance = importance[:valid_length]
    return normalize_scores(importance) if normalize else importance

def mean_metric_dicts(metric_rows: list[dict]) -> dict:
    return {
        "auroc": float(np.nanmean([row["auroc"] for row in metric_rows])),
        "auprc": float(np.mean([row["auprc"] for row in metric_rows])),
        "precision_at_k": float(np.mean([row["precision_at_k"] for row in metric_rows])),
    }
