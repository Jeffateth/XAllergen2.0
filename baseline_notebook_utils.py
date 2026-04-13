from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from captum.attr import IntegratedGradients
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
    ref_path = repo_dir / "refs" / "main"
    if ref_path.exists():
        revision = ref_path.read_text().strip()
        snapshot_dir = repo_dir / "snapshots" / revision
        if snapshot_dir.exists():
            return str(snapshot_dir)

    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])

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
    return "mps" if torch.backends.mps.is_available() else "cpu"


def print_runtime_context(device: str, project_root: Path) -> None:
    print("RUN_TARGET: local")
    print(f"Device: {device}")
    print(f"Project root: {project_root}")
    if device == "mps":
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
        mask = attention_mask.bool()
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
        residue_embeddings = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state
        return self.forward_from_residue_embeddings(residue_embeddings, attention_mask)


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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint.get(
        "architecture_hyperparameters",
        {"hidden_dim": hidden_dim, "dropout": dropout},
    )
    model = FrozenESMAllergenClassifier(
        model_name,
        hidden_dim=architecture.get("hidden_dim", hidden_dim),
        dropout=architecture.get("dropout", dropout),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, checkpoint


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


def compute_integrated_gradients(
    model,
    tokenizer,
    sequence: str,
    device: str,
    steps: int = IG_STEPS,
    normalize: bool = False,
) -> np.ndarray:
    encodings = tokenize_sequence(tokenizer, sequence, device)
    input_embeds = model.backbone.get_input_embeddings()(encodings["input_ids"]).detach()
    baseline = torch.zeros_like(input_embeds)

    def ig_forward(inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return model.forward_from_inputs_embeds(inputs_embeds, attention_mask)["logits"]

    attributions = IntegratedGradients(ig_forward).attribute(
        inputs=input_embeds,
        baselines=baseline,
        additional_forward_args=(encodings["attention_mask"],),
        n_steps=steps,
    )
    importance = attributions.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
    valid_length = int(encodings["attention_mask"].sum().item())
    importance = importance[:valid_length]
    return normalize_scores(importance) if normalize else importance


def mean_metric_dicts(metric_rows: list[dict]) -> dict:
    return {
        "auroc": float(np.nanmean([row["auroc"] for row in metric_rows])),
        "auprc": float(np.mean([row["auprc"] for row in metric_rows])),
        "precision_at_k": float(np.mean([row["precision_at_k"] for row in metric_rows])),
    }
