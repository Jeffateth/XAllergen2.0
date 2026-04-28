from __future__ import annotations

import contextlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from tqdm.auto import tqdm
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


def cudnn_backward_safe_context(device: str):
    if str(device).startswith("cuda"):
        return torch.backends.cudnn.flags(enabled=False)
    return contextlib.nullcontext()


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


def _find_final_backbone_encoder_block(backbone: nn.Module) -> tuple[str, nn.Module]:
    encoder = getattr(backbone, "encoder", None)
    if encoder is not None:
        for layer_attr in ("layer", "layers"):
            layer_container = getattr(encoder, layer_attr, None)
            if isinstance(layer_container, nn.ModuleList) and len(layer_container) > 0:
                return f"encoder.{layer_attr}.{len(layer_container) - 1}", layer_container[-1]

    candidate_names = [
        name
        for name, module in backbone.named_modules()
        if isinstance(module, nn.ModuleList)
        and len(module) > 0
        and ("encoder" in name)
        and name.endswith(("layer", "layers"))
    ]
    for name in sorted(candidate_names):
        module_list = dict(backbone.named_modules())[name]
        return f"{name}.{len(module_list) - 1}", module_list[-1]

    raise RuntimeError(
        "Could not locate the final ESM encoder block inside model.backbone. "
        "Expected an encoder layer ModuleList such as encoder.layer or encoder.layers."
    )


def _final_block_parameter_names(backbone: nn.Module, final_block_name: str) -> set[str]:
    prefix = f"{final_block_name}."
    return {
        name
        for name, _ in backbone.named_parameters()
        if name == final_block_name or name.startswith(prefix)
    }


def _compact_backbone_prefix(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "encoder" and parts[1] in {"layer", "layers"}:
        return ".".join(parts[:3])
    if len(parts) >= 4 and parts[0] == "esm" and parts[1] == "encoder" and parts[2] in {"layer", "layers"}:
        return ".".join(parts[:4])
    return ".".join(parts[: min(len(parts), 3)])


def configure_backbone_trainability(
    model: nn.Module,
    backbone_train_mode: str,
) -> dict[str, object]:
    if not hasattr(model, "backbone"):
        raise AttributeError("Model does not expose a `backbone` module.")

    backbone = model.backbone
    for param in backbone.parameters():
        param.requires_grad = False

    final_block_name = None
    if backbone_train_mode == "top1_unfrozen":
        final_block_name, final_block = _find_final_backbone_encoder_block(backbone)
        for param in final_block.parameters():
            param.requires_grad = True
    elif backbone_train_mode != "frozen":
        raise ValueError(
            f"Unsupported backbone_train_mode={backbone_train_mode!r}. "
            "Supported modes: 'frozen', 'top1_unfrozen'."
        )

    backbone_named_params = list(backbone.named_parameters())
    total_backbone_params = sum(param.numel() for _, param in backbone_named_params)
    trainable_backbone = [
        (name, param) for name, param in backbone_named_params if param.requires_grad
    ]
    trainable_backbone_params = sum(param.numel() for _, param in trainable_backbone)
    trainable_pct = (
        100.0 * trainable_backbone_params / total_backbone_params if total_backbone_params else 0.0
    )
    trainable_prefixes = sorted(
        {f"model.backbone.{_compact_backbone_prefix(name)}" for name, _ in trainable_backbone}
    )

    print("Backbone trainability summary:")
    print(f"  mode: {backbone_train_mode}")
    print(f"  total backbone params: {total_backbone_params:,}")
    print(f"  trainable backbone params: {trainable_backbone_params:,}")
    print(f"  percent trainable backbone: {trainable_pct:.2f}%")
    print("  trainable backbone submodules:")
    if trainable_prefixes:
        for prefix in trainable_prefixes:
            print(f"    {prefix}")
    else:
        print("    <none>")
    if final_block_name is not None:
        print(f"  detected final encoder block: model.backbone.{final_block_name}")

    return {
        "mode": backbone_train_mode,
        "final_block_name": final_block_name,
        "final_block_path": (
            f"model.backbone.{final_block_name}" if final_block_name is not None else None
        ),
        "total_backbone_params": total_backbone_params,
        "trainable_backbone_params": trainable_backbone_params,
        "trainable_backbone_pct": trainable_pct,
        "trainable_backbone_prefixes": trainable_prefixes,
    }


def assert_backbone_trainability_mode(
    model: nn.Module,
    backbone_train_mode: str,
) -> dict[str, object]:
    if not hasattr(model, "backbone"):
        raise AttributeError("Model does not expose a `backbone` module.")

    backbone = model.backbone
    trainable_names = {
        name for name, param in backbone.named_parameters() if param.requires_grad
    }

    if backbone_train_mode == "frozen":
        assert not trainable_names, "Expected all backbone parameters to remain frozen."
        return {"final_block_path": None, "trainable_parameter_names": sorted(trainable_names)}

    if backbone_train_mode != "top1_unfrozen":
        raise ValueError(
            f"Unsupported backbone_train_mode={backbone_train_mode!r}. "
            "Supported modes: 'frozen', 'top1_unfrozen'."
        )

    final_block_name, _ = _find_final_backbone_encoder_block(backbone)
    expected_trainable_names = _final_block_parameter_names(backbone, final_block_name)
    all_backbone_names = {name for name, _ in backbone.named_parameters()}

    assert trainable_names, "Expected some backbone parameters to be trainable in top1_unfrozen mode."
    assert trainable_names != all_backbone_names, (
        "Expected only part of the backbone to be trainable in top1_unfrozen mode."
    )
    assert trainable_names == expected_trainable_names, (
        "Expected only the final encoder block to be trainable within the backbone."
    )
    return {
        "final_block_path": f"model.backbone.{final_block_name}",
        "trainable_parameter_names": sorted(trainable_names),
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


def initialize_mtl_from_baseline_checkpoint(
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
        {"hidden_dim": hidden_dim, "dropout": dropout},
    )
    model = FrozenESMAllergenMTLClassifier(
        model_name,
        hidden_dim=architecture.get("hidden_dim", hidden_dim),
        dropout=architecture.get("dropout", dropout),
        epitope_hidden_dim=epitope_hidden_dim,
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    missing_keys = [k for k in incompatible.missing_keys if "position_embeddings" not in k]
    epitope_head_missing_keys = [k for k in missing_keys if k.startswith("epitope_head.")]
    unexpected_missing_keys = [k for k in missing_keys if not k.startswith("epitope_head.")]
    if unexpected_missing_keys:
        raise RuntimeError(
            "Unexpected missing keys when initializing MTL model from baseline checkpoint:\n"
            + "\n".join(f"  {k}" for k in unexpected_missing_keys)
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected extra keys when initializing MTL model from baseline checkpoint:\n"
            + "\n".join(f"  {k}" for k in incompatible.unexpected_keys)
        )

    loaded_keys = sorted(
        key for key in checkpoint["model_state_dict"].keys() if key in model.state_dict()
    )
    print(f"Loaded baseline checkpoint: {checkpoint_path}")
    print("Loaded shared keys:")
    for key in loaded_keys:
        print(f"  {key}")
    print("Missing keys:")
    if epitope_head_missing_keys:
        for key in epitope_head_missing_keys:
            print(f"  {key}")
    else:
        print("  <none>")
    print("Unexpected keys:")
    if incompatible.unexpected_keys:
        for key in incompatible.unexpected_keys:
            print(f"  {key}")
    else:
        print("  <none>")
    if epitope_head_missing_keys:
        print("Epitope head remains newly initialized:")
        for key in epitope_head_missing_keys:
            print(f"  {key}")

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
    internal_batch_size: int | None = 1,
) -> np.ndarray:
    from captum.attr import IntegratedGradients

    encodings = tokenize_sequence(tokenizer, sequence, device)
    attention_mask = encodings["attention_mask"]  # shape: (1, seq_len), dtype: long

    input_embeds = model.backbone.get_input_embeddings()(encodings["input_ids"]).detach()
    baseline = torch.zeros_like(input_embeds)

    def ig_forward(inputs_embeds: torch.Tensor) -> torch.Tensor:
        return model.forward_from_inputs_embeds(inputs_embeds, attention_mask)["logits"]

    with cudnn_backward_safe_context(device):
        attributions = IntegratedGradients(ig_forward).attribute(
            inputs=input_embeds,
            baselines=baseline,
            n_steps=steps,
            internal_batch_size=internal_batch_size,
        )
    importance = attributions.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
    valid_length = int(attention_mask.sum().item())
    importance = importance[:valid_length]
    return normalize_scores(importance) if normalize else importance


def compute_gradient_x_input_scores(
    model,
    tokenizer,
    sequence: str,
    device: str,
) -> np.ndarray:
    """
    Gradient x Input attribution against the protein-level classification logit.
    Returns one absolute embedding-dot-gradient score per residue.
    """
    model.eval()
    encodings = tokenize_sequence(tokenizer, sequence, device)
    attention_mask = encodings["attention_mask"]

    model.zero_grad(set_to_none=True)
    input_embeds = model.backbone.get_input_embeddings()(encodings["input_ids"]).detach()
    input_embeds.requires_grad_(True)

    with cudnn_backward_safe_context(device):
        logits = model.forward_from_inputs_embeds(input_embeds, attention_mask)["logits"]
        gradients = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=input_embeds,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
        )[0]

    scores = (input_embeds * gradients).sum(dim=-1).abs().squeeze(0)
    valid_length = int(attention_mask.sum().item())
    scores = scores[:valid_length].detach().cpu().numpy().astype(np.float32)
    model.zero_grad(set_to_none=True)
    return scores


def compute_smoothgrad_ig_scores(
    model,
    tokenizer,
    sequence: str,
    device: str,
    steps: int,
    n_samples: int = 10,
    noise_std: float = 0.05,
    internal_batch_size: int = 10,
) -> np.ndarray:
    """
    SmoothGrad over Integrated Gradients in embedding space.
    Noise is added to ESM input embeddings; tokens and labels are unchanged.
    """
    from captum.attr import IntegratedGradients

    if n_samples <= 0:
        raise ValueError("n_samples must be positive for SmoothGrad-IG.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative for SmoothGrad-IG.")

    model.eval()
    encodings = tokenize_sequence(tokenizer, sequence, device)
    attention_mask = encodings["attention_mask"]
    base_embeds = model.backbone.get_input_embeddings()(encodings["input_ids"]).detach()
    baseline = torch.zeros_like(base_embeds)
    noise_scale = float(noise_std) * float(base_embeds.detach().std().item())
    total_scores = np.zeros(int(attention_mask.sum().item()), dtype=np.float64)

    def ig_forward(inputs_embeds: torch.Tensor) -> torch.Tensor:
        return model.forward_from_inputs_embeds(inputs_embeds, attention_mask)["logits"]

    ig = IntegratedGradients(ig_forward)
    for _ in range(n_samples):
        model.zero_grad(set_to_none=True)
        if noise_scale > 0:
            noisy_embeds = base_embeds + torch.randn_like(base_embeds) * noise_scale
        else:
            noisy_embeds = base_embeds
        with cudnn_backward_safe_context(device):
            attributions = ig.attribute(
                inputs=noisy_embeds.detach(),
                baselines=baseline,
                n_steps=steps,
                internal_batch_size=internal_batch_size,
            )
        scores = attributions.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
        total_scores += scores[: total_scores.shape[0]]

    model.zero_grad(set_to_none=True)
    return (total_scores / n_samples).astype(np.float32)


def serialize_score_array(scores: np.ndarray) -> str:
    array = np.asarray(scores, dtype=np.float32)
    return json.dumps(array.tolist(), separators=(",", ":"))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray) -> float:
    k = int(np.asarray(y_true).sum())
    if k == 0:
        return float("nan")
    top_k = np.argsort(scores)[::-1][:k]
    return float(np.asarray(y_true)[top_k].sum() / k)


def compute_probe_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if len(np.unique(y_true)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, scores))
    return {
        "auroc": auroc,
        "auprc": float(average_precision_score(y_true, scores)),
        "precision_at_k": precision_at_k(y_true, scores),
    }


def prepare_baseline_probe_frame(positives_csv: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(positives_csv)
    raw_df["accession"] = raw_df["accession"].astype(str)
    raw_df["sequence"] = raw_df["sequence"].astype(str).str.strip().str.upper()

    records = []
    for _, row in raw_df.iterrows():
        label_vec = parse_epitope_label(row["sequence"], row["epitope_start"], row["epitope_end"])
        n_epitope = int(label_vec.sum())
        seq_len = len(row["sequence"])
        if n_epitope == 0 or n_epitope == seq_len:
            continue
        records.append(
            {
                "accession": row["accession"],
                "sequence": row["sequence"],
                "epitope_label": label_vec,
                "seq_len": seq_len,
                "n_epitope_residues": n_epitope,
                "epitope_density": n_epitope / seq_len,
            }
        )
    return pd.DataFrame(records).reset_index(drop=True)


def run_baseline_probe_suite(
    model,
    tokenizer,
    eval_df: pd.DataFrame,
    device: str,
    ig_steps: int = IG_STEPS,
    n_random_draws: int = 100,
    max_seq_len: int = MAX_SEQ_LEN,
    ig_internal_batch_size: int | None = 1,
    smoothgrad_ig_samples: int = 10,
    smoothgrad_ig_noise_std: float = 0.05,
    include_shuffled_mean: bool = False,
    progress_label: str | None = None,
    enabled_methods: set[str] | None = None,
    progress_print_every: int = 5,
) -> pd.DataFrame:
    from .mtl_epitope_notebook_utils import compute_occlusion_scores_mtl

    rng = np.random.default_rng(RANDOM_STATE)
    results_rows = []
    enabled_methods = None if enabled_methods is None else set(enabled_methods)

    def method_enabled(method: str) -> bool:
        return enabled_methods is None or method in enabled_methods

    progress_desc = progress_label or "Evaluating proteins"
    total_rows = len(eval_df)
    for idx, (_, row) in enumerate(
        tqdm(eval_df.iterrows(), total=total_rows, desc=progress_desc),
        start=1,
    ):
        sequence = row["sequence"]
        epitope_labels = row["epitope_label"]
        accession = row["accession"]
        seq_len = row["seq_len"]

        tok_len = tokenizer(sequence, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
        if tok_len > max_seq_len:
            continue

        base = {
            "accession": accession,
            "seq_len": seq_len,
            "epitope_density": row["epitope_density"],
            "n_epitope_residues": row["n_epitope_residues"],
        }

        attn_scores = None
        if method_enabled("attention_weights"):
            try:
                attn_scores = compute_attention_weights(model, tokenizer, sequence, device)
                results_rows.append(
                    {**base, "method": "attention_weights", **compute_probe_metrics(epitope_labels, attn_scores)}
                )
            except Exception as exc:
                print(f"[attention] {accession}: {exc}")

        if method_enabled("integrated_gradients"):
            try:
                ig_scores = compute_integrated_gradients(
                    model,
                    tokenizer,
                    sequence,
                    device,
                    steps=ig_steps,
                    normalize=False,
                    internal_batch_size=ig_internal_batch_size,
                )
                results_rows.append(
                    {
                        **base,
                        "method": "integrated_gradients",
                        "ig_scores_json": serialize_score_array(ig_scores),
                        **compute_probe_metrics(epitope_labels, ig_scores),
                    }
                )
            except Exception as exc:
                print(f"[IG] {accession}: {exc}")

        if method_enabled("gradient_x_input"):
            try:
                gradient_x_input_scores = compute_gradient_x_input_scores(
                    model,
                    tokenizer,
                    sequence,
                    device,
                )
                results_rows.append(
                    {
                        **base,
                        "method": "gradient_x_input",
                        "gradient_x_input_scores_json": serialize_score_array(gradient_x_input_scores),
                        **compute_probe_metrics(epitope_labels, gradient_x_input_scores),
                    }
                )
            except Exception as exc:
                print(f"[Grad x Input] {accession}: {exc}")

        if method_enabled("smoothgrad_ig"):
            try:
                smoothgrad_ig_scores = compute_smoothgrad_ig_scores(
                    model,
                    tokenizer,
                    sequence,
                    device,
                    steps=ig_steps,
                    n_samples=smoothgrad_ig_samples,
                    noise_std=smoothgrad_ig_noise_std,
                    internal_batch_size=ig_internal_batch_size if ig_internal_batch_size is not None else 1,
                )
                results_rows.append(
                    {
                        **base,
                        "method": "smoothgrad_ig",
                        "smoothgrad_ig_scores_json": serialize_score_array(smoothgrad_ig_scores),
                        **compute_probe_metrics(epitope_labels, smoothgrad_ig_scores),
                    }
                )
            except Exception as exc:
                print(f"[SmoothGrad-IG] {accession}: {exc}")

        if method_enabled("occlusion"):
            try:
                occlusion_scores = normalize_scores(
                    compute_occlusion_scores_mtl(model, tokenizer, sequence, device)
                )
                results_rows.append(
                    {
                        **base,
                        "method": "occlusion",
                        "occlusion_scores_json": serialize_score_array(occlusion_scores),
                        **compute_probe_metrics(epitope_labels, occlusion_scores),
                    }
                )
            except Exception as exc:
                print(f"[Occlusion] {accession}: {exc}")

        if method_enabled("random_mean"):
            rand_metrics = [
                compute_probe_metrics(epitope_labels, rng.uniform(0.0, 1.0, size=seq_len))
                for _ in range(n_random_draws)
            ]
            results_rows.append(
                {
                    **base,
                    "method": "random_mean",
                    **mean_metric_dicts(rand_metrics),
                }
            )

        if include_shuffled_mean and attn_scores is not None:
            try:
                shuffled_metrics = [
                    compute_probe_metrics(rng.permutation(epitope_labels), attn_scores)
                    for _ in range(n_random_draws)
                ]
                results_rows.append(
                    {
                        **base,
                        "method": "shuffled_mean",
                        **mean_metric_dicts(shuffled_metrics),
                    }
                )
            except Exception as exc:
                print(f"[shuffled] {accession}: {exc}")

        if progress_print_every > 0 and (idx % progress_print_every == 0 or idx == total_rows):
            print(f"{progress_desc}: processed {idx}/{total_rows} proteins")

    return pd.DataFrame(results_rows)


# ── Stage 1 ────────────────────────────────────────────────────────────

def get_top_k_indices(ig_scores: np.ndarray, k_pct: float) -> list[int]:
    """
    Return indices of top-k% residues by IG score.
    k_pct is a fraction in (0, 1], e.g. 0.05 for top 5%.
    k is at least 1.
    """
    k = max(1, int(np.ceil(len(ig_scores) * k_pct)))
    return np.argsort(ig_scores)[::-1][:k].tolist()


def validate_ig_residues_by_masking(
    model,
    tokenizer,
    sequence: str,
    ig_scores: np.ndarray,
    device: str,
    k_pct: float,
) -> dict:
    """
    Mask the top-k% IG-important residues simultaneously and record
    the drop in allergenicity probability.

    Returns:
        {
          "k_pct": float,
          "k_absolute": int,
          "p_base": float,
          "p_masked": float,
          "delta_p": float,
          "top_k_indices": list[int],
          "validated": bool,         # True if delta_p > 0
        }
    """
    model.eval()

    assert tokenizer.mask_token is not None, (
        "Tokenizer has no mask token. Rebuild with add_special_tokens=True."
    )

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

    top_k_indices = get_top_k_indices(ig_scores, k_pct)

    residues = list(sequence)
    masked = residues.copy()
    for idx in top_k_indices:
        masked[idx] = tokenizer.mask_token

    p_base = _forward(sequence)
    p_masked = _forward("".join(masked))
    delta_p = p_base - p_masked

    return {
        "k_pct": k_pct,
        "k_absolute": len(top_k_indices),
        "p_base": float(p_base),
        "p_masked": float(p_masked),
        "delta_p": float(delta_p),
        "top_k_indices": top_k_indices,
        "validated": bool(delta_p > 0),
    }


# ── Stage 2 ────────────────────────────────────────────────────────────

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def run_saturation_mutagenesis(
    model,
    tokenizer,
    sequence: str,
    target_indices: list[int],
    device: str,
    p_base: float | None = None,
) -> pd.DataFrame:
    """
    For each residue index in target_indices, substitute all 20 amino
    acids one at a time and record the change in allergenicity probability.

    Returns a DataFrame with columns:
        position, original_aa, mutant_aa, p_base, p_mutant, delta_p,
        reduces_allergenicity
    """
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

    if p_base is None:
        p_base = _forward(sequence)

    residues = list(sequence)
    rows = []

    for idx in target_indices:
        original_aa = residues[idx]
        for mutant_aa in AMINO_ACIDS:
            if mutant_aa == original_aa:
                continue
            mutated = residues.copy()
            mutated[idx] = mutant_aa
            p_mutant = _forward("".join(mutated))
            delta_p = p_base - p_mutant
            rows.append(
                {
                    "position": idx,
                    "original_aa": original_aa,
                    "mutant_aa": mutant_aa,
                    "p_base": float(p_base),
                    "p_mutant": float(p_mutant),
                    "delta_p": float(delta_p),
                    "reduces_allergenicity": bool(delta_p > 0),
                }
            )

    return pd.DataFrame(rows)


# ── Stage 3 ────────────────────────────────────────────────────────────

AA_PROPERTIES = {
    "charge_positive": set("KRH"),
    "charge_negative": set("DE"),
    "hydrophobic": set("VILMFYWC"),
    "polar_uncharged": set("STNQ"),
    "special": set("GAP"),
}


def get_aa_property(aa: str) -> str:
    for prop, aa_set in AA_PROPERTIES.items():
        if aa in aa_set:
            return prop
    return "unknown"


def annotate_mutagenesis_results(mut_df: pd.DataFrame) -> pd.DataFrame:
    df = mut_df.copy()
    df["original_property"] = df["original_aa"].map(get_aa_property)
    df["mutant_property"] = df["mutant_aa"].map(get_aa_property)
    df["property_change"] = df.apply(
        lambda row: "same"
        if row["original_property"] == row["mutant_property"]
        else f"{row['original_property']} → {row['mutant_property']}",
        axis=1,
    )
    return df


def mean_metric_dicts(metric_rows: list[dict]) -> dict:
    return {
        "auroc": float(np.nanmean([row["auroc"] for row in metric_rows])),
        "auprc": float(np.mean([row["auprc"] for row in metric_rows])),
        "precision_at_k": float(np.mean([row["precision_at_k"] for row in metric_rows])),
    }
