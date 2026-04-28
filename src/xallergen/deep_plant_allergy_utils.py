from __future__ import annotations

import contextlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from .baseline_notebook_utils import (
    compute_probe_metrics,
    prepare_baseline_probe_frame,
    serialize_score_array,
)


RANDOM_STATE = 42
ESM_MODEL_NAME = "esm1b_t33_650M_UR50S"
HF_MODEL_NAME = f"facebook/{ESM_MODEL_NAME}"
EMBEDDING_DIM = 1280
HIDDEN_DIM = 128
OUTPUT_DIM = 1
NUM_LSTM_LAYERS = 3
NUM_FC_LAYERS = 3
NUM_ATTENTION_HEADS = 8
NUM_FILTERS = 64
KERNEL_SIZE = 5
IG_STEPS = 50
MAX_SEQ_LEN = 1000


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
            if snapshot.is_dir() and any(list(snapshot.glob(pat)) for pat in ("*.safetensors", "*.bin")):
                return str(snapshot)

    return model_name


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root. Make sure VSCode is opened from inside the XAllergen2.0 folder."
    )


def cudnn_backward_safe_context(device: str):
    if str(device).startswith("cuda"):
        return torch.backends.cudnn.flags(enabled=False)
    return contextlib.nullcontext()


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
        print("  backend: CUDA")
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
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(resolve_hf_model_source(model_name))


def build_embedding_model(model_name: str = HF_MODEL_NAME, device: str = "cpu"):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(resolve_hf_model_source(model_name)).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        if self.head_dim * heads != embed_size:
            raise ValueError("Embedding size must be divisible by heads")

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.attention_weights = None

    def forward(self, values, keys, query, mask=None):
        batch_size = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(batch_size, value_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim)
        queries = query.reshape(batch_size, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(energy / (self.embed_size ** 0.5), dim=3)
        self.attention_weights = attention
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            batch_size, query_len, self.embed_size
        )

        return self.fc_out(out)

    def get_attention_weights(self):
        return self.attention_weights


class EnhancedProteinModelFull(nn.Module):
    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
        num_lstm_layers: int = NUM_LSTM_LAYERS,
        num_fc_layers: int = NUM_FC_LAYERS,
        num_attention_heads: int = NUM_ATTENTION_HEADS,
        num_filters: int = NUM_FILTERS,
        kernel_size: int = KERNEL_SIZE,
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.batch_norm_conv = nn.BatchNorm1d(num_filters)

        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )

        self.attention = SelfAttention(embed_size=hidden_dim * 2, heads=num_attention_heads)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc_layers = nn.ModuleList()
        self.batch_norm_fc = nn.ModuleList()
        input_dim = hidden_dim * 2

        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norm_fc.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim

        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.batch_norm_conv(x)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = self.attention(x, x, x)
        x = self.pool(x.permute(0, 2, 1)).squeeze(-1)

        for fc, bn in zip(self.fc_layers, self.batch_norm_fc):
            x = self.relu(bn(fc(x)))
            x = self.dropout(x)

        return self.fc_output(x)


def load_deep_plant_allergy_checkpoint(
    checkpoint_path: Path,
    device: str,
) -> tuple[EnhancedProteinModelFull, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}")

    metadata = checkpoint.get("metadata", {}) if "state_dict" in checkpoint else {}
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected checkpoint state_dict dict, got {type(state_dict)!r}")

    architecture = metadata.get("architecture_hyperparameters", {})
    model = EnhancedProteinModelFull(
        embedding_dim=int(architecture.get("embedding_dim", EMBEDDING_DIM)),
        hidden_dim=int(architecture.get("hidden_dim", HIDDEN_DIM)),
        output_dim=int(architecture.get("output_dim", OUTPUT_DIM)),
        num_lstm_layers=int(architecture.get("num_lstm_layers", NUM_LSTM_LAYERS)),
        num_fc_layers=int(architecture.get("num_fc_layers", NUM_FC_LAYERS)),
        num_attention_heads=int(architecture.get("num_attention_heads", NUM_ATTENTION_HEADS)),
        num_filters=int(architecture.get("num_filters", NUM_FILTERS)),
        kernel_size=int(architecture.get("kernel_size", KERNEL_SIZE)),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    resolved_metadata = {
        "architecture_hyperparameters": {
            "embedding_dim": int(architecture.get("embedding_dim", EMBEDDING_DIM)),
            "hidden_dim": int(architecture.get("hidden_dim", HIDDEN_DIM)),
            "output_dim": int(architecture.get("output_dim", OUTPUT_DIM)),
            "num_lstm_layers": int(architecture.get("num_lstm_layers", NUM_LSTM_LAYERS)),
            "num_fc_layers": int(architecture.get("num_fc_layers", NUM_FC_LAYERS)),
            "num_attention_heads": int(architecture.get("num_attention_heads", NUM_ATTENTION_HEADS)),
            "num_filters": int(architecture.get("num_filters", NUM_FILTERS)),
            "kernel_size": int(architecture.get("kernel_size", KERNEL_SIZE)),
        },
        "esm_model_name": metadata.get("esm_model_name", ESM_MODEL_NAME),
        "hf_model_name": metadata.get("hf_model_name", HF_MODEL_NAME),
        "embedding_source": metadata.get("embedding_source", "huggingface_transformers"),
        "checkpoint_format": "metadata_wrapped" if "state_dict" in checkpoint else "raw_state_dict",
    }
    return model, resolved_metadata


def clean_protein_sequence(sequence: str) -> str:
    return "".join(str(sequence).strip().split()).upper()


def compute_residue_embeddings(
    embedding_model,
    tokenizer,
    sequence: str,
    device: str,
    max_seq_len: int = MAX_SEQ_LEN,
) -> torch.Tensor:
    sequence = clean_protein_sequence(sequence)
    encodings = tokenizer(
        sequence,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_len + 2,
    )
    encodings = {key: value.to(device) for key, value in encodings.items()}

    with torch.no_grad():
        outputs = embedding_model(**encodings)

    valid_length = int(encodings["attention_mask"].sum().item())
    residue_embeddings = outputs.last_hidden_state[:, 1 : valid_length - 1, :]
    return residue_embeddings[:, :max_seq_len, :].detach()


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.maximum(scores, 0.0)
    total = scores.sum()
    return scores / total if total > 0 else scores


def compute_attention_weights(
    model,
    embedding_model,
    tokenizer,
    sequence: str,
    device: str,
) -> np.ndarray:
    residue_embeddings = compute_residue_embeddings(embedding_model, tokenizer, sequence, device)
    with torch.no_grad():
        model(residue_embeddings)
    attention = model.attention.get_attention_weights()
    if attention is None:
        raise RuntimeError("Model did not expose self-attention weights.")
    scores = attention.squeeze(0).mean(dim=(0, 1)).detach().cpu().numpy()
    return scores[: residue_embeddings.shape[1]]


def compute_integrated_gradients(
    model,
    embedding_model,
    tokenizer,
    sequence: str,
    device: str,
    steps: int = IG_STEPS,
    normalize: bool = False,
    internal_batch_size: int | None = 1,
) -> np.ndarray:
    from captum.attr import IntegratedGradients

    residue_embeddings = compute_residue_embeddings(embedding_model, tokenizer, sequence, device)
    baseline = torch.zeros_like(residue_embeddings)

    lstm_module = getattr(model, "lstm", None)
    lstm_training_state = None if lstm_module is None else lstm_module.training

    def ig_forward(inputs_embeds: torch.Tensor) -> torch.Tensor:
        return model(inputs_embeds).squeeze(-1)

    try:
        if lstm_module is not None:
            lstm_module.train()
        with cudnn_backward_safe_context(device):
            attributions = IntegratedGradients(ig_forward).attribute(
                inputs=residue_embeddings,
                baselines=baseline,
                n_steps=steps,
                internal_batch_size=internal_batch_size,
            )
    finally:
        if lstm_module is not None and lstm_training_state is not None:
            lstm_module.train(lstm_training_state)
    importance = attributions.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
    return normalize_scores(importance) if normalize else importance


def compute_gradient_x_input_scores(
    model,
    embedding_model,
    tokenizer,
    sequence: str,
    device: str,
) -> np.ndarray:
    """
    Gradient x Input attribution against the protein-level classification logit.
    Returns one absolute embedding-dot-gradient score per residue.
    """
    model.eval()
    residue_embeddings = compute_residue_embeddings(embedding_model, tokenizer, sequence, device)
    residue_embeddings = residue_embeddings.detach()
    residue_embeddings.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    lstm_module = getattr(model, "lstm", None)
    lstm_training_state = None if lstm_module is None else lstm_module.training

    try:
        if lstm_module is not None:
            lstm_module.train()
        with cudnn_backward_safe_context(device):
            logits = model(residue_embeddings)
            gradients = torch.autograd.grad(
                outputs=logits.sum(),
                inputs=residue_embeddings,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0]
    finally:
        if lstm_module is not None and lstm_training_state is not None:
            lstm_module.train(lstm_training_state)

    scores = (residue_embeddings * gradients).sum(dim=-1).abs().squeeze(0)
    model.zero_grad(set_to_none=True)
    return scores.detach().cpu().numpy().astype(np.float32)


def compute_smoothgrad_ig_scores(
    model,
    embedding_model,
    tokenizer,
    sequence: str,
    device: str,
    steps: int,
    n_samples: int = 10,
    noise_std: float = 0.05,
    internal_batch_size: int = 10,
) -> np.ndarray:
    """
    SmoothGrad over Integrated Gradients in residue-embedding space.
    Noise is added to the frozen ESM embeddings before attribution.
    """
    from captum.attr import IntegratedGradients

    if n_samples <= 0:
        raise ValueError("n_samples must be positive for SmoothGrad-IG.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative for SmoothGrad-IG.")

    model.eval()
    residue_embeddings = compute_residue_embeddings(embedding_model, tokenizer, sequence, device)
    baseline = torch.zeros_like(residue_embeddings)
    noise_scale = float(noise_std) * float(residue_embeddings.detach().std().item())
    total_scores = np.zeros(residue_embeddings.shape[1], dtype=np.float64)

    lstm_module = getattr(model, "lstm", None)
    lstm_training_state = None if lstm_module is None else lstm_module.training

    def ig_forward(inputs_embeds: torch.Tensor) -> torch.Tensor:
        return model(inputs_embeds).squeeze(-1)

    ig = IntegratedGradients(ig_forward)
    try:
        if lstm_module is not None:
            lstm_module.train()
        for _ in range(n_samples):
            model.zero_grad(set_to_none=True)
            if noise_scale > 0:
                noisy_embeds = residue_embeddings + torch.randn_like(residue_embeddings) * noise_scale
            else:
                noisy_embeds = residue_embeddings
            with cudnn_backward_safe_context(device):
                attributions = ig.attribute(
                    inputs=noisy_embeds.detach(),
                    baselines=baseline,
                    n_steps=steps,
                    internal_batch_size=internal_batch_size,
                )
            scores = attributions.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
            total_scores += scores[: total_scores.shape[0]]
    finally:
        if lstm_module is not None and lstm_training_state is not None:
            lstm_module.train(lstm_training_state)

    model.zero_grad(set_to_none=True)
    return (total_scores / n_samples).astype(np.float32)


def compute_occlusion_scores(
    model,
    embedding_model,
    tokenizer,
    sequence: str,
    device: str,
) -> np.ndarray:
    """
    Single-residue occlusion for DeepPlantAllergy.
    Replaces residue i with the tokenizer's mask token, then records the
    drop in predicted allergenicity probability.
    """
    if tokenizer.mask_token is None:
        raise ValueError("Tokenizer has no mask token; cannot compute occlusion scores.")

    model.eval()

    def _forward(seq: str) -> float:
        residue_embeddings = compute_residue_embeddings(embedding_model, tokenizer, seq, device)
        with torch.no_grad():
            logits = model(residue_embeddings)
        return float(torch.sigmoid(logits).item())

    p_base = _forward(sequence)
    residues = list(clean_protein_sequence(sequence))
    delta_p = np.zeros(len(residues), dtype=np.float32)

    for idx in range(len(residues)):
        masked = residues.copy()
        masked[idx] = tokenizer.mask_token
        delta_p[idx] = p_base - _forward("".join(masked))

    return delta_p


def mean_metric_dicts(metric_rows: list[dict]) -> dict:
    return {
        "auroc": float(np.nanmean([row["auroc"] for row in metric_rows])),
        "auprc": float(np.mean([row["auprc"] for row in metric_rows])),
        "precision_at_k": float(np.mean([row["precision_at_k"] for row in metric_rows])),
    }


def run_deep_plant_probe_suite(
    model,
    embedding_model,
    tokenizer,
    eval_df: pd.DataFrame,
    device: str,
    ig_steps: int = IG_STEPS,
    n_random_draws: int = 100,
    max_seq_len: int = MAX_SEQ_LEN,
    ig_internal_batch_size: int | None = 1,
    smoothgrad_ig_samples: int = 10,
    smoothgrad_ig_noise_std: float = 0.05,
    progress_label: str | None = None,
    enabled_methods: set[str] | None = None,
    progress_print_every: int = 5,
) -> pd.DataFrame:
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
                attn_scores = compute_attention_weights(model, embedding_model, tokenizer, sequence, device)
                results_rows.append(
                    {**base, "method": "attention_weights", **compute_probe_metrics(epitope_labels, attn_scores)}
                )
            except Exception as exc:
                print(f"[attention] {accession}: {exc}")

        if method_enabled("integrated_gradients"):
            try:
                ig_scores = compute_integrated_gradients(
                    model,
                    embedding_model,
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
                    embedding_model,
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
                    embedding_model,
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
                    compute_occlusion_scores(
                        model,
                        embedding_model,
                        tokenizer,
                        sequence,
                        device,
                    )
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

        if attn_scores is not None:
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
