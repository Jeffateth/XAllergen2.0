from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


RANDOM_STATE = 42
ESM_MODEL_NAME = "esm2_t6_8M_UR50D"
HF_MODEL_NAME = f"facebook/{ESM_MODEL_NAME}"
EMBEDDING_DIM = 320
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
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected checkpoint state_dict dict, got {type(state_dict)!r}")

    model = EnhancedProteinModelFull().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    metadata = {
        "architecture_hyperparameters": {
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": OUTPUT_DIM,
            "num_lstm_layers": NUM_LSTM_LAYERS,
            "num_fc_layers": NUM_FC_LAYERS,
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "num_filters": NUM_FILTERS,
            "kernel_size": KERNEL_SIZE,
        },
        "esm_model_name": ESM_MODEL_NAME,
    }
    return model, metadata


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

    def ig_forward(inputs_embeds: torch.Tensor) -> torch.Tensor:
        return model(inputs_embeds).squeeze(-1)

    attributions = IntegratedGradients(ig_forward).attribute(
        inputs=residue_embeddings,
        baselines=baseline,
        n_steps=steps,
        internal_batch_size=internal_batch_size,
    )
    importance = attributions.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
    return normalize_scores(importance) if normalize else importance


def mean_metric_dicts(metric_rows: list[dict]) -> dict:
    return {
        "auroc": float(np.nanmean([row["auroc"] for row in metric_rows])),
        "auprc": float(np.mean([row["auprc"] for row in metric_rows])),
        "precision_at_k": float(np.mean([row["precision_at_k"] for row in metric_rows])),
    }
