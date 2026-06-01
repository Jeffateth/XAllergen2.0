"""Tests for src/xallergen/physics_head/."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

# ── path setup ───────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from xallergen.physics_head.features import (
    CHARGE_PH7,
    HB_ACCEPTORS,
    HB_DONORS,
    KYTE_DOOLITTLE,
    PhysicsScaler,
    build_physics_vector,
)
from xallergen.physics_head.model import (
    CHANNEL_NAMES,
    EXPECTED_SIGNS,
    PHYSICS_INIT_WEIGHTS,
    FrozenESM2WithPhysics,
    PhysicsProjection,
    get_weight_summary,
)

# ── fixtures ─────────────────────────────────────────────────────────────────

_STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
_SEQ = "ACDEFGHIKLMNPQRSTVWY"  # one of each standard AA
_L = len(_SEQ)


@pytest.fixture()
def rsa():
    return np.random.default_rng(0).uniform(0, 1, _L).astype(np.float32)


@pytest.fixture()
def disorder():
    return np.random.default_rng(1).uniform(0, 1, _L).astype(np.float32)


@pytest.fixture()
def phi():
    # NetSurfP outputs degrees in (-180, 180)
    return np.random.default_rng(2).uniform(-180, 180, _L).astype(np.float32)


@pytest.fixture()
def psi():
    return np.random.default_rng(3).uniform(-180, 180, _L).astype(np.float32)


@pytest.fixture()
def physics_vec(rsa, disorder, phi, psi):
    return build_physics_vector(_SEQ, rsa, disorder, phi, psi)


# ── lookup dict completeness ─────────────────────────────────────────────────

def test_charge_all_aas():
    assert set(CHARGE_PH7.keys()) == set(_STANDARD_AAS)


def test_kyte_doolittle_all_aas():
    assert set(KYTE_DOOLITTLE.keys()) == set(_STANDARD_AAS)


def test_hb_donors_all_aas():
    assert set(HB_DONORS.keys()) == set(_STANDARD_AAS)


def test_hb_acceptors_all_aas():
    assert set(HB_ACCEPTORS.keys()) == set(_STANDARD_AAS)


# ── build_physics_vector ─────────────────────────────────────────────────────

def test_build_physics_vector_shape(physics_vec):
    assert physics_vec.shape == (_L, 10)


def test_build_physics_vector_channel_2_equals_1_times_0(rsa, disorder, phi, psi):
    vec = build_physics_vector(_SEQ, rsa, disorder, phi, psi)
    np.testing.assert_allclose(vec[:, 2], vec[:, 1] * vec[:, 0], rtol=1e-5)


def test_build_physics_vector_channel_4_equals_burial_times_hydrophob(rsa, disorder, phi, psi):
    vec = build_physics_vector(_SEQ, rsa, disorder, phi, psi)
    np.testing.assert_allclose(vec[:, 4], (1 - vec[:, 0]) * vec[:, 3], rtol=1e-5)


def test_build_physics_vector_sin_cos_bounded(physics_vec):
    assert np.all(np.abs(physics_vec[:, 8]) <= 1.0 + 1e-6)
    assert np.all(np.abs(physics_vec[:, 9]) <= 1.0 + 1e-6)


def test_build_physics_vector_length_mismatch_raises(rsa, disorder, phi, psi):
    bad_rsa = rsa[:-1]
    with pytest.raises(ValueError, match="Array lengths"):
        build_physics_vector(_SEQ, bad_rsa, disorder, phi, psi)


# ── PhysicsScaler ────────────────────────────────────────────────────────────

def _random_vectors(seed: int = 0, n: int = 1000, L: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rsa = rng.uniform(0, 1, (n, L)).astype(np.float32)
    disorder = rng.uniform(0, 1, (n, L)).astype(np.float32)
    phi = rng.uniform(-180, 180, (n, L)).astype(np.float32)
    psi = rng.uniform(-180, 180, (n, L)).astype(np.float32)
    seq = _SEQ  # 20 AAs
    all_vecs = []
    for i in range(n):
        v = build_physics_vector(seq, rsa[i], disorder[i], phi[i], psi[i])
        all_vecs.append(v)
    return np.concatenate(all_vecs, axis=0)  # (n*L, 10)


def test_scaler_fit_transform_mean_zero_std_one():
    vecs = _random_vectors()
    scaler = PhysicsScaler().fit(vecs)
    transformed = scaler.transform(vecs)
    # channels 0-7 should have approximately mean 0 and std 1
    np.testing.assert_allclose(transformed[:, :8].mean(axis=0), 0.0, atol=0.05)
    np.testing.assert_allclose(transformed[:, :8].std(axis=0), 1.0, atol=0.05)


def test_scaler_channels_8_9_unchanged():
    vecs = _random_vectors()
    scaler = PhysicsScaler().fit(vecs)
    transformed = scaler.transform(vecs)
    np.testing.assert_array_equal(transformed[:, 8], vecs[:, 8])
    np.testing.assert_array_equal(transformed[:, 9], vecs[:, 9])


def test_scaler_save_load_roundtrip():
    vecs = _random_vectors()
    scaler = PhysicsScaler().fit(vecs)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    scaler.save(path)
    restored = PhysicsScaler.load(path)
    np.testing.assert_array_equal(scaler.mean, restored.mean)
    np.testing.assert_array_equal(scaler.std, restored.std)


def test_scaler_transform_without_fit_raises():
    scaler = PhysicsScaler()
    vecs = _random_vectors(n=2)
    with pytest.raises(RuntimeError, match="not been fitted"):
        scaler.transform(vecs[:20])


# ── PhysicsProjection ────────────────────────────────────────────────────────

def test_physics_projection_weight_shape():
    proj = PhysicsProjection()
    assert proj.linear.weight.shape == (1, 10)


def test_physics_projection_matches_init_weights():
    proj = PhysicsProjection()
    np.testing.assert_allclose(
        proj.linear.weight.detach().numpy(),
        PHYSICS_INIT_WEIGHTS.numpy(),
        rtol=1e-6,
    )


def test_physics_projection_forward_shape():
    proj = PhysicsProjection()
    x = torch.randn(2, 15, 10)
    out = proj(x)
    assert out.shape == (2, 15, 1)


# ── FrozenESM2WithPhysics (mock backbone) ────────────────────────────────────

class _MockConfig:
    hidden_size = 8  # tiny for speed


class _MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _MockConfig()
        self.embed = nn.Embedding(30, 8)

    def forward(self, input_ids, attention_mask):
        B, L = input_ids.shape
        out = MagicMock()
        out.last_hidden_state = torch.randn(B, L, 8)
        return out

    def parameters(self):
        for p in super().parameters():
            p.requires_grad = False
            yield p


def _make_mock_baseline():
    """Build a minimal FrozenESMAllergenClassifier-like object for testing."""
    from xallergen.baseline_notebook_utils import AttentionPooling
    baseline = MagicMock()
    backbone = _MockBackbone()
    baseline.backbone = backbone
    baseline.attention_pool = AttentionPooling(embed_dim=8)
    baseline.classifier = nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Dropout(0.0),
        nn.Linear(4, 1),
    )
    return baseline


@pytest.fixture()
def mock_physics_model():
    baseline = _make_mock_baseline()
    return FrozenESM2WithPhysics(baseline, physics_dim=10, dropout=0.0)


def test_frozen_esm2_with_physics_output_logit_shape(mock_physics_model):
    B, L = 2, _L
    input_ids = torch.randint(0, 20, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    physics = torch.randn(B, L, 10)
    mock_physics_model.eval()
    with patch.object(
        mock_physics_model.backbone,
        "forward",
        side_effect=lambda input_ids, attention_mask: _mock_esm_output(B, L, 8),
    ):
        out = mock_physics_model(input_ids, attention_mask, physics)
    assert out["logits"].shape == (B,)


def _mock_esm_output(B, L, H):
    result = MagicMock()
    result.last_hidden_state = torch.randn(B, L, H)
    return result


def test_shared_alpha_perturb_changes_both_streams(mock_physics_model):
    """Verify pooled_esm and pooled_physics both depend on the same alpha.

    We run two forward passes with different attention_mask inputs that force
    different alpha distributions, then assert both pooled streams change.
    """
    B, L = 1, _L
    physics = torch.randn(B, L, 10)

    mock_physics_model.eval()

    def _run(mask_ones: int):
        mask = torch.zeros(B, L, dtype=torch.long)
        mask[0, :mask_ones] = 1
        input_ids = torch.zeros(B, L, dtype=torch.long)
        with patch.object(
            mock_physics_model.backbone,
            "forward",
            side_effect=lambda **kw: _mock_esm_output(B, L, 8),
        ):
            # Patch backbone to be called with keyword args
            def _backbone_call(input_ids, attention_mask):
                return _mock_esm_output(B, L, 8)
            mock_physics_model.backbone.forward = _backbone_call
            out = mock_physics_model(input_ids, mask, physics)
        return out["pooled_esm"].detach().clone(), out["pooled_physics"].detach().clone()

    pooled_esm_1, pooled_physics_1 = _run(mask_ones=5)
    pooled_esm_2, pooled_physics_2 = _run(mask_ones=15)

    # Different masks → different alpha → both pooled outputs change
    assert not torch.allclose(pooled_esm_1, pooled_esm_2), (
        "pooled_esm should change when attention mask changes"
    )
    assert not torch.allclose(pooled_physics_1, pooled_physics_2), (
        "pooled_physics should change when attention mask changes (shared alpha)"
    )


# ── get_weight_summary ───────────────────────────────────────────────────────

def test_get_weight_summary_shape(mock_physics_model):
    df = get_weight_summary(mock_physics_model)
    assert len(df) == 10
    assert set(df.columns) == {"channel_name", "learned_weight", "expected_sign", "sign_preserved"}


def test_get_weight_summary_channel_names(mock_physics_model):
    df = get_weight_summary(mock_physics_model)
    assert list(df["channel_name"]) == CHANNEL_NAMES


def test_get_weight_summary_expected_signs(mock_physics_model):
    df = get_weight_summary(mock_physics_model)
    assert list(df["expected_sign"]) == EXPECTED_SIGNS


def test_get_weight_summary_sign_preserved_for_init_weights(mock_physics_model):
    """After PHYSICS_INIT_WEIGHTS init, non-zero channels should have sign preserved."""
    df = get_weight_summary(mock_physics_model)
    for _, row in df.iterrows():
        if row["expected_sign"] != 0:
            assert row["sign_preserved"], (
                f"Channel {row['channel_name']}: expected sign {row['expected_sign']} "
                f"but learned weight is {row['learned_weight']:.4f}"
            )
