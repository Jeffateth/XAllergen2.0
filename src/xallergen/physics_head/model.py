from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from xallergen.baseline_notebook_utils import (
    DROPOUT,
    FrozenESMAllergenClassifier,
)

# ── Channel metadata ─────────────────────────────────────────────────────────

CHANNEL_NAMES: list[str] = [
    "rsa",
    "charge",
    "charge_x_rsa",
    "hydrophobicity",
    "hydrophob_x_burial",
    "hb_count",
    "hb_x_rsa",
    "disorder",
    "sin_phi",
    "cos_phi",
]

# Expected sign for each channel's learned weight.
# +1 = positive contribution to allergenicity prediction
# -1 = negative contribution
#  0 = no strong prior (sin/cos phi have no directional prior)
EXPECTED_SIGNS: list[int] = [+1, +1, +1, -1, -1, +1, +1, -1, 0, 0]

# Physics-informed initial weights: [+1,+1,+1,-1,-1,+1,+1,-1,0,0] * 0.1
# RSA(+): surface-exposed residues are more accessible to antibodies.
# charge(+): charged residues attract complementary charges on antibody CDRs.
# charge×RSA(+): exposed charge is most relevant for electrostatic binding.
# hydrophob(-): hydrophobic residues favour burial, reducing epitope probability.
# hydrophob×burial(-): buried hydrophobic core is strongly anti-epitope.
# HB(+): H-bond-capable residues stabilise Ab-epitope contacts.
# HB×RSA(+): surface H-bond donors/acceptors can engage directly with CDRs.
# disorder(-): disordered regions are generally poor MHC-II binders (less stable).
# sin/cos phi(0): backbone torsion has no directional prior without context.
PHYSICS_INIT_WEIGHTS: torch.Tensor = (
    torch.tensor([[+1.0, +1.0, +1.0, -1.0, -1.0, +1.0, +1.0, -1.0, 0.0, 0.0]])
    * 0.1
)


class PhysicsProjection(nn.Module):
    """Linear projection from 10 physics channels to a single scalar.

    Acts as a learnable linear free-energy equation: the scalar output
    represents the net physics-based contribution of each residue to the
    allergenicity logit.  Initialised with PHYSICS_INIT_WEIGHTS so the
    model starts from physically motivated priors.
    """

    def __init__(self, n_channels: int = 10) -> None:
        super().__init__()
        self.linear = nn.Linear(n_channels, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(PHYSICS_INIT_WEIGHTS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, 10)

        Returns
        -------
        (B, L, 1)
        """
        return self.linear(x)


class FrozenESM2WithPhysics(nn.Module):
    """Frozen ESM-2 allergenicity classifier extended with a physics channel.

    Architecture
    ------------
    The existing frozen ESM-2 backbone produces per-residue embeddings
    (B, L, embed_dim).  The shared learned attention module produces alpha
    (B, L) from those embeddings.  Alpha is computed **once** and used to
    pool both the ESM embedding stream and the physics scalar stream:

        pooled_esm     = Σ_i alpha_i · esm_i       → (B, embed_dim)
        pooled_physics = Σ_i alpha_i · physics_i   → (B, 1)
        combined       = cat([pooled_esm, pooled_physics], dim=-1)
                                                    → (B, embed_dim + 1)
        logit          = MLP(combined)              → (B,)

    The MLP is the same two-layer classifier from the baseline, with its
    input dimension incremented by one.  No second attention module is added.
    """

    def __init__(
        self,
        baseline_model: FrozenESMAllergenClassifier,
        physics_dim: int = 10,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.backbone = baseline_model.backbone  # stays frozen
        self.attention_pool = baseline_model.attention_pool
        self.physics_proj = PhysicsProjection(physics_dim)

        embed_dim = self.backbone.config.hidden_size
        hidden_dim = baseline_model.classifier[0].out_features

        # Classifier with input dim = embed_dim + 1.
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Warm-start from baseline weights.  The first Linear gets an extra
        # zero column (for the physics scalar); all other layers copy exactly.
        with torch.no_grad():
            old_w = baseline_model.classifier[0].weight  # (hidden_dim, embed_dim)
            pad = torch.zeros(hidden_dim, 1, device=old_w.device, dtype=old_w.dtype)
            self.classifier[0].weight.copy_(torch.cat([old_w, pad], dim=1))
            self.classifier[0].bias.copy_(baseline_model.classifier[0].bias)
            self.classifier[3].weight.copy_(baseline_model.classifier[3].weight)
            self.classifier[3].bias.copy_(baseline_model.classifier[3].bias)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        device: str,
        physics_dim: int = 10,
        dropout: float = DROPOUT,
    ) -> "FrozenESM2WithPhysics":
        """Convenience constructor: load baseline checkpoint then wrap it."""
        from xallergen.baseline_notebook_utils import load_baseline_checkpoint

        baseline, _ = load_baseline_checkpoint(checkpoint_path, device)
        return cls(baseline, physics_dim=physics_dim, dropout=dropout).to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        physics_features: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        input_ids       : (B, L)
        attention_mask  : (B, L)
        physics_features: (B, L, 10)  scaled physics vectors, zero-padded
        """
        # --- frozen ESM-2 embeddings ---
        with torch.no_grad():
            residue_embeddings = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state  # (B, L, embed_dim)

        # --- shared alpha from attention pooling (called once) ---
        pooled_esm, alpha = self.attention_pool(residue_embeddings, attention_mask)
        # pooled_esm : (B, embed_dim)
        # alpha      : (B, L)

        # --- physics projection and pooling with the SAME alpha ---
        phys_scalar = self.physics_proj(physics_features)  # (B, L, 1)
        pooled_physics = torch.sum(alpha.unsqueeze(-1) * phys_scalar, dim=1)  # (B, 1)

        # --- classification ---
        combined = torch.cat([pooled_esm, pooled_physics], dim=-1)  # (B, embed_dim+1)
        logits = self.classifier(combined).squeeze(-1)  # (B,)

        return {
            "logits": logits,
            "attention_weights": alpha,
            "residue_embeddings": residue_embeddings,
            "pooled_esm": pooled_esm,
            "pooled_physics": pooled_physics,
        }


def get_weight_summary(model: FrozenESM2WithPhysics) -> pd.DataFrame:
    """Return a DataFrame summarising PhysicsProjection learned weights.

    Columns
    -------
    channel_name   : str
    learned_weight : float
    expected_sign  : int  (+1, -1, or 0)
    sign_preserved : bool  True if expected_sign == 0 or
                           np.sign(learned_weight) == expected_sign
    """
    weights = model.physics_proj.linear.weight.detach().cpu().numpy().flatten()
    rows = []
    for name, w, es in zip(CHANNEL_NAMES, weights, EXPECTED_SIGNS):
        if es == 0:
            preserved = True
        else:
            preserved = bool(np.sign(float(w)) == es)
        rows.append(
            {
                "channel_name": name,
                "learned_weight": float(w),
                "expected_sign": int(es),
                "sign_preserved": preserved,
            }
        )
    return pd.DataFrame(rows)
