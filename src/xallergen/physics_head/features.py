from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ── Amino-acid lookup tables ─────────────────────────────────────────────────
# Net formal charge at pH 7 (side-chain only; R/K +1, D/E -1, H partial).
CHARGE_PH7: dict[str, float] = {
    "A":  0.0, "R":  1.0, "N":  0.0, "D": -1.0, "C":  0.0,
    "Q":  0.0, "E": -1.0, "G":  0.0, "H":  0.1, "I":  0.0,
    "L":  0.0, "K":  1.0, "M":  0.0, "F":  0.0, "P":  0.0,
    "S":  0.0, "T":  0.0, "W":  0.0, "Y":  0.0, "V":  0.0,
}

# Kyte-Doolittle hydrophobicity scale (Kyte & Doolittle, 1982).
KYTE_DOOLITTLE: dict[str, float] = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

# Side-chain hydrogen-bond donors (–NH, –OH, –SH groups that can donate H).
HB_DONORS: dict[str, int] = {
    "A": 0, "R": 3, "N": 1, "D": 0, "C": 1,
    "Q": 1, "E": 0, "G": 0, "H": 1, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 1, "T": 1, "W": 1, "Y": 1, "V": 0,
}

# Side-chain hydrogen-bond acceptors (lone-pair O/N atoms).
HB_ACCEPTORS: dict[str, int] = {
    "A": 0, "R": 0, "N": 1, "D": 2, "C": 0,
    "Q": 1, "E": 2, "G": 0, "H": 1, "I": 0,
    "L": 0, "K": 0, "M": 0, "F": 0, "P": 0,
    "S": 1, "T": 1, "W": 0, "Y": 1, "V": 0,
}

_STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def build_physics_vector(
    sequence: str,
    rsa: np.ndarray,
    disorder: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
) -> np.ndarray:
    """Return (L, 10) raw unscaled physics feature matrix.

    Channel order
    -------------
    0  rsa
    1  charge
    2  charge × rsa
    3  hydrophobicity (Kyte-Doolittle)
    4  (1 - rsa) × hydrophobicity  (burial-weighted hydrophobicity)
    5  hb_count  (donors + acceptors)
    6  hb_count × rsa
    7  disorder
    8  sin(phi)   phi converted from degrees to radians
    9  cos(phi)   phi converted from degrees to radians

    Parameters
    ----------
    sequence : str
        Amino-acid sequence (single-letter codes, case-insensitive).
    rsa : np.ndarray shape (L,)
        Relative solvent accessibility, values in [0, 1].
    disorder : np.ndarray shape (L,)
        Per-residue disorder probability, values in [0, 1].
    phi : np.ndarray shape (L,)
        Backbone phi dihedral in **degrees** (NetSurfP convention).
    psi : np.ndarray shape (L,)
        Backbone psi dihedral in **degrees** (unused here, included for
        forward-compatibility with callers that always pass all four arrays).
    """
    seq = sequence.upper()
    L = len(seq)
    if rsa.shape[0] != L or disorder.shape[0] != L or phi.shape[0] != L:
        raise ValueError(
            f"Array lengths do not match sequence length {L}: "
            f"rsa={rsa.shape[0]}, disorder={disorder.shape[0]}, phi={phi.shape[0]}"
        )

    vec = np.empty((L, 10), dtype=np.float32)
    phi_rad = np.radians(phi.astype(np.float64))

    for i, aa in enumerate(seq):
        charge = CHARGE_PH7.get(aa, 0.0)
        hydro = KYTE_DOOLITTLE.get(aa, 0.0)
        hb = float(HB_DONORS.get(aa, 0) + HB_ACCEPTORS.get(aa, 0))
        r = float(rsa[i])
        d = float(disorder[i])

        vec[i, 0] = r
        vec[i, 1] = charge
        vec[i, 2] = charge * r
        vec[i, 3] = hydro
        vec[i, 4] = (1.0 - r) * hydro
        vec[i, 5] = hb
        vec[i, 6] = hb * r
        vec[i, 7] = d
        vec[i, 8] = float(np.sin(phi_rad[i]))
        vec[i, 9] = float(np.cos(phi_rad[i]))

    return vec


class PhysicsScaler:
    """Per-channel standardiser for the 10-channel physics feature matrix.

    Channels 0–7 are standardised to zero mean and unit variance using
    statistics from the training residue pool.  Channels 8–9 (sin/cos phi)
    are already bounded in [−1, 1] and are passed through unchanged.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, vectors: np.ndarray) -> "PhysicsScaler":
        """Fit on concatenated (N_residues_total, 10) training matrix."""
        if vectors.ndim != 2 or vectors.shape[1] != 10:
            raise ValueError(f"Expected (N, 10) array, got {vectors.shape}")
        self.mean = vectors[:, :8].mean(axis=0).astype(np.float64)
        self.std = vectors[:, :8].std(axis=0).astype(np.float64)
        return self

    def transform(self, vector: np.ndarray) -> np.ndarray:
        """Standardise (L, 10) vector; channels 8–9 returned unchanged."""
        if self.mean is None or self.std is None:
            raise RuntimeError("PhysicsScaler has not been fitted yet.")
        result = vector.copy().astype(np.float32)
        result[:, :8] = (
            (vector[:, :8].astype(np.float64) - self.mean) / (self.std + 1e-8)
        ).astype(np.float32)
        return result

    def save(self, path: str | Path) -> None:
        """Serialise mean and std to JSON."""
        payload = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PhysicsScaler":
        """Restore from a JSON file produced by :meth:`save`."""
        payload = json.loads(Path(path).read_text())
        scaler = cls()
        scaler.mean = np.array(payload["mean"], dtype=np.float64)
        scaler.std = np.array(payload["std"], dtype=np.float64)
        return scaler
