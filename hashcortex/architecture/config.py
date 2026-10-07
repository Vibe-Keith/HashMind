"""HashCortex architecture configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class HashCortexConfig:
    """Hyperparameters of the HashCortex v1 architecture.

    Data path per token (host = PC, asic = S9 or simulator)::

        token id
          -> [host] embedding lookup e = E[token]            (PRESERVED)
          -> [host] z = P^T (rmsnorm(e) - mu)                (TRANSFORMED from FFN/attn-V SVD)
          -> [host] sign bits s = (z > 0), k bits
          -> [host] u = s ++ reservoir r, k + c bits
          -> [host] n-tuple selection: n_tuples groups of tuple_bits bits
          -> [asic] per tuple: SHA-256d over 80-byte header, nonce range R,
                    report nonces meeting difficulty -> R binary features
          -> [host] f in {0,1}^(n_tuples*R); reservoir r <- update(r, f)
          -> [host] h = e + W_r^T (2f - 1) + b_r              (new readout, ridge-fit)
          -> [host] logits = W_out rmsnorm(h)                 (PRESERVED LM head)

    The hash layer is a random boolean function of small bit tuples (an n-tuple
    / WiSARD-style random lookup layer). Using small tuples is essential:
    a hash of the full input would map neighbouring inputs to uncorrelated
    features and the readout could only memorize, not generalize.
    """

    hidden_dim: int
    vocab_size: int
    proj_dim: int = 64  # k: sign bits from host projection
    reservoir_bits: int = 64  # c: recurrent binary context state
    n_tuples: int = 256  # number of ASIC jobs per token
    tuple_bits: int = 8  # bits of u hashed per job
    nonces_per_tuple: int = 4  # R: nonce range per job -> features per job
    difficulty_bits: int = 1  # 1 => p(feature=1)=0.5 (idealized); real ticket floor TBD
    reservoir_keep: float = 0.75  # fraction of reservoir bits shifted vs refreshed each step
    seed: int = 0x5EED_C0DE
    challenge_salt: str = "hashcortex-s9-v1"
    projection_source_layers: int = 4
    ridge_lambda: float = 1e-2
    norm_eps: float = 1e-5

    @property
    def input_bits(self) -> int:
        return self.proj_dim + self.reservoir_bits

    @property
    def n_features(self) -> int:
        return self.n_tuples * self.nonces_per_tuple

    @property
    def hashes_per_token(self) -> int:
        return self.n_features

    def validate(self) -> None:
        if not 1 <= self.tuple_bits <= min(self.input_bits, 256):
            raise ValueError("tuple_bits must be in [1, min(input_bits, 256)]")
        if self.proj_dim > self.hidden_dim:
            raise ValueError("proj_dim cannot exceed hidden_dim")
        if not 0.0 <= self.reservoir_keep <= 1.0:
            raise ValueError("reservoir_keep must be in [0, 1]")
        if self.reservoir_bits and self.n_features < self.reservoir_bits:
            raise ValueError("need n_features >= reservoir_bits to refresh the reservoir")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HashCortexConfig":
        return cls(**d)
