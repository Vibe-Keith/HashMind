"""HashMind (HashMind-S9): GGUF models re-expressed around SHA-256 ASIC features.

HashMind is a research prototype. It takes a GGUF language model and builds a
new, experimental architecture in which the nonlinear feature layer is SHA-256d
computation of the kind an Antminer S9 / BM1387 performs. It does NOT claim the
result is equivalent to the source network; it measures how much useful
information survives the transformation.

Formerly named HashCortex; ``import hashcortex`` and ``.hcmodel`` files still work.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "HASHMIND_VERSION",
    "HMMODEL_FORMAT_VERSION",
    "CONVERSION_VERSION",
    "HASHCORTEX_VERSION",
    "HCMODEL_FORMAT_VERSION",
]

__version__ = "0.2.0"
HASHMIND_VERSION = __version__

# On-disk .hmmodel container format version. v1 = phase-1 .hcmodel layout;
# v2 adds the weight-preservation plan and the HashMind layer spec.
HMMODEL_FORMAT_VERSION = 2

# Bump when conversion math changes results for the same input model + seeds.
CONVERSION_VERSION = 2

# Backwards-compatible aliases (phase 1 names).
HASHCORTEX_VERSION = HASHMIND_VERSION
HCMODEL_FORMAT_VERSION = HMMODEL_FORMAT_VERSION
