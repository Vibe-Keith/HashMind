"""HashCortex-S9: convert GGUF models into an ASIC-native feature architecture.

HashCortex is a research prototype that takes a GGUF language model and
re-expresses as much of its learned information as is realistically possible
inside an architecture whose nonlinearity is produced by SHA-256 computation
(as performed by an Antminer S9 / BM1387 ASIC, or a CPU simulator).

This package deliberately does NOT claim that a BM1387 can run neural-network
math. It cannot. See ``docs/ARCHITECTURE.md`` and the ``README`` for an honest
account of what is preserved, what is approximated, and what is discarded.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "HASHCORTEX_VERSION",
    "HCMODEL_FORMAT_VERSION",
    "CONVERSION_VERSION",
]

# Semantic version of the HashCortex software.
__version__ = "0.1.0"
HASHCORTEX_VERSION = __version__

# On-disk .hcmodel container format version. Bump on breaking format changes.
HCMODEL_FORMAT_VERSION = 1

# Conversion-pipeline version. Bump when the conversion math changes in a way
# that would change results for the same input model + seeds.
CONVERSION_VERSION = 1
