"""AttnDiff: Attention-based Differential Fingerprinting for LLMs."""

__version__ = "0.1.0"

from attndiff.core.fingerprint import compute_fingerprint, load_fingerprint
from attndiff.core.similarity import compare_fingerprints, linear_cka

__all__ = [
    "__version__",
    "compute_fingerprint",
    "load_fingerprint",
    "compare_fingerprints",
    "linear_cka",
]
