"""Public API for SIGS.

Objects are imported lazily so using the symbolic grammar does not immediately
import training, clustering, JAX, and plotting dependencies.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

try:
    __version__ = version("sigs")
except PackageNotFoundError:  # source checkout without an editable install
    __version__ = "0.0.0+local"

_EXPORTS = {
    "GCFG": ("sigs.grammar", "GCFG"),
    "S": ("sigs.grammar", "S"),
    "T": ("sigs.grammar", "T"),
    "D": ("sigs.grammar", "D"),
    "get_mask": ("sigs.grammar", "get_mask"),
    "GrammarVAE": ("sigs.model", "GrammarVAE"),
    "FlexibleVectorSampler": ("sigs.sampler", "FlexibleVectorSampler"),
    "MathClass": ("sigs.utils", "MathClass"),
    "ExpressionUtils": ("sigs.utils", "ExpressionUtils"),
    "ModelUtils": ("sigs.utils", "ModelUtils"),
    "FileUtils": ("sigs.utils", "FileUtils"),
    "SIGSConfig": ("sigs.huggingface", "SIGSConfig"),
    "SIGSModel": ("sigs.huggingface", "SIGSModel"),
    "load_hdf5_dataset": ("sigs.hf_data", "load_hdf5_dataset"),
}

_TRANSFORMERS_EXPORTS = {"SIGSConfig", "SIGSModel"}
_DATASETS_EXPORTS = {"load_hdf5_dataset"}
_OPTIONAL_EXPORTS = _TRANSFORMERS_EXPORTS | _DATASETS_EXPORTS
__all__ = [
    "__version__",
    *(name for name in _EXPORTS if name not in _OPTIONAL_EXPORTS),
]
if find_spec("transformers") is not None:
    __all__.extend(sorted(_TRANSFORMERS_EXPORTS))
if find_spec("datasets") is not None and find_spec("h5py") is not None:
    __all__.extend(sorted(_DATASETS_EXPORTS))


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'sigs' has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
