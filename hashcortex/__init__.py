"""Deprecated alias: HashCortex was renamed to HashMind. Use ``import hashmind``.

``import hashcortex.x.y`` returns the very same module object as
``hashmind.x.y``, so classes and isinstance checks are shared.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings
from types import ModuleType

import hashmind as _hashmind

warnings.warn("'hashcortex' is deprecated; use 'hashmind'", DeprecationWarning, stacklevel=2)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str) -> None:
        self.target = target

    def create_module(self, spec: object) -> ModuleType:
        return importlib.import_module(self.target)

    def exec_module(self, module: ModuleType) -> None:
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object, target: object = None):  # type: ignore[override]
        if fullname.startswith("hashcortex."):
            real = "hashmind." + fullname.removeprefix("hashcortex.")
            if importlib.util.find_spec(real) is None:
                return None
            return importlib.util.spec_from_loader(fullname, _AliasLoader(real))
        return None


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

sys.modules[__name__] = _hashmind
