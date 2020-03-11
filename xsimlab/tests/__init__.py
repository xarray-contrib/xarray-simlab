import importlib
import pytest


def _importorskip(modname):
    try:
        mod = importlib.import_module(modname)
        has = True
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason=f"requires {modname}")
    return has, func


has_tqdm, requires_tqdm = _importorskip("tqdm")
