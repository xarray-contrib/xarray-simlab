import importlib
import os

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


if os.environ.get("DASK_SINGLE_THREADED"):
    # CI do not always support parallel code
    use_dask_schedulers = ["single-threaded"]
else:
    # Still useful to test threads/processes (pickle issues) locally
    use_dask_schedulers = ["threads", "processes", "distributed", "distributed-threads"]
