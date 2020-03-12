import importlib

import pytest

from ..monitoring import ProgressBar
from . import has_tqdm


@pytest.mark.skipif(not has_tqdm, reason="requires tqdm")
@pytest.mark.parametrize(
    "frontend,tqdm_module",
    [
        ("auto", "tqdm"),  # assume tests are run in a terminal evironment
        ("console", "tqdm"),
        ("gui", "tqdm.gui"),
        ("notebook", "tqdm.notebook"),
    ],
)
def test_progress_bar_init(frontend, tqdm_module):
    pbar = ProgressBar(frontend=frontend)
    tqdm = importlib.import_module(tqdm_module)

    assert pbar.tqdm is tqdm.tqdm


@pytest.mark.skipif(not has_tqdm, reason="requires tqdm")
@pytest.mark.parametrize("kw", [{}, {"bar_format": "{bar}"}])
def test_progress_bar_init_kwargs(kw):
    pbar = ProgressBar(**kw)

    assert "bar_format" in pbar.tqdm_kwargs

    if "bar_format" in kw:
        assert pbar.tqdm_kwargs["bar_format"] == kw["bar_format"]


@pytest.mark.skipif(not has_tqdm, reason="requires tqdm")
def test_progress_bar_init_error(in_dataset, model):
    with pytest.raises(ValueError, match=r".*not supported.*"):
        ProgressBar(frontend="invalid_frontend")
