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


@pytest.mark.parametrize("kw", [{}, {"desc": "custom description"}])
@pytest.mark.skipif(not has_tqdm, reason="requires tqdm")
def test_progress_bar_init_bar(kw):
    pbar = ProgressBar(**kw)
    pbar.init_bar(None, {"nsteps": 10}, {})

    assert pbar.pbar_model.format_dict["total"] == 12
    if kw:
        assert pbar.pbar_model.format_dict["prefix"] == "custom description"
    else:
        assert pbar.pbar_model.format_dict["prefix"] == "initialize"


@pytest.mark.skipif(not has_tqdm, reason="requires tqdm")
def test_progress_bar_update_init():
    pbar = ProgressBar()
    pbar.init_bar(None, {"nsteps": 10}, {})
    pbar.update_init(None, {}, {})

    assert pbar.pbar_model.format_dict["n"] == 1


@pytest.mark.skipif(not has_tqdm, reason="requires tqdm")
def test_progress_bar_update_run_step():
    pbar = ProgressBar()
    pbar.init_bar(None, {"nsteps": 10}, {})
    pbar.update_init(None, {}, {})
    pbar.update_run_step(None, {"nsteps": 10, "step": 1}, {})

    assert pbar.pbar_model.format_dict["n"] == 2
    assert pbar.pbar_model.format_dict["prefix"] == "run step 1/10"


@pytest.mark.skipif(not has_tqdm, reason="requires tqdm")
def test_progress_bar_update_finalize():
    pbar = ProgressBar()
    pbar.init_bar(None, {"nsteps": 10}, {})
    pbar.update_finalize(None, {}, {})

    assert pbar.pbar_model.format_dict["prefix"] == "finalize"


@pytest.mark.skipif(not has_tqdm, reason="requires tqdm")
def test_progress_bar_close_bar():
    pbar = ProgressBar()
    pbar.init_bar(None, {"nsteps": 10}, {})
    pbar.close_bar(None, {}, {})

    assert pbar.pbar_model.format_dict["n"] == 1
    assert pbar.pbar_model.format_dict["prefix"].startswith("Simulation finished")
