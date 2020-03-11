import pytest
import tqdm

from . import has_tqdm
import xsimlab as xs


@pytest.mark.skipif(not has_tqdm, reason="requires tqdm")
@pytest.mark.parametrize("test_input", ["auto", "console", "gui", "notebook"])
def test_progress_bar_init(test_input):
    pbar = xs.ProgressBar()
    assert test_input in ["auto", "console", "gui", "notebook"]


@pytest.mark.skipif(not has_tqdm, reason="requires tqdm")
def test_progress_bar_init_error(in_dataset, model):
    with pytest.raises(ValueError, match=r".*not supported.*"):
        pbar = xs.ProgressBar(frontend="invalid_frontend")
