import pytest
import sys
import tqdm

import xsimlab as xs
from .fixture_model import in_dataset, model


@pytest.mark.skipif(
    sys.version_info[0] == 3 and sys.version_info[1] != 7,
    reason="test performed in python3.7",
)
@pytest.mark.parametrize("test_input", ["auto", "console", "gui", "notebook"])
def test_progress_bar_init(test_input):
    pbar = xs.ProgressBar()
    assert test_input in pbar.env_list


@pytest.mark.skipif(
    sys.version_info[0] == 3 and sys.version_info[1] != 7,
    reason="test performed in python3.7",
)
def test_progress_bar_init_error(in_dataset, model):
    with pytest.raises(ValueError, match=r".*not supported.*"):
        pbar = xs.ProgressBar(frontend="invalid_frontend")
        out_ds = in_dataset.xsimlab.run(model=model, hooks=[pbar])
