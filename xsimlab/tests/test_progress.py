import pytest
import tqdm

import xsimlab as xs
from .fixture_model import in_dataset, model


def test_tqdm():
    pbar = tqdm.tqdm(range(5))

    assert pbar.desc == ""
    assert pbar.total == 5

    pbar.set_description_str("update")

    assert pbar.desc == "update"


def test_bar():
    pbar_default = xs.progress.ProgressBar()
    pbar_auto = xs.progress.ProgressBar(frontend="auto")
    pbar_console = xs.progress.ProgressBar(frontend="console")
    pbar_gui = xs.progress.ProgressBar(frontend="gui")
    pbar_notebook = xs.progress.ProgressBar(frontend="notebook")

    assert pbar_auto.frontend in pbar_default.env_list
    assert pbar_console.frontend in pbar_default.env_list
    assert pbar_gui.frontend in pbar_default.env_list
    assert pbar_notebook.frontend in pbar_default.env_list
    assert "{bar}" in pbar_default.pbar_dict["bar_format"]


def test_environment_error(in_dataset, model):
    with pytest.raises(ValueError) as env:
        pbar = xs.progress.ProgressBar(frontend="ipython")
        out_ds = in_dataset.xsimlab.run(model=model, hooks=[pbar])

    assert "not supported" in str(env.value)
