import pytest
import tqdm

import xsimlab as xs


def test_tqdm():
    pbar = tqdm.tqdm(range(5))

    assert pbar.desc == ""
    assert pbar.total == 5

    pbar.set_description_str("update")

    assert pbar.desc == "update"


def test_bar():
    pbar = xs.progress.ProgressBar()

    assert pbar.frontend == "auto"
    assert "{bar}" in pbar.pbar_dict["bar_format"]
