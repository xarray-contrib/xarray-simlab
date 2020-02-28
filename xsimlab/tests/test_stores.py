from tempfile import mkdtemp

import numpy as np
from numpy.testing import assert_array_equal
import pytest
import xarray as xr
import zarr

from xsimlab.stores import (
    _get_output_steps_by_clock,
    ZarrOutputStore
)


def test_get_output_steps_by_clock(in_dataset):
    expected = {
        "clock": np.array([True, True, True, True, True]),
        "out": np.array([True, False, True, False, True]),
        None: np.array([False, False, False, False, True]),
    }

    actual = _get_output_steps_by_clock(in_dataset)

    assert actual.keys() == expected.keys()
    for k in expected:
        assert_array_equal(actual[k], expected[k])


class TestZarrOutputStore:

    @pytest.mark.parametrize(
        "zobject", [None, mkdtemp(), zarr.MemoryStore(), zarr.group()]
    )
    def test_constructor(self, in_dataset, model, zobject):
        out_store = ZarrOutputStore(in_dataset, model, zobject)

        assert out_store.zgroup.store is not None

    def test_write_input_xr_dataset(self, in_dataset, model):
        out_store = ZarrOutputStore(in_dataset, model, None)

        out_store.write_input_xr_dataset()
        ds = xr.open_zarr(out_store.zgroup.store, chunks=None)

        xr.testing.assert_equal(ds, in_dataset)

        # check output variables attrs removed before saving input dataset
        assert not ds.xsimlab.output_vars
