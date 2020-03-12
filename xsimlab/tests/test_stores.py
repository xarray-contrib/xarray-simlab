from tempfile import mkdtemp

import numpy as np
from numpy.testing import assert_array_equal
import pytest
import xarray as xr
import zarr

import xsimlab as xs
from xsimlab.stores import ZarrOutputStore


def _bind_store(model):
    store = {}
    model.store = store

    for p_obj in model.values():
        p_obj.__xsimlab_store__ = store


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

    def test_write_output_vars(self, in_dataset, model):
        _bind_store(model)
        out_store = ZarrOutputStore(in_dataset, model, None)

        model.store[("profile", "u")] = np.array([1.0, 2.0, 3.0])
        model.store[("roll", "u_diff")] = np.array([-1.0, 1.0, 0.0])
        model.store[("add", "offset")] = 2.0

        out_store.write_output_vars(0)

        ztest = zarr.open_group(out_store.zgroup.store, mode="r")

        assert ztest.profile__u.shape == (in_dataset.clock.size, 3)
        assert_array_equal(ztest.profile__u[0], np.array([1.0, 2.0, 3.0]))

        assert ztest.roll__u_diff.shape == (in_dataset.out.size, 3)
        assert_array_equal(ztest.roll__u_diff[0], np.array([-1.0, 1.0, 0.0]))

        assert ztest.add__u_diff.shape == (in_dataset.out.size,)
        assert_array_equal(ztest.add__u_diff, np.array([2.0, np.nan, np.nan]))

        # test save master clock but not out clock
        out_store.write_output_vars(1)
        assert_array_equal(ztest.profile__u[1], np.array([1.0, 2.0, 3.0]))
        assert_array_equal(ztest.roll__u_diff[1], np.array([np.nan, np.nan, np.nan]))

        # test save no-clock outputs
        out_store.write_output_vars(-1)
        assert_array_equal(ztest.profile__u_opp, np.array([-1.0, -2.0, -3.0]))

    def test_write_output_vars_error(self, in_dataset, model):
        _bind_store(model)
        out_store = ZarrOutputStore(in_dataset, model, None)

        model.store[("profile", "u")] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        model.store[("roll", "u_diff")] = np.array([-1.0, 1.0, 0.0])
        model.store[("add", "offset")] = 2.0

        with pytest.raises(ValueError, match=r".*accepted dimension.*"):
            out_store.write_output_vars(0)

    def test_write_index_vars(self, in_dataset, model):
        _bind_store(model)
        out_store = ZarrOutputStore(in_dataset, model, None)

        model.store[("init_profile", "x")] = np.array([1.0, 2.0, 3.0])

        out_store.write_index_vars()
        ztest = zarr.open_group(out_store.zgroup.store, mode="r")

        assert_array_equal(ztest.x, np.array([1.0, 2.0, 3.0]))

    def test_resize_zarr_dataset(self):
        @xs.process
        class P:
            arr = xs.variable(dims="x", intent="out")

        model = xs.Model({"p": P})

        in_ds = xs.create_setup(
            model=model, clocks={"clock": [0, 1, 2]}, output_vars={"p__arr": "clock"},
        )

        _bind_store(model)
        out_store = ZarrOutputStore(in_ds, model, None)

        for step in range(3):
            model.store[("p", "arr")] = np.ones(step + 1)
            out_store.write_output_vars(step)

        ztest = zarr.open_group(out_store.zgroup.store, mode="r")

        expected = np.array(
            [[1.0, np.nan, np.nan], [1.0, 1.0, np.nan], [1.0, 1.0, 1.0]]
        )
        assert_array_equal(ztest.p__arr, expected)

    def test_open_as_xr_dataset(self, in_dataset, model):
        _bind_store(model)
        out_store = ZarrOutputStore(in_dataset, model, None)

        model.store[("profile", "u")] = np.array([1.0, 2.0, 3.0])
        model.store[("roll", "u_diff")] = np.array([-1.0, 1.0, 0.0])
        model.store[("add", "offset")] = 2.0

        out_store.write_output_vars(0)

        ds = out_store.open_as_xr_dataset()
        assert ds.profile__u.chunks is None

    def test_open_as_xr_dataset_chunks(self, in_dataset, model):
        _bind_store(model)
        out_store = ZarrOutputStore(in_dataset, model, mkdtemp())

        model.store[("profile", "u")] = np.array([1.0, 2.0, 3.0])
        model.store[("roll", "u_diff")] = np.array([-1.0, 1.0, 0.0])
        model.store[("add", "offset")] = 2.0

        out_store.write_output_vars(0)

        ds = out_store.open_as_xr_dataset()
        assert ds.profile__u.chunks is not None
