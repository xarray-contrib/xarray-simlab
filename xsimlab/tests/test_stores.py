from tempfile import mkdtemp

import numpy as np
from numpy.testing import assert_array_equal
import pytest
import xarray as xr
import zarr

import xsimlab as xs
from xsimlab.stores import ZarrSimulationStore


def _bind_state(model):
    state = {}
    model.state = state

    for p_obj in model.values():
        p_obj.__xsimlab_state__ = state


@pytest.fixture
def store(in_dataset, model):
    _bind_state(model)

    zstore = ZarrSimulationStore(in_dataset, model)

    # init cache for the case of a single simulation
    zstore.init_var_cache(-1, zstore.model)

    return zstore


class TestZarrSimulationStore:
    @pytest.mark.parametrize(
        "zobject", [None, mkdtemp(), zarr.MemoryStore(), zarr.group()]
    )
    def test_constructor(self, in_dataset, model, zobject):
        store = ZarrSimulationStore(in_dataset, model)

        assert store.zgroup.store is not None

    def test_write_input_xr_dataset(self, in_dataset, store):
        store.write_input_xr_dataset()
        ds = xr.open_zarr(store.zgroup.store, chunks=None)

        xr.testing.assert_equal(ds, in_dataset)

        # check output variables attrs removed before saving input dataset
        assert not ds.xsimlab.output_vars

    def test_write_output_vars(self, in_dataset, store):
        model = store.model
        model.state[("profile", "u")] = np.array([1.0, 2.0, 3.0])
        model.state[("roll", "u_diff")] = np.array([-1.0, 1.0, 0.0])
        model.state[("add", "offset")] = 2.0

        store.write_output_vars(-1, 0)

        ztest = zarr.open_group(store.zgroup.store, mode="r")

        assert ztest.profile__u.shape == (in_dataset.clock.size, 3)
        assert_array_equal(ztest.profile__u[0], np.array([1.0, 2.0, 3.0]))

        assert ztest.roll__u_diff.shape == (in_dataset.out.size, 3)
        assert_array_equal(ztest.roll__u_diff[0], np.array([-1.0, 1.0, 0.0]))

        assert ztest.add__u_diff.shape == (in_dataset.out.size,)
        assert_array_equal(ztest.add__u_diff, np.array([2.0, np.nan, np.nan]))

        # test save master clock but not out clock
        store.write_output_vars(-1, 1)
        assert_array_equal(ztest.profile__u[1], np.array([1.0, 2.0, 3.0]))
        assert_array_equal(ztest.roll__u_diff[1], np.array([np.nan, np.nan, np.nan]))

        # test save no-clock outputs
        store.write_output_vars(-1, -1)
        assert_array_equal(ztest.profile__u_opp, np.array([-1.0, -2.0, -3.0]))

    def test_write_output_vars_error(self, store):
        model = store.model
        model.state[("profile", "u")] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        model.state[("roll", "u_diff")] = np.array([-1.0, 1.0, 0.0])
        model.state[("add", "offset")] = 2.0

        with pytest.raises(ValueError, match=r".*accepted dimension.*"):
            store.write_output_vars(-1, 0)

    def test_write_index_vars(self, store):
        store.model.state[("init_profile", "x")] = np.array([1.0, 2.0, 3.0])

        store.write_index_vars()
        ztest = zarr.open_group(store.zgroup.store, mode="r")

        assert_array_equal(ztest.x, np.array([1.0, 2.0, 3.0]))

    def test_resize_zarr_dataset(self):
        @xs.process
        class P:
            arr = xs.variable(dims="x", intent="out")

        model = xs.Model({"p": P})

        in_ds = xs.create_setup(
            model=model, clocks={"clock": [0, 1, 2]}, output_vars={"p__arr": "clock"},
        )

        _bind_state(model)
        store = ZarrSimulationStore(in_ds, model)
        store.init_var_cache(-1, model)

        for step, size in zip([0, 1, 2], [1, 3, 2]):
            model.state[("p", "arr")] = np.ones(size)
            store.write_output_vars(-1, step)

        ztest = zarr.open_group(store.zgroup.store, mode="r")

        expected = np.array(
            [[1.0, np.nan, np.nan], [1.0, 1.0, 1.0], [1.0, 1.0, np.nan]]
        )
        assert_array_equal(ztest.p__arr, expected)

    def test_encoding(self):
        @xs.process
        class P:
            v1 = xs.variable(dims="x", intent="out", encoding={"chunks": (10,)})
            v2 = xs.on_demand(dims="x", encoding={"fill_value": 0})
            v3 = xs.index(dims="x")

            @v2.compute
            def _get_v2(self):
                return [0]

        model = xs.Model({"p": P})

        in_ds = xs.create_setup(
            model=model,
            clocks={"clock": [0]},
            output_vars={"p__v1": None, "p__v2": None, "p__v3": None},
        )

        _bind_state(model)
        store = ZarrSimulationStore(
            in_ds,
            model,
            encoding={"p__v2": {"fill_value": -1}, "p__v3": {"compressor": None}},
        )
        store.init_var_cache(-1, model)

        model.state[("p", "v1")] = [0]
        model.state[("p", "v3")] = [0]
        store.write_output_vars(-1, -1)

        ztest = zarr.open_group(store.zgroup.store, mode="r")

        assert ztest.p__v1.chunks == (10,)
        # test encoding precedence ZarrSimulationStore > model variable
        assert ztest.p__v2.fill_value == -1
        assert ztest.p__v3.compressor is None

    def test_open_as_xr_dataset(self, store):
        model = store.model
        model.state[("profile", "u")] = np.array([1.0, 2.0, 3.0])
        model.state[("roll", "u_diff")] = np.array([-1.0, 1.0, 0.0])
        model.state[("add", "offset")] = 2.0

        store.write_output_vars(-1, 0)

        ds = store.open_as_xr_dataset()
        assert ds.profile__u.chunks is None

    def test_open_as_xr_dataset_chunks(self, in_dataset, model):
        _bind_state(model)
        store = ZarrSimulationStore(in_dataset, model, zobject=mkdtemp())
        store.init_var_cache(-1, model)

        model = store.model
        model.state[("profile", "u")] = np.array([1.0, 2.0, 3.0])
        model.state[("roll", "u_diff")] = np.array([-1.0, 1.0, 0.0])
        model.state[("add", "offset")] = 2.0

        store.write_output_vars(-1, 0)

        ds = store.open_as_xr_dataset()
        assert ds.profile__u.chunks is not None
