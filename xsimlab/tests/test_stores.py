import numpy as np
import pytest
import xarray as xr
import zarr

import xsimlab as xs
from xsimlab.stores import DummyLock, ZarrSimulationStore


@pytest.fixture(params=["directory", zarr.MemoryStore])
def zobject(request, tmpdir):
    if request.param == "directory":
        return str(tmpdir)
    else:
        return request.param()


@pytest.fixture
def in_ds(in_dataset, model):
    # need to test for scalar output variables
    return in_dataset.xsimlab.update_vars(
        model=model, output_vars={"add__offset": None}
    )


@pytest.fixture
def store(in_ds, model, zobject):
    zstore = ZarrSimulationStore(in_ds, model, zobject=zobject)

    return zstore


@pytest.fixture
def in_ds_batch(in_ds):
    # a batch of two simulations
    in_ds["roll__shift"] = ("batch", [1, 2])

    return in_ds


@pytest.fixture
def model_batch1(model):
    return model.clone()


@pytest.fixture
def model_batch2(model):
    return model.clone()


@pytest.fixture
def store_batch(in_ds_batch, model, zobject):
    return ZarrSimulationStore(in_ds_batch, model, zobject=zobject, batch_dim="batch")


def test_dummy_lock():
    lock = DummyLock()

    lock.acquire()
    assert not lock.locked()
    lock.release()

    with lock:
        assert not lock.locked()


class TestZarrSimulationStore:
    @pytest.mark.parametrize("zobj", [None, "dir", zarr.MemoryStore(), zarr.group()])
    def test_constructor(self, in_ds, model, zobj, tmpdir):
        if zobj == "dir":
            zobj = str(tmpdir)

        store = ZarrSimulationStore(in_ds, model, zobject=zobj)

        assert store.zgroup.store is not None
        assert store.batch_size == -1

        if zobj is None:
            assert store.in_memory is True

    def test_constructor_batch(self, store_batch):
        assert store_batch.batch_size == 2

    def test_constructor_conflict(self, in_ds, model):
        zgroup = zarr.group()
        zgroup.create_dataset("profile__u", shape=(1, 1))

        with pytest.raises(ValueError, match=r".*already contains.*"):
            ZarrSimulationStore(in_ds, model, zobject=zgroup)

    def test_write_input_xr_dataset(self, in_ds, store):
        store.write_input_xr_dataset()
        ds = xr.open_zarr(store.zgroup.store, chunks=None)

        # output variables removed
        del in_ds["add__offset"]

        xr.testing.assert_equal(ds, in_ds)

        # check output variables attrs removed before saving input dataset
        assert not ds.xsimlab.output_vars

    def test_write_output_vars(self, in_ds, store):
        model = store.model
        model.state[("profile", "u")] = np.array([1.0, 2.0, 3.0])
        model.state[("roll", "u_diff")] = np.array([-1.0, 1.0, 0.0])
        model.state[("add", "offset")] = 2.0

        store.write_output_vars(-1, 0)

        ztest = zarr.open_group(store.zgroup.store, mode="r")

        assert ztest.profile__u.shape == (in_ds.clock.size, 3)
        np.testing.assert_array_equal(ztest.profile__u[0], np.array([1.0, 2.0, 3.0]))

        assert ztest.roll__u_diff.shape == (in_ds.out.size, 3)
        np.testing.assert_array_equal(ztest.roll__u_diff[0], np.array([-1.0, 1.0, 0.0]))

        assert ztest.add__u_diff.shape == (in_ds.out.size,)
        np.testing.assert_array_equal(
            ztest.add__u_diff, np.array([2.0, np.nan, np.nan])
        )

        # test save master clock but not out clock
        store.write_output_vars(-1, 1)
        np.testing.assert_array_equal(ztest.profile__u[1], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(
            ztest.roll__u_diff[1], np.array([np.nan, np.nan, np.nan])
        )

        # test save no-clock outputs
        store.write_output_vars(-1, -1)
        np.testing.assert_array_equal(
            ztest.profile__u_opp, np.array([-1.0, -2.0, -3.0])
        )
        assert ztest.add__offset[()] == 2.0

    def test_write_output_vars_error(self, store):
        model = store.model
        model.state[("profile", "u")] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        model.state[("roll", "u_diff")] = np.array([-1.0, 1.0, 0.0])
        model.state[("add", "offset")] = 2.0

        with pytest.raises(ValueError, match=r".*accepted dimension.*"):
            store.write_output_vars(-1, 0)

    def test_write_output_vars_batch(self, store_batch, model_batch1, model_batch2):
        model_batch1.state[("profile", "u")] = np.array([1.0, 2.0, 3.0])
        model_batch2.state[("profile", "u")] = np.array([4.0, 5.0, 6.0])

        model_batch1.state[("roll", "u_diff")] = np.array([-1.0, 1.0, 0.0])
        model_batch2.state[("roll", "u_diff")] = np.array([0.0, 1.0, -1.0])

        model_batch1.state[("add", "offset")] = 2.0
        model_batch2.state[("add", "offset")] = 3.0

        store_batch.write_output_vars(0, 0, model=model_batch1)
        store_batch.write_output_vars(1, 0, model=model_batch2)

        ztest = zarr.open_group(store_batch.zgroup.store, mode="r")

        assert ztest.profile__u.ndim == 3
        np.testing.assert_array_equal(
            ztest.profile__u[:, 0, :], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )

        store_batch.write_output_vars(0, -1, model=model_batch1)
        store_batch.write_output_vars(1, -1, model=model_batch2)

        np.testing.assert_array_equal(ztest.add__offset[:], np.array([2.0, 3.0]))

        # test default chunk size along batch dim
        assert ztest.profile__u.chunks[0] == 1

    def test_write_index_vars(self, store):
        store.model.state[("init_profile", "x")] = np.array([1.0, 2.0, 3.0])

        store.write_index_vars()
        ztest = zarr.open_group(store.zgroup.store, mode="r")

        np.testing.assert_array_equal(ztest.x, np.array([1.0, 2.0, 3.0]))

    def test_write_index_vars_batch(self, store_batch, model_batch1):
        # ensure that no batch dim is created
        model_batch1.state[("init_profile", "x")] = np.array([1.0, 2.0, 3.0])

        store_batch.write_index_vars(model=model_batch1)
        ztest = zarr.open_group(store_batch.zgroup.store, mode="r")

        np.testing.assert_array_equal(ztest.x, np.array([1.0, 2.0, 3.0]))

    def test_resize_zarr_dataset(self):
        @xs.process
        class P:
            arr = xs.variable(dims="x", intent="out")

        model = xs.Model({"p": P})

        in_ds = xs.create_setup(
            model=model, clocks={"clock": [0, 1, 2]}, output_vars={"p__arr": "clock"},
        )

        store = ZarrSimulationStore(in_ds, model)

        for step, size in zip([0, 1, 2], [1, 3, 2]):
            model.state[("p", "arr")] = np.ones(size)
            store.write_output_vars(-1, step)

        ztest = zarr.open_group(store.zgroup.store, mode="r")

        expected = np.array(
            [[1.0, np.nan, np.nan], [1.0, 1.0, 1.0], [1.0, 1.0, np.nan]]
        )
        np.testing.assert_array_equal(ztest.p__arr, expected)

    def test_encoding(self):
        @xs.process
        class P:
            v1 = xs.variable(dims="x", intent="out", encoding={"dtype": np.int32})
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

        store = ZarrSimulationStore(
            in_ds,
            model,
            encoding={"p__v2": {"fill_value": -1}, "p__v3": {"chunks": (10,)}},
        )

        model.state[("p", "v1")] = [0]
        model.state[("p", "v3")] = [0]
        store.write_output_vars(-1, -1)

        ztest = zarr.open_group(store.zgroup.store, mode="r")

        assert ztest.p__v1.dtype == np.int32
        # test encoding precedence ZarrSimulationStore > model variable
        assert ztest.p__v2.fill_value == -1
        assert ztest.p__v3.chunks == (10,)

    def test_open_as_xr_dataset(self, store):
        model = store.model
        model.state[("profile", "u")] = np.array([1.0, 2.0, 3.0])
        model.state[("roll", "u_diff")] = np.array([-1.0, 1.0, 0.0])
        model.state[("add", "offset")] = 2.0

        store.write_output_vars(-1, 0)
        store.write_output_vars(-1, -1)

        ds = store.open_as_xr_dataset()

        if store.in_memory:
            assert ds.profile__u.chunks is None
        else:
            assert ds.profile__u.chunks is not None

            # test scalars still loaded in memory
            assert isinstance(ds.variables["add__offset"]._data, np.ndarray)
