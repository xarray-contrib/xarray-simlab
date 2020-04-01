import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xsimlab as xs
from xsimlab.drivers import (
    BaseSimulationDriver,
    RuntimeContext,
    XarraySimulationDriver,
)


@pytest.fixture
def base_driver(model):
    return BaseSimulationDriver(model)


@pytest.fixture
def xarray_driver(in_dataset, model):
    return XarraySimulationDriver(in_dataset, model)


def test_runtime_context():
    with pytest.raises(KeyError, match=".*Invalid key.*"):
        RuntimeContext(invalid=False)

    assert len(RuntimeContext()) == len(RuntimeContext._context_keys)

    # test iter
    assert set(RuntimeContext()) == set(RuntimeContext._context_keys)

    assert repr(RuntimeContext()).startswith("RuntimeContext({")


class TestBaseSimulationDriver:
    def test_constructor(self, model):
        driver = BaseSimulationDriver(model)

        assert driver.model is model

    def test_run_model(self, base_driver):
        with pytest.raises(NotImplementedError):
            base_driver.run_model()

    def test_get_results(self, base_driver):
        with pytest.raises(NotImplementedError):
            base_driver.get_results()


class TestXarraySimulationDriver:
    def test_constructor(self, in_dataset, model):
        invalid_ds = in_dataset.drop("clock")
        with pytest.raises(ValueError, match=r"Missing master clock.*"):
            XarraySimulationDriver(invalid_ds, model)

        invalid_ds = in_dataset.drop("init_profile__n_points")
        with pytest.raises(KeyError, match=r"Missing variables.*"):
            XarraySimulationDriver(invalid_ds, model)

    @pytest.mark.parametrize(
        "value,is_scalar", [(1, True), (("x", [1, 1, 1, 1, 1]), False)]
    )
    def test_get_input_vars_scalar(self, in_dataset, xarray_driver, value, is_scalar):
        in_dataset["add__offset"] = value
        in_vars = xarray_driver._get_input_vars(in_dataset)

        actual = in_vars[("add", "offset")]
        expected = in_dataset["add__offset"].data

        if is_scalar:
            assert actual == expected
            assert np.isscalar(actual)

        else:
            np.testing.assert_array_equal(actual, expected)
            assert not np.isscalar(actual)

    def test_run_model_get_results(self, in_dataset, out_dataset, xarray_driver):
        xarray_driver.run_model()
        out_ds_actual = xarray_driver.get_results()

        # skip attributes added by xr.open_zarr from check
        for xr_var in out_ds_actual.variables.values():
            xr_var.attrs.pop("_FillValue", None)

        assert out_ds_actual is not out_dataset
        xr.testing.assert_identical(out_ds_actual.load(), out_dataset)

    def test_multi_index(self, in_dataset, model):
        # just check that multi-index pass through model run (reset -> zarr -> rebuilt)
        midx = pd.MultiIndex.from_tuples([(0, 1), (0, 2)], names=["a", "b"])

        in_dataset["dummy"] = ("dummy", midx)

        driver = XarraySimulationDriver(in_dataset, model)
        driver.run_model()
        out_dataset = driver.get_results()

        pd.testing.assert_index_equal(out_dataset.indexes["dummy"], midx)

    def test_static_var_as_scalar_coord(self, in_dataset, out_dataset, model):
        # test that a model input (static variable) given as a scalar coordinate
        # doesn't cause any trouble
        in_dataset.coords["init_profile__n_points"] = in_dataset[
            "init_profile__n_points"
        ]

        driver = XarraySimulationDriver(in_dataset, model)
        driver.run_model()
        out_ds = driver.get_results()

        xr.testing.assert_equal(out_ds.reset_coords(), out_dataset)

    def test_runtime_context(self, in_dataset, model):
        @xs.process
        class P:
            @xs.runtime(args="not_a_runtime_arg")
            def run_step(self, arg):
                pass

        m = model.update_processes({"p": P})

        driver = XarraySimulationDriver(in_dataset, m)

        with pytest.raises(KeyError, match="'not_a_runtime_arg'"):
            driver.run_model()
