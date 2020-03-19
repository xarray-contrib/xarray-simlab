import numpy as np
import pytest
from numpy.testing import assert_array_equal
from xarray.testing import assert_identical

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


class TestBaseDriver:
    def test_bind_state(self, base_driver):
        base_driver.state[("init_profile", "n_points")] = 10
        assert base_driver.model.init_profile.n_points == 10

    def test_set_state_ignore(self, base_driver):
        input_vars = {("not-a-model", "input"): 0}
        base_driver.initialize_state(input_vars)

        assert ("not-a-model", "input") not in base_driver.state

    def test_set_state_copy(self, base_driver):
        n = np.array([1, 2, 3, 4])
        input_vars = {("add", "offset"): n}
        base_driver.initialize_state(input_vars)

        actual = base_driver.state[("add", "offset")]
        assert_array_equal(actual, n)
        assert actual is not n

    def test_set_state_converter(self, base_driver):
        input_vars = {("init_profile", "n_points"): 10.2}
        base_driver.initialize_state(input_vars)

        actual = base_driver.state[("init_profile", "n_points")]
        assert actual == 10
        assert type(actual) is int

    def test_initialize_state(self, base_driver):
        input_vars = {("init_profile", "n_points"): 10}
        base_driver.initialize_state(input_vars)

        assert base_driver.state[("init_profile", "n_points")] == 10

    def test_update_state(self, base_driver):
        input_vars = {("init_profile", "n_points"): 10}

        with pytest.raises(RuntimeError, match=r".* static variable .*"):
            base_driver.update_state(input_vars)

    def test_validate(self, base_driver):
        base_driver.state[("roll", "shift")] = 2.5

        with pytest.raises(TypeError, match=r".*'int'.*"):
            base_driver.validate(["roll"])

    def test_run_model(self, base_driver):
        with pytest.raises(NotImplementedError):
            base_driver.run_model()


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
            assert_array_equal(actual, expected)
            assert not np.isscalar(actual)

    def test_get_output_dataset(self, in_dataset, xarray_driver):
        # regression test: make sure a copy of input dataset is used
        out_ds = xarray_driver.run_model()
        assert not in_dataset.identical(out_ds)

    def test_run_model(self, in_dataset, out_dataset, xarray_driver):
        out_ds_actual = xarray_driver.run_model()

        # skip attributes added by xr.open_zarr from check
        for xr_var in out_ds_actual.variables.values():
            xr_var.attrs.pop("_FillValue", None)

        assert out_ds_actual is not out_dataset
        assert_identical(out_ds_actual.load(), out_dataset)

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
