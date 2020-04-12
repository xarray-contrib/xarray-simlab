import pytest
from dask.distributed import Client
import xarray as xr
import numpy as np
import zarr

import xsimlab as xs
from xsimlab import xr_accessor, create_setup
from xsimlab.xr_accessor import (
    as_variable_key,
    _flatten_inputs,
    _flatten_outputs,
    _maybe_get_model_from_context,
)

from . import use_dask_schedulers
from .fixture_model import Profile, Roll


@pytest.fixture(params=[True, False])
def parallel(request):
    return request.param


@pytest.yield_fixture(scope="module", params=use_dask_schedulers)
def scheduler(request):
    if request.param.startswith("distributed"):
        kw = {"dashboard_address": None}

        if request.param == "distributed-threads":
            kw["processes"] = False

        client = Client(**kw)
        yield client
        client.close()

    else:
        yield request.param


def test_filter_accessor():
    ds = xr.Dataset(
        data_vars={"var1": ("x", [1, 2]), "var2": ("y", [3, 4])},
        coords={"x": [1, 2], "y": [3, 4]},
    )
    filtered = ds.filter(lambda var: "x" in var.dims)
    assert "var1" in filtered and "var2" not in filtered
    assert "x" in filtered.coords and "y" not in filtered.coords


def test_get_model_from_context(model):
    with pytest.raises(ValueError, match="No model found in context"):
        _maybe_get_model_from_context(None)

    with model as m:
        assert _maybe_get_model_from_context(None) is m

    with pytest.raises(TypeError, match=r".*is not an instance of.*"):
        _maybe_get_model_from_context("not a model")


def test_as_variable_key():
    assert as_variable_key(("foo", "bar")) == ("foo", "bar")
    assert as_variable_key("foo__bar") == ("foo", "bar")
    assert as_variable_key("foo_bar__baz") == ("foo_bar", "baz")

    with pytest.raises(ValueError) as excinfo:
        as_variable_key("foo__bar__baz")
    assert "not a valid input variable" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        as_variable_key("foo__")
    assert "not a valid input variable" in str(excinfo.value)


@pytest.mark.parametrize(
    "input_vars,expected",
    [
        ({("foo", "bar"): 1}, {("foo", "bar"): 1}),
        ({"foo__bar": 1}, {("foo", "bar"): 1}),
        ({"foo": {"bar": 1}}, {("foo", "bar"): 1}),
    ],
)
def test_flatten_inputs(input_vars, expected):
    assert _flatten_inputs(input_vars) == expected


@pytest.mark.parametrize(
    "output_vars,expected",
    [
        ({"clock": "foo__bar"}, {"clock": [("foo", "bar")]}),
        ({"clock": ("foo", "bar")}, {"clock": [("foo", "bar")]}),
        ({"clock": [("foo", "bar")]}, {"clock": [("foo", "bar")]}),
        ({"clock": [("foo__bar")]}, {"clock": [("foo", "bar")]}),
        ({"clock": {"foo": "bar"}}, {"clock": [("foo", "bar")]}),
        (
            {"clock": {"foo": ["bar", "baz"]}},
            {"clock": [("foo", "bar"), ("foo", "baz")]},
        ),
    ],
)
def test_flatten_outputs(output_vars, expected):
    assert _flatten_outputs(output_vars) == expected


def test_flatten_outputs_error():
    with pytest.raises(ValueError) as excinfo:
        _flatten_outputs({"clock": 2})
    assert "Cannot interpret" in str(excinfo.value)


class TestSimlabAccessor:

    _clock_key = xr_accessor.SimlabAccessor._clock_key
    _master_clock_key = xr_accessor.SimlabAccessor._master_clock_key
    _output_vars_key = xr_accessor.SimlabAccessor._output_vars_key

    def test_clock_coords(self):
        ds = xr.Dataset(
            coords={
                "mclock": (
                    "mclock",
                    [0, 1, 2],
                    {self._clock_key: 1, self._master_clock_key: 1},
                ),
                "sclock": ("sclock", [0, 2], {self._clock_key: 1}),
                "no_clock": ("no_clock", [3, 4]),
            }
        )
        assert set(ds.xsimlab.clock_coords) == {"mclock", "sclock"}

    def test_clock_sizes(self):
        ds = xr.Dataset(
            coords={
                "clock1": ("clock1", [0, 1, 2], {self._clock_key: 1}),
                "clock2": ("clock2", [0, 2], {self._clock_key: 1}),
                "no_clock": ("no_clock", [3, 4]),
            }
        )

        assert ds.xsimlab.clock_sizes == {"clock1": 3, "clock2": 2}

    def test_master_clock_dim(self):
        attrs = {self._clock_key: 1, self._master_clock_key: 1}
        ds = xr.Dataset(coords={"clock": ("clock", [1, 2], attrs)})

        assert ds.xsimlab.master_clock_dim == "clock"
        assert ds.xsimlab._master_clock_dim == "clock"  # cache
        assert ds.xsimlab.master_clock_dim == "clock"  # get cached value

        ds = xr.Dataset()
        assert ds.xsimlab.master_clock_dim is None

    def test_nsteps(self):
        attrs = {self._clock_key: 1, self._master_clock_key: 1}
        ds = xr.Dataset(coords={"clock": ("clock", [1, 2, 3], attrs)})

        assert ds.xsimlab.nsteps == 2

        ds = xr.Dataset()
        assert ds.xsimlab.nsteps == 0

    def test_get_output_save_steps(self):
        attrs = {self._clock_key: 1, self._master_clock_key: 1}
        ds = xr.Dataset(
            coords={
                "clock": ("clock", [0, 1, 2, 3, 4], attrs),
                "clock1": ("clock1", [0, 2, 4], {self._clock_key: 1}),
                "clock2": ("clock2", [0, 4], {self._clock_key: 1}),
            }
        )

        expected = xr.Dataset(
            coords={"clock": ("clock", [0, 1, 2, 3, 4], attrs)},
            data_vars={
                "clock1": ("clock", [True, False, True, False, True]),
                "clock2": ("clock", [True, False, False, False, True]),
            },
        )

        xr.testing.assert_identical(ds.xsimlab.get_output_save_steps(), expected)

    def test_set_input_vars(self, model, in_dataset):
        in_vars = {
            ("init_profile", "n_points"): 5,
            ("roll", "shift"): 1,
            ("add", "offset"): ("clock", [1, 2, 3, 4, 5]),
        }

        ds = xr.Dataset(coords={"clock": [0, 2, 4, 6, 8]})
        ds.xsimlab._set_input_vars(model, in_vars)

        for vname in ("init_profile__n_points", "roll__shift", "add__offset"):
            # xr.testing.assert_identical also checks attrs of coordinates
            # (not needed here)
            xr.testing.assert_equal(ds[vname], in_dataset[vname])
            assert ds[vname].attrs == in_dataset[vname].attrs

        # test errors
        in_vars[("not_an", "input_var")] = None

        with pytest.raises(KeyError) as excinfo:
            ds.xsimlab._set_input_vars(model, in_vars)
        assert "not valid key(s)" in str(excinfo.value)

        # test implicit dimension label
        in_vars = {("add", "offset"): [1, 2, 3, 4, 5]}
        ds.xsimlab._set_input_vars(model, in_vars)

        assert ds["add__offset"].dims == ("x",)

        # test implicit dimension label error
        in_vars = {("roll", "shift"): [1, 2]}

        with pytest.raises(TypeError, match=r"Could not get dimension labels.*"):
            ds.xsimlab._set_input_vars(model, in_vars)

    def test_update_clocks(self, model):
        ds = xr.Dataset()
        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab.update_clocks(model=model, clocks={})
        assert "Cannot determine which clock" in str(excinfo.value)

        ds = xr.Dataset()
        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab.update_clocks(
                model=model, clocks={"clock": [0, 1, 2], "out": [0, 2]}
            )
        assert "Cannot determine which clock" in str(excinfo.value)

        ds = xr.Dataset()
        with pytest.raises(KeyError) as excinfo:
            ds.xsimlab.update_clocks(
                model=model,
                clocks={"clock": [0, 1, 2]},
                master_clock="non_existing_clock_dim",
            )
        assert "Master clock dimension name" in str(excinfo.value)

        ds = xr.Dataset()
        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab.update_clocks(
                model=model, clocks={"clock": ("x", [0, 1, 2])},
            )
        assert "Invalid dimension" in str(excinfo.value)

        ds = xr.Dataset()
        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab.update_clocks(
                model=model,
                clocks={"clock": [0, 1, 2], "out": [0, 0.5, 2]},
                master_clock="clock",
            )
        assert "not synchronized" in str(excinfo.value)

        ds = xr.Dataset()
        ds = ds.xsimlab.update_clocks(model=model, clocks={"clock": [0, 1, 2]})
        assert ds.xsimlab.master_clock_dim == "clock"

        ds.clock.attrs[self._output_vars_key] = "profile__u"

        ds = ds.xsimlab.update_clocks(
            model=model,
            clocks={"clock": [0, 1, 2]},
            master_clock={
                "dim": "clock",
                "units": "days since 1-1-1 0:0:0",
                "calendar": "365_days",
            },
        )
        np.testing.assert_array_equal(ds.clock.values, [0, 1, 2])
        assert "units" in ds.clock.attrs
        assert "calendar" in ds.clock.attrs
        assert ds.clock.attrs[self._output_vars_key] == "profile__u"

        new_ds = ds.xsimlab.update_clocks(
            model=model, clocks={"clock2": [0, 0.5, 1, 1.5, 2]}, master_clock="clock2",
        )
        assert new_ds.xsimlab.master_clock_dim == "clock2"

        new_ds = ds.xsimlab.update_clocks(model=model, clocks={"out2": [0, 2]})
        assert new_ds.xsimlab.master_clock_dim == "clock"

    def test_update_vars(self, model, in_dataset):
        ds = in_dataset.xsimlab.update_vars(
            model=model,
            input_vars={("roll", "shift"): 2},
            output_vars={("profile", "u"): "out"},
        )

        assert not ds["roll__shift"].equals(in_dataset["roll__shift"])
        assert not ds["out"].identical(in_dataset["out"])

    def test_update_vars_promote_to_coords(self, model, in_dataset):
        # It should be possible to update an input variable with a dimension
        # label that cooresponds to its name (turned into a coordinate). This
        # should not raise any merge conflict error
        ds = in_dataset.xsimlab.update_vars(
            model=model, input_vars={"roll__shift": ("roll__shift", [1, 2])},
        )

        assert "roll__shift" in ds.coords

    def test_reset_vars(self, model, in_dataset):
        # add new variable
        ds = xr.Dataset().xsimlab.reset_vars(model)
        assert ds["roll__shift"] == 2

        # overwrite existing variable
        reset_ds = in_dataset.xsimlab.reset_vars(model)
        assert reset_ds["roll__shift"] == 2

    def test_filter_vars(self, simple_model, in_dataset):
        in_dataset["not_a_xsimlab_model_input"] = 1

        filtered_ds = in_dataset.xsimlab.filter_vars(model=simple_model)

        assert "add__offset" not in filtered_ds
        assert "not_a_xsimlab_model_input" not in filtered_ds
        assert sorted(filtered_ds.xsimlab.clock_coords) == ["clock", "out"]
        assert filtered_ds.out.attrs[self._output_vars_key] == "roll__u_diff"

        # test unchanged attributes in original dataset
        assert in_dataset.out.attrs[self._output_vars_key] == "roll__u_diff,add__u_diff"
        assert in_dataset.attrs[self._output_vars_key] == "profile__u_opp"

    def test_set_output_vars(self, model):
        ds = xr.Dataset()
        ds["clock"] = (
            "clock",
            [0, 2, 4, 6, 8],
            {self._clock_key: 1, self._master_clock_key: 1},
        )
        ds["out"] = ("out", [0, 4, 8], {self._clock_key: 1})
        ds["not_a_clock"] = ("not_a_clock", [0, 1])

        with pytest.raises(KeyError, match=r".*not valid key.*"):
            ds.xsimlab._set_output_vars(model, {("invalid", "var"): None})

        ds.xsimlab._set_output_vars(model, {("profile", "u_opp"): None})
        assert ds.attrs[self._output_vars_key] == "profile__u_opp"

        ds.xsimlab._set_output_vars(
            model, {("roll", "u_diff"): "out", ("add", "u_diff"): "out"}
        )
        expected = "roll__u_diff,add__u_diff"
        assert ds["out"].attrs[self._output_vars_key] == expected

        with pytest.raises(ValueError, match=r".not a valid clock.*"):
            ds.xsimlab._set_output_vars(model, {("profile", "u"): "not_a_clock"})

        with pytest.warns(FutureWarning):
            ds.xsimlab._set_output_vars(model, {None: ("profile", "u_opp")})

        with pytest.warns(FutureWarning):
            ds.xsimlab._set_output_vars(model, {"out": ("profile", "u_opp")})

    def test_set_output_object_vars(self):
        @xs.process
        class P:
            obj = xs.any_object()

        m = xs.Model({"p": P})
        ds = xr.Dataset()

        with pytest.raises(ValueError, match=r"Object variables can't be set.*"):
            ds.xsimlab._set_output_vars(m, {("p", "obj"): None})

    def test_output_vars(self, model):
        o_vars = {
            ("profile", "u_opp"): None,
            ("profile", "u"): "clock",
            ("roll", "u_diff"): "out",
            ("add", "u_diff"): "out",
        }

        ds = xs.create_setup(
            model=model,
            clocks={
                "clock": [0, 2, 4, 6, 8],
                "out": [0, 4, 8],
                # snapshot clock with no output variable
                "out2": [0, 8],
            },
            master_clock="clock",
            output_vars=o_vars,
        )

        assert ds.xsimlab.output_vars == o_vars

    def test_output_vars_by_clock(self, model):
        o_vars = {("roll", "u_diff"): "clock", ("add", "u_diff"): None}

        ds = xs.create_setup(
            model=model, clocks={"clock": [0, 2, 4, 6, 8]}, output_vars=o_vars,
        )

        expected = {"clock": [("roll", "u_diff")], None: [("add", "u_diff")]}

        assert ds.xsimlab.output_vars_by_clock == expected

    def test_run(self, model, in_dataset, out_dataset, parallel, scheduler):
        @xs.process
        class ProfileFix(Profile):
            # limitation of using distributed for single-model parallelism
            # internal instance attributes created and used in multiple stage
            # methods are not supported.
            u_change = xs.any_object()

        m = model.update_processes({"profile": ProfileFix})

        out_ds = in_dataset.xsimlab.run(model=m, parallel=parallel, scheduler=scheduler)

        xr.testing.assert_equal(out_ds.load(), out_dataset)

    def test_run_safe_mode(self, model, in_dataset):
        # safe mode True: ensure model is cloned (empty state)
        _ = in_dataset.xsimlab.run(model=model, safe_mode=True)
        assert model.state == {}

        # safe mode False: model not cloned (non empty state)
        _ = in_dataset.xsimlab.run(model=model, safe_mode=False)
        assert model.state != {}

    def test_run_check_dims(self):
        @xs.process
        class P:
            var = xs.variable(dims=["x", ("x", "y")])

        m = xs.Model({"p": P})

        arr = np.array([[1, 2], [3, 4]])

        in_ds = xs.create_setup(
            model=m,
            clocks={"clock": [1, 2]},
            input_vars={"p__var": (("y", "x"), arr)},
            output_vars={"p__var": None},
        )

        out_ds = in_ds.xsimlab.run(model=m, check_dims=None)
        actual = out_ds.p__var.values
        np.testing.assert_array_equal(actual, arr)

        with pytest.raises(ValueError, match=r"Invalid dimension.*"):
            in_ds.xsimlab.run(model=m, check_dims="strict")

        out_ds = in_ds.xsimlab.run(model=m, check_dims="transpose", safe_mode=False)
        actual = out_ds.p__var.values
        np.testing.assert_array_equal(actual, arr)
        np.testing.assert_array_equal(m.p.var, arr.transpose())

        in_ds2 = in_ds.xsimlab.update_vars(model=m, output_vars={"p__var": "clock"})
        # TODO: fix update output vars time-independet -> dependent
        # currently need the workaround below
        in_ds2.attrs = {}

        out_ds = in_ds2.xsimlab.run(model=m, check_dims="transpose")
        actual = out_ds.p__var.isel(clock=-1).values
        np.testing.assert_array_equal(actual, arr)

    def test_run_validate(self, model, in_dataset):
        in_dataset["roll__shift"] = 2.5

        # no input validation -> raises within np.roll()
        with pytest.raises(TypeError, match=r"slice indices must be integers.*"):
            in_dataset.xsimlab.run(model=model, validate=None)

        # input validation at initialization -> raises within attr.validate()
        with pytest.raises(TypeError, match=r".*'int'.*"):
            in_dataset.xsimlab.run(model=model, validate="inputs")

        in_dataset["roll__shift"] = ("clock", [1, 2.5, 1, 1, 1])

        # input validation at runtime -> raises within attr.validate()
        with pytest.raises(TypeError, match=r".*'int'.*"):
            in_dataset.xsimlab.run(model=model, validate="inputs")

        @xs.process
        class SetRollShift:
            shift = xs.foreign(Roll, "shift", intent="out")

            def initialize(self):
                self.shift = 2.5

        m = model.update_processes({"set_shift": SetRollShift})

        # no internal validation -> raises within np.roll()
        with pytest.raises(TypeError, match=r"slice indices must be integers.*"):
            in_dataset.xsimlab.run(model=m, validate="inputs")

        # internal validation -> raises within attr.validate()
        with pytest.raises(TypeError, match=r".*'int'.*"):
            in_dataset.xsimlab.run(model=m, validate="all")

    @pytest.mark.parametrize(
        "dims,data,clock",
        [
            ("batch", [1, 2], None),
            (("batch", "clock"), [[1, 1, 1], [2, 2, 2]], "clock"),
            (("batch", "x"), [[1, 1], [2, 2]], None),
        ],
    )
    def test_run_batch_dim(self, dims, data, clock, parallel, scheduler):
        @xs.process
        class P:
            in_var = xs.variable(dims=[(), "x"])
            out_var = xs.variable(dims=[(), "x"], intent="out")
            idx_var = xs.index(dims="x")

            def initialize(self):
                self.idx_var = [0, 1]

            def run_step(self):
                self.out_var = self.in_var * 2

        m = xs.Model({"p": P})

        in_ds = xs.create_setup(
            model=m,
            clocks={"clock": [0, 1, 2]},
            input_vars={"p__in_var": (dims, data)},
            output_vars={"p__out_var": clock},
        )

        out_ds = in_ds.xsimlab.run(
            model=m,
            batch_dim="batch",
            parallel=parallel,
            scheduler=scheduler,
            store=zarr.TempStore(),
        )

        if clock is None:
            coords = {}
        else:
            coords = {"clock": in_ds["clock"]}

        expected = xr.DataArray(data, dims=dims, coords=coords) * 2
        xr.testing.assert_equal(out_ds["p__out_var"], expected)


def test_create_setup(model, in_dataset):
    expected = xr.Dataset()
    actual = create_setup(model=model, fill_default=False)
    xr.testing.assert_identical(actual, expected)

    expected = xr.Dataset({"roll__shift": 2})
    actual = create_setup(model=model, fill_default=True)
    xr.testing.assert_equal(actual, expected)

    ds = create_setup(
        model=model,
        input_vars={
            "init_profile": {"n_points": 5},
            ("roll", "shift"): 1,
            "add__offset": ("clock", [1, 2, 3, 4, 5]),
        },
        clocks={"clock": [0, 2, 4, 6, 8], "out": [0, 4, 8]},
        master_clock="clock",
        output_vars={
            "profile__u": "clock",
            ("roll", "u_diff"): "out",
            ("add", "u_diff"): "out",
            "profile": {"u_opp": None},
        },
    )
    xr.testing.assert_identical(ds, in_dataset)
