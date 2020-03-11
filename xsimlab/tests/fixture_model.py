from collections import OrderedDict
from textwrap import dedent

import attr
import numpy as np
import xarray as xr
import pytest

import xsimlab as xs
from xsimlab.xr_accessor import SimlabAccessor


@xs.process
class Profile:
    u = xs.variable(dims="x", description="quantity u", intent="inout")
    u_diffs = xs.group("diff")
    u_opp = xs.on_demand(dims="x")

    def initialize(self):
        self.u_change = np.zeros_like(self.u)

    def run_step(self):
        self.u_change[:] = sum((d for d in self.u_diffs))

    def finalize_step(self):
        self.u += self.u_change

    def finalize(self):
        self.u[:] = 0.0

    @u_opp.compute
    def _get_u_opposite(self):
        return -self.u


@xs.process
class InitProfile:
    n_points = xs.variable(
        description="nb. of profile points", converter=int, static=True
    )

    x = xs.index(dims="x")
    u = xs.foreign(Profile, "u", intent="out")

    def initialize(self):
        self.x = np.arange(self.n_points)

        self.u = np.zeros(self.n_points)
        self.u[0] = 1.0


@xs.process
class Roll:
    shift = xs.variable(
        default=2,
        validator=attr.validators.instance_of(int),
        description=("shift profile by a nb. of points"),
        attrs={"units": "unitless"},
    )
    u = xs.foreign(Profile, "u")
    u_diff = xs.variable(dims="x", groups="diff", intent="out")

    def run_step(self):
        self.u_diff = np.roll(self.u, self.shift) - self.u


@xs.process
class Add:
    offset = xs.variable(
        description=("offset * dt added every time step " "to profile u")
    )
    u_diff = xs.variable(groups="diff", intent="out")

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        self.u_diff = self.offset * dt


@xs.process
class AddOnDemand:
    offset = xs.variable(dims=[(), "x"], description="offset added to profile u")
    u_diff = xs.on_demand(dims=[(), "x"], groups="diff")

    @u_diff.compute
    def _compute_u_diff(self):
        return self.offset


@pytest.fixture
def model():
    return xs.Model(
        {
            "roll": Roll,
            "add": AddOnDemand,
            "profile": Profile,
            "init_profile": InitProfile,
        }
    )


@pytest.fixture
def no_init_model():
    return xs.Model({"roll": Roll, "add": Add, "profile": Profile})


@pytest.fixture
def simple_model():
    return xs.Model({"roll": Roll, "profile": Profile})


@pytest.fixture(scope="session")
def simple_model_repr():
    return dedent(
        """\
    <xsimlab.Model (2 processes, 2 inputs)>
    roll
        shift       [in] shift profile by a nb. of points
    profile
        u        [inout] ('x',) quantity u
    """
    )


@pytest.fixture
def in_dataset():
    clock_key = SimlabAccessor._clock_key
    mclock_key = SimlabAccessor._master_clock_key
    svars_key = SimlabAccessor._output_vars_key

    ds = xr.Dataset()

    ds["clock"] = (
        "clock",
        [0, 2, 4, 6, 8],
        {clock_key: np.uint8(True), mclock_key: np.uint8(True)},
    )
    ds["out"] = ("out", [0, 4, 8], {clock_key: np.uint8(True)})

    ds["init_profile__n_points"] = (
        (),
        5,
        {"description": "nb. of profile points"},
    )
    ds["roll__shift"] = (
        (),
        1,
        OrderedDict(
            [
                ("description", "shift profile by a nb. of points"),
                ("units", "unitless"),
            ]
        ),
    )
    ds["add__offset"] = (
        "clock",
        [1, 2, 3, 4, 5],
        {"description": "offset added to profile u"},
    )

    ds["clock"].attrs[svars_key] = "profile__u"
    ds["out"].attrs[svars_key] = "roll__u_diff,add__u_diff"
    ds.attrs[svars_key] = "profile__u_opp"

    return ds


@pytest.fixture
def out_dataset(in_dataset):
    out_ds = in_dataset.copy()

    del out_ds.attrs[SimlabAccessor._output_vars_key]
    del out_ds.clock.attrs[SimlabAccessor._output_vars_key]
    del out_ds.out.attrs[SimlabAccessor._output_vars_key]
    out_ds["profile__u_opp"] = ("x", [-10.0, -10.0, -10.0, -10.0, -11.0])
    out_ds["profile__u"] = (
        ("clock", "x"),
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0, 1.0, 1.0],
                [3.0, 3.0, 4.0, 3.0, 3.0],
                [6.0, 6.0, 6.0, 7.0, 6.0],
                [10.0, 10.0, 10.0, 10.0, 11.0],
            ]
        ),
        {"description": "quantity u"},
    )
    out_ds["roll__u_diff"] = (
        ("out", "x"),
        np.array(
            [
                [-1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 1.0],
            ]
        ),
    )
    out_ds["add__u_diff"] = ("out", [1, 3, 4])

    out_ds["x"] = ("x", [0.0, 1.0, 2.0, 3.0, 4.0])

    return out_ds
