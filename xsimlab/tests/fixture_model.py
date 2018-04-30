from textwrap import dedent

import numpy as np
import pytest

import xsimlab as xs


@xs.process
class Profile(object):
    u = xs.variable(dims='x', description='quantity u', intent='inout')
    u_diffs = xs.group('diff')
    u_opp = xs.on_demand(dims='x')

    def initialize(self):
        self.u_change = np.zeros_like(self.u)

    def run_step(self, *args):
        self.u_change[:] = np.sum((d for d in self.u_diffs))

    def finalize_step(self):
        self.u += self.u_change

    def finalize(self):
        self.u[:] = 0.

    @u_opp.compute
    def _get_u_opposite(self):
        return -self.u


@xs.process
class InitProfile(object):
    n_points = xs.variable(description='nb. of profile points')
    u = xs.foreign(Profile, 'u', intent='out')

    def initialize(self):
        self.u_init = np.zeros(self.n_points)
        self.u_init[0] = 1.


@xs.process
class Roll(object):
    shift = xs.variable(description=('shift profile by a nb. of points'))
    u = xs.foreign(Profile, 'u')
    u_diff = xs.variable(dims='x', group='diff', intent='out')

    def run_step(self, *args):
        self.u_diff = np.roll(self.u, self.shift) - self.u


@xs.process
class Add(object):
    offset = xs.variable(description=('offset * dt added every time step '
                                      'to profile u'))
    u_diff = xs.variable(dims='x', group='diff', intent='out')

    def run_step(self, dt):
        self.u_diff = self.offset * dt


@xs.process
class AddOnDemand(object):
    offset = xs.variable(description='offset added to profile u')
    u_diff = xs.on_demand(group='diff')

    @u_diff.compute
    def _compute_u_diff(self):
        self.u_diff = self.offset


@pytest.fixture
def model():
    return xs.Model({'roll': Roll,
                     'add': AddOnDemand,
                     'profile': Profile,
                     'init_profile': InitProfile})


@pytest.fixture(scope='session')
def model_repr():
    return dedent("""\
    <xsimlab.Model (4 processes, 3 inputs)>
    init_profile
        n_points     [in] nb. of profile points
    roll
        shift        [in] shift profile by a nb. of points
    add
        offset       [in] offset added to profile u
    profile""")


@pytest.fixture
def no_init_model():
    return xs.Model({'roll': Roll,
                     'add': Add,
                     'profile': Profile})


@pytest.fixture
def simple_model():
    return xs.Model({'roll': Roll,
                     'profile': Profile})
