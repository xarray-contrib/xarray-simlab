from collections import OrderedDict
from textwrap import dedent

import numpy as np
import xarray as xr
import pytest

import xsimlab as xs
from xsimlab.xr_accessor import SimlabAccessor


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
        self.u = np.zeros(self.n_points)
        self.u[0] = 1.


@xs.process
class Roll(object):
    shift = xs.variable(description=('shift profile by a nb. of points'),
                        attrs={'units': 'unitless'})
    u = xs.foreign(Profile, 'u')
    u_diff = xs.variable(dims='x', group='diff', intent='out')

    def run_step(self, *args):
        self.u_diff = np.roll(self.u, self.shift) - self.u


@xs.process
class Add(object):
    offset = xs.variable(description=('offset * dt added every time step '
                                      'to profile u'))
    u_diff = xs.variable(group='diff', intent='out')

    def run_step(self, dt):
        self.u_diff = self.offset * dt


@xs.process
class AddOnDemand(object):
    offset = xs.variable(description='offset added to profile u')
    u_diff = xs.on_demand(group='diff')

    @u_diff.compute
    def _compute_u_diff(self):
        return self.offset


@pytest.fixture
def model():
    return xs.Model({'roll': Roll,
                     'add': AddOnDemand,
                     'profile': Profile,
                     'init_profile': InitProfile})


@pytest.fixture
def no_init_model():
    return xs.Model({'roll': Roll,
                     'add': Add,
                     'profile': Profile})


@pytest.fixture
def simple_model():
    return xs.Model({'roll': Roll,
                     'profile': Profile})


@pytest.fixture(scope='session')
def simple_model_repr():
    return dedent("""\
    <xsimlab.Model (2 processes, 2 inputs)>
    roll
        shift       [in] shift profile by a nb. of points
    profile
        u        [inout] ('x',) quantity u
    """)


@pytest.fixture
def in_dataset():
    clock_key = SimlabAccessor._clock_key
    mclock_key = SimlabAccessor._master_clock_key
    svars_key = SimlabAccessor._output_vars_key

    ds = xr.Dataset()

    ds['clock'] = ('clock', [0, 2, 4, 6, 8],
                   {clock_key: np.uint8(True), mclock_key: np.uint8(True)})
    ds['out'] = ('out', [0, 4, 8], {clock_key: np.uint8(True)})

    ds['init_profile__n_points'] = (
        (), 5, {'description': 'nb. of profile points'})
    ds['roll__shift'] = (
        (), 1,
        OrderedDict([('description', 'shift profile by a nb. of points'),
                     ('units', 'unitless')]))
    ds['add__offset'] = (
        'clock', [1, 2, 3, 4, 5], {'description': 'offset added to profile u'})

    ds['clock'].attrs[svars_key] = 'profile__u'
    ds['out'].attrs[svars_key] = ('roll__u_diff,'
                                  'add__u_diff')
    ds.attrs[svars_key] = 'profile__u_opp'

    return ds


@pytest.fixture
def out_dataset(in_dataset):
    out_ds = in_dataset

    del out_ds.attrs[SimlabAccessor._output_vars_key]
    del out_ds.clock.attrs[SimlabAccessor._output_vars_key]
    del out_ds.out.attrs[SimlabAccessor._output_vars_key]
    out_ds['profile__u_opp'] = ('x', [-10., -10., -10., -10., -11.])
    out_ds['profile_u'] = (
        ('clock', 'x'),
        np.array([[1., 0., 0., 0., 0.],
                  [1., 2., 1., 1., 1.],
                  [3., 3., 4., 3., 3.],
                  [6., 6., 6., 7., 6.],
                  [10., 10., 10., 10., 11.]]),
        {'description': 'quantity u'}
    )
    out_ds['roll_u_diff'] = (
        ('out', 'x'),
        np.array([[-1., 1., 0., 0., 0.],
                  [0., 0., -1., 1., 0.],
                  [0., 0., 0., -1., 1.]])
    )
    out_ds['add__u_diff'] = ('out', [1, 3, 4])

    return out_ds
