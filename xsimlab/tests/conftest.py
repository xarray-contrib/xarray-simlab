"""
This module provides a set of Process subclasses and pytest fixtures that are
used across the tests.
"""
from textwrap import dedent

import pytest
import numpy as np
import xarray as xr

from xsimlab.variable.base import (Variable, ForeignVariable, VariableGroup,
                                   VariableList, diagnostic)
from xsimlab.process import Process
from xsimlab.model import Model
from xsimlab.xr_accessor import SimlabAccessor
from xsimlab.xr_interface import DatasetModelInterface


class ExampleProcess(Process):
    """A full example of process interface.
    """
    var = Variable((), provided=True)
    var_list = VariableList([Variable('x'), Variable(((), 'x'))])
    var_group = VariableGroup('group')
    no_var = 'this is not a variable object'

    class Meta:
        time_dependent = False

    @diagnostic
    def diag(self):
        return 1


@pytest.fixture
def process():
    return ExampleProcess()


@pytest.fixture(scope='session')
def process_repr():
    return dedent("""\
    Variables:
      * diag       DiagnosticVariable
      * var        Variable ()
        var_group  VariableGroup 'group'
        var_list   VariableList
        -          Variable ('x')
        -          Variable (), ('x')
    Meta:
        time_dependent: False""")


class Grid(Process):
    x_size = Variable((), optional=True, description='grid size')
    x = Variable('x', provided=True)

    class Meta:
        time_dependent = False

    def validate(self):
        if np.asscalar(self.x_size.value) is None:
            self.x_size.value = 5

    def initialize(self):
        self.x.value = np.arange(self.x_size.value)


class Quantity(Process):
    quantity = Variable('x', description='a quantity')
    all_effects = VariableGroup('effect')

    def run_step(self, *args):
        self.quantity.change = sum((var.value for var in self.all_effects))

    def finalize_step(self):
        self.quantity.state += self.quantity.change

    @diagnostic
    def some_derived_quantity(self):
        """some derived quantity."""
        return 1

    @diagnostic({'units': 'm'})
    def other_derived_quantity(self):
        """other derived quantity."""
        return 2


class SomeProcess(Process):
    some_param = Variable((), default_value=1, description='some parameter')
    x = ForeignVariable(Grid, 'x')
    quantity = ForeignVariable(Quantity, 'quantity')
    some_effect = Variable('x', group='effect', provided=True)

    # SomeProcess always appears before OtherProcess in a model
    copy_param = Variable((), provided=True)

    def initialize(self):
        self.copy_param.value = self.some_param.value

    def run_step(self, dt):
        self.some_effect.value = self.x.value * self.some_param.value + dt

    def finalize(self):
        self.some_effect.rate = 0


class OtherProcess(Process):
    x = ForeignVariable(Grid, 'x')
    quantity = ForeignVariable(Quantity, 'quantity')
    other_param = Variable((), default_value=1, description='other parameter')
    other_effect = Variable('x', group='effect', provided=True)

    # OtherProcess should always appear after SomeProcess in a model
    copy_param = ForeignVariable(SomeProcess, 'copy_param')

    def run_step(self, dt):
        self.other_effect.value = self.x.value * self.copy_param.value - dt

    @diagnostic
    def x2(self):
        return self.x * 2


class PlugProcess(Process):
    meta_param = Variable(())
    some_param = ForeignVariable(SomeProcess, 'some_param', provided=True)
    x = ForeignVariable(Grid, 'x')

    def run_step(self, *args):
        self.some_param.value = self.meta_param.value


@pytest.fixture
def model():
    model = Model({'grid': Grid,
                   'some_process': SomeProcess,
                   'other_process': OtherProcess,
                   'quantity': Quantity})
    return model


@pytest.fixture(scope='session')
def model_repr():
    return dedent("""\
        <xsimlab.Model (4 processes, 4 inputs)>
        grid
            x_size       (in) grid size
        some_process
            some_param   (in) some parameter
        other_process
            other_param  (in) other parameter
        quantity
            quantity     (in) a quantity""")


@pytest.fixture
def input_dataset():
    clock_key = SimlabAccessor._clock_key
    mclock_key = SimlabAccessor._master_clock_key
    svars_key = SimlabAccessor._snapshot_vars_key

    ds = xr.Dataset()

    ds['clock'] = ('clock', [0, 2, 4, 6, 8],
                   {clock_key: np.uint8(True), mclock_key: np.uint8(True)})
    ds['out'] = ('out', [0, 4, 8], {clock_key: np.uint8(True)})

    ds['grid__x_size'] = ((), 10, {'description': 'grid size'})
    ds['quantity__quantity'] = ('x', np.zeros(10),
                                {'description': 'a quantity'})
    ds['some_process__some_param'] = ((), 1, {'description': 'some parameter'})
    ds['other_process__other_param'] = ('clock', [1, 2, 3, 4, 5],
                                        {'description': 'other parameter'})

    ds['clock'].attrs[svars_key] = 'quantity__quantity'
    ds['out'].attrs[svars_key] = ('other_process__other_effect,'
                                  'some_process__some_effect')
    ds.attrs[svars_key] = 'grid__x'

    return ds


@pytest.fixture
def ds_model_interface(model, input_dataset):
    return DatasetModelInterface(model, input_dataset)
