import unittest
from collections import OrderedDict

import pytest
import xarray as xr

from xsimlab.variable.base import (Variable, ForeignVariable, diagnostic,
                                   DiagnosticVariable, VariableList,
                                   VariableGroup, ValidationError)
from xsimlab.variable.custom import (NumberVariable, FloatVariable,
                                     IntegerVariable)
from xsimlab.process import Process


class MyProcess(Process):
    var = Variable((), group='mygroup')

    @diagnostic
    def diag(self):
        """diagnostic."""
        return 1

    @diagnostic({'units': 'm'})
    def diag2(self):
        """diagnostic 2."""
        return 2


class MyProcess2(Process):
    var = Variable((), group='mygroup')


class MyProcess3(Process):
    var = VariableGroup('mygroup')


class TestVariable(unittest.TestCase):

    def test_constructor(self):
        # verify allowed_dims
        for allowed_dims in (tuple(), list(), ''):
            var = Variable(allowed_dims)
            assert var.allowed_dims == ((),)

        for allowed_dims in ('x', ['x'], tuple(['x'])):
            var = Variable(allowed_dims)
            assert var.allowed_dims == (('x',),)

        var = Variable([(), 'x', ('x', 'y')])
        assert var.allowed_dims, ((), ('x',), ('x', 'y'))

    def test_validators(self):
        # verify default validators + user supplied validators
        validator_func = lambda xr_var: xr_var is not None

        class MyVariable(Variable):
            default_validators = [validator_func]

        var = MyVariable((), validators=[validator_func])
        assert var.validators == [validator_func, validator_func]

    def test_validate_dimensions(self):
        var = Variable([(), 'x', ('x', 'y')])

        with pytest.raises(ValidationError) as excinfo:
            var.validate_dimensions(('x', 'z'))
        assert 'invalid dimensions' in str(excinfo.value)

        var.validate_dimensions(('time', 'x'), ignore_dims=['time'])

    def test_to_xarray_variable(self):
        attrs = {'units': 'm'}
        description = 'x var'
        xr_var_attrs = attrs.copy()
        xr_var_attrs.update({'description': description})

        var = Variable('x', description=description, attrs=attrs)
        xr_var = var.to_xarray_variable(('x', [1, 2]))
        expected_xr_var = xr.Variable('x', data=[1, 2], attrs=xr_var_attrs)
        xr.testing.assert_identical(xr_var, expected_xr_var)

        var = Variable((), default_value=1)

        xr_var = var.to_xarray_variable(2)
        expected_xr_var = xr.Variable((), data=2)
        xr.testing.assert_identical(xr_var, expected_xr_var)

        # test default value
        xr_var = var.to_xarray_variable(None)
        expected_xr_var = xr.Variable((), data=1)
        xr.testing.assert_identical(xr_var, expected_xr_var)

        # test variable name
        xr_var = var.to_xarray_variable([1, 2])
        expected_xr_var = xr.Variable('this_variable', data=[1, 2])
        expected_xr_var = expected_xr_var.to_index_variable()
        xr.testing.assert_identical(xr_var, expected_xr_var)

    def test_repr(self):
        var = Variable(((), 'x', ('x', 'y')))
        expected_repr = "<xsimlab.Variable (), ('x'), ('x', 'y')>"
        assert repr(var) == expected_repr


class TestForeignVariable(unittest.TestCase):

    def setUp(self):
        self.fvar = ForeignVariable(MyProcess, 'var')
        self.other_process = MyProcess()
        self.fvar._other_process_obj = self.other_process

    def test_ref_process(self):
        fvar = ForeignVariable(MyProcess, 'var')
        assert fvar.ref_process is MyProcess
        assert self.fvar.ref_process is self.other_process

    def test_ref_var(self):
        assert self.fvar.ref_var is self.other_process.var

    def test_properties(self):
        for prop in ('state', 'value', 'rate', 'change'):
            # test foreign getter
            setattr(self.other_process.var, prop, 1)
            assert getattr(self.fvar, prop) == 1

            # test foreign setter
            setattr(self.fvar, prop, 2)
            assert getattr(self.other_process.var, prop) == 2

    def test_repr(self):
        expected_repr = "<xsimlab.ForeignVariable (MyProcess.var)>"
        assert repr(self.fvar) == expected_repr

        fvar = ForeignVariable(MyProcess, 'var')
        assert repr(fvar) == expected_repr


class TestDiagnosticVariable(unittest.TestCase):

    def setUp(self):
        self.process = MyProcess()
        self.process.diag.assign_process_obj(self.process)
        self.process.diag2.assign_process_obj(self.process)

    def test_decorator(self):
        assert isinstance(self.process.diag, DiagnosticVariable)
        assert isinstance(self.process.diag2, DiagnosticVariable)

        assert self.process.diag.description == "diagnostic."
        assert self.process.diag2.attrs == {'units': 'm'}

    def test_state(self):
        assert self.process.diag.state == 1
        assert self.process.diag2.state == 2

    def test_call(self):
        assert self.process.diag() == 1
        assert self.process.diag2() == 2

    def test_repr(self):
        expected_repr = "<xsimlab.DiagnosticVariable>"
        assert repr(self.process.diag) == expected_repr
        assert repr(self.process.diag2) == expected_repr


class TestVariableList(unittest.TestCase):

    def test_constructor(self):
        var_list = VariableList([Variable(()), Variable(('x'))])
        assert isinstance(var_list, tuple)

        with pytest.raises(ValueError) as excinfo:
            _ = VariableList([2, Variable(())])
        assert "found variables mixed" in str(excinfo.value)


class TestVariableGroup(unittest.TestCase):

    def test_iter(self):
        myprocess = MyProcess()
        myprocess2 = MyProcess2()
        myprocess3 = MyProcess3()

        with pytest.raises(ValueError) as excinfo:
            _ = list(myprocess3.var)
        assert "cannot retrieve variables" in str(excinfo.value)

        processes_dict = OrderedDict([('p1', myprocess),
                                      ('p2', myprocess2),
                                      ('p3', myprocess3)])
        myprocess3.var._set_variables(processes_dict)

        expected = [myprocess.var, myprocess2.var]
        for var, proc in zip(myprocess3.var, processes_dict.values()):
            var._other_process_obj = proc

        fvar_list = [var.ref_var for var in myprocess3.var]
        assert fvar_list == expected

    def test_repr(self):
        myprocess3 = MyProcess3()

        expected_repr = "<xsimlab.VariableGroup 'mygroup'>"
        assert repr(myprocess3.var) == expected_repr


class TestNumberVariable(unittest.TestCase):

    def test_validate(self):
        var = NumberVariable((), bounds=(0, 1))
        for data in (-1, [-1, 0], [-1, 1], [0, 2], 2):
            xr_var = var.to_xarray_variable(data)
            with pytest.raises(ValidationError) as excinfo:
                var.validate(xr_var)
            assert "out of bounds" in str(excinfo.value)

        for ib in [(True, False), (False, True), (False, False)]:
            var = NumberVariable((), bounds=(0, 1), inclusive_bounds=ib)
            xr_var = var.to_xarray_variable([0, 1])
            with pytest.raises(ValidationError) as excinfo:
                var.validate(xr_var)
            assert "out of bounds" in str(excinfo.value)


class TestFloatVariable(unittest.TestCase):

    def test_validators(self):
        var = FloatVariable(())

        for val in [1, 1.]:
            xr_var = xr.Variable((), val)
            var.run_validators(xr_var)

        xr_var = xr.Variable((), '1')
        with pytest.raises(ValidationError) as excinfo:
            var.run_validators(xr_var)
        assert "invalid dtype" in str(excinfo.value)


class TestIntegerVariable(unittest.TestCase):

    def test_validators(self):
        var = IntegerVariable(())

        xr_var = xr.Variable((), 1)
        var.run_validators(xr_var)

        for val in [1., '1']:
            xr_var = xr.Variable((), val)
            with pytest.raises(ValidationError) as excinfo:
                var.run_validators(xr_var)
            assert "invalid dtype" in str(excinfo.value)
