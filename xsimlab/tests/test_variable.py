from collections import OrderedDict

import pytest
import xarray as xr

from xsimlab.variable.base import (Variable, ForeignVariable,
                                   DiagnosticVariable, VariableList,
                                   ValidationError)
from xsimlab.variable.custom import (NumberVariable, FloatVariable,
                                     IntegerVariable)
from xsimlab.tests.conftest import SomeProcess, OtherProcess, Quantity


class TestVariable(object):

    def test_constructor(self):
        # verify allowed_dims
        for allowed_dims in (tuple(), list(), ''):
            var = Variable(allowed_dims)
            assert var.allowed_dims == ((),)

        for allowed_dims in ('x', ['x'], ('x')):
            var = Variable(allowed_dims)
            assert var.allowed_dims == (('x',),)

        var = Variable(('x', 'y'))
        assert var.allowed_dims == (('x', 'y'),)

        var = Variable([(), 'x', ('x', 'y')])
        assert var.allowed_dims == ((), ('x',), ('x', 'y'))

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
        var = Variable([(), 'x', ('x', 'y')])
        expected_repr = "<xsimlab.Variable (), ('x'), ('x', 'y')>"
        assert repr(var) == expected_repr


class TestForeignVariable(object):

    @pytest.fixture
    def some_process(self):
        """A instance of the process in which the original variable is
        declared.
        """
        return SomeProcess()

    @pytest.fixture
    def foreign_var_cls(self):
        """A foreign variable with no Process instance assigned."""
        return ForeignVariable(SomeProcess, 'some_param')

    @pytest.fixture
    def foreign_var(self, some_process):
        """A foreign variable with an assigned instance of SomeProcess."""
        fvar = ForeignVariable(SomeProcess, 'some_param')
        fvar._other_process_obj = some_process
        return fvar

    def test_ref_process(self, foreign_var, foreign_var_cls, some_process):
        assert foreign_var.ref_process is some_process
        assert foreign_var_cls.ref_process is SomeProcess

    def test_ref_var(self, foreign_var, some_process):
        assert foreign_var.ref_var is some_process.some_param

    def test_properties(self, foreign_var, some_process):
        for prop in ('state', 'value', 'rate', 'change'):
            # test foreign getter
            setattr(some_process.some_param, prop, 1)
            assert getattr(foreign_var, prop) == 1

            # test foreign setter
            setattr(foreign_var, prop, 2)
            assert getattr(some_process.some_param, prop) == 2

    def test_repr(self, foreign_var, foreign_var_cls):
        expected_repr = "<xsimlab.ForeignVariable (SomeProcess.some_param)>"
        assert repr(foreign_var) == expected_repr
        assert repr(foreign_var_cls) == expected_repr


class TestDiagnosticVariable(object):

    @pytest.fixture
    def quantity(self):
        """An instance of the Quantity process that defines some diagnostics."""
        proc = Quantity()
        proc.some_derived_quantity.assign_process_obj(proc)
        proc.other_derived_quantity.assign_process_obj(proc)
        return proc

    def test_decorator(self, quantity):
        assert isinstance(quantity.some_derived_quantity, DiagnosticVariable)
        assert isinstance(quantity.other_derived_quantity, DiagnosticVariable)

        assert quantity.some_derived_quantity.description == (
            "some derived quantity.")
        assert quantity.other_derived_quantity.attrs == {'units': 'm'}

    def test_state(self, quantity):
        assert quantity.some_derived_quantity.state == 1
        assert quantity.other_derived_quantity.state == 2

    def test_call(self, quantity):
        assert quantity.some_derived_quantity() == 1
        assert quantity.other_derived_quantity() == 2

    def test_repr(self, quantity):
        expected_repr = "<xsimlab.DiagnosticVariable>"
        assert repr(quantity.some_derived_quantity) == expected_repr
        assert repr(quantity.other_derived_quantity) == expected_repr


class TestVariableList(object):

    def test_constructor(self):
        var_list = VariableList([Variable(()), Variable(('x'))])
        assert isinstance(var_list, tuple)

        with pytest.raises(ValueError) as excinfo:
            _ = VariableList([2, Variable(())])
        assert "found variables mixed" in str(excinfo.value)


class TestVariableGroup(object):

    def test_iter(self):
        some_process = SomeProcess()
        other_process = OtherProcess()
        quantity = Quantity()

        with pytest.raises(ValueError) as excinfo:
            _ = list(quantity.all_effects)
        assert "cannot retrieve variables" in str(excinfo.value)

        processes_dict = OrderedDict([('some_process', some_process),
                                      ('other_process', other_process),
                                      ('quantity', quantity)])
        quantity.all_effects._set_variables(processes_dict)

        expected = [some_process.some_effect, other_process.other_effect]
        for var, proc in zip(quantity.all_effects, processes_dict.values()):
            var._other_process_obj = proc

        fvar_list = [var.ref_var for var in quantity.all_effects]
        assert fvar_list == expected

    def test_repr(self):
        quantity = Quantity()

        expected_repr = "<xsimlab.VariableGroup 'effect'>"
        assert repr(quantity.all_effects) == expected_repr


class TestNumberVariable(object):

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


class TestFloatVariable(object):

    def test_validators(self):
        var = FloatVariable(())

        for val in [1, 1.]:
            xr_var = xr.Variable((), val)
            var.run_validators(xr_var)

        xr_var = xr.Variable((), '1')
        with pytest.raises(ValidationError) as excinfo:
            var.run_validators(xr_var)
        assert "invalid dtype" in str(excinfo.value)


class TestIntegerVariable(object):

    def test_validators(self):
        var = IntegerVariable(())

        xr_var = xr.Variable((), 1)
        var.run_validators(xr_var)

        for val in [1., '1']:
            xr_var = xr.Variable((), val)
            with pytest.raises(ValidationError) as excinfo:
                var.run_validators(xr_var)
            assert "invalid dtype" in str(excinfo.value)
