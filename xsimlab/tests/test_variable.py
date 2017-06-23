import unittest
from collections import OrderedDict

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
            self.assertEqual(var.allowed_dims, ((),))

        for allowed_dims in ('x', ['x'], tuple(['x'])):
            var = Variable(allowed_dims)
            self.assertEqual(var.allowed_dims, (('x',),))

        var = Variable([(), 'x', ('x', 'y')])
        self.assertEqual(var.allowed_dims, ((), ('x',), ('x', 'y')))

    def test_validators(self):
        # verify default validators + user supplied validators
        validator_func = lambda xr_var: xr_var is not None

        class MyVariable(Variable):
            default_validators = [validator_func]

        var = MyVariable((), validators=[validator_func])
        self.assertEqual(var.validators, [validator_func, validator_func])

    def test_validate_dimensions(self):
        var = Variable([(), 'x', ('x', 'y')])

        with self.assertRaisesRegex(ValidationError, 'invalid dimensions'):
            var.validate_dimensions(('x', 'z'))

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
        self.assertEqual(repr(var), expected_repr)


class TestForeignVariable(unittest.TestCase):

    def setUp(self):
        self.fvar = ForeignVariable(MyProcess, 'var')
        self.other_process = MyProcess()
        self.fvar._other_process_obj = self.other_process

    def test_ref_process(self):
        fvar = ForeignVariable(MyProcess, 'var')
        self.assertIs(fvar.ref_process, MyProcess)

        self.assertIs(self.fvar.ref_process, self.other_process)

    def test_ref_var(self):
        self.assertIs(self.fvar.ref_var, self.other_process.var)

    def test_properties(self):
        for prop in ('state', 'value', 'rate', 'change'):
            # test foreign getter
            setattr(self.other_process.var, prop, 1)
            self.assertEqual(getattr(self.fvar, prop), 1)

            # test foreign setter
            setattr(self.fvar, prop, 2)
            self.assertEqual(getattr(self.other_process.var, prop), 2)

    def test_repr(self):
        expected_repr = "<xsimlab.ForeignVariable (MyProcess.var)>"
        self.assertEqual(repr(self.fvar), expected_repr)

        fvar = ForeignVariable(MyProcess, 'var')
        self.assertEqual(repr(fvar), expected_repr)


class TestDiagnosticVariable(unittest.TestCase):

    def setUp(self):
        self.process = MyProcess()
        self.process.diag.assign_process_obj(self.process)
        self.process.diag2.assign_process_obj(self.process)

    def test_decorator(self):
        self.assertIsInstance(self.process.diag, DiagnosticVariable)
        self.assertIsInstance(self.process.diag2, DiagnosticVariable)

        self.assertEqual(self.process.diag.description, "diagnostic.")
        self.assertEqual(self.process.diag2.attrs, {'units': 'm'})

    def test_state(self):
        self.assertEqual(self.process.diag.state, 1)
        self.assertEqual(self.process.diag2.state, 2)

    def test_call(self):
        self.assertEqual(self.process.diag(), 1)
        self.assertEqual(self.process.diag2(), 2)

    def test_repr(self):
        expected_repr = "<xsimlab.DiagnosticVariable>"
        self.assertEqual(repr(self.process.diag), expected_repr)
        self.assertEqual(repr(self.process.diag2), expected_repr)


class TestVariableList(unittest.TestCase):

    def test_constructor(self):
        with self.assertRaisesRegex(ValueError, "found variables mixed"):
            _ = VariableList([2, Variable(())])


class TestVariableGroup(unittest.TestCase):

    def test_iter(self):
        myprocess = MyProcess()
        myprocess2 = MyProcess2()
        myprocess3 = MyProcess3()

        with self.assertRaisesRegex(ValueError, "cannot retrieve variables"):
            _ = list(myprocess3.var)

        processes_dict = OrderedDict((('p1', myprocess), ('p2', myprocess2)))
        myprocess3.var._set_variables(processes_dict)

        expected = [myprocess.var, myprocess2.var]
        for var, proc in zip(myprocess3.var, processes_dict.values()):
            var._other_process_obj = proc

        fvar_list = [var.ref_var for var in myprocess3.var]
        self.assertEqual(fvar_list, expected)

    def test_repr(self):
        myprocess3 = MyProcess3()

        expected_repr = "<xsimlab.VariableGroup 'mygroup'>"
        self.assertEqual(repr(myprocess3.var), expected_repr)


class TestNumberVariable(unittest.TestCase):

    def test_validate(self):
        var = NumberVariable((), bounds=(0, 1))
        for data in (-1, [-1, 0], [-1, 1], [0, 2], 2):
            xr_var = var.to_xarray_variable(data)
            with self.assertRaisesRegex(ValidationError, "out of bounds"):
                var.validate(xr_var)

        for ib in [(True, False), (False, True), (False, False)]:
            var = NumberVariable((), bounds=(0, 1), inclusive_bounds=ib)
            xr_var = var.to_xarray_variable([0, 1])
            with self.assertRaisesRegex(ValidationError, "out of bounds"):
                var.validate(xr_var)


class TestFloatVariable(unittest.TestCase):

    def test_validators(self):
        var = FloatVariable(())

        for val in [1, 1.]:
            xr_var = xr.Variable((), val)
            var.run_validators(xr_var)

        xr_var = xr.Variable((), '1')
        with self.assertRaisesRegex(ValidationError, "invalid dtype"):
            var.run_validators(xr_var)


class TestIntegerVariable(unittest.TestCase):

    def test_validators(self):
        var = IntegerVariable(())

        xr_var = xr.Variable((), 1)
        var.run_validators(xr_var)

        for val in [1., '1']:
            xr_var = xr.Variable((), val)
            with self.assertRaisesRegex(ValidationError, "invalid dtype"):
                var.run_validators(xr_var)
