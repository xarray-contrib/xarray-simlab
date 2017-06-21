import unittest

import xarray as xr

from xsimlab import Variable, ValidationError


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
