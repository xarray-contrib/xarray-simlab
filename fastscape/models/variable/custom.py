"""
Custom Variable sub-classes.

"""

from functools import partial

import numpy as np

from .base import Variable, ValidationError


def dtype_validator(variable, expected_dtypes):
    if not isinstance(expected_dtypes, (list, tuple)):
        expected_dtypes = [expected_dtypes]

    test_dtype = any([np.issubdtype(variable.dtype, dtype)
                      for dtype in expected_dtypes])

    if not test_dtype:
        raise ValidationError(
            "invalid dtype, expected one between %s, found %r)"
            % ([np.dtype(dtype) for dtype in expected_dtypes], variable.dtype))


floating_validator = partial(dtype_validator,
                             expected_dtypes=[np.floating, np.integer])
integer_validator = partial(dtype_validator, expected_dtypes=np.integer)


class NumberVariable(Variable):
    """A variable that accept numbers as values."""

    def __init__(self, allowed_dims, bounds=(None, None),
                 inclusive_bounds=(True, True), **kwargs):
        """
        Parameters
        ----------
        allowed_dims : str or tuple or list
            Dimension label(s) allowed for the variable. An empty tuple
            corresponds to a scalar variable, a string or a 1-length tuple
            corresponds to a 1-d variable and a n-length tuple corresponds to a
            n-d variable. A list of str or tuple items may also be provided if
            the variable accepts different numbers of dimensions.
            This should not include a time dimension, which is always allowed.
        bounds : tuple or None, optional
            (lower, upper) value bounds (default: no bounds).
        inclusive_bounds : tuple, optional
            Whether the given (lower, upper) bounds are inclusive or not.
            Default: (True, True).
        **kwargs
            Keyword arguments of Variable.

        See Also
        --------
        Variable

        """
        super(NumberVariable, self).__init__(allowed_dims, **kwargs)
        self.bounds = bounds
        self.inclusive_bounds = inclusive_bounds

    def _check_bounds(self, xr_variable):
        vmin, vmax = self.bounds
        incmin, incmax = self.inclusive_bounds

        tmin = ((vmin is not None)
                and (xr_variable < vmin if incmin else xr_variable <= vmin))
        tmax = ((vmax is not None)
                and (xr_variable > vmax if incmax else xr_variable >= vmax))

        if np.any(tmin) or np.any(tmax):
            smin = '[' if incmin else ']'
            smax = ']' if incmax else '['
            strbounds = '{}{}, {}{}'.format(smin, vmin, vmax, smax)
            raise ValidationError("found value(s) out of bounds %s"
                                  % strbounds)

    def validate(self, xr_variable):
        self._check_bounds(xr_variable)
        super(NumberVariable, self).validate(xr_variable)


class FloatVariable(NumberVariable):
    """A variable that accepts floating point numbers as values."""

    default_validators = [floating_validator]


class IntegerVariable(NumberVariable):
    """A variable that accepts integer numbers as values."""

    default_validators = [integer_validator]
