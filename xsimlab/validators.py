from typing import Any, Tuple

import attr
import numpy as np


__all__ = ["in_bounds", "is_subdtype"]


@attr.s(auto_attribs=True, repr=False, hash=True)
class _InBoundsValidator:
    bounds: Tuple[Any, Any]
    closed: Tuple[bool, bool]

    def __attrs_post_init__(self):
        delim_left = "[" if self.closed[0] else "("
        delim_right = "]" if self.closed[1] else ")"

        self.bounds_str = f"{delim_left}{self.bounds[0]}, {self.bounds[1]}{delim_right}"

        if (
            self.bounds[0] is not None
            and self.bounds[1] is not None
            and self.bounds[1] < self.bounds[0]
        ):
            raise ValueError(
                f"Invalid bounds {self.bounds_str}: "
                "upper limit should be higher than lower limit"
            )

    def __call__(self, inst, attr, value):
        out_lower = self.bounds[0] is not None and (
            value < self.bounds[0] if self.closed[0] else value <= self.bounds[0]
        )
        out_upper = self.bounds[1] is not None and (
            value > self.bounds[1] if self.closed[1] else value >= self.bounds[1]
        )

        if np.any(out_lower) or np.any(out_upper):
            common_msg = f"out of bounds {self.bounds_str}"

            if np.isscalar(value):
                msg = f"Value {value} of variable '{attr.name}' is " + common_msg
            else:
                msg = f"Found value(s) in array '{attr.name}' " + common_msg

            raise ValueError(msg)

    def __repr__(self):
        return f"<in_bounds validator with bounds {self.bounds_str}>"


def in_bounds(bounds, closed=(True, True)):
    """A validator that raises a :exc:`ValueError` if a given value is out of
    the given bounded interval.

    It works with scalar values as well as with arrays (element-wise check).

    Parameters
    ----------
    bounds : tuple
        Lower and upper value bounds. Use ``None`` for either lower or upper
        value to set half-bounded intervals.
    closed : tuple, optional
        Set an open, half-open or closed interval, i.e., whether the
        lower and/or upper bound is included or not in the interval.
        Default: closed interval (i.e., includes both lower and upper bounds).

    """
    return _InBoundsValidator(tuple(bounds), tuple(closed))


@attr.s(repr=False, hash=True)
class _IsSubdtypeValidator:
    dtype = attr.ib()

    def __call__(self, inst, attr, value):
        if not np.issubdtype(value.dtype, self.dtype):
            raise TypeError(
                f"'{attr.name}' array has {value.dtype!r}, which is not "
                f"a sub-dtype of {self.dtype!r}"
            )

    def __repr__(self):
        return f"<is_subdtype validator with type: {self.dtype!r}>"


def is_subdtype(dtype):
    """A validator that raises a :exc:`TypeError` if a given array has a wrong
    dtype.

    Parameters
    ----------
    dtype : dtype_like
        dtype or string representing the typecode of the highest type
        allowed in the hierarchy.

    See Also
    --------
    :func:`numpy.issubdtype`

    """
    return _IsSubdtypeValidator(dtype)
