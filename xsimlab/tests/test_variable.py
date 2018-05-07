import pytest

from xsimlab.tests.fixture_process import ExampleProcess
from xsimlab.variable import _as_dim_tuple, foreign


@pytest.mark.parametrize("dims,expected", [
    ((), ((),)),
    ([], ((),)),
    ('', ((),)),
    (('x'), (('x',),)),
    (['x'], (('x',),)),
    ('x', (('x',),)),
    (('x', 'y'), (('x', 'y'),)),
    ([(), 'x', ('x', 'y')], ((), ('x',), ('x', 'y')))
])
def test_as_dim_tuple(dims, expected):
    assert _as_dim_tuple(dims) == expected


def test_as_dim_tuple_invalid():
    invalid_dims = ['x', 'y', ('x', 'y'), ('y', 'x')]

    with pytest.raises(ValueError) as excinfo:
        _as_dim_tuple(invalid_dims)
    assert "following combinations" in str(excinfo)
    assert "('x',), ('y',) and ('x', 'y'), ('y', 'x')" in str(excinfo)


def test_foreign():
    with pytest.raises(ValueError) as excinfo:
        foreign(ExampleProcess, 'some_var', intent='inout')

    assert "intent='inout' is not supported" in str(excinfo.value)
