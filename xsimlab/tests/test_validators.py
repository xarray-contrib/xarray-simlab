import attr
import numpy as np
import pytest

from xsimlab.validators import in_bounds, is_subdtype


@pytest.fixture
def simple_attr():
    """
    Return an attribute with a name just for testing purpose.
    """

    @attr.attrs
    class C:
        test = attr.attrib()

    return attr.fields_dict(C)["test"]


def test_in_bounds_init():
    with pytest.raises(ValueError, match=r"Invalid bounds.*"):
        in_bounds((5, 0))


@pytest.mark.parametrize(
    "bounds,value",
    [
        ((0, 5), 2),
        ((0, 5), 0),
        ((0, 5), 5),
        ((0, 5), np.array([0, 1, 2, 3, 4, 5])),
        ((None, 5), -1000),
        ((0, None), 1000),
        ((None, None), 1000),
    ],
)
def test_in_bounds_success(simple_attr, bounds, value):
    v = in_bounds(bounds)

    # nothing happens
    v(None, simple_attr, value)


@pytest.mark.parametrize(
    "bounds,closed,value",
    [
        ((0, 5), (True, True), 6),
        ((0, 5), (True, False), 5),
        ((0, 5), (False, True), 0),
        ((0, 5), (False, False), np.array([0, 1, 2, 3, 4, 5])),
    ],
)
def test_in_bounds_fail(simple_attr, bounds, closed, value):
    v = in_bounds(bounds, closed=closed)

    with pytest.raises(ValueError, match=r".*out of bounds.*"):
        v(None, simple_attr, value)


@pytest.mark.parametrize(
    "closed,interval_str",
    [
        ((True, True), "[0, 5]"),
        ((True, False), "[0, 5)"),
        ((False, True), "(0, 5]"),
        ((False, False), "(0, 5)"),
    ],
)
def test_in_bounds_repr(closed, interval_str):
    v = in_bounds((0, 5), closed=closed)

    expected = f"<in_bounds validator with bounds {interval_str}>"
    assert repr(v) == expected


def test_is_subdtype_success(simple_attr):
    v = is_subdtype(np.number)

    # nothing happends
    v(None, simple_attr, np.array([1, 2, 3]))
    v(None, simple_attr, np.array([1.0, 2.0, 3.0]))


def test_is_subdtype_fail(simple_attr):
    v = is_subdtype(np.number)

    with pytest.raises(TypeError, match=r".*not a sub-dtype of.*"):
        v(None, simple_attr, np.array(["1", "2", "3"]))


def test_is_subdtype_repr():
    v = is_subdtype(np.number)

    assert repr(v) == "<is_subdtype validator with type: <class 'numpy.number'>>"
