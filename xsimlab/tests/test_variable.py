import pytest
import attr

from xsimlab.tests.fixture_process import AnotherProcess, ExampleProcess
from xsimlab.variable import _as_dim_tuple, _as_group_tuple, foreign, index


@pytest.mark.parametrize(
    "dims,expected",
    [
        ((), ((),)),
        ([], ((),)),
        ("", ((),)),
        (("x"), (("x",),)),
        (["x"], (("x",),)),
        ("x", (("x",),)),
        (("x", "y"), (("x", "y"),)),
        ([(), "x", ("x", "y")], ((), ("x",), ("x", "y"))),
    ],
)
def test_as_dim_tuple(dims, expected):
    assert _as_dim_tuple(dims) == expected


def test_as_dim_tuple_invalid():
    invalid_dims = ["x", "y", ("x", "y"), ("y", "x")]

    with pytest.raises(ValueError) as excinfo:
        _as_dim_tuple(invalid_dims)
    assert "following combinations" in str(excinfo.value)
    assert "('x',), ('y',) and ('x', 'y'), ('y', 'x')" in str(excinfo.value)


@pytest.mark.parametrize(
    "groups,group,expected",
    [
        (None, None, ()),
        ("group1", None, ("group1",)),
        (["group1", "group2"], None, ("group1", "group2")),
        ("group1", "group2", ("group1", "group2")),
        ("group1", "group1", ("group1",)),
    ],
)
def test_as_group_tuple(groups, group, expected):
    if group is not None:
        with pytest.warns(FutureWarning):
            actual = _as_group_tuple(groups, group)

    else:
        actual = _as_group_tuple(groups, group)

    assert actual == expected


def test_index():
    with pytest.raises(ValueError, match=r".*not accept scalar values.*"):
        index(())


def test_foreign():
    with pytest.raises(ValueError) as excinfo:
        foreign(ExampleProcess, "some_var", intent="inout")
    assert "intent='inout' is not supported" in str(excinfo.value)

    var = attr.fields(ExampleProcess).out_foreign_var
    ref_var = attr.fields(AnotherProcess).another_var

    for k in ("description", "attrs"):
        assert var.metadata[k] == ref_var.metadata[k]
