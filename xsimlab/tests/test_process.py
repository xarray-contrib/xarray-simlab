from io import StringIO
import inspect

import pytest

import xsimlab as xs
from xsimlab.variable import VarIntent, VarType
from xsimlab.process import (
    filter_variables,
    get_process_cls,
    get_process_obj,
    get_target_variable,
    NotAProcessClassError,
    process_info,
    variable_info,
)
from xsimlab.utils import variables_dict
from xsimlab.tests.fixture_process import ExampleProcess, SomeProcess


def test_get_process_cls(example_process_obj):
    p_cls = get_process_cls(ExampleProcess)
    assert get_process_cls(example_process_obj) is p_cls


def test_get_process_obj(example_process_obj):
    p_cls = get_process_cls(ExampleProcess)
    assert type(get_process_obj(ExampleProcess)) is p_cls

    # get_process_obj returns a new instance
    assert get_process_obj(example_process_obj) is not example_process_obj


def test_get_process_raise():
    class NotAProcess:
        pass

    with pytest.raises(NotAProcessClassError) as excinfo:
        get_process_cls(NotAProcess)
    assert "is not a process-decorated class" in str(excinfo.value)

    with pytest.raises(NotAProcessClassError) as excinfo:
        get_process_obj(NotAProcess)
    assert "is not a process-decorated class" in str(excinfo.value)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            {},
            {
                "in_var",
                "out_var",
                "inout_var",
                "in_foreign_var",
                "in_foreign_var2",
                "out_foreign_var",
                "in_foreign_od_var",
                "group_var",
                "od_var",
            },
        ),
        ({"var_type": "variable"}, {"in_var", "out_var", "inout_var"}),
        (
            {"intent": "in"},
            {
                "in_var",
                "in_foreign_var",
                "in_foreign_var2",
                "in_foreign_od_var",
                "group_var",
            },
        ),
        ({"intent": "out"}, {"out_var", "out_foreign_var", "od_var"}),
        ({"group": "example_group"}, {"out_var"}),
        (
            {
                "func": lambda var: (
                    var.metadata["var_type"] != VarType.GROUP
                    and var.metadata["intent"] != VarIntent.OUT
                )
            },
            {
                "in_var",
                "inout_var",
                "in_foreign_var",
                "in_foreign_var2",
                "in_foreign_od_var",
            },
        ),
    ],
)
def test_filter_variables(kwargs, expected):
    assert set(filter_variables(ExampleProcess, **kwargs)) == expected


@pytest.mark.parametrize(
    "var_name,expected_cls,expected_var_name",
    [
        ("in_var", ExampleProcess, "in_var"),
        ("in_foreign_var", SomeProcess, "some_var"),
        ("in_foreign_var2", SomeProcess, "some_var"),  # test foreign of foreign
    ],
)
def test_get_target_variable(var_name, expected_cls, expected_var_name):
    _ExampleProcess = get_process_cls(ExampleProcess)
    expected_p_cls = get_process_cls(expected_cls)

    var = variables_dict(_ExampleProcess)[var_name]
    expected_var = variables_dict(expected_p_cls)[expected_var_name]

    actual_cls, actual_var = get_target_variable(var)

    if expected_p_cls is _ExampleProcess:
        assert actual_cls is None
    else:
        actual_p_cls = get_process_cls(actual_cls)
        assert actual_p_cls is expected_p_cls

    assert actual_var is expected_var


@pytest.mark.parametrize(
    "cls,var_name,prop_is_read_only",
    [
        (ExampleProcess, "in_var", True),
        (ExampleProcess, "in_foreign_var", True),
        (ExampleProcess, "group_var", True),
        (ExampleProcess, "od_var", True),
        (ExampleProcess, "inout_var", False),
        (ExampleProcess, "out_var", False),
        (ExampleProcess, "out_foreign_var", False),
    ],
)
def test_process_properties_readonly(cls, var_name, prop_is_read_only):
    p_cls = get_process_cls(cls)

    if prop_is_read_only:
        assert getattr(p_cls, var_name).fset is None
    else:
        assert getattr(p_cls, var_name).fset is not None


def test_process_properties_errors():
    with pytest.raises(ValueError) as excinfo:

        @xs.process
        class Process1:
            invalid_var = xs.foreign(ExampleProcess, "group_var")

    assert "links to group variable" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:

        @xs.process
        class Process2:
            invalid_var = xs.foreign(ExampleProcess, "out_var", intent="out")

    assert "both have intent='out'" in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:

        @xs.process
        class Process3:
            var = xs.on_demand()

    assert "No compute method found" in str(excinfo.value)


def test_process_properties_docstrings(in_var_details):
    # order of lines in string is not ensured (printed from a dictionary)
    to_lines = lambda details_str: sorted(details_str.split("\n"))

    _ExampleProcess = get_process_cls(ExampleProcess)

    assert to_lines(_ExampleProcess.in_var.__doc__) == to_lines(in_var_details)


def test_process_properties_values(processes_with_store):
    some_process, another_process, example_process = processes_with_store

    assert example_process.od_var == 0
    assert example_process.in_foreign_od_var == 1

    example_process.inout_var = 2
    assert example_process.inout_var == 2

    example_process.out_foreign_var = 3
    assert another_process.another_var == 3

    some_process.some_var = 4
    assert another_process.some_var == 4
    assert example_process.in_foreign_var == 4
    assert example_process.in_foreign_var2 == 4

    assert set(example_process.group_var) == {1, 4}


def test_runtime_decorator_noargs():
    @xs.runtime
    def meth(self):
        return 1

    assert meth.__xsimlab_executor__.execute(None, {}) == 1


@pytest.mark.parametrize("args", ["p1,p2", ["p1", "p2"], ("p1", "p2")])
def test_runtime_decorator(args):
    @xs.runtime(args=args)
    def meth(self, a, b):
        return a + b

    d = {"p1": 1, "p2": 2, "other": 3}

    assert meth.__xsimlab_executor__.execute(None, d) == 3


def test_runtime_decorator_raise():
    with pytest.raises(ValueError, match=r".*args must be either.*"):

        @xs.runtime(args=1)
        def meth(self):
            pass


def test_runtime_function():
    def meth(self):
        return 1

    rmeth = xs.runtime(meth)

    assert rmeth.__xsimlab_executor__.execute(None, {}) == 1


def test_process_executor_raise():
    # TODO: remove (depreciated)
    with pytest.warns(FutureWarning):

        @xs.process
        class P:
            def run_step(self, dt):
                pass

    with pytest.raises(TypeError, match=r"Process runtime methods.*"):

        @xs.process
        class P2:
            def run_step(self, a, b):
                pass


def test_process_decorator():
    @xs.process(autodoc=True)
    class Dummy_t:
        pass

    @xs.process(autodoc=False)
    class Dummy_f:
        pass

    assert "Attributes" in Dummy_t.__doc__
    assert Dummy_f.__doc__ is None


def test_process_no_model():
    params = inspect.signature(ExampleProcess.__init__).parameters

    expected_params = [
        "self",
        "in_var",
        "inout_var",
        "in_foreign_var",
        "in_foreign_var2",
        "in_foreign_od_var",
        "group_var",
    ]

    assert list(params.keys()) == expected_params

    @xs.process
    class P:
        invar = xs.variable()
        outvar = xs.variable(intent="out")

        def initialize(self):
            self.outvar = self.invar + 2

    p = P(invar=1)
    p.initialize()

    assert p.outvar == 3


def test_process_info(example_process_obj, example_process_repr):
    buf = StringIO()
    process_info(example_process_obj, buf=buf)

    assert buf.getvalue() == example_process_repr


def test_variable_info(in_var_details):
    buf = StringIO()
    variable_info(ExampleProcess, "in_var", buf=buf)

    # order of lines in string is not ensured (printed from a dictionary)
    to_lines = lambda details_str: sorted(details_str.split("\n"))

    assert to_lines(buf.getvalue()) == to_lines(in_var_details)
