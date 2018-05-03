from io import StringIO

import pytest

import xsimlab as xs
from xsimlab.variable import VarIntent, VarType
from xsimlab.process import (ensure_process_decorated, filter_variables,
                             get_process_cls, get_process_obj,
                             get_target_variable, NotAProcessClassError,
                             process_info, variable_info)
from xsimlab.utils import variables_dict
from xsimlab.tests.fixture_process import ExampleProcess, SomeProcess


def test_ensure_process_decorated():
    class NotAProcess(object):
        pass

    with pytest.raises(NotAProcessClassError) as excinfo:
        ensure_process_decorated(NotAProcess)
    assert "is not a process-decorated class" in str(excinfo.value)


def test_get_process_cls(example_process_obj):
    assert get_process_cls(ExampleProcess) is ExampleProcess
    assert get_process_cls(example_process_obj) is ExampleProcess


def test_get_process_obj(example_process_obj):
    assert get_process_obj(example_process_obj) is example_process_obj
    assert type(get_process_obj(ExampleProcess)) is ExampleProcess


@pytest.mark.parametrize('kwargs,expected', [
    ({}, {'in_var', 'out_var', 'inout_var', 'in_foreign_var',
          'in_foreign_var2', 'out_foreign_var', 'in_foreign_od_var',
          'group_var', 'od_var'}),
    ({'var_type': 'variable'}, {'in_var', 'out_var', 'inout_var'}),
    ({'intent': 'in'}, {'in_var', 'in_foreign_var', 'in_foreign_var2',
                        'in_foreign_od_var', 'group_var'}),
    ({'intent': 'out'}, {'out_var', 'out_foreign_var', 'od_var'}),
    ({'group': 'example_group'}, {'out_var'}),
    ({'func': lambda var: (
        var.metadata['var_type'] != VarType.GROUP and
        var.metadata['intent'] != VarIntent.OUT)},
     {'in_var', 'inout_var', 'in_foreign_var', 'in_foreign_var2',
      'in_foreign_od_var'})
])
def test_filter_variables(kwargs, expected):
    assert set(filter_variables(ExampleProcess, **kwargs)) == expected


@pytest.mark.parametrize('var_name,expected_p_cls,expected_var_name', [
    ('in_var', ExampleProcess, 'in_var'),
    ('in_foreign_var', SomeProcess, 'some_var'),
    ('in_foreign_var2', SomeProcess, 'some_var')  # test foreign of foreign
])
def test_get_target_variable(var_name, expected_p_cls, expected_var_name):
    var = variables_dict(ExampleProcess)[var_name]
    expected_var = variables_dict(expected_p_cls)[expected_var_name]

    actual_p_cls, actual_var = get_target_variable(var)

    if expected_p_cls is ExampleProcess:
        assert actual_p_cls is None
    else:
        assert actual_p_cls is expected_p_cls

    assert actual_var is expected_var


@pytest.mark.parametrize('p_cls,var_name,prop_is_read_only', [
    (ExampleProcess, 'in_var', True),
    (ExampleProcess, 'in_foreign_var', True),
    (ExampleProcess, 'group_var', True),
    (ExampleProcess, 'od_var', True),
    (ExampleProcess, 'inout_var', False),
    (ExampleProcess, 'out_var', False),
    (ExampleProcess, 'out_foreign_var', False)
])
def test_process_properties_readonly(p_cls, var_name, prop_is_read_only):
    if prop_is_read_only:
        assert getattr(p_cls, var_name).fset is None
    else:
        assert getattr(p_cls, var_name).fset is not None


def test_process_properties_errors():
    with pytest.raises(ValueError) as excinfo:
        @xs.process
        class Process1(object):
            invalid_var = xs.foreign(ExampleProcess, 'group_var')

    assert "links to group variable" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        @xs.process
        class Process2(object):
            invalid_var = xs.foreign(ExampleProcess, 'out_var', intent='out')

    assert "both have intent='out'" in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:
        @xs.process
        class Process3(object):
            var = xs.on_demand()

    assert "No compute method found" in str(excinfo.value)


def test_process_properties_docstrings(in_var_details):
    # order of lines in string is not ensured (printed from a dictionary)
    to_lines = lambda details_str: sorted(details_str.split('\n'))

    assert to_lines(ExampleProcess.in_var.__doc__) == to_lines(in_var_details)


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


def test_process_decorator():
    with pytest.raises(NotImplementedError):
        @xs.process(autodoc=True)
        class Dummy(object):
            pass


def test_process_info(example_process_obj, example_process_repr):
    buf = StringIO()
    process_info(example_process_obj, buf=buf)

    assert buf.getvalue() == example_process_repr


def test_variable_info(in_var_details):
    buf = StringIO()
    variable_info(ExampleProcess, 'in_var', buf=buf)

    # order of lines in string is not ensured (printed from a dictionary)
    to_lines = lambda details_str: sorted(details_str.split('\n'))

    assert to_lines(buf.getvalue()) == to_lines(in_var_details)
