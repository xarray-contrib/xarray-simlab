from textwrap import dedent
from io import StringIO

import attr
import pytest

from xsimlab.variable import VarIntent, VarType
from xsimlab.process import (filter_variables, get_process_cls, get_process_obj,
                             NotAProcessClassError)
from xsimlab.tests.conftest import ExampleProcess

# from xsimlab.variable.base import Variable
# from xsimlab.process import Process
# from xsimlab.tests.conftest import ExampleProcess


def test_get_process_cls(example_process_obj):
    assert get_process_cls(ExampleProcess) is ExampleProcess
    assert get_process_cls(example_process_obj) is ExampleProcess


def test_get_process_obj(example_process_obj):
    assert get_process_obj(example_process_obj) is example_process_obj
    assert type(get_process_obj(ExampleProcess)) is ExampleProcess


def test_get_process_invalid():
    class NotAProcess(object):
        pass

    with pytest.raises(NotAProcessClassError) as excinfo:
        get_process_cls(NotAProcess)
        get_process_obj(NotAProcess)
    assert "is not a process-decorated class" in str(excinfo.value)


def test_filter_variables():
    func = lambda kw: set(filter_variables(ExampleProcess, **kw).keys())

    expected = {'in_var', 'out_var', 'inout_var',
                'in_foreign_var', 'out_foreign_var',
                'group_var', 'od_var'}
    assert func({}) == expected

    expected = {'in_var', 'out_var', 'inout_var'}
    assert func({'var_type': 'variable'}) == expected

    expected = {'in_var', 'in_foreign_var', 'group_var'}
    assert func({'intent': 'in'})

    expected = {'out_var', 'out_foreign_var', 'od_var'}
    assert func({'intent': 'out'})

    expected = {'out_var'}
    assert func({'group': 'group1'})

    expected = {'in_var', 'inout_var', 'in_foreign_var', 'od_var'}
    ff = lambda var: (
        var.metadata['var_type'] != VarType.GROUP and
        var.metadata['intent'] != VarIntent.OUT
    )
    assert func({'func': ff})


# class TestProcessBase(object):

#     def test_new(self):
#         with pytest.raises(TypeError) as excinfo:
#             class InvalidProcess(ExampleProcess):
#                 var = Variable(())
#         assert "subclassing a subclass" in str(excinfo.value)

#         with pytest.raises(AttributeError) as excinfo:
#             class InvalidProcess2(Process):
#                 class Meta:
#                     time_dependent = True
#                     invalid_meta_attr = 'invalid'
#         assert "invalid attribute" in str(excinfo.value)

#         # test extract variable objects vs. other attributes
#         assert getattr(ExampleProcess, 'no_var', False)
#         assert not getattr(ExampleProcess, 'var', False)
#         assert set(['var', 'var_list', 'var_group', 'diag']) == (
#             set(ExampleProcess._variables.keys()))

#         # test Meta attributes
#         assert ExampleProcess._meta == {'time_dependent': False}


# class TestProcess(object):

#     def test_constructor(self, process):
#         # test dict-like vs. attribute access
#         assert process['var'] is process._variables['var']
#         assert process.var is process._variables['var']

#         # test deep copy variable objects
#         ExampleProcess._variables['var'].state = 2
#         assert process._variables['var'].state != (
#             ExampleProcess._variables['var'].state)

#         # test assign process to diagnostics
#         assert process['diag']._process_obj is process

#     def test_clone(self, process):
#         cloned_process = process.clone()
#         assert process['var'] is not cloned_process['var']

#     def test_variables(self, process):
#         assert set(['var', 'var_list', 'var_group', 'diag']) == (
#             set(process.variables.keys()))

#     def test_meta(self, process):
#         assert process.meta == {'time_dependent': False}

#     def test_name(self, process):
#         assert process.name == "ExampleProcess"

#         process._name = "my_process"
#         assert process.name == "my_process"

#     def test_run_step(self, process):
#         with pytest.raises(NotImplementedError) as excinfo:
#             process.run_step(1)
#         assert "no method" in str(excinfo.value)

#     def test_info(self, process, process_repr):
#         for cls_or_obj in [ExampleProcess, process]:
#             buf = StringIO()
#             cls_or_obj.info(buf=buf)
#             actual = buf.getvalue()
#             assert actual == process_repr

#         class EmptyProcess(Process):
#             pass

#         expected = dedent("""\
#         Variables:
#             *empty*
#         Meta:
#             time_dependent: True""")

#         buf = StringIO()
#         EmptyProcess.info(buf=buf)
#         actual = buf.getvalue()
#         assert actual == expected

#     def test_repr(self, process, process_repr):
#         expected = '\n'.join(
#             ["<xsimlab.Process 'xsimlab.tests.conftest.ExampleProcess'>",
#              process_repr]
#         )
#         assert repr(process) == expected
