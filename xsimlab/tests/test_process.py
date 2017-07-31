import unittest
from textwrap import dedent
from io import StringIO

import pytest

from xsimlab.variable.base import (Variable, VariableList, VariableGroup,
                                   diagnostic)
from xsimlab.process import Process


class MyProcess(Process):
    var = Variable((), provided=True)
    var_list = VariableList([Variable('x'), Variable(((), 'x'))])
    var_group = VariableGroup('group')
    no_var = 'this is not a variable object'

    class Meta:
        time_dependent = False

    @diagnostic
    def diag(self):
        return 1


class TestProcessBase(unittest.TestCase):

    def test_new(self):
        with pytest.raises(TypeError) as excinfo:
            class InvalidProcess(MyProcess):
                var = Variable(())
        assert "subclassing a subclass" in str(excinfo.value)

        with pytest.raises(AttributeError) as excinfo:
            class InvalidProcess2(Process):
                class Meta:
                    time_dependent = True
                    invalid_meta_attr = 'invalid'
        assert "invalid attribute" in str(excinfo.value)

        # test extract variable objects vs. other attributes
        assert getattr(MyProcess, 'no_var', False)
        assert not getattr(MyProcess, 'var', False)
        assert set(['var', 'var_list', 'var_group', 'diag']) == (
            set(MyProcess._variables.keys()))

        # test Meta attributes
        assert MyProcess._meta == {'time_dependent': False}


class TestProcess(unittest.TestCase):

    def setUp(self):
        self.my_process = MyProcess()
        self.my_process_str = dedent("""\
        Variables:
          * diag       DiagnosticVariable
          * var        Variable ()
            var_group  VariableGroup 'group'
            var_list   VariableList
            -          Variable ('x')
            -          Variable (), ('x')
        Meta:
            time_dependent: False""")

    def test_constructor(self):
        # test dict-like vs. attribute access
        assert self.my_process['var'] is self.my_process._variables['var']
        assert self.my_process.var is self.my_process._variables['var']

        # test deep copy variable objects
        MyProcess._variables['var'].state = 2
        assert self.my_process._variables['var'].state != (
            MyProcess._variables['var'].state)

        # test assign process to diagnostics
        assert self.my_process['diag']._process_obj is self.my_process

    def test_clone(self):
        cloned_process = self.my_process.clone()
        assert self.my_process['var'] is not cloned_process['var']

    def test_variables(self):
        assert set(['var', 'var_list', 'var_group', 'diag']) == (
            set(self.my_process.variables.keys()))

    def test_meta(self):
        assert self.my_process.meta == {'time_dependent': False}

    def test_name(self):
        assert self.my_process.name == "MyProcess"

        self.my_process._name = "my_process"
        assert self.my_process.name == "my_process"

    def test_run_step(self):
        with pytest.raises(NotImplementedError) as excinfo:
            self.my_process.run_step(1)
        assert "no method" in str(excinfo.value)

    def test_info(self):
        expected = self.my_process_str

        for cls_or_obj in [MyProcess, self.my_process]:
            buf = StringIO()
            cls_or_obj.info(buf=buf)
            actual = buf.getvalue()
            assert actual == expected

        class OtherProcess(Process):
            pass

        expected = dedent("""\
        Variables:
            *empty*
        Meta:
            time_dependent: True""")

        buf = StringIO()
        OtherProcess.info(buf=buf)
        actual = buf.getvalue()
        assert actual == expected

    def test_repr(self):
        expected = '\n'.join(
            ["<xsimlab.Process 'xsimlab.tests.test_process.MyProcess'>",
             self.my_process_str]
        )
        assert repr(self.my_process) == expected
