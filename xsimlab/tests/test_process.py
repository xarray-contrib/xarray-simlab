from textwrap import dedent
from io import StringIO

import pytest

from xsimlab.variable.base import Variable
from xsimlab.process import Process
from xsimlab.tests.conftest import ExampleProcess


class TestProcessBase(object):

    def test_new(self):
        with pytest.raises(TypeError) as excinfo:
            class InvalidProcess(ExampleProcess):
                var = Variable(())
        assert "subclassing a subclass" in str(excinfo.value)

        with pytest.raises(AttributeError) as excinfo:
            class InvalidProcess2(Process):
                class Meta:
                    time_dependent = True
                    invalid_meta_attr = 'invalid'
        assert "invalid attribute" in str(excinfo.value)

        # test extract variable objects vs. other attributes
        assert getattr(ExampleProcess, 'no_var', False)
        assert not getattr(ExampleProcess, 'var', False)
        assert set(['var', 'var_list', 'var_group', 'diag']) == (
            set(ExampleProcess._variables.keys()))

        # test Meta attributes
        assert ExampleProcess._meta == {'time_dependent': False}


class TestProcess(object):

    def test_constructor(self, process):
        # test dict-like vs. attribute access
        assert process['var'] is process._variables['var']
        assert process.var is process._variables['var']

        # test deep copy variable objects
        ExampleProcess._variables['var'].state = 2
        assert process._variables['var'].state != (
            ExampleProcess._variables['var'].state)

        # test assign process to diagnostics
        assert process['diag']._process_obj is process

    def test_clone(self, process):
        cloned_process = process.clone()
        assert process['var'] is not cloned_process['var']

    def test_variables(self, process):
        assert set(['var', 'var_list', 'var_group', 'diag']) == (
            set(process.variables.keys()))

    def test_meta(self, process):
        assert process.meta == {'time_dependent': False}

    def test_name(self, process):
        assert process.name == "ExampleProcess"

        process._name = "my_process"
        assert process.name == "my_process"

    def test_run_step(self, process):
        with pytest.raises(NotImplementedError) as excinfo:
            process.run_step(1)
        assert "no method" in str(excinfo.value)

    def test_info(self, process, process_repr):
        for cls_or_obj in [ExampleProcess, process]:
            buf = StringIO()
            cls_or_obj.info(buf=buf)
            actual = buf.getvalue()
            assert actual == process_repr

        class EmptyProcess(Process):
            pass

        expected = dedent("""\
        Variables:
            *empty*
        Meta:
            time_dependent: True""")

        buf = StringIO()
        EmptyProcess.info(buf=buf)
        actual = buf.getvalue()
        assert actual == expected

    def test_repr(self, process, process_repr):
        expected = '\n'.join(
            ["<xsimlab.Process 'xsimlab.tests.conftest.ExampleProcess'>",
             process_repr]
        )
        assert repr(process) == expected
