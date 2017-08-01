import pytest
import numpy as np
from numpy.testing import assert_array_equal

from xsimlab.variable.base import Variable
from xsimlab.process import Process
from xsimlab.model import Model
from xsimlab.tests.conftest import (Grid, SomeProcess, OtherProcess, Quantity,
                                    PlugProcess)


class TestModel(object):

    def test_constructor(self, model):
        # test invalid processes
        with pytest.raises(TypeError):
            Model({'not_a_class': Grid()})

        class OtherClass(object):
            pass

        with pytest.raises(TypeError) as excinfo:
            Model({'invalid_class': Process})
        assert "is not a subclass" in str(excinfo.value)

        with pytest.raises(TypeError) as excinfo:
            Model({'invalid_class': OtherClass})
        assert "is not a subclass" in str(excinfo.value)

        # test process ordering
        expected = ['grid', 'some_process', 'other_process', 'quantity']
        assert list(model) == expected

        # test dict-like vs. attribute access
        assert model['grid'] is model.grid

    def test_input_vars(self, model):
        expected = {'grid': ['x_size'],
                    'some_process': ['some_param'],
                    'quantity': ['quantity']}
        actual = {k: list(v.keys()) for k, v in model.input_vars.items()}
        assert expected == actual

    def test_is_input(self, model):
        assert model.is_input(model.grid.x_size) is True
        assert model.is_input(('grid', 'x_size')) is True
        assert model.is_input(('quantity', 'all_effects')) is False
        assert model.is_input(('other_process', 'copy_param')) is False

        external_variable = Variable(())
        assert model.is_input(external_variable) is False

    def test_initialize(self, model):
        model.initialize()
        expected = np.arange(10)
        assert_array_equal(model.grid.x.value, expected)

    def test_run_step(self, model):
        model.initialize()
        model.run_step(100)

        expected = model.grid.x.value * 2
        assert_array_equal(model.quantity.quantity.change, expected)

    def test_finalize_step(self, model):
        model.initialize()
        model.run_step(100)
        model.finalize_step()

        expected = model.grid.x.value * 2
        assert_array_equal(model.quantity.quantity.state, expected)

    def test_finalize(self, model):
        model.finalize()
        assert model.some_process.some_effect.rate == 0

    def test_clone(self, model):
        cloned = model.clone()

        for (ck, cp), (k, p) in zip(cloned.items(), model.items()):
            assert ck == k
            assert cp is not p

    def test_update_processes(self, model):
        expected = Model({'grid': Grid,
                          'plug_process': PlugProcess,
                          'some_process': SomeProcess,
                          'other_process': OtherProcess,
                          'quantity': Quantity})
        actual = model.update_processes({'plug_process': PlugProcess})

        # TODO: more advanced (public?) test function to compare two models?
        assert list(actual) == list(expected)

    def test_drop_processes(self, model):

        expected = Model({'grid': Grid,
                          'some_process': SomeProcess,
                          'quantity': Quantity})
        actual = model.drop_processes('other_process')
        assert list(actual) == list(expected)

        expected = Model({'grid': Grid,
                          'quantity': Quantity})
        actual = model.drop_processes(['some_process', 'other_process'])
        assert list(actual) == list(expected)

    def test_repr(self, model, model_repr):
        assert repr(model) == model_repr
