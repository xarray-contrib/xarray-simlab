import unittest
from textwrap import dedent

import numpy as np
from numpy.testing import assert_array_equal

from xsimlab.variable.base import (Variable, ForeignVariable, VariableGroup,
                                   diagnostic)
from xsimlab.process import Process
from xsimlab.model import Model


class Grid(Process):
    x_size = Variable((), optional=True, description='grid size')
    x = Variable('x', provided=True)

    class Meta:
        time_dependent = False

    def validate(self):
        if self.x_size.value is None:
            self.x_size.value = 5

    def initialize(self):
        self.x.value = np.arange(self.x_size.value)


class Quantity(Process):
    quantity = Variable('x', description='a quantity')
    all_effects = VariableGroup('effect')

    def run_step(self, *args):
        self.quantity.change = sum((var.value for var in self.all_effects))

    def finalize_step(self):
        self.quantity.state += self.quantity.change


class SomeProcess(Process):
    some_param = Variable((), description='some parameter')
    copy_param = Variable((), provided=True)
    x = ForeignVariable(Grid, 'x')
    quantity = ForeignVariable(Quantity, 'quantity')
    some_effect = Variable('x', group='effect', provided=True)

    def initialize(self):
        self.copy_param.value = self.some_param.value

    def run_step(self, dt):
        self.some_effect.value = self.x.value * self.some_param.value + dt

    def finalize(self):
        self.some_effect.rate = 0


class OtherProcess(Process):
    x = ForeignVariable(Grid, 'x')
    copy_param = ForeignVariable(SomeProcess, 'copy_param')
    quantity = ForeignVariable(Quantity, 'quantity')
    other_effect = Variable('x', group='effect', provided=True)

    def run_step(self, dt):
        self.other_effect.value = self.x.value * self.copy_param.value - dt

    @diagnostic
    def x2(self):
        return self.x * 2


class PlugProcess(Process):
    meta_param = Variable(())
    some_param = ForeignVariable(SomeProcess, 'some_param', provided=True)

    def run_step(self, *args):
        self.some_param.value = self.meta_param.value


def get_test_model():
    model = Model({'grid': Grid,
                   'some_process': SomeProcess,
                   'other_process': OtherProcess,
                   'quantity': Quantity})

    model.grid.x_size.value = 10
    model.quantity.quantity.state = np.zeros(10)
    model.some_process.some_param.value = 1

    return model


class TestModel(unittest.TestCase):

    def test_constructor(self):
        # test invalid processes
        with self.assertRaises(TypeError):
            Model({'not_a_class': Grid()})

        class OtherClass(object):
            pass

        with self.assertRaisesRegex(TypeError, "is not a subclass"):
            Model({'invalid_class': Process})

        with self.assertRaisesRegex(TypeError, "is not a subclass"):
            Model({'invalid_class': OtherClass})

        # test process ordering
        model = get_test_model()
        expected = ['grid', 'some_process', 'other_process', 'quantity']
        self.assertEqual(list(model), expected)

        # test dict-like vs. attribute access
        self.assertIs(model['grid'], model.grid)

    def test_input_vars(self):
        model = get_test_model()
        expected = {'grid': ['x_size'],
                    'some_process': ['some_param'],
                    'quantity': ['quantity']}
        actual = {k: list(v.keys()) for k, v in model.input_vars.items()}
        self.assertDictEqual(expected, actual)

    def test_is_input(self):
        model = get_test_model()
        self.assertTrue(model.is_input(model.grid.x_size))
        self.assertTrue(model.is_input(('grid', 'x_size')))
        self.assertFalse(model.is_input(('quantity', 'all_effects')))

        external_variable = Variable(())
        self.assertFalse(model.is_input(external_variable))

    def test_initialize(self):
        model = get_test_model()
        model.initialize()
        expected = np.arange(10)
        assert_array_equal(model.grid.x.value, expected)

    def test_run_step(self):
        model = get_test_model()
        model.initialize()
        model.run_step(100)

        expected = model.grid.x.value * 2
        assert_array_equal(model.quantity.quantity.change, expected)

    def test_finalize_step(self):
        model = get_test_model()
        model.initialize()
        model.run_step(100)
        model.finalize_step()

        expected = model.grid.x.value * 2
        assert_array_equal(model.quantity.quantity.state, expected)

    def test_finalize(self):
        model = get_test_model()
        model.finalize()
        self.assertEqual(model.some_process.some_effect.rate, 0)

    def test_clone(self):
        model = get_test_model()
        cloned = model.clone()

        for (ck, cp), (k, p) in zip(cloned.items(), model.items()):
            self.assertEqual(ck, k)
            self.assertIsNot(cp, p)

    def test_update_processes(self):
        model = get_test_model()
        expected = Model({'grid': Grid,
                          'plug_process': PlugProcess,
                          'some_process': SomeProcess,
                          'other_process': OtherProcess,
                          'quantity': Quantity})
        actual = model.update_processes({'plug_process': PlugProcess})

        # TODO: more advanced (public?) test function to compare two models?
        self.assertEqual(list(actual), list(expected))

    def test_drop_processes(self):
        model = get_test_model()

        expected = Model({'grid': Grid,
                          'some_process': SomeProcess,
                          'quantity': Quantity})
        actual = model.drop_processes('other_process')
        self.assertEqual(list(actual), list(expected))

        expected = Model({'grid': Grid,
                          'quantity': Quantity})
        actual = model.drop_processes(['some_process', 'other_process'])
        self.assertEqual(list(actual), list(expected))

    def test_repr(self):
        model = get_test_model()
        expected = dedent("""\
        <xsimlab.Model (4 processes, 3 inputs)>
        grid
            x_size      (in) grid size
        some_process
            some_param  (in) some parameter
        other_process
        quantity
            quantity    (in) a quantity""")

        self.assertEqual(repr(model), expected)
