import unittest

import numpy as np
from numpy.testing import assert_array_equal

from xsimlab.variable.base import (Variable, ForeignVariable, VariableGroup,
                                   diagnostic)
from xsimlab.process import Process
from xsimlab.model import Model


class Grid(Process):
    x_size = Variable((), optional=True)
    x = Variable('x', provided=True)

    class Meta:
        time_dependent = False

    def validate(self):
        if self.x_size.value is None:
            self.x_size.value = 5

    def initialize(self):
        self.x.value = np.arange(self.x_size.value)


class Quantity(Process):
    quantity = Variable('x')
    all_effects = VariableGroup('effect')

    def run_step(self, *args):
        self.quantity.change = sum((var.value for var in self.all_effects))

    def finalize_step(self):
        self.quantity.state += self.quantity.change


class SomeProcess(Process):
    some_param = Variable(())
    x = ForeignVariable(Grid, 'x')
    quantity = ForeignVariable(Quantity, 'quantity')
    some_effect = Variable('x', group='effect', provided=True)

    def run_step(self, dt):
        self.some_effect.value = self.x.value * self.some_param.value + dt


class OtherProcess(Process):
    other_param = Variable(())
    x = ForeignVariable(Grid, 'x')
    quantity = ForeignVariable(Quantity, 'quantity')
    other_effect = Variable('x', group='effect', provided=True)

    def run_step(self, dt):
        self.other_effect.value = self.x.value * self.other_param.value - dt

    @diagnostic
    def x2(self):
        return self.x * 2


class MetaProcess(Process):
    meta_param = Variable(())
    some_param = ForeignVariable(SomeProcess, 'some_param')
    other_param = ForeignVariable(OtherProcess, 'other_param')

    def run_step(self, *args):
        self.some_param.value = self.param.value
        self.other_param.value = self.param.value


class TestModel(unittest.TestCase):

    def setUp(self):
        self.model = Model({'grid': Grid,
                            'some_process': SomeProcess,
                            'other_process': OtherProcess,
                            'quantity': Quantity})

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
        sorted_process_names = list(self.model.keys())
        self.assertEqual(sorted_process_names[0], 'grid')
        self.assertEqual(sorted_process_names[-1], 'quantity')
        self.assertIn('some_process', sorted_process_names[1:-1])
        self.assertIn('other_process', sorted_process_names[1:-1])

        # test dict-like vs. attribute access
        self.assertIs(self.model['grid'], self.model.grid)

    def test_input_vars(self):
        expected = {'grid': ['x_size'],
                    'some_process': ['some_param'],
                    'other_process': ['other_param'],
                    'quantity': ['quantity']}
        actual = {k: list(v.keys()) for k, v in self.model.input_vars.items()}
        self.assertDictEqual(expected, actual)

    def test_is_input(self):
        self.assertTrue(self.model.is_input(self.model.grid.x_size))
        self.assertTrue(self.model.is_input(('grid', 'x_size')))
        self.assertFalse(self.model.is_input(('quantity', 'all_effects')))

    def test_initialize(self):
        model = self.model.clone()
        model.grid.x_size.value = 10
        model.initialize()
        expected = np.arange(10)
        assert_array_equal(model.grid.x.value, expected)

    def test_run_step(self):
        model = self.model.clone()
        model.grid.x_size.value = 10
        model.some_process.some_param.value = 1
        model.other_process.other_param.value = 1

        model.initialize()
        model.run_step(100)

        expected = model.grid.x.value * 2
        assert_array_equal(model.quantity.quantity.change, expected)
