import unittest

import numpy as np

from xsimlab.variable.base import (Variable, ForeignVariable, VariableList,
                                   VariableGroup, diagnostic)
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
        self.some_effect.value = self.x * self.some_param.value + dt


class OtherProcess(Process):
    other_param = Variable(())
    x = ForeignVariable(Grid, 'x')
    quantity = ForeignVariable(Quantity, 'quantity')
    other_effect = Variable('x', group='effect', provided=True)

    def run_step(self, dt):
        self.other_effect.value = self.x * self.other_param.value - dt

    @diagnostic
    def x2(self):
        return self.x * 2


class MetaProcess(Process):
    param = Variable(())
    some_param = ForeignVariable(SomeProcess, 'some_param')
    other_param = ForeignVariable(OtherProcess, 'other_param')

    def run_step(self, *args):
        self.some_param.value = self.param.value
        self.other_param.value = self.param.value


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

        model = Model({'grid': Grid, 'some_process': SomeProcess,
                       'other_process': OtherProcess, 'quantity': Quantity})

        # test process ordering
        sorted_process_names = list(model.keys())
        self.assertEqual(sorted_process_names[0], 'grid')
        self.assertEqual(sorted_process_names[-1], 'quantity')
        self.assertIn('some_process', sorted_process_names[1:-1])
        self.assertIn('other_process', sorted_process_names[1:-1])

        # test dict-like vs. attribute access
        self.assertIs(model['grid'], model.grid)
