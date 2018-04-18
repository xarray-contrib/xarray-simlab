"""This module provides a set of process classes and models as pytest
fixtures that are used across the tests.

"""
import attr
import pytest

import xsimlab as xs


@xs.process
class SomeProcess(object):
    """Just used for foreign variable in ExampleProcess."""
    ref_var = xs.variable()


@xs.process
class OtherProcess(object):
    """Just used for foreign variable in ExampleProcess."""
    ref_var = xs.variable()


@xs.process
class ExampleProcess(object):
    """A process with complete interface for testing."""
    in_var = xs.variable()
    out_var = xs.variable(group='group1', intent='out')
    inout_var = xs.variable(intent='inout')
    in_foreign_var = xs.foreign(SomeProcess, 'ref_var')
    out_foreign_var = xs.foreign(OtherProcess, 'ref_var', intent='out')
    group_var = xs.group('group2')
    od_var = xs.on_demand()

    other_attrib = attr.attrib(init=False, cmp=False, repr=False)
    other_attr = "this is not a xsimlab variable attribute"

    @od_var.compute
    def compute_od_var(self):
        return 1


@pytest.fixture
def example_process_obj():
    return ExampleProcess()
