from textwrap import dedent

import attr
import pytest

import xsimlab as xs


@xs.process
class SomeProcess(object):
    """Just used for foreign variables in ExampleProcess."""
    some_var = xs.variable(group='some_group', intent='out')
    some_od_var = xs.on_demand(group='some_group')

    @some_od_var.compute
    def compute_some_od_var(self):
        return 1


@xs.process
class AnotherProcess(object):
    """Just used for foreign variables in ExampleProcess."""
    another_var = xs.variable()
    some_var = xs.foreign(SomeProcess, 'some_var')


@xs.process
class ExampleProcess(object):
    """A process with complete interface for testing."""
    in_var = xs.variable(dims=['x', ('x', 'y')], description='input variable')
    out_var = xs.variable(group='example_group', intent='out')
    inout_var = xs.variable(intent='inout')
    od_var = xs.on_demand()

    in_foreign_var = xs.foreign(SomeProcess, 'some_var')
    in_foreign_var2 = xs.foreign(AnotherProcess, 'some_var')
    out_foreign_var = xs.foreign(AnotherProcess, 'another_var', intent='out')
    in_foreign_od_var = xs.foreign(SomeProcess, 'some_od_var')

    group_var = xs.group('some_group')

    other_attrib = attr.attrib(init=False, cmp=False, repr=False)
    other_attr = "this is not a xsimlab variable attribute"

    @od_var.compute
    def compute_od_var(self):
        return 0


@pytest.fixture
def example_process_obj():
    return ExampleProcess()


@pytest.fixture(scope='session')
def example_process_repr():
    return dedent("""\
    <ExampleProcess  (xsimlab process)>
    Variables:
        in_var                [in] ('x',) or ('x', 'y') input variable
        out_var              [out]
        inout_var          [inout]
        od_var               [out]
        in_foreign_var        [in] <--- SomeProcess.some_var
        in_foreign_var2       [in] <--- AnotherProcess.some_var
        out_foreign_var      [out] ---> AnotherProcess.another_var
        in_foreign_od_var     [in] <--- SomeProcess.some_od_var
        group_var             [in] <--- group 'some_group'
    Simulation stages:
        *no stage implemented*
    """)


@pytest.fixture(scope='session')
def in_var_details():
    return dedent("""\
    Input variable

    - type : variable
    - intent : in
    - dims : (('x',), ('x', 'y'))
    - group : None
    - attrs : {}
    """)


def _init_process(p_cls, p_name, model, store, store_keys=None, od_keys=None):
    p_obj = p_cls()
    p_obj.__xsimlab_name__ = p_name
    p_obj.__xsimlab_model__ = model
    p_obj.__xsimlab_store__ = store
    p_obj.__xsimlab_store_keys__ = store_keys or {}
    p_obj.__xsimlab_od_keys__ = od_keys or {}
    return p_obj


@pytest.fixture
def processes_with_store():
    class FakeModel(object):
        def __init__(self):
            self._processes = {}

    model = FakeModel()
    store = {}

    some_process = _init_process(
        SomeProcess, 'some_process', model, store,
        store_keys={'some_var': ('some_process', 'some_var')}
    )
    another_process = _init_process(
        AnotherProcess, 'another_process', model, store,
        store_keys={'another_var': ('another_process', 'another_var'),
                    'some_var': ('some_process', 'some_var')}
    )
    example_process = _init_process(
        ExampleProcess, 'example_process', model, store,
        store_keys={'in_var': ('example_process', 'in_var'),
                    'out_var': ('example_process', 'out_var'),
                    'inout_var': ('example_process', 'inout_var'),
                    'in_foreign_var': ('some_process', 'some_var'),
                    'in_foreign_var2': ('some_process', 'some_var'),
                    'out_foreign_var': ('another_process', 'another_var'),
                    'group_var': [('some_process', 'some_var')]},
        od_keys={'in_foreign_od_var': ('some_process', 'some_od_var'),
                 'group_var': [('some_process', 'some_od_var')]}
    )

    model._processes.update({'some_process': some_process,
                             'another_process': another_process,
                             'example_process': example_process})

    return some_process, another_process, example_process


@pytest.fixture(scope='session')
def example_process_in_model_repr():
    return dedent("""\
    <ExampleProcess 'example_process' (xsimlab process)>
    Variables:
        in_var                [in] ('x',) or ('x', 'y') input variable
        out_var              [out]
        inout_var          [inout]
        od_var               [out]
        in_foreign_var        [in] <--- some_process.some_var
        in_foreign_var2       [in] <--- some_process.some_var
        out_foreign_var      [out] ---> another_process.another_var
        in_foreign_od_var     [in] <--- some_process.some_od_var
        group_var             [in] <--- group 'some_group'
    Simulation stages:
        *no stage implemented*
    """)
