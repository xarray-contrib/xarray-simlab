from textwrap import dedent

import xsimlab as xs
from xsimlab.formatting import (maybe_truncate, pretty_print,
                                repr_process, var_details, wrap_indent)


def test_maybe_truncate():
    assert maybe_truncate('test', 10) == 'test'
    assert maybe_truncate('longteststring', 10) == 'longtes...'


def test_pretty_print():
    assert pretty_print('test', 10) == 'test' + ' ' * 6


def test_wrap_indent():
    text = "line1\nline2"

    expected = '-line1\n line2'
    assert wrap_indent(text, start='-') == expected

    expected = 'line1\n line2'
    assert wrap_indent(text, length=1) == expected


def test_var_details(example_process_obj):
    var = xs.variable(dims='x', description='a variable')

    expected = dedent("""\
    A variable

    - type : variable
    - intent : in
    - dims : (('x',),)
    - group : None
    - attrs : {}""")

    assert var_details(var) == expected


def test_process_repr(example_process_obj, processes_with_store,
                      example_process_repr, example_process_in_model_repr):
    assert repr_process(example_process_obj) == example_process_repr

    _, _, process_in_model = processes_with_store
    assert repr_process(process_in_model) == example_process_in_model_repr

    @xs.process
    class Dummy(object):
        def initialize(self):
            pass

        def run_step(self):
            pass

    expected = dedent("""\
    <Dummy  (xsimlab process)>
    Variables:
        *empty*
    Simulation stages:
        initialize
        run_step""")

    assert repr_process(Dummy()) == expected
