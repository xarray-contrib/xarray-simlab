from textwrap import dedent

import attr

import xsimlab as xs
from xsimlab.formatting import (
    add_attribute_section,
    maybe_truncate,
    pretty_print,
    repr_process,
    repr_model,
    var_details,
    wrap_indent,
)
from xsimlab.process import get_process_obj


def test_maybe_truncate():
    assert maybe_truncate("test", 10) == "test"
    assert maybe_truncate("longteststring", 10) == "longtes..."


def test_pretty_print():
    assert pretty_print("test", 10) == "test" + " " * 6


def test_wrap_indent():
    text = "line1\nline2"

    expected = "-line1\n line2"
    assert wrap_indent(text, start="-") == expected

    expected = "line1\n line2"
    assert wrap_indent(text, length=1) == expected


def test_var_details():
    @xs.process
    class P:
        var = xs.variable(
            dims=[(), "x"],
            description="a variable",
            default=0,
            groups=["g1", "g2"],
            static=True,
            attrs={"units": "m"},
            encoding={"fill_value": -1},
        )
        var2 = xs.variable()

    var_details_str = var_details(attr.fields(P).var)

    expected = dedent(
        """\
        A variable

        Variable properties:

        - type : ``variable``
        - intent : ``in``
        - dimensions : () or ('x',)
        - groups : g1, g2
        - default value : 0
        - static : ``True``

        Other attributes:

        - units : m

        Encoding options:

        - fill_value : -1
        """
    )

    assert var_details_str == expected

    @xs.process
    class PP:
        var = xs.foreign(P, "var2")

    var_details_str = var_details(attr.fields(PP).var)

    expected = dedent(
        """\
        No description given

        Variable properties:

        - type : ``foreign``
        - reference variable : :attr:`test_var_details.<locals>.P.var2`
        - intent : ``in``
        - dimensions : ()
        """
    )

    assert var_details_str == expected


@xs.process(autodoc=False)
class WithoutPlaceHolder:
    """My process"""

    var1 = xs.variable(dims="x", description="a variable")
    var2 = xs.variable()


@xs.process(autodoc=False)
class WithPlaceholder:
    """My process

    {{attributes}}

    """

    var1 = xs.variable(dims="x", description="a variable")
    var2 = xs.variable()


def test_add_attribute_section():
    # For testing, autodoc is set to False to avoid redundancy
    expected = """My process

    Attributes
    ----------
    var1 : :class:`attr.Attribute`
        A variable

        Variable properties:

        - type : ``variable``
        - intent : ``in``
        - dimensions : ('x',)

    var2 : :class:`attr.Attribute`
        No description given

        Variable properties:

        - type : ``variable``
        - intent : ``in``
        - dimensions : ()
    """

    assert add_attribute_section(WithoutPlaceHolder).strip() == expected.strip()
    assert add_attribute_section(WithPlaceholder).strip() == expected.strip()


def test_process_repr(
    example_process_obj,
    processes_with_state,
    example_process_repr,
    example_process_in_model_repr,
):
    assert repr_process(example_process_obj) == example_process_repr

    _, _, process_in_model = processes_with_state
    assert repr_process(process_in_model) == example_process_in_model_repr

    @xs.process
    class Dummy:
        def initialize(self):
            pass

        def run_step(self):
            pass

    expected = dedent(
        """\
    <Dummy  (xsimlab process)>
    Variables:
        *empty*
    Simulation stages:
        initialize
        run_step
    """
    )

    assert repr_process(get_process_obj(Dummy)) == expected


def test_model_repr(simple_model, simple_model_repr):
    assert repr_model(simple_model) == simple_model_repr

    expected = "<xsimlab.Model (0 processes, 0 inputs)>\n"
    assert repr(xs.Model({})) == expected
