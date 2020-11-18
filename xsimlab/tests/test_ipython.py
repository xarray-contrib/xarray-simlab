import pytest

pytest.importorskip("IPython")
from IPython.testing.globalipapp import start_ipython


@pytest.fixture(scope="session")
def session_ip():
    yield start_ipython()


@pytest.fixture(scope="function")
def ip(session_ip):
    session_ip.run_line_magic(magic_name="load_ext", line="xsimlab.ipython")
    yield session_ip
    session_ip.run_line_magic(magic_name="unload_ext", line="xsimlab.ipython")
    session_ip.run_line_magic(magic_name="reset", line="-f")


model_cell = """
import xsimlab as xs

@xs.process
class Foo:
    v1 = xs.variable(dims='x', description="v1 description", default=1.2, static=True)
    v2 = xs.variable(description="v2 description", attrs={'units': 'm'})

@xs.process
class Bar:
    v1 = xs.variable(default='z')

my_model = xs.Model({'foo': Foo, 'bar': Bar})
"""


@pytest.fixture
def model_ip(ip):
    ip.run_cell(raw_cell=model_cell)
    yield ip


def test_create_setup_magic_error(ip):
    with pytest.raises(KeyError, match=".*not defined or not imported"):
        ip.run_line_magic(magic_name="create_setup", line="missing_model")

    ip.run_cell(raw_cell="model = 'not a Model object'")

    with pytest.raises(TypeError, match=".*not a xsimlab.Model object"):
        ip.run_line_magic(magic_name="create_setup", line="model")


cell_input = """
import xsimlab as xs

ds_in = xs.create_setup(
    model=my_model,
    clocks={},
    input_vars={
        'foo__v1': ,
        'foo__v2': ,
        'bar__v1': ,
    },
    output_vars={}
)
"""


cell_input_skip_default = """
import xsimlab as xs

ds_in = xs.create_setup(
    model=my_model,
    clocks={},
    input_vars={
        'foo__v2': ,
    },
    output_vars={}
)
"""


cell_input_default = """
import xsimlab as xs

ds_in = xs.create_setup(
    model=my_model,
    clocks={},
    input_vars={
        'foo__v1': 1.2,
        'foo__v2': ,
        'bar__v1': 'z',
    },
    output_vars={}
)
"""


cell_input_nested = """
import xsimlab as xs

ds_in = xs.create_setup(
    model=my_model,
    clocks={},
    input_vars={
        'foo': {
            'v1': ,
            'v2': ,
        },
        'bar': {
            'v1': ,
        },
    },
    output_vars={}
)
"""


cell_input_verbose1 = """
import xsimlab as xs

ds_in = xs.create_setup(
    model=my_model,
    clocks={},
    input_vars={
        # v1 description
        'foo__v1': ,
        # v2 description
        'foo__v2': ,
        # ---
        'bar__v1': ,
    },
    output_vars={}
)
"""


cell_input_verbose2 = """
import xsimlab as xs

ds_in = xs.create_setup(
    model=my_model,
    clocks={},
    input_vars={
        # v1 description
        #     dimensions: ('x',)
        #     static: master clock dimension not supported
        'foo__v1': ,
        # v2 description
        'foo__v2': ,
        # ---
        'bar__v1': ,
    },
    output_vars={}
)
"""


cell_input_verbose3 = """
import xsimlab as xs

ds_in = xs.create_setup(
    model=my_model,
    clocks={},
    input_vars={
        # v1 description
        #     dimensions: ('x',)
        #     static: master clock dimension not supported
        'foo__v1': ,
        # v2 description
        #     units: m
        'foo__v2': ,
        # ---
        'bar__v1': ,
    },
    output_vars={}
)
"""


@pytest.mark.parametrize(
    "line,expected_cell_input",
    [
        ("my_model", cell_input),
        ("my_model --skip-default", cell_input_skip_default),
        ("my_model --default", cell_input_default),
        ("my_model --nested", cell_input_nested),
        ("my_model --verbose", cell_input_verbose1),
        ("my_model -vv", cell_input_verbose2),
        ("my_model -vvv", cell_input_verbose3),
    ],
)
def test_create_setup_magic(model_ip, mocker, line, expected_cell_input):
    import IPython.core.interactiveshell

    mocker.patch("IPython.core.interactiveshell.InteractiveShell.set_next_input")

    model_ip.run_line_magic("create_setup", line)

    expected = "# %create_setup " + line + expected_cell_input

    patched_func = IPython.core.interactiveshell.InteractiveShell.set_next_input
    patched_func.assert_called_once_with(expected, replace=True)
