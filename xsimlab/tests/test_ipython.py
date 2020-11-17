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
    v1 = xs.variable(description="v1 description", default=1.2)
    v2 = xs.variable(description="v2 description")

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


cell_input_all = """
import xsimlab as xs

ds_in = xs.create_setup(
    model=my_model,
    clocks={},
    input_vars={
        # v1 description
        'foo__v1': 1.2,
        # v2 description
        'foo__v2': ,
        # ---
        'bar__v1': 'z',
    },
    output_vars={}
)
"""


cell_input_all_nested = """
import xsimlab as xs

ds_in = xs.create_setup(
    model=my_model,
    clocks={},
    input_vars={
        'foo': {
            # v1 description
            'v1': 1.2,
            # v2 description
            'v2': ,
        },
        'bar': {
            # ---
            'v1': 'z',
        },
    },
    output_vars={}
)
"""


@pytest.mark.parametrize(
    "line,expected_cell_input",
    [
        ("my_model", cell_input),
        ("my_model --default --comment", cell_input_all),
        ("my_model -d -c", cell_input_all),
        ("my_model --default --comment --nested", cell_input_all_nested),
        ("my_model -d -c -n", cell_input_all_nested),
    ],
)
def test_create_setup_magic(model_ip, mocker, line, expected_cell_input):
    import IPython.core.interactiveshell

    mocker.patch("IPython.core.interactiveshell.InteractiveShell.set_next_input")

    model_ip.run_line_magic("create_setup", line)

    expected = "# %create_setup " + line + expected_cell_input

    patched_func = IPython.core.interactiveshell.InteractiveShell.set_next_input
    patched_func.assert_called_once_with(expected, replace=True)
