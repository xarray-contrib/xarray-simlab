import re
import os
from errno import ENOENT

import pytest
pytest.importorskip("graphviz")

from xsimlab.dot import to_graphviz, dot_graph, _hash_variable
from xsimlab.utils import variables_dict


# need to parse elements of graphivz's Graph object
g_node_label_re = re.compile(r'.*\[label=([\w<>\\"]*?)\s+.*\]')
g_edge_labels_re = re.compile(r'\s*([-\w]*?)\s+->\s+([-\w]*?)\s+.*]')


def _get_node_label(line):
    m = g_node_label_re.match(line)
    if m:
        return m.group(1)


def _get_edge_labels(line):
    m = g_edge_labels_re.match(line)
    if m:
        return m.group(1, 2)


def _get_graph_nodes(g):
    return list(filter(None, map(_get_node_label, g.body)))


def _get_graph_edges(g):
    return list(filter(None, map(_get_edge_labels, g.body)))


def _ensure_not_exists(filename):
    """
    Ensure that a file does not exist.
    """
    try:
        os.unlink(filename)
    except OSError as e:
        if e.errno != ENOENT:
            raise


def test_to_graphviz(model):
    g = to_graphviz(model)
    actual_nodes = _get_graph_nodes(g)
    actual_edges = _get_graph_edges(g)
    expected_nodes = list(model)
    expected_edges = [(dep_p_name, p_name)
                      for p_name, p_deps in model.dependent_processes.items()
                      for dep_p_name in p_deps]
    assert sorted(actual_nodes) == sorted(expected_nodes)
    assert set(actual_edges) == set(expected_edges)

    g = to_graphviz(model, show_inputs=True)
    actual_nodes = _get_graph_nodes(g)
    actual_edges = _get_graph_edges(g)
    expected_nodes += [var_name for _, var_name in model.input_vars]
    expected_edges += [
        (_hash_variable(variables_dict(type(model[p_name]))[var_name]), p_name)
        for p_name, var_name in model.input_vars
    ]
    assert sorted(actual_nodes) == sorted(expected_nodes)
    assert set(actual_edges) == set(expected_edges)

    g = to_graphviz(model, show_variables=True)
    actual_nodes = _get_graph_nodes(g)
    expected_nodes = list(model) + [var_name for _, var_name in model.all_vars]
    assert sorted(actual_nodes) == sorted(expected_nodes)

    g = to_graphviz(model, show_only_variable=('profile', 'u'))
    actual_nodes = _get_graph_nodes(g)
    expected_nodes = list(model) + ['u'] * 3
    assert sorted(actual_nodes) == sorted(expected_nodes)


def test_to_graphviz_attributes(model):
    assert to_graphviz(model).graph_attr['rankdir'] == 'LR'
    assert to_graphviz(model, rankdir='BT').graph_attr['rankdir'] == 'BT'


def test_dot_graph(model, tmpdir):
    ipydisp = pytest.importorskip('IPython.display')

    # Use a name that the shell would interpret specially to ensure that we're
    # not vulnerable to shell injection when interacting with `dot`.
    filename = str(tmpdir.join('$(touch should_not_get_created.txt)'))

    # Map from format extension to expected return type.
    result_types = {
        'png': ipydisp.Image,
        'jpeg': ipydisp.Image,
        'dot': type(None),
        'pdf': type(None),
        'svg': ipydisp.SVG,
    }
    for format in result_types:
        target = '.'.join([filename, format])
        _ensure_not_exists(target)
        try:
            result = dot_graph(model, filename=filename, format=format)

            assert not os.path.exists('should_not_get_created.txt')
            assert os.path.isfile(target)
            assert isinstance(result, result_types[format])
        finally:
            _ensure_not_exists(target)

    # format supported by graphviz but not by IPython
    with pytest.raises(ValueError) as excinfo:
        dot_graph(model, filename=filename, format='ps')
    assert "Unknown format" in str(excinfo.value)


def test_dot_graph_no_ipython(model):
    try:
        import IPython.display
    except ImportError:
        result = dot_graph(model)
        assert result is None


def test_dot_graph_no_filename(tmpdir, model):
    ipydisp = pytest.importorskip('IPython.display')

    # Map from format extension to expected return type.
    result_types = {
        'png': ipydisp.Image,
        'jpeg': ipydisp.Image,
        'dot': type(None),
        'pdf': type(None),
        'svg': ipydisp.SVG,
    }
    for format in result_types:
        before = tmpdir.listdir()
        result = dot_graph(model, filename=None, format=format)
        # We shouldn't write any files if filename is None.
        after = tmpdir.listdir()
        assert before == after
        assert isinstance(result, result_types[format])


def test_filenames_and_formats(model):
    ipydisp = pytest.importorskip('IPython.display')

    # Test with a variety of user provided args
    filenames = ['modelpdf', 'model.pdf', 'model.pdf', 'modelpdf',
                 'model.pdf.svg']
    formats = ['svg', None, 'svg', None, None]
    targets = ['modelpdf.svg', 'model.pdf', 'model.pdf.svg', 'modelpdf.png',
               'model.pdf.svg']

    result_types = {
        'png': ipydisp.Image,
        'jpeg': ipydisp.Image,
        'dot': type(None),
        'pdf': type(None),
        'svg': ipydisp.SVG,
    }

    for filename, format, target in zip(filenames, formats, targets):
        expected_result_type = result_types[target.split('.')[-1]]
        result = dot_graph(model, filename=filename, format=format)
        assert os.path.isfile(target)
        assert isinstance(result, expected_result_type)
        _ensure_not_exists(target)
