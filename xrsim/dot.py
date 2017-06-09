"""
Model visualization using graphviz/dot.

Part of the code below is copied and modified from:

- dask 0.14.3 (Copyright (c) 2014-2015, Continuum Analytics, Inc.
  and contributors)
  Licensed under the BSD 3 License
  http://dask.pydata.org

"""
from __future__ import absolute_import, division, print_function

import os
from functools import partial

from .utils import import_required
from .variable.base import (AbstractVariable, ForeignVariable,
                            DiagnosticVariable, VariableGroup)


graphviz = import_required("graphviz", "Drawing dask graphs requires the "
                                       "`graphviz` python library and the "
                                       "`graphviz` system library to be "
                                       "installed.")


PROC_NODE_ATTRS = {'shape': 'oval', 'color': '#3454b4', 'fontcolor': '#131f43',
                   'style': 'filled', 'fillcolor': '#c6d2f6'}
PROC_EDGE_ATTRS = {'color': '#3454b4', 'style': 'bold'}
INPUT_NODE_ATTRS = {'shape': 'box', 'color': '#b49434', 'fontcolor': '#2d250d',
                    'style': 'filled', 'fillcolor': '#f3e3b3'}
INPUT_EDGE_ATTRS = {'arrowhead': 'none', 'color': '#b49434'}
VAR_NODE_ATTRS = {'shape': 'box', 'color': '#555555', 'fontcolor': '#555555'}
VAR_EDGE_ATTRS = {'arrowhead': 'none', 'color': '#555555'}


def hash_variable(var):
    return str(hash(var))


def _add_processes(g, model):
    seen = set()

    for proc_name, proc in model._processes.items():
        label = proc_name
        if proc_name not in seen:
            seen.add(proc_name)
            g.node(proc_name, label=label, **PROC_NODE_ATTRS)

        for dep_proc_name in model._dep_processes[proc_name]:
            if dep_proc_name not in seen:
                seen.add(dep_proc_name)
                dep_label = dep_proc_name
                g.node(dep_proc_name, label=dep_label, **PROC_NODE_ATTRS)
            g.edge(dep_proc_name, proc_name, **PROC_EDGE_ATTRS)


def _add_var(g, var, label, link_2_node, is_input=False):
    node_attrs = VAR_NODE_ATTRS.copy()
    edge_attrs = VAR_EDGE_ATTRS.copy()
    var_key = hash_variable(var)

    if is_input:
        node_attrs = INPUT_NODE_ATTRS.copy()
        edge_attrs = INPUT_EDGE_ATTRS.copy()
    elif isinstance(var, DiagnosticVariable):
        node_attrs['style'] = 'diagonals'
    elif isinstance(var, ForeignVariable):
        node_attrs['style'] = 'dashed'
        edge_attrs['style'] = 'dashed'
    elif isinstance(var, (tuple, VariableGroup)):
        node_attrs['shape'] = 'box3d'

    if not isinstance(var, tuple) and var.provided:
        edge_attrs.update({'arrowhead': 'empty'})
        edge_ends = link_2_node, var_key
    else:
        edge_ends = var_key, link_2_node

    g.node(var_key, label=label, **node_attrs)
    g.edge(*edge_ends, weight='200', **edge_attrs)


def _add_inputs(g, model):
    for proc_name, variables in model._input_vars.items():
        for var_name, var in variables.items():
            _add_var(g, var, var_name, proc_name, is_input=True)


def _add_variables(g, model):
    for proc_name, variables in model._processes.items():
        for var_name, var in variables.items():
            if model.is_input(var):
                continue
            _add_var(g, var, var_name, proc_name)

            if isinstance(var, (tuple, VariableGroup)):
                for v in var:
                    _add_var(g, v, '\<no_name\>', hash_variable(var))


def _add_var_and_foreign_vars(g, model, proc_name, var_name):
    variable = model[proc_name][var_name]

    if isinstance(variable, ForeignVariable):
        variable = variable.ref_var

    for p_name, variables in model._processes.items():
        for v_name, var in variables.items():
            if model.is_input(var):
                is_input = True
            else:
                is_input = False

            if var is variable or getattr(var, 'ref_var', None) is variable:
                _add_var(g, var, v_name, p_name, is_input=is_input)
            elif isinstance(var, (tuple, VariableGroup)):
                for v in var:
                    if v is variable or getattr(v, 'ref_var', None) is variable:
                        _add_var(g, var, v_name, p_name, is_input=is_input)
                        _add_var(g, v, '\<no_name\>', hash_variable(var))


def to_graphviz(model, rankdir='LR', show_only_variable=None,
                show_inputs=False, show_variables=False, graph_attr={},
                **kwargs):
    graph_attr = graph_attr or {}
    graph_attr['rankdir'] = rankdir
    graph_attr.update(kwargs)
    g = graphviz.Digraph(graph_attr=graph_attr)

    _add_processes(g, model)

    if show_only_variable is not None:
        if isinstance(show_only_variable, AbstractVariable):
            proc_name, var_name = model._get_proc_var_name(show_only_variable)
        else:
            proc_name, var_name = show_only_variable
        _add_var_and_foreign_vars(g, model, proc_name, var_name)

    else:
        if show_inputs:
            _add_inputs(g, model)

        if show_variables:
            _add_variables(g, model)

    return g


IPYTHON_IMAGE_FORMATS = frozenset(['jpeg', 'png'])
IPYTHON_NO_DISPLAY_FORMATS = frozenset(['dot', 'pdf'])


def _get_display_cls(format):
    """
    Get the appropriate IPython display class for `format`.

    Returns `IPython.display.SVG` if format=='svg', otherwise
    `IPython.display.Image`.

    If IPython is not importable, return dummy function that swallows its
    arguments and returns None.
    """
    dummy = lambda *args, **kwargs: None
    try:
        import IPython.display as display
    except ImportError:
        # Can't return a display object if no IPython.
        return dummy

    if format in IPYTHON_NO_DISPLAY_FORMATS:
        # IPython can't display this format natively, so just return None.
        return dummy
    elif format in IPYTHON_IMAGE_FORMATS:
        # Partially apply `format` so that `Image` and `SVG` supply a uniform
        # interface to the caller.
        return partial(display.Image, format=format)
    elif format == 'svg':
        return display.SVG
    else:
        raise ValueError("Unknown format '%s' passed to `dot_graph`" % format)


def dot_graph(model, filename=None, format=None, show_only_variable=None,
              show_inputs=False, show_variables=False, **kwargs):
    """
    Render a model as a graph using dot.
    If `filename` is not None, write a file to disk with that name in the
    format specified by `format`.  `filename` should not include an extension.

    Parameters
    ----------
    model : object
        The Model instance to display.
    filename : str or None, optional
        The name (without an extension) of the file to write to disk.  If
        `filename` is None (default), no file will be written, and we
        communicate with dot using only pipes.
    format : {'png', 'pdf', 'dot', 'svg', 'jpeg', 'jpg'}, optional
        Format in which to write output file.  Default is 'png'.
    show_only_variable : object or tuple, optional
        Show only a variable (and all other linked variables) given either
        as a Variable object or a tuple corresponding to process name and
        variable name. Deactivated by default.
    show_inputs : bool, optional
        If True, show all input variables in the graph (default: False).
        Ignored if `show_only_variable` is not None.
    show_variabless : bool, optional
        If True, show also the other variables (default: False).
        Ignored if `show_only_variable` is not None.
    **kwargs
        Additional keyword arguments to forward to `to_graphviz`.

    Returns
    -------
    result : None or IPython.display.Image or IPython.display.SVG  (See below.)

    Notes
    -----
    If IPython is installed, we return an IPython.display object in the
    requested format.  If IPython is not installed, we just return None.
    We always return None if format is 'pdf' or 'dot', because IPython can't
    display these formats natively. Passing these formats with filename=None
    will not produce any useful output.

    See Also
    --------
    to_graphviz

    """
    g = to_graphviz(model, show_only_variable=show_only_variable,
                    show_inputs=show_inputs, show_variables=show_variables,
                    **kwargs)

    if filename is None:
        filename = ''

    fmts = ['.png', '.pdf', '.dot', '.svg', '.jpeg', '.jpg']
    if format is None and any(filename.lower().endswith(fmt) for fmt in fmts):
        filename, format = os.path.splitext(filename)
        format = format[1:].lower()

    if format is None:
        format = 'png'

    data = g.pipe(format=format)
    if not data:
        raise RuntimeError("Graphviz failed to properly produce an image. "
                           "This probably means your installation of graphviz "
                           "is missing png support. See: "
                           "https://github.com/ContinuumIO/anaconda-issues/"
                           "issues/485 for more information.")

    display_cls = _get_display_cls(format)

    if not filename:
        return display_cls(data=data)

    full_filename = '.'.join([filename, format])
    with open(full_filename, 'wb') as f:
        f.write(data)

    return display_cls(filename=full_filename)
