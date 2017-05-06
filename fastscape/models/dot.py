"""
Part of the code below is copied and modified from:

- dask 0.14.3 (Copyright (c) 2014-2015, Continuum Analytics, Inc.
  and contributors)
  Licensed under the BSD 3 License
  http://dask.pydata.org

"""
from __future__ import absolute_import, division, print_function

import os
from functools import partial

from dask.core import get_dependencies
from dask.utils import import_required


graphviz = import_required("graphviz", "Drawing dask graphs requires the "
                                       "`graphviz` python library and the "
                                       "`graphviz` system library to be "
                                       "installed.")


def name(x):
    try:
        return str(hash(x))
    except TypeError:
        return str(hash(str(x)))


def to_graphviz(model, data_attributes=None, function_attributes=None,
                rankdir='BT', graph_attr={}, node_attr=None, edge_attr=None,
                **kwargs):
    if data_attributes is None:
        data_attributes = {}
    if function_attributes is None:
        function_attributes = {}

    graph_attr = graph_attr or {}
    graph_attr['rankdir'] = rankdir
    graph_attr.update(kwargs)
    g = graphviz.Digraph(graph_attr=graph_attr,
                         node_attr=node_attr,
                         edge_attr=edge_attr)

    seen = set()
    dsk = model._get_dsk()

    for k, v in dsk.items():
        k_name = name(k)
        k_label = "'{}'\n{}()".format(
            k, model._processes[k].__class__.__name__
        )
        if k_name not in seen:
            seen.add(k_name)
            g.node(k_name, label=k_label, shape='ellipse',
                   **data_attributes.get(k, {}))

        for dep in get_dependencies(dsk, k):
            dep_name = name(dep)
            if dep_name not in seen:
                seen.add(dep_name)
                dep_label = "'{}'\n{}()".format(
                    dep, model._processes[dep].__class__.__name__
                )
                g.node(dep_name, label=dep_label, shape='ellipse',
                       **data_attributes.get(dep, {}))
            g.edge(dep_name, k_name)
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


def dot_graph(model, filename='mymodel', format=None, **kwargs):
    """
    Render a task graph using dot.
    If `filename` is not None, write a file to disk with that name in the
    format specified by `format`.  `filename` should not include an extension.

    Parameters
    ----------
    dsk : dict
        The graph to display.
    filename : str or None, optional
        The name (without an extension) of the file to write to disk.  If
        `filename` is None, no file will be written, and we communicate with
        dot using only pipes.  Default is 'mydask'.
    format : {'png', 'pdf', 'dot', 'svg', 'jpeg', 'jpg'}, optional
        Format in which to write output file.  Default is 'png'.
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
    dask.dot.to_graphviz
    """
    g = to_graphviz(model, **kwargs)

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
