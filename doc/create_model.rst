.. _create_model:

Creating models
===============

xarray-simlab's framework provides a few Python base classes, e.g.,
:class:`~xrsimlab.Variable`, :class:`~xrsimlab.Process` and
:class:`~xrsimlab.Model` that can used togheter to create fully operational
models.

A ``Model`` is a collection of processes that each define an interface with the
all the "public" variables it needs, update or provide for its proper
computation.

Process
-------




Variable
--------

Generally, the ``state``, ``value``, ``rate`` and ``change`` properties should
not be used (either get or set the value) outside of Process subclasses,
or maybe only for debugging.

Foreign Variable
~~~~~~~~~~~~~~~~

ForeignVariable.state return the same object (usually a numpy array) than
Variable.state (replace class names by variable names in processes).
ForeignVariable.state is actually a shortcut to ForeignVariable.ref_var.state.

Variable List and Group
~~~~~~~~~~~~~~~~~~~~~~~


Model
-----

The latter class represents a model with which we interact using the xarray's
:class:`~xarray.Dataset` structure.
