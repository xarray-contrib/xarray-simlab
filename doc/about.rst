.. _about:

About xarray-simlab
===================

xarray-simlab provides a framework for easily building custom models from a set
of re-usable components (i.e., Python classes), called processes.

The framework handles issues that scientists who develop models should not care
too much about, like a model interface and its data structure as well as its
overall workflow management. Both are automatically determined from the
succint, declarative-like interfaces of the model processes.

Next versions of xarray-simlab will hopefully also handle other technical issues
like logging, command line integration and running (many) simulations in
parallel.

Motivation
----------
xarray-simlab is being developped with the idea of reducing the gap between the
environment used for building and running a computational model and the one(s)
used for processing and analysing simulation results. It also encourages
building new models from re-usable components and avoid reinventing the wheel.

Ultimately, it should reduce the time lag between the ideas that scientists have
on how to model a particular phenomenon and the insights they get from exploring
the behavior of the model, which often itself lead to new modeling ideas.

Sources of inspiration
----------------------

- xarray_: data structure
- dask_: build and run task graphs (DAGs)
- luigi_: use Python classes as re-usable units that help building complex
  workflows.
- django_: especially The Django's ORM part (i.e., django.db.models) for the
  design of Process and Variable classes
- param_: another source of inspiration for Process interface and Variable objects.
- climlab_: another python package for process-oriented modeling, applied to
  climate.

.. _xarray: https://github.com/pydata/xarray
.. _dask: https://github.com/dask/dask
.. _luigi: https://github.com/spotify/luigi
.. _django: https://github.com/django/django
.. _param: https://github.com/ioam/param
.. _climlab: https://github.com/brian-rose/climlab

**Draft: put this in create models section**

The framework consists of a few base classes, e.g., :py:class:`xrsimlab.Variable`,
:py:class:`xrsimlab.Process` and :py:class:`xrsimlab.Model`. The latter class
represents a model with which we interact using the xarray's
:py:class:`xarray.Dataset` structure.

Variable.state (or value, rate, change) should not be used (get/set) outside
of Process subclasses.

ForeignVariable.state return the same object (usually a numpy array) than
Variable.state (replace class names by variable names in processes).
ForeignVariable.state is actually a shortcut to ForeignVariable.ref_var.state.
