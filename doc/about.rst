.. _about:

About xarray-simlab
===================

xarray-simlab provides a framework for easily building custom computational
models from a set of re-usable components (i.e., Python classes), called
processes.

The framework handles issues that scientists who are developing models should
not care too much about, like the model interface - including the data structure
used - and the overall workflow management. Both are automatically determined
from the succint, declarative-like interfaces of the model processes.

Next versions of xarray-simlab will hopefully handle other technical issues
like logging simulation progress, command line integration and running (many)
simulations in parallel.

Motivation
----------

xarray-simlab is being developped with the idea of reducing the gap between the
environments used for building and running computational models and the ones
used for processing and analysing simulation results. It also encourages
building new models from re-usable components and avoid reinventing the wheel.

The design of this tool is mainly focused on fast development and simulation
setting(s), which would ultimately optimize the iterative, back-and-forth
process between ideas on how to model a particular phenomenon and insights
from the exploration of model behavior.

Sources of inspiration
----------------------

xarray-simlab leverages the great number of packages that are part of the
Python scientific ecosystem. More specifically, the packages below have been
great sources of inspiration for this project.

- xarray_: xarray-simlab actually provides an xarray extension for setting and
  running models.
- luigi_: the concept of Luigi is to use Python classes as re-usable units that
  help building complex workflows. xarray-simlab's concept is similar, but
  here it is specific to computational (numerical) modeling.
- django_ (not really a scientific package): the way that model processes are
  designed in xarray-simlab is heavily inspired from Django's ORM (i.e., the
  ``django.db.models`` part).
- param_: another source of inspiration for the interface of processes
  (more specifically the variables that it defines).
- climlab_: another python package for process-oriented modeling, which uses
  the same approach although having a slightly different design/API, and which
  is applied to climate modeling.
- dask_: represents fine-grained processing tasks as Directed Acyclic Graphs
  (DAGs). xarray-simlab models are DAGs too, where the nodes are interdepent
  processes. In this project we actually borrow some code from dask
  for resolving process dependencies and for model visualization.

.. _xarray: https://github.com/pydata/xarray
.. _dask: https://github.com/dask/dask
.. _luigi: https://github.com/spotify/luigi
.. _django: https://github.com/django/django
.. _param: https://github.com/ioam/param
.. _climlab: https://github.com/brian-rose/climlab
