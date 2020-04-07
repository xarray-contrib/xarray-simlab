.. _about:

About xarray-simlab
===================

xarray-simlab provides a framework to easily build custom
computational models from a collection of modular components, called
processes.

It also provides an extension to `xarray <https://xarray.pydata.org>`_ (i.e.,
labeled arrays and datasets), that connects it to a wide range of Python
libraries for processing, analysis, visualization, etc.

xarray-simlab is well integrated with other libraries of the PyData
ecosystem such as `dask <https://docs.dask.org>`_ and `zarr
<https://zarr.readthedocs.io>`_.

In a nutshell
-------------

The Conway's Game of Life example shown below is adapted from this
`blog post <https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/>`_
by Jake VanderPlas.

1. Create new model components by writing compact Python classes,
   i.e., very much like dataclasses_ (note: more features and Python <
   3.7 support are also available through the `attrs
   <https://www.attrs.org>`_ library):

.. ipython::

    In [1]: import numpy as np
       ...: import xsimlab as xs
       ...:
       ...:
       ...: @xs.process
       ...: class GameOfLife:
       ...:     world = xs.variable(dims=('x', 'y'), intent='inout')
       ...:
       ...:     def run_step(self):
       ...:         nbrs_count = sum(
       ...:             np.roll(np.roll(self.world, i, 0), j, 1)
       ...:             for i in (-1, 0, 1) for j in (-1, 0, 1)
       ...:             if (i != 0 or j != 0)
       ...:         )
       ...:         self._world_next = (nbrs_count == 3) | (self.world & (nbrs_count == 2))
       ...:
       ...:     def finalize_step(self):
       ...:         self.world[:] = self._world_next
       ...:
       ...:
       ...: @xs.process
       ...: class Glider:
       ...:     pos = xs.variable(dims='point_xy', description='glider position')
       ...:     world = xs.foreign(GameOfLife, 'world', intent='out')
       ...:
       ...:     def initialize(self):
       ...:         x, y = self.pos
       ...:
       ...:         kernel = [[1, 0, 0],
       ...:                   [0, 1, 1],
       ...:                   [1, 1, 0]]
       ...:
       ...:         self.world = np.zeros((10, 10), dtype=bool)
       ...:         self.world[x:x+3, y:y+3] = kernel

2. Create a new model just by providing a dictionary of model components:

.. ipython::

    In [2]: model = xs.Model({'gol': GameOfLife,
       ...:                   'init': Glider})

3. Create an input :py:class:`xarray.Dataset`, run the model and get an
   output Dataset:

.. ipython::

    In [3]: input_dataset = xs.create_setup(
       ...:     model=model,
       ...:     clocks={'step': np.arange(9)},
       ...:     input_vars={'init__pos': ('point_xy', [4, 5])},
       ...:     output_vars={'gol__world': 'step'}
       ...: )
       ...:
       ...: output_dataset = input_dataset.xsimlab.run(model=model)
       ...:
       ...: output_dataset

4. Perform model setup, pre-processing, run, post-processing and
   visualization in a functional style, using method chaining:

.. ipython::

    @savefig gol.png width=4in
    In [5]: import matplotlib.pyplot as plt
       ...:
       ...: with model:
       ...:     (input_dataset
       ...:      .xsimlab.update_vars(
       ...:          input_vars={'init__pos': ('point_xy', [2, 2])}
       ...:      )
       ...:      .xsimlab.run()
       ...:      .gol__world.plot.imshow(
       ...:          col='step', col_wrap=3, figsize=(5, 5),
       ...:          xticks=[], yticks=[],
       ...:          add_colorbar=False, cmap=plt.cm.binary)
       ...:     )

.. _dataclasses: https://docs.python.org/3/library/dataclasses.html

Motivation
----------

xarray-simlab is a tool for *fast model development* and *easy,
interactive model exploration*. It aims at empowering scientists to do
better research in less time, collaborate efficiently and make new
discoveries.

**Fast model development**: xarray-simlab allows building new models
from re-usable sets of components, with minimal effort. Models are
created dynamically and instantly just by plugging in/out components,
always keeping the model structure and interface tidy even in
situations where the model development workflow is highly experimental
or organic.

**Interactive model exploration**: xarray-simlab is being developed
with the idea of reducing the gap between the environments used for
building and running computational models and the ones used for
processing, analyzing and visualizing simulation results. Users may
fully leverage powerful environments like jupyter_ at all stages of
their modeling workflow.

.. _jupyter: https://jupyter.org/

Sources of inspiration
----------------------

xarray-simlab leverages the great number of packages that are part of the
Python scientific ecosystem. More specifically, the packages below have been
great sources of inspiration for this project.

- xarray_: xarray-simlab actually provides an xarray extension for
  setting up and running models.
- attrs_: a package that allows writing Python classes without
  boilerplate. xarray-simlab uses and extends attrs for writing
  processes as succinct Python classes.
- luigi_: the concept of Luigi is to use Python classes as re-usable units that
  help building complex workflows. xarray-simlab's concept is similar, but
  here it is specific to computational (numerical) modeling.
- django_ (not really a scientific package): the way that model
  processes are designed in xarray-simlab has been initially inspired
  from Django's ORM (i.e., the ``django.db.models`` part).
- param_: another source of inspiration for the interface of processes
  (more specifically the variables that it defines).
- climlab_: another python package for process-oriented modeling, which uses
  the same approach although having a slightly different design/API, and which
  is applied to climate modeling.
- landlab_: like climlab it provides a framework for building model
  components but it is here applied to landscape evolution
  modeling. It already has a great list of components ready to use.
- dask_: represents fine-grained processing tasks as Directed Acyclic Graphs
  (DAGs). xarray-simlab models are DAGs too, where the nodes are interdepent
  processes. In this project we actually borrow some code from dask
  for resolving process dependencies and for model visualization.

.. _luigi: https://github.com/spotify/luigi
.. _django: https://github.com/django/django
.. _param: https://github.com/ioam/param
.. _climlab: https://github.com/brian-rose/climlab
.. _landlab: https://github.com/landlab/landlab
