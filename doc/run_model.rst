.. _run_model:

Setup and Run Models
====================

This section shows how to create new settings (either from scratch or
from existing settings) and run simulations with :class:`~xsimlab.Model`
instances, using the xarray extension provided by xarray-simlab. We'll
use here the simple advection models that we have created in Section
:doc:`create_model`.

.. ipython:: python
   :suppress:

    import sys
    sys.path.append('scripts')
    from advection_model import advect_model, advect_model_src

The following imports are necessary for the examples below.

.. ipython:: python

    import numpy as np
    import xsimlab as xs
    import matplotlib.pyplot as plt

.. note::

   When the ``xsimlab`` package is imported, it registers a new
   namespace named ``xsimlab`` for :class:`xarray.Dataset` objects. As
   shown below, this namespace is used to access all xarray-simlab
   methods that can be applied to those objects.


Create a new setup from scratch
-------------------------------

In this example we use the ``advect_model`` Model instance:

.. ipython:: python

    advect_model

The convenient :func:`~xsimlab.create_setup` function can be used to
create a new setup in a very declarative way:

.. ipython:: python

    in_ds = xs.create_setup(
        model=advect_model,
        clocks={
            'time': np.linspace(0., 1., 101),
            'otime': [0, 0.5, 1]
        },
        master_clock='time',
        input_vars={
            'grid': {'length': 1.5, 'spacing': 0.01},
            'init': {'loc': 0.3, 'scale': 0.1},
            'advect__v': 1.
        },
        output_vars={
            'profile__u': 'otime'
        }
    )

A setup consists in:

- one or more time dimensions ("clocks") and their given coordinate
  values ;
- one of these time dimensions, defined as master clock, which will be
  used to define the simulation time steps (the other time dimensions
  usually serve to take snapshots during a simulation on a different
  but synchronized clock) ;
- values given for input variables ;
- one or more variables for which we want to take snapshots on given
  clocks (time dimension) or just once at the end of the simulation
  (``None``).

In the example above, we set ``time`` as the master clock dimension
and ``otime`` as another dimension for taking snapshots of :math:`u`
along the grid at three given times of the simulation (beginning,
middle and end).

``create_setup`` returns all these settings packaged into a
:class:`xarray.Dataset` :

.. ipython:: python

    in_ds

If defined in the model, variable metadata such as description are
also added in the dataset as attributes of the corresponding data
variables, e.g.,

.. ipython:: python

    in_ds.advect__v

Run a simulation
----------------

A new simulation is run by simply calling the :meth:`.xsimlab.run`
method from the input dataset created above. It returns a new dataset:

.. ipython:: python

    out_ds = in_ds.xsimlab.run(model=advect_model)

The returned dataset contains all the variables of the input
dataset. It also contains simulation outputs as new or updated data
variables, e.g., ``profile__u`` in this example:

.. ipython:: python

    out_ds

Note also the ``x`` coordinate present in this output dataset. ``x`` is declared
in ``advect_model.grid`` as an index variable and therefore has been
automatically added as a coordinate in the dataset.

Post-processing and plotting
----------------------------

A great advantage of using xarray Datasets is that it is straightforward to
include the simulation as part of a processing pipeline, i.e., by chaining
``xsimlab.run()`` with other methods that can also be applied on Dataset (or
DataArray) objects.

For example, we can extract the values of ``profile__u`` at a given position on
the grid (and clearly notice the advection of the pulse):

.. ipython:: python

    out_ds.profile__u.sel(x=0.75)

Or plot the whole profile for all snapshots:

.. ipython:: python

    @savefig run_advect_model.png width=100%
    out_ds.profile__u.plot(col='otime', figsize=(9, 3));

There is a huge number of features available for selecting data, computation,
plotting, I/O, and more, see `xarray's documentation`_!

.. _`xarray's documentation`: https://xarray.pydata.org/en/stable/

Reuse existing settings
-----------------------

Update inputs
~~~~~~~~~~~~~

In the following example, we set and run another simulation in which
we decrease the advection velocity down to 0.5. Instead of creating a
new setup from scratch, we can reuse the one created previously and
update only the value of velocity, thanks to
:meth:`.xsimlab.update_vars`.

.. ipython:: python

    in_vars = {('advect', 'v'): 0.5}
    with advect_model:
        out_ds2 = (in_ds.xsimlab.update_vars(input_vars=in_vars)
                        .xsimlab.run())

.. note::

   For convenience, a Model instance may be used in a context instead
   of providing it repeatedly as an argument of xarray-simlab's
   functions or methods in which it is required.

We plot the results to compare this simulation with the previous one
(note the numerical dissipation as a side-effect of the Lax scheme,
which is more visible here):

.. ipython:: python

    @savefig run_advect_model_input.png width=100%
    out_ds2.profile__u.plot(col='otime', figsize=(9, 3));

Update time dimensions
~~~~~~~~~~~~~~~~~~~~~~

:meth:`.xsimlab.update_clocks` allows to only update the time
dimensions and/or their coordinates. Here below we set other values
for the ``otime`` coordinate (which serves to take snapshots of
:math:`u`):

.. ipython:: python

    clocks = {'otime': [0, 0.25, 0.5]}
    with advect_model:
        out_ds3 = (in_ds.xsimlab.update_clocks(clocks=clocks,
                                               master_clock='time')
                        .xsimlab.run())
    @savefig run_advect_model_clock.png width=100%
    out_ds3.profile__u.plot(col='otime', figsize=(9, 3));

Use an alternative model
~~~~~~~~~~~~~~~~~~~~~~~~

A model and its alternative versions often keep inputs in common. It
this case too, it would make sense to create an input dataset from an
existing dataset, e.g., by dropping data variables that are irrelevant
(see :meth:`.xsimlab.filter_vars`) and by adding data variables for
inputs that are present only in the alternative model.

Here is an example of simulation using ``advect_model_src`` (source point and
flat initial profile for :math:`u`) instead of ``advect_model`` :

.. ipython:: python

    in_vars = {'source': {'loc': 1., 'flux': 100.}}
    with advect_model_src:
        out_ds4 = (in_ds.xsimlab.filter_vars()
                        .xsimlab.update_vars(input_vars=in_vars)
                        .xsimlab.run())
    @savefig run_advect_model_alt.png width=100%
    out_ds4.profile__u.plot(col='otime', figsize=(9, 3));

.. _time_varying_inputs:

Time-varying input values
-------------------------

Except for static variables, all model inputs accept arrays which have a
dimension that corresponds to the master clock. This is useful for adding
external forcing.

The example below is based on the last example above, but instead of
being fixed, the flux of :math:`u` at the source point decreases over
time at a fixed rate:

.. ipython:: python

    flux = 100. - 100. * in_ds.time
    in_vars = {'source': {'loc': 1., 'flux': flux}}
    with advect_model_src:
        out_ds5 = (in_ds.xsimlab.filter_vars()
                        .xsimlab.update_vars(input_vars=in_vars)
                        .xsimlab.run())
    @savefig run_advect_model_time.png width=100%
    out_ds5.profile__u.plot(col='otime', figsize=(9, 3));

.. _run_batch:

Run multiple simulations
------------------------

Besides a time dimension, model inputs may also accept another extra dimension
that is used to run batches of simulations. This is very convenient for
sensitivity analyses: the inputs and results from all simulations are neatly
combined into one xarray Dataset object. Another advantage is that those
simulations can be run in parallel easily, see Section
:ref:`run_parallel_multi`.

.. note::

   Because of the limitations of the xarray data model, model inputs with a
   "batch" dimension may not work well if these directly or indirectly affect
   the shape of other variables defined in the model (e.g., grid size).

As a simple example, let's update the setup for the advection model and set
different values for velocity:

.. ipython:: python

    in_ds_vel = in_ds.xsimlab.update_vars(
        model=advect_model,
        input_vars={'advect__v': ('batch', [1.0, 0.5, 0.2])}
    )

Those values are defined along a dimension named "batch", that we need to
explicitly pass to :func:`xarray.Dataset.xsimlab.run` via its ``batch_dim``
parameter in order to run one simulation for each value of velocity:

.. ipython:: python

    out_ds_vel = in_ds_vel.xsimlab.run(model=advect_model, batch_dim='batch')
    out_ds_vel

Note the additional ``batch`` dimension in the resulting dataset for the
``profile__u`` variable.

Having all simulations results in a single Dataset allows to fully leverage
xarray's powerful capabilities for analysis and plotting those results. For
example, the one-liner expression below plots the profile of all snapshots
(columns) from all simulations (rows):

.. ipython:: python

    @savefig run_advect_model_batch.png width=100%
    out_ds_vel.profile__u.plot(row='batch', col='otime', figsize=(9, 6));

Advanced examples
~~~~~~~~~~~~~~~~~

Running batches of simulations works well with time-varying input values,
since the time and batch dimensions are orthogonal.

It is also possible to run multiple simulations by varying the value of several
model inputs, e.g., with different value combinations for the advection velocity
and the initial location of the pulse:

.. ipython:: python

    in_ds_comb = in_ds.xsimlab.update_vars(
        model=advect_model,
        input_vars={'init__loc': ('batch', [0.3, 0.6, 0.9]),
                    'advect__v': ('batch', [1.0, 0.5, 0.2])}
    )
    out_ds_comb = in_ds_comb.xsimlab.run(model=advect_model, batch_dim='batch')

    @savefig run_advect_model_comb.png width=100%
     out_ds_comb.profile__u.plot(row='batch', col='otime', figsize=(9, 6));

Using :meth:`xarray.Dataset.stack` and :meth:`xarray.Dataset.unstack`
respectively before and after ``run``, it is straightforward to regularly sample
a n-dimensional parameter space (i.e., from combinations obtained by the cartesian
product of values along each parameter dimension). Note the dimensions of
``profile__u`` in the example below, which include the parameter space:

.. ipython:: python

    in_vars = {'init__loc': ('init__loc', [0.3, 0.6, 0.9]),
               'advect__v': ('advect__v', [1.0, 0.5, 0.2])}
    with advect_model:
        out_ds_nparams = (
            in_ds
            .xsimlab.update_vars(input_vars=in_vars)
            .stack(batch=['init__loc', 'advect__v'])
            .xsimlab.run(batch_dim='batch')
            .unstack('batch')
        )
    out_ds_nparams

