.. _run_model:

Setup and Run Models
====================

This section shows how to create new settings (either from scratch or
from existing settings) and run simulations with :class:`~xsimlab.Model`
instances, using the xarray extension provided by xarray-simlab. We'll
use here the simple advection models that we have created in section
:doc:`create_model`.

.. ipython:: python
   :suppress:

    import sys
    sys.path.append('scripts')
    from advection_model import model2, model3

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

In this example we use the ``model2`` Model instance:

.. ipython:: python

    model2

The convenient :func:`~xsimlab.create_setup` function can be used to
create a new setup in a very declarative way:

.. ipython:: python

    in_ds = xs.create_setup(
        model=model2,
        clocks={'time': np.linspace(0., 1., 101),
                'otime': [0, 0.5, 1]},
        master_clock='time',
        input_vars={'grid': {'length': 1.5, 'spacing': 0.01},
                    'init': {'loc': 0.3, 'scale': 0.1},
                    'advect': {'v': 1.}},
        output_vars={None: {'grid': 'x'},
                     'otime': {'profile': 'u'}}
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
middle and end). The time-independent x-coordinate values of the grid
will be saved as well.

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

    out_ds = in_ds.xsimlab.run(model=model2)

The returned dataset contains all the variables of the input
dataset. It also contains simulation outputs as new or updated data
variables, e.g., ``grid__x`` and ``profile__u`` in this example:

.. ipython:: python

    out_ds

Post-processing and plotting
----------------------------

A great advantage of using xarray Datasets is that it is
straightforward to include the simulation as part of a processing
pipeline, i.e., by chaining ``xsimlab.run()`` with other methods
that can also be applied on Dataset objects.

As an example, instead of a data variable ``grid__x`` it would be
nicer to save the grid :math:`x` values as a coordinate in the output
dataset:

.. ipython:: python

    out_ds = (in_ds.xsimlab.run(model=model2)
                   .set_index(x='grid__x'))
    out_ds

All convenient methods provided by xarray are directly accessible,
e.g., for plotting snapshots:

.. ipython:: python

    def plot_u(ds):
        fig, axes = plt.subplots(ncols=3, figsize=(10, 3))
        for t, ax in zip(ds.otime, axes):
            ds.profile__u.sel(otime=t).plot(ax=ax)
        fig.tight_layout()
        return fig

    @savefig run_model.png width=100%
    plot_u(out_ds);

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
    with model2:
        out_ds2 = (in_ds.xsimlab.update_vars(input_vars=in_vars)
                        .xsimlab.run()
                        .set_index(x='grid__x'))

.. note::

   For convenience, a Model instance may be used in a context instead
   of providing it repeatedly as an argument of xarray-simlab's
   functions or methods in which it is required.

We plot the results to compare this simulation with the previous one
(note the numerical dissipation as a side-effect of the Lax scheme,
which is more visible here):

.. ipython:: python

    @savefig run_model2.png width=100%
    plot_u(out_ds2);

Update time dimensions
~~~~~~~~~~~~~~~~~~~~~~

:meth:`.xsimlab.update_clocks` allows to only update the time
dimensions and/or their coordinates. Here below we set other values
for the ``otime`` coordinate (which serves to take snapshots of
:math:`u`):

.. ipython:: python

    clocks = {'otime': [0, 0.25, 0.5]}
    with model2:
        out_ds3 = (in_ds.xsimlab.update_clocks(clocks=clocks,
                                               master_clock='time')
                        .xsimlab.run()
                        .set_index(x='grid__x'))
    @savefig run_model3.png width=100%
    plot_u(out_ds3);

Use an alternative model
~~~~~~~~~~~~~~~~~~~~~~~~

A model and its alternative versions often keep inputs in common. It
this case too, it would make sense to create an input dataset from an
existing dataset, e.g., by dropping data variables that are irrelevant
(see :meth:`.xsimlab.filter_vars`) and by adding data variables for
inputs that are present only in the alternative model.

Here is an example of simulation using ``model3`` (source point and
flat initial profile for :math:`u`) instead of ``model2`` :

.. ipython:: python

    in_vars = {'source': {'loc': 1., 'flux': 100.}}
    with model3:
        out_ds4 = (in_ds.xsimlab.filter_vars()
                        .xsimlab.update_vars(input_vars=in_vars)
                        .xsimlab.run()
                        .set_index(x='grid__x'))
    @savefig run_model4.png width=100%
    plot_u(out_ds4);

Time-varying input values
-------------------------

All model inputs accept arrays which have a dimension that corresponds
to the master clock.

The example below is based on the last example above, but instead of
being fixed, the flux of :math:`u` at the source point decreases over
time at a fixed rate:

.. ipython:: python

    flux = 100. - 100. * in_ds.time
    in_vars = {'source': {'loc': 1., 'flux': flux}}
    with model3:
        out_ds5 = (in_ds.xsimlab.filter_vars()
                        .xsimlab.update_vars(input_vars=in_vars)
                        .xsimlab.run()
                        .set_index(x='grid__x'))
    @savefig run_model5.png width=100%
    plot_u(out_ds5);
