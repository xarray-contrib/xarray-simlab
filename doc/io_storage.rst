.. _io_storage:

Store Model Inputs and Outputs
==============================

Simulation inputs and/or outputs can be kept in memory or saved on disk using
either `xarray`_'s or `zarr`_'s I/O capabilities.

.. _xarray: http://xarray.pydata.org
.. _zarr: https://zarr.readthedocs.io/en/stable

.. ipython:: python
   :suppress:

    import sys
    sys.path.append('scripts')
    from advection_model import advect_model

Using xarray
------------

The :class:`xarray.Dataset` structure, used for both simulation inputs and
outputs, already supports serialization and I/O to several file formats, among
which netCDF_ is the recommended format. For more information, see Section
`reading and writing files`_ in xarray's docs.

.. _netCDF: https://www.unidata.ucar.edu/software/netcdf/
.. _`reading and writing files`: http://xarray.pydata.org/en/stable/io.html

Before showing some examples, let's first create the same initial setup than the
one used in Section :doc:`run_model`:

.. ipython:: python

    advect_model

.. ipython:: python

    import xsimlab as xs
    ds = xs.create_setup(
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
            'grid__x': None,
            'profile__u': 'otime'
        }
    )
    ds

You can save the dataset here above, e.g., using :meth:`xarray.Dataset.to_netcdf`

.. ipython:: python

   ds.to_netcdf("advect_model_setup.nc")

You can then reload this setup later or elsewhere before starting a new
simulation:

.. ipython:: python

   import xarray as xr
   in_ds = xr.load_dataset("advect_model_setup.nc")
   out_ds = in_ds.xsimlab.run(model=advect_model)
   out_ds

The latter dataset ``out_ds`` contains both the inputs and the outputs of this
model run. Likewise, You can write it to a netCDF file or any other format
supported by xarray, e.g.,

.. ipython:: python

   out_ds.to_netcdf("advect_model_run.nc")

.. _io_storage_zarr:

Using zarr
----------

When :meth:`xarray.Dataset.xsimlab.run` is called, xarray-simlab uses the zarr_
library to efficiently store (i.e., with compression) both simulation input and
output data. Input data is stored before the simulation starts and output data
is stored progressively as the simulation proceeds.

By default all this data is saved into memory. For large amounts of model I/O
data, however, it is recommended to save the data on disk. For example, you can
specify a directory where to store it:

.. ipython:: python

   out_ds = in_ds.xsimlab.run(model=advect_model, store="advect_model_run.zarr")

You can also store the data in a temporary directory:

.. ipython:: python

   import zarr
   out_ds = in_ds.xsimlab.run(model=advect_model, store=zarr.TempStore())

Or you can directly use :func:`zarr.group` for more options, e.g., if you want
to overwrite a directory that has been used for old model runs:

.. ipython:: python

   zg = zarr.group("advect_model_run.zarr", overwrite=True)
   out_ds = in_ds.xsimlab.run(model=advect_model, store=zg)

.. note::

   The zarr library provides many storage alternatives, including support for
   distributed/cloud and database storage systems (see `storage alternatives`_
   in zarr's tutorial). Note, however, that a few alternatives won't work well
   with xarray-simlab. For example, :class:`zarr.ZipStore` doesn't support
   feeding a zarr dataset once it has been created.

Regardless of the chosen alternative, :meth:`xarray.Dataset.xsimlab.run` returns
a ``xarray.Dataset`` object that contains the data (lazily) loaded from the zarr
store:

.. ipython:: python

   out_ds

Zarr stores large multi-dimensional arrays as contiguous chunks. When opened as
a ``xarray.Dataset``, xarray keeps track of those chunks, which enables efficient
and parallel post-processing via the dask_ library (see Section `parallel
computing with Dask`_ in xarray's docs).

.. _`storage alternatives`: https://zarr.readthedocs.io/en/stable/tutorial.html#storage-alternatives
.. _`parallel computing with Dask`: http://xarray.pydata.org/en/stable/dask.html
.. _dask: https://dask.org/

.. ipython:: python
   :suppress:

   import os
   import shutil
   os.remove("advect_model_setup.nc")
   os.remove("advect_model_run.nc")
   shutil.rmtree("advect_model_run.zarr")

Advanced usage
--------------

Dynamically sized arrays
~~~~~~~~~~~~~~~~~~~~~~~~

Model variables may have one or several of their dimension(s) dynamically
resized during a simulation. When saving those variables as outputs, the
corresponding zarr datasets may be resized so that, at the end of the
simulation, all values are stored in large arrays of fixed shape and possibly
containing missing values (note: depending on chunk size, zarr doesn't need to
physically store all regions of contiguous missing values).

The example below illustrates how such variables are returned as outputs:

.. ipython::

   In [2]: import numpy as np

   In [3]: @xs.process
      ...: class Particles:
      ...:     """Generate at each step a random number of particles
      ...:     at random positions along an axis.
      ...:     """
      ...:
      ...:     position = xs.variable(dims='pt', intent='out')
      ...:
      ...:     def initialize(self):
      ...:         self._rng = np.random.default_rng(123)
      ...:
      ...:     def run_step(self):
      ...:         nparticles = self._rng.integers(1, 4)
      ...:         self.position = self._rng.uniform(0, 10, size=nparticles)
      ...:

   In [4]: model = xs.Model({'pt': Particles})

   In [5]: with model:
      ...:     in_ds = xs.create_setup(clocks={'steps': range(4)},
      ...:                             output_vars={'pt__position': 'steps'})
      ...:     out_ds = in_ds.xsimlab.run()
      ...:

   In [6]: out_ds.pt__position

N-dimensional arrays with missing values might not be the best format for
dealing with this kind of output data. It could still be converted into a denser
format, like for example a :class:`pandas.DataFrame` with a multi-index thanks
to the xarray Dataset or DataArray :meth:`~xarray.Dataset.stack`,
:meth:`~xarray.Dataset.dropna` and :meth:`~xarray.Dataset.to_dataframe` methods:

.. ipython::

   In [7]: (out_ds.stack(particles=('steps', 'pt'))
      ...:        .dropna('particles')
      ...:        .to_dataframe())

.. _io_storage_encoding:

Encoding options
~~~~~~~~~~~~~~~~

It is possible to control via some encoding options how Zarr stores simulation
data.

Those options can be set for variables declared in process classes. See the
``encoding`` parameter of :func:`~xsimlab.variable` for all available options.
In the example below we specify a custom fill value for the ``position``
variable, which will be used to replace missing values:

.. ipython::

   In [4]: @xs.process
      ...: class Particles:
      ...:     position = xs.variable(dims='pt', intent='out',
      ...:                            encoding={'fill_value': -1.0})
      ...:
      ...:     def initialize(self):
      ...:         self._rng = np.random.default_rng(123)
      ...:
      ...:     def run_step(self):
      ...:         nparticles = self._rng.integers(1, 4)
      ...:         self.position = self._rng.uniform(0, 10, size=nparticles)
      ...:

   In [5]: model = xs.Model({'pt': Particles})

   In [6]: with model:
      ...:     in_ds = xs.create_setup(clocks={'steps': range(4)},
      ...:                             output_vars={'pt__position': 'steps'})
      ...:     out_ds = in_ds.xsimlab.run()
      ...:

   In [7]: out_ds.pt__position

Encoding options may also be set or overridden when calling
:func:`~xarray.Dataset.xsimlab.run`, e.g.,

.. ipython::

   In [8]: out_ds = in_ds.xsimlab.run(
      ...:     model=model,
      ...:     encoding={'pt__position': {'fill_value': -10.0}}
      ...: )
      ...:

   In [9]: out_ds.pt__position
