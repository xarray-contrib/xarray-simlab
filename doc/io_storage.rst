.. _io_storage:

Store Model Inputs and Outputs
==============================

Model inputs and/or outputs can be kept in memory or saved on disk using either
`xarray`_'s or `zarr`_'s I/O capabilities.

.. _xarray: http://xarray.pydata.org
.. _zarr: https://zarr.readthedocs.io/en/stable

.. ipython:: python
   :suppress:

    import sys
    sys.path.append('scripts')
    from advection_model import model2

Using xarray
------------

The :class:`xarray.Dataset` structure, used for both simulation inputs and
outputs, already supports serialization and I/O to several file formats, among
which netCDF_ is the recommended format. For more information, see the `reading
and writing files`_ section of xarray's docs.

.. _netCDF: https://www.unidata.ucar.edu/software/netcdf/
.. _`reading and writing files`: http://xarray.pydata.org/en/stable/io.html

Before showing some examples, let's first create the same initial setup than the
one used in section :doc:`run_model`:

.. ipython:: python

    model2

.. ipython:: python

    import xsimlab as xs
    ds = xs.create_setup(
        model=model2,
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

   ds.to_netcdf("model2_setup.nc")

So that you can reload this setup later or elsewhere before running a
simulation:

.. ipython:: python

   import xarray as xr
   in_ds = xr.load_dataset("model2_setup.nc")
   out_ds = in_ds.xsimlab.run(model=model2)
   out_ds

The latter dataset ``out_ds`` contains both the inputs and outputs of this model
run, that you can also write to a netCDF file or any other format supported by
xarray, e.g.,

.. ipython:: python

   out_ds.to_netcdf("model2_run.nc")

Using zarr
----------

When :meth:`xarray.Dataset.xsimlab.run` is called, xarray-simlab uses the zarr_
library to efficiently store (i.e., with compression) both the inputs and the
outputs. The output data is stored progressively as the simulation proceeds.

By default all this data is saved into memory. When the amount of model I/O data
is bigger, it is recommended to save the data on disk. For example, you can
specify a directory where to store it:

.. ipython:: python

   out_ds = in_ds.xsimlab.run(model=model2, output_store="model2_run.zarr")

You can also store the data in a temporary directory:

.. ipython:: python

   import zarr
   out_ds = in_ds.xsimlab.run(model=model2, output_store=zarr.TempStore())

Or you can directly use :func:`zarr.group` for more options, e.g., if you want
to overwrite a directory that has been used for old model runs:

.. ipython:: python

   zg = zarr.group("model2_run.zarr", overwrite=True)
   out_ds = in_ds.xsimlab.run(model=model2, output_store=zg)

.. note::

   The zarr library provides many storage alternatives, including support for
   distributed/cloud and database storage systems (see `storage alternatives`_
   in zarr's tutorial). Note, however, that a few alternatives won't work well
   with xarray-simlab. For example, :class:`zarr.ZipStore` doesn't support
   feeding a dataset once it has been created.

Regardless of the chosen alternative, :meth:`xarray.Dataset.xsimlab.run` returns
a ``xarray.Dataset`` containing the data (lazily) loaded from the zarr store:

.. ipython:: python

   out_ds

Zarr stores large multi-dimensional arrays as contiguous chunks. Xarray keeps track of
those chunks, which enables efficient, parallel post-processing or visualization
via the dask_ library (see the `parallel computing with Dask`_ section in
xarray's docs).

.. _`storage alternatives`: https://zarr.readthedocs.io/en/stable/tutorial.html#storage-alternatives
.. _`parallel computing with Dask`: http://xarray.pydata.org/en/stable/dask.html
.. _dask: https://dask.org/

.. ipython:: python
   :suppress:

   import os
   import shutil
   os.remove("model2_setup.nc")
   os.remove("model2_run.nc")
   shutil.rmtree("model2_run.zarr")
