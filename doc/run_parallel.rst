.. _run_parallel:

Run Model(s) in Parallel
========================

xarray-simlab allows running one or more models in parallel via the Dask_
library. There are two parallel modes:

- run one simulation in parallel (single-model parallelism)
- run a batch of simulations in parallel (multi-model parallelism)

.. warning::

   This is an experimental feature.

.. note::

   Dask is a versatile library that provides many ways of executing tasks in
   parallel (i.e., threads vs. processes, single machine vs. distributed
   infrastructure). xarray-simlab lets you choose which alternative best suits
   your needs. Beware, however, that not all alternatives are optimal or
   supported depending on your case. More details below.

.. _Dask: https://docs.dask.org/en/latest/

.. _run_parallel_single:

Single-model parallelism
------------------------

This mode runs the processes in a model in parallel.

A :class:`~xsimlab.Model` object can be viewed as a Directed Acyclic Graph (DAG)
built from a collection of processes (i.e., process-decorated classes) as nodes
and their inter-dependencies as oriented edges. At each simulation stage, a task
graph is built from this DAG, which is then executed by one of the schedulers
available in Dask.

To activate this parallel mode, simply set ``parallel=True`` when calling
:func:`xarray.Dataset.xsimlab.run`:

.. code:: python

   >>> in_ds.xsimlab.run(model=my_model, parallel=True)

The default Dask scheduler used here is ``"threads"``. The code in the
process-decorated classes must thus be thread-safe. It should also release
CPython's Global Interpreter Lock (GIL) as much as possible in order to see a
gain in performance. For example, most Numpy functions release the GIL.

Other schedulers may be selected via the ``scheduler`` argument of
:func:`~xarray.Dataset.xsimlab.run`.

.. code:: python

   >>> in_ds.xsimlab.run(model=my_model, parallel=True, scheduler="processes")

Dask also provides other ways to select a scheduler, see `here
<https://docs.dask.org/en/latest/setup/single-machine.html>`_.

Note, however, that multi-processes or distributed schedulers are not well
supported and may have very poor performance for this mode, depending on how
much simulation active data needs to be shared between the model components. See
:meth:`xsimlab.Model.execute` for more information.

The gain in performance compared to sequential execution of the model processes
will also depend on how the DAG is structured, i.e., how many processes can be
executed in parallel. Visualizing the DAG helps a lot, see Section
:ref:`inspect_model_visualize`.

.. _run_parallel_multi:

Multi-models parallelism
------------------------

This mode runs multiple simulations in parallel, using the same model but
different input values.

.. note::

   This mode should scale well from a few dozen to a few thousand of
   simulations but it has not been tested yet beyond that level.

.. note::

   It may not work well with dynamic-sized arrays.

This parallel mode is automatically selected when a batch dimension label is set
while calling :func:`xarray.Dataset.xsimlab.run` (see Section
:ref:`run_batch`). You still need to explicitly set ``Parallel=True``:

.. code:: python

   >>> in_ds.xsimlab.run(model=my_model, batch_dim="batch", parallel=True, store="output.zarr")

Both multi-threads and multi-processes Dask schedulers are well supported for
this embarrassingly parallel problem. Like for single-model parallelism, the
default scheduler used here is ``"threads"``.

If you use a multi-threads scheduler, the same precautions apply regarding
thread-safety and CPython's GIL.

If you use a multi-processes scheduler, beware of the following:

- The code in the process-decorated classes must be serializable.
- Not all Zarr stores are supported for model outputs, see `Zarr's documentation
  <https://zarr.readthedocs.io/en/stable/api/storage.html>`_. For example, the
  default in-memory store is not supported. See Section :ref:`io_storage_zarr`
  on how to specify an alternative store.
- By default, the chunk size of Zarr datasets along the batch dimension is equal
  to 1 in order to prevent race conditions during parallel writes. This might
  not be optimal for further post-processing, though. It is possible to override
  this default value and set larger chunk sizes via the ``encoding`` parameter
  of :func:`~xarray.Dataset.xsimlab.run`, but then you should also use one of
  the Zarr's synchronizers (either :class:`zarr.sync.ThreadSynchronizer` or
  :class:`zarr.sync.ProcessSynchronizer`) to ensure that all output values will
  be properly saved.
