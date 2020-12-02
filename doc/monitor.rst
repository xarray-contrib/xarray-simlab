.. _monitor:

Monitor Model Runs
==================

Models may be complex, built from many processes and may take a while to
run. xarray-simlab provides functionality to help in monitoring model runs.

This section demonstrates how to use the built-in progress bar. Moreover, 
it exemplifies how to create your own custom monitoring.

.. ipython:: python
   :suppress:

    import sys
    sys.path.append('scripts')
    from advection_model import advect_model, advect_model_src

Let's use the following setup for the examples below. It is based on the
``advect_model`` created in Section :ref:`create_model`.

.. ipython:: python

    import xsimlab as xs

    in_ds = xs.create_setup(
        model=advect_model,
        clocks={
            'time': np.linspace(0., 1., 6),
        },
        input_vars={
            'grid': {'length': 1.5, 'spacing': 0.01},
            'init': {'loc': 0.3, 'scale': 0.1},
            'advect__v': 1.
        },
        output_vars={'profile__u': 'time'}
    )


Progress bar
------------

:class:`~xsimlab.monitoring.ProgressBar` is based on the `Tqdm`_ package and
allows to track the progress of simulation runs in ``xarray-simlab``. It can be
used as a context manager around simulation calls:

.. _Tqdm: https://tqdm.github.io

.. ipython:: python

   from xsimlab.monitoring import ProgressBar

.. ipython:: python
   :suppress:

   from progress_bar_hack import ProgressBarHack as ProgressBar

.. ipython:: python

   with ProgressBar():
       out_ds = in_ds.xsimlab.run(model=advect_model)

Alternatively, you can pass the progress bar via the ``hooks`` argument of
``Dataset.xsimlab.run()`` or you can use the ``register`` method (for more
information, refer to Section :ref:`custom_runtime_hooks`).

``ProgressBar`` and the underlying Tqdm tool are built to work with different
Python front-ends. Use the optional argument ``frontend`` depending on your
environment:

- ``auto``: automatically selects the front-end (default)
- ``console``: renders the progress bar as text
- ``gui``: progress rich rendering (experimental), which needs matplotlib_ to be
  installed
- ``notebook``: for use within IPython/Jupyter notebooks, which needs
  ipywidgets_ to be installed

.. _matplotlib: https://matplotlib.org/
.. _ipywidgets: https://ipywidgets.readthedocs.io/en/stable/

Additionally, you can customize the built-in progress bar by supplying
keyword arguments list to ``ProgressBar``, e.g.:

.. ipython:: python

   with ProgressBar(bar_format="{desc}|{bar}{r_bar}"):
       out_ds = in_ds.xsimlab.run(model=advect_model)

For a full list of customization options, refer to the `Tqdm documentation`_.

.. _Tqdm documentation: https://tqdm.github.io

.. note::

   Currently this progress bar doesn't support tracking the progress of batches
   of simulations. If those batches are run in parallel you can
   use Dask's diagnostics instead.

.. _custom_runtime_hooks:

Custom runtime hooks
--------------------

Custom monitoring can be implemented using simulation runtime hooks.

The :func:`~xsimlab.runtime_hook` decorator allows a function to be called once
or repeatedly at specific times during a simulation. The simple example below
prints the step number as the simulation proceeds:

.. ipython::

   In [2]: @xs.runtime_hook("run_step", "model", "pre")
      ...: def print_step_start(model, context, state):
      ...:     print(f"Starting execution of step {context['step']}")
      ...:

   In [3]: out_ds = in_ds.xsimlab.run(model=advect_model, hooks=[print_step_start])

Runtime hook functions are always called with the following 3 arguments:

- ``model``: the instance of :class:`~xsimlab.Model` that is running
- ``context``: a read-only dictionary that contains information about simulation
  runtime (see :func:`~xsimlab.runtime` for a list of available keys)
- ``state``: a read-only dictionary that contains the simulation state, where
  keys are tuples in the form ``('process_name', 'variable_name')``.

An alternative to the ``runtime_hook`` decorator is the
:class:`~xsimlab.RuntimeHook` class. You can create new instances with any
number of hook functions, e.g.,

.. ipython::

   In [4]: @xs.runtime_hook("run_step", "model", "post")
      ...: def print_step_end(model, context, state):
      ...:     print(f"Finished execution of step {context['step']}")
      ...:

   In [5]: print_steps = xs.RuntimeHook(print_step_start, print_step_end)

   In [6]: out_ds = in_ds.xsimlab.run(model=advect_model, hooks=[print_steps])

An advantage over directly using hook functions is that you can also use an
instance of ``RuntimeHook`` either as a context manager over a model run
call or globally with its ``register`` method:

.. ipython::

   In [7]: with print_steps:
      ...:     out_ds = in_ds.xsimlab.run(model=advect_model)

   In [8]: print_steps.register()

   In [9]: out_ds = in_ds.xsimlab.run(model=advect_model)

   In [10]: print_steps.unregister()

   In [11]: out_ds = in_ds.xsimlab.run(model=advect_model)  # no print

Another advantage is that you can subclass ``RuntimeHook`` and add decorated
methods that may share some state:

.. ipython:: python
   :suppress:

   from runtime_hook_subclass import PrintStepTime

.. literalinclude:: scripts/runtime_hook_subclass.py
   :lines: 7-18

.. ipython:: python

   with PrintStepTime():
       in_ds.xsimlab.run(model=advect_model)


Control simulation runtime
--------------------------

Runtime hook functions may return a :class:`~xsimlab.RuntimeSignal` so that you
can control the simulation workflow (e.g., skip the current stage or process,
break the simulation time steps) based on some condition or some computed value.

In the example below, the simulation stops as soon as the gaussian pulse (peak
value) has been advected past ``x = 0.4``.

.. ipython::

   In [2]: @xs.runtime_hook("run_step", "model", "post")
      ...: def maybe_stop(model, context, state):
      ...:     peak_idx = np.argmax(state[('profile', 'u')])
      ...:     peak_x = state[('grid', 'x')][peak_idx]
      ...:
      ...:     if peak_x > 0.4:
      ...:         print("Peak crossed x=0.4, stop simulation!")
      ...:         return xs.RuntimeSignal.BREAK
      ...:

   In [3]: out_ds = in_ds.xsimlab.run(
      ...:     model=advect_model,
      ...:     hooks=[print_step_start, maybe_stop]
      ...: )

Even when a simulation stops early like in the example above, the resulting
xarray Dataset still contains all time steps defined in the input Dataset.
Output variables have fill (masked) values for the time steps that were not run,
as shown below with the ``nan`` values for ``profile__u`` (fill values are not
stored physically in the Zarr output store).

.. ipython:: python

   out_ds
