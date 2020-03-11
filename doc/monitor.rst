.. _monitor:

Monitor Model Runs
==================

Models may be complex, built from many processes and may take a while to
run. xarray-simlab provides functionality to help in monitoring model runs.

This section demonstrates how to use the built-in progress bar. Moreover, 
it exemplifies how customize your own custom monitoring.

.. ipython:: python
   :suppress:

    import sys
    sys.path.append('scripts')
    from advection_model import model2, model3

The following imports are necessary for the examples below.

.. ipython:: python

    import xsimlab as xs

.. ipython:: python
   :suppress:

    in_ds = xs.create_setup(
        model=model2,
        clocks={
            'time': np.linspace(0., 1., 5),
        },
        input_vars={
            'grid': {'length': 1.5, 'spacing': 0.01},
            'init': {'loc': 0.3, 'scale': 0.1},
            'advect__v': 1.
        },
    )


Progress bar
------------

:class:`~xsimlab.ProgressBar` is based on the `Tqdm`_ package and allows to track
the progress of simulation runs in ``xarray-simlab``.
It can be used as a context manager around simulation calls:

.. _Tqdm: https://github.com/tqdm/tqdm/

.. ipython::

   In [2]: with xs.ProgressBar():
      ...:     out_ds = in_ds.xsimlab.run(model=model2)

Alternatively, you can pass the progress bar via the ``hooks`` argument or use the
``register`` method (for more information, refer to the :ref:`custom_runtime_hooks` subsection)

``ProgressBar`` and the underlying Tqdm is built to work with different Python
interfaces. Use the optional argument ``frontend`` according to your
development environment.

- ``auto``: (default) Automatically detects environment.
- ``console``: When Python is run from the command line.
- ``gui``: Tqdm provides a gui version. According to the developers, this is
  still an experimental feature.
- ``notebook``: For use in a IPython/Jupyter notebook.

Additionally, you can customize the built-in progress bar, by supplying a
keyworded argument list to ``ProgressBar``, e.g.:

.. ipython::

   In [4]: with xs.ProgressBar(bar_format="{r_bar}"):
      ...:     out_ds = in_ds.xsimlab.run(model=model2)

For a full list of customization options, refer to the `Tqdm documentation`_

Note: The ``total`` argument cannot be changed to ensure best performance and
functionality.

.. _Tqdm documentation: https://tqdm.github.io

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

   In [3]: out_ds = in_ds.xsimlab.run(model=model2, hooks=[print_step_start])

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

   In [6]: out_ds = in_ds.xsimlab.run(model=model2, hooks=[print_steps])

An advantage over directly using hook functions is that you can also use an
instance of ``RuntimeHook`` either as a context manager over a model run
call or globally with its ``register`` method:

.. ipython::

   In [7]: with print_steps:
      ...:     out_ds = in_ds.xsimlab.run(model=model2)

   In [8]: print_steps.register()

   In [9]: out_ds = in_ds.xsimlab.run(model=model2)

   In [10]: print_steps.unregister()

   In [11]: out_ds = in_ds.xsimlab.run(model=model2)  # no print

Another advantage is that you can subclass ``RuntimeHook`` and add decorated
methods that may share some state:

.. ipython:: python
   :suppress:

   from runtime_hook_subclass import PrintStepTime

.. literalinclude:: scripts/runtime_hook_subclass.py
   :lines: 7-18

.. ipython:: python

   with PrintStepTime():
       in_ds.xsimlab.run(model=model2)
