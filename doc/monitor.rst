.. _monitor:

Monitor Model Runs
==================

Models may be complex, built from many processes and may take a while to
run. xarray-simlab provides functionality to help in monitoring model runs.

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

- ``model``: the Model instance currently running
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

   In [10]: print_steps.unregister()  # no print

   In [11]: out_ds = in_ds.xsimlab.run(model=model2)

Another advantage is that you can subclass ``RuntimeHook`` and add decorated
methods that may share some state

.. ipython::

   In [12]: import time

.. ipython:: python

   class PrintStepTime:
       @runtime_hook('run_step', 'model', 'pre')
       def start_step(self, model, context, state):
           self._start_time = time.time()
       @runtime_hook('run_step', 'model', 'post')
       def finish_step(self, model, context, state):
           step_time = time.time() - self._start_time
           print(f"Step {context['step']} took {step_time} seconds")

.. ipython::

   In [14]: #with PrintStepTime():
       ...: #    in_ds.xsimlab.run(model=model2)
