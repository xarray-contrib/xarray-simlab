.. _progress:

Track simulation
================

It is useful to be able to track simulations, especially in longer more
convoluted setups.

First, import ``xsimlab``

.. ipython:: python

    import xsimlab as xs

.. ipython:: python
   :suppress:

    import fastscape
    from fastscape.models import basic_model

    nx = 101
    ny = 101

    in_ds = xs.create_setup(
        model=basic_model,
        clocks={
            'time': np.linspace(0., 1e9, 1001),
            'out': np.linspace(0., 1e9, 11)
        },
        master_clock='time',
        input_vars={
            'grid__shape': ('shape_yx', [101, 201]),
            'grid__length': ('shape_yx', [1e4, 2e4]),
            'boundary__status': ('border', ['looped', 'looped', 'fixed_value', 'fixed_value']),
            'uplift__rate': 1e-3,
            'spl': {'k_coef': 1e-4, 'area_exp': 0.4, 'slope_exp': 1.},
            'diffusion__diffusivity': 1e-1
        },
        output_vars={
            'out': ['topography__elevation',
                    'drainage__area',
                    'flow__basin'],
            None: ['boundary__border',
                   'grid__x',
                   'grid__y',
                   'spl__chi']
        }
    )

Tqdm progress bar
-------------------------

:class:`~xsimlab.ProgressBar` is based on the `Tqdm`_ package and allows to track
the progress of simulation runs in ``xarray-simlab``. ``ProgressBar`` was
implemented using simulation runtime hooks and thus, are easy to incorporate in
your computer models:

.. _Tqdm: https://github.com/tqdm/tqdm/

.. ipython::

   In [2]: pbar = xs.progress.ProgressBar()

   In [3]: out_ds = in_ds.xsimlab.run(model=basic_model, hooks=[pbar])

Similarly to ``RuntimeHook``, an instance of ``ProgressBar`` can be used either
as a context manager over a model run call or globally with its ``register``
method:

.. ipython::

   In [4]: with pbar:
      ...:     out_ds = in_ds.xsimlab.run(model=basic_model)

   In [5]: pbar.register()
      ...: out_ds = in_ds.xsimlab.run(model=basic_model)
      ...: pbar.unregister()

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

   In [4]: with xs.progress.ProgressBar(bar_format="{r_bar}"):
      ...:     out_ds = in_ds.xsimlab.run(model=basic_model)

For a full list of customization options, refer to the `Tqdm documentation`_

Note: The ``total`` argument cannot be changed to ensure best performance and
functionality.

.. _Tqdm documentation: https://tqdm.github.io
