.. _whats_new:

Release Notes
=============

v0.4.1 (17 April 2020)
----------------------

Enhancements
~~~~~~~~~~~~

- Added :attr:`xsimlab.Model.cache` public property (:issue:`125`).
- Parameter ``input_vars`` of :func:`~xsimlab.create_setup` and
  :func:`xarray.Dataset.xsimlab.update_vars` now accepts array-like values
  with no explicit dimension label(s), in this case those labels are inferred
  from model variables' metadata (:issue:`126`).
- Single-model parallelism now supports Dask's multi-processes or distributed
  schedulers, although this is still limited and rarely optimal (:issue:`127`).
- Improved auto-generated docstrings of variables declared in process classes
  (:issue:`130`).

Bug fixes
~~~~~~~~~

- Fix running batches of simulations using ``dask.distributed`` (:issue:`124`).
- Fix rendering of auto-generated docstrings of process classes (:issue:`128`).
- Fix tests with ``attr`` v20.1.0 (:issue:`129`).

v0.4.0 (7 April 2020)
---------------------

This is a big release which includes many exciting new features built on top of
great Python libraries. It is now possible to set default, validate or convert
model input values just like regular attributes in `attrs
<https://www.attrs.org>`_, save model input/outputs with `zarr
<https://zarr.readthedocs.io>`_, run model(s) in parallel using `dask
<https://docs.dask.org>`_, monitor model runs with a `tqdm
<https://tqdm.github.io/>`_ progress bar, and much more!

Breaking changes
~~~~~~~~~~~~~~~~

- Python 3.6 is now the oldest supported version (:issue:`70`).
- The keys of the dictionary returned by
  :attr:`xarray.Dataset.xsimlab.output_vars` now correspond to variable names,
  and the values are clock dimension labels or ``None`` (previously the
  dictionary was formatted the other way around).
  :attr:`xarray.Dataset.xsimlab.output_vars_by_clock` has been added for
  convenience (:issue:`85`, :issue:`103`).

Depreciations
~~~~~~~~~~~~~

- Using the ``group`` parameter in :func:`xsimlab.variable` and
  :func:`xsimlab.on_demand` is depreciated; use ``groups`` instead.
- Providing a dictionary with clock dimensions or ``None`` as keys to
  ``output_vars`` in :func:`xarray.Dataset.xsimlab.update_vars()` and
  :func:`xsimlab.create_setup()` is depreciated. Variable names should be used
  instead (:issue:`85`).

Enhancements
~~~~~~~~~~~~

- It is now possible to assign multiple groups to a single variable
  (:issue:`71`).
- The xarray interface may now handle default values that are defined in model
  variables (:issue:`72`). A new method
  :func:`xarray.Dataset.xsimlab.reset_vars` allows to (re)populate an input
  Dataset with variables and their default values. :func:`~xsimlab.create_setup`
  has also a new ``fill_default`` parameter.
- Added static variables, i.e., variables that don't accept time-varying input
  values (:issue:`73`).
- Added support for the validation of variable values (given as inputs and/or
  set through foreign variables), reusing :func:`attr.validate` (:issue:`74`,
  :issue:`79`). Validation is optional and is controlled by the parameter
  ``validate`` added to :func:`xarray.Dataset.xsimlab.run`.
- Check or automatically transpose the dimensions of the variables given in
  input xarray Datasets to match those defined in model variables (:issue:`76`).
  This is optional and controlled by the parameter ``check_dims`` added
  to :func:`xarray.Dataset.xsimlab.run`.
- More consistent dictionary format for output variables in the xarray
  extension (:issue:`85`).
- %-formatting and str.format() code has been converted into formatted string
  literals (f-strings) (:issue:`90`).
- :func:`~xsimlab.foreign` has been updated so that it sets its description and
  its metadata from the variable it refers to (:issue:`91`, :issue:`107`).
- The ``autodoc`` parameter of the :func:`xsimlab.process` decorator now allows
  to automatically add an attributes section to the docstring of the class to
  which the decorator is applied, using the metadata of each variable declared
  in the class (:issue:`67`).
- Added :func:`~xsimlab.validators.in_bounds` and
  :func:`~xsimlab.validators.is_subdtype` validators (:issue:`87`).
- :func:`xsimlab.variable` has now a ``converter`` parameter that can be used to
  convert any input value before (maybe) validating it and setting the variable
  (:issue:`92`).
- Added :func:`xsimlab.index` for setting index variables (e.g., coordinate
  labels). Using the xarray extension, those variables are automatically added
  in the output Dataset as coordinates (:issue:`94`).
- Added simulation runtime hooks (:issue:`95`). Hooks can be created by using
  either the :func:`~xsimlab.runtime_hook` decorator or the
  :class:`~xsimlab.RuntimeHook` class.
- Added some useful properties and methods to the ``xarray.Dataset.xsimlab``
  extension (:issue:`103`).
- Save model inputs/outputs using zarr (:issue:`102`, :issue:`111`,
  :issue:`113`).
- Added :class:`~xsimlab.monitoring.ProgressBar` to track simulation progress
  (:issue:`104`, :issue:`110`).
- Added the ability to easily run batches of simulations using the ``batch_dim``
  parameter of :func:`xarray.Dataset.xsimlab.run` (:issue:`115`).
- Added 'object' variables :func:`~xsimlab.any_object` for sharing arbitrary
  Python objects between processes (:issue:`118`).
- Run one or multiple simulations in parallel using Dask (:issue:`119`).

Bug fixes
~~~~~~~~~

- Remove ``attrs`` 19.2.0 depreciation warning (:issue:`68`).
- Fix compatibility with xarray 0.14.1 (:issue:`69`).
- Avoid update in-place attributes in original/input xarray Datasets
  (:issue:`101`).

Maintenance
~~~~~~~~~~~

- Switched to GitHub Actions for continuous integration and Codecov for
  coverage (:issue:`86`).

v0.3.0 (30 September 2019)
--------------------------

Breaking changes
~~~~~~~~~~~~~~~~

- It is now possible to use class inheritance to customize a process
  without re-writing the class from scratch and without breaking the
  links between (foreign) variables when replacing the process in a
  model (:issue:`45`). Although it should work just fine in most
  cases, there are potential caveats. This should be considered as an
  experimental, possibly breaking change.
- ``Model.initialize``, ``Model.run_step``, ``Model.finalize_step``
  and ``Model.finalize`` have been removed in favor of
  ``Model.execute`` (:issue:`59`).

Depreciations
~~~~~~~~~~~~~

- ``run_step`` methods defined in process classes won't accept anymore
  current step duration as a positional argument by default. Use the
  ``runtime`` decorator if you need current step duration (and/or
  other runtime information) inside the method (:issue:`59`).

Enhancements
~~~~~~~~~~~~

- Ensure that there is no ``intent`` conflict between the variables
  declared in a model. This check is explicit at Model creation and a
  more meaningful error message is shown when it fails (:issue:`57`).
- Added ``runtime`` decorator to pass simulation runtime information
  to the (runtime) methods defined in process classes (:issue:`59`).
- Better documentation with a minimal, yet illustrative example based
  on Game of Life (:issue:`61`).
- A class decorated with ``process`` can now be instantiated
  independently of any Model object. This is very useful for testing
  and debugging (:issue:`63`).

Bug fixes
~~~~~~~~~

- Fixed compatibility with xarray 0.13.0 (:issue:`54`).
- Fixed compatibility with pytest >= 4 (:issue:`56`).

v0.2.1 (7 November 2018)
------------------------

Bug fixes
~~~~~~~~~

- Fix an issue after a change in attrs 0.18.2 (:issue:`47`).

v0.2.0 (9 May 2018)
-------------------

Highlights
~~~~~~~~~~

This release includes a major refactoring of both the internals and
the API on how processes and variables are defined and depends on
each other in a model. xarray-simlab now uses and extends
attrs_ (:issue:`33`).

Also, Python 3.4 support has been dropped. It may still work with that
version but it is not actively tested anymore and it is not packaged
with conda.

Breaking changes
~~~~~~~~~~~~~~~~

As xarray-simlab is still at an early development stage and hasn't
been adopted "in production" yet (to our knowledge), we haven't gone
through any depreciation cycle, which by the way would have been
almost impossible for such a major refactoring. The following breaking
changes are effective now!

- ``Variable``, ``ForeignVariable`` and ``VariableGroup`` classes have
  been replaced by ``variable``, ``foreign`` and ``group`` factory
  functions (wrappers around ``attr.ib``), respectively.
- ``VariableList`` has been removed and has not been replaced by
  anything equivalent.
- ``DiagnosticVariable`` has been replaced by ``on_demand`` and the
  ``diagnostic`` decorator has been replaced by the variable's
  ``compute`` decorator.
- The ``provided`` (``bool``) argument (variable constructors) has
  been replaced by ``intent`` (``{'in', 'out', 'inout'}``).
- The ``allowed_dims`` argument has been renamed to ``dims`` and is
  now optional (a scalar value is expected by default).
- The ``validators`` argument has been renamed to ``validator`` to be
  consistent with ``attr.ib``.
- The ``optional`` argument has been removed. Variables that don't
  require an input value may be defined using a special validator
  function (see ``attrs`` documentation).
- Variable values are not anymore accessed using three different
  properties ``state``, ``rate`` and ``change`` (e.g.,
  ``self.foo.state``). Instead, all variables accept a unique value,
  which one can get/set by simply using the variable name (e.g.,
  ``self.foo``). Now multiple variables have to be declared for
  holding different values.

- Process classes are now defined using the ``process`` decorator
  instead of inheriting from a ``Process`` base class.
- It is not needed anymore to explicitly define whether or not a
  process is time dependent (it is now deducted from the methods
  implemented in the process class).
- Using ``class Meta`` inside a process class to define some metadata
  is not used anymore.

- ``Model.input_vars`` now returns a list of ``(process_name,
  variable_name)`` tuples instead of a dict of dicts.
  ``Model.input_vars_dict`` has been added for convenience
  (i.e., to get input variables grouped by process as a dictionary).
- ``Model.is_input`` has been removed. Use ``Model.input_vars``
  instead to check if a variable is a model input.

- ``__repr__`` has slightly changed for variables, processes and
  models.  Process classes don't have an ``.info()`` method anymore,
  which has been replaced by the ``process_info()`` top-level
  function. Another helper function ``variable_info()`` has been
  added.

- In ``Model.visualize()`` and ``xsimlab.dot.dot_graph()``,
  ``show_variables=True`` now shows all model variables including
  inputs. Items of group variables are not shown anymore as nodes.
- ``Model.visualize()`` and ``xsimlab.dot.dot_graph()`` now only
  accept tuples for ``show_only_variable``.

- For simplicity, ``Dataset.xsimlab.snapshot_vars`` has been renamed to
  ``output_vars``. The corresponding arguments in ``create_setup`` and
  ``Dataset.xsimlab.update_vars`` have been renamed accordingly.
- Values for all model inputs must be provided when creating or
  updating a setup using ``create_setup`` or
  ``Dataset.xsimlab.update_vars``. this is a regression that will be
  fixed in the next releases.
- Argument values for generating clock data in ``create_setup`` and
  ``Dataset.xsimlab.update_clocks`` have changed and are now more
  consistent with how coordinates are set in xarray. Additionally,
  ``auto_adjust`` has been removed (an error is raised instead when
  clock coordinate labels are not synchronized).

- Scalar values from a input ``xarray.Dataset`` are now converted into
  scalars (instead of a 0-d numpy array) when setting input model
  variables during a simulation.

Enhancements
~~~~~~~~~~~~

- The major refactoring in this release should reduce the overhead
  caused by the indirect access to variable values in process objects.
- Another benefit of the refactoring is that a process-decorated class
  may now inherit from other classes (possibly also
  process-decorated), which allows more flexibility in model
  customization.
- By creating read-only properties in specific cases (i.e., when
  ``intent='in'``), the ``process`` decorator applied on a class adds
  some safeguards to prevent setting variable values where it is not
  intended.
- Some more sanity checks have been added when creating process
  classes.
- Simulation active and output data r/w access has been refactored
  internally so that it should be easy to later support alternative
  data storage backends (e.g., on-disk, distributed).
- Added ``Model.dependent_processes`` property (so far this was not
  in public API).
- Added ``Model.all_vars`` and ``Model.all_vars_dict`` properties that
  are similar to ``Model.input_vars`` and ``Model.input_vars_dict``
  but return all variable names in the model.
- ``input_vars`` and ``output_vars`` arguments of ``create_setup`` and
  ``Dataset.xsimlab.update_vars`` now accepts different formats.
- It is now possible to update only some clocks with
  ``Dataset.xsimlab.update_clocks`` (previously all existing clock
  coordinates were dropped first).

Regressions (will be fixed in future releases)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Although it is possible to set validators, converters and/or default
  values for variables (this is directly supported by ``attrs``), these
  are not handled by xarray-simlab yet.
- Variables don't accept anymore a dimension that corresponds to their
  own name. This may be useful, e.g., for sensitivity analysis, but as
  the latter is not implemented yet this feature has been removed and
  will be added back in a next release.
- High-level API for generating clock coordinate data (i.e.,
  ``start``, ``end``, ``step`` and ``auto_adjust`` arguments) is not
  supported anymore. This could be added back in a future release in a
  cleaner form.

v0.1.1 (20 November 2017)
-------------------------

Bug fixes
~~~~~~~~~

- Fix misinterpreted tuples passed as ``allowed_dims`` argument of
  ``Variable`` init (:issue:`17`).
- Better error message when a Model instance is expected but no object
  is found or a different object is provided (:issue:`13`).

v0.1.0 (8 October 2017)
-----------------------

Initial release.
