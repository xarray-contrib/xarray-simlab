.. _whats_new:

Release Notes
=============

v0.3.0 (Unreleased)
-------------------

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

Deprecations
~~~~~~~~~~~~

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

.. _attrs: http://www.attrs.org

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
