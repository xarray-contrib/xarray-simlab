.. _api:
.. currentmodule:: xrsimlab

#############
API reference
#############

This page provides an auto-generated summary of xarray-simlab's API. For more
details and examples, refer to the relevant sections in the main part of the
documentation.

Dataset.simlab (xarray accessor)
================================

Note: ``SimLabAccessor`` is a class that is decorated using
:py:func:`xarray.register_dataset_accessor`. This allows extending
:py:class:`xarray.Dataset` with all the methods and properties listed below,
which can then be accessed via the ``Dataset.simlab`` property.
Proper use of this accessor should be like:

.. code-block:: python

   >>> import xarray as xr   # first import xarray
   >>> import xrsimlab       # import xrsimlab (simlab accessor is registered)
   >>> ds = xr.Dataset()
   >>> ds.simlab.<meth_or_prop>   # access to the methods and properties listed below

.. autosummary::
   :toctree: _api_generated/

   SimLabAccessor.model
   SimLabAccessor.use_model
   SimLabAccessor.dim_master_clock
   SimLabAccessor.set_master_clock
   SimLabAccessor.set_snapshot_clock
   SimLabAccessor.set_input_vars
   SimLabAccessor.set_snapshot_vars
   SimLabAccessor.snapshot_vars
   SimLabAccessor.run

Model
=====

Creating a model
----------------

.. currentmodule:: xrsimlab
.. autosummary::
   :toctree: _api_generated/

   Model

Creating a new model from an existing one
-----------------------------------------

.. autosummary::
   :toctree: _api_generated/

   Model.clone
   Model.update_processes
   Model.drop_processes

Model introspection
-------------------

``Model`` implements an immutable mapping interface where keys are
process names and values are objects of ``Process`` subclasses (attribute-style
access is also supported).

.. autosummary::
   :toctree: _api_generated/

   Model.input_vars
   Model.is_input
   Model.visualize

Running a model
---------------

In most cases, the methods listed below should not be used directly.
For running simulations, it is preferable to use the ``Dataset.simlab`` accessor
instead. These methods might be useful for debugging, though.

.. autosummary::
   :toctree: _api_generated/

   Model.initialize
   Model.run_step
   Model.finalize_step
   Model.finalize
   Model.run

Process
=======

Note: ``Process`` is a base class that should be subclassed.

.. autosummary::
   :toctree: _api_generated/

   Process

Clone a process
---------------

.. autosummary::
   :toctree: _api_generated/

   Process.clone

Process interface and introspection
-----------------------------------

``Process`` implements an immutable mapping interface where keys are
variable names and values are Variable objects (attribute-style
access is also supported).

.. autosummary::
   :toctree: _api_generated/

   Process.variables
   Process.meta
   Process.name
   Process.info

Process "abstract" methods
--------------------------

Subclasses of ``Process`` usually implement at least some of the methods below.

.. autosummary::
   :toctree: _api_generated/

   Process.validate
   Process.initialize
   Process.run_step
   Process.finalize_step
   Process.finalize

Variable
========

Base variable class
-------------------

.. autosummary::
   :toctree: _api_generated/

   Variable
   Variable.to_xarray_variable
   Variable.value
   Variable.state
   Variable.rate
   Variable.change

Derived variable classes
------------------------

These classes inherit from ``Variable``.

.. autosummary::
   :toctree: _api_generated/

   NumberVariable
   FloatVariable
   IntegerVariable

Foreign variable
----------------

.. autosummary::
   :toctree: _api_generated/

   ForeignVariable
   ForeignVariable.ref_process
   ForeignVariable.ref_var
   ForeignVariable.value
   ForeignVariable.state
   ForeignVariable.rate
   ForeignVariable.change

Diagnostic variable
-------------------

.. autosummary::
   :toctree: _api_generated/

   diagnostic

Collections of variables
------------------------

.. autosummary::
   :toctree: _api_generated/

   VariableList
   VariableGroup
