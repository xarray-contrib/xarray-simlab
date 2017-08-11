.. _api:

#############
API Reference
#############

This page provides an auto-generated summary of xarray-simlab's API. For more
details and examples, refer to the relevant sections in the main part of the
documentation.

Top-level functions
===================

.. currentmodule:: xsimlab
.. autosummary::
   :toctree: _api_generated/

   create_setup

Dataset.xsimlab (xarray accessor)
=================================

This accessor extends :py:class:`xarray.Dataset` with all the methods and
properties listed below. Proper use of this accessor should be like:

.. code-block:: python

   >>> import xarray as xr         # first import xarray
   >>> import xsimlab              # import xsimlab (the 'xsimlab' accessor is registered)
   >>> ds = xr.Dataset()           # create or load an xarray Dataset
   >>> ds.xsimlab.<meth_or_prop>   # access to the methods and properties listed below

.. currentmodule:: xarray

**Properties**

.. autosummary::
   :toctree: _api_generated/
   :template: autosummary/accessor_attribute.rst

   Dataset.xsimlab.model
   Dataset.xsimlab.clock_coords
   Dataset.xsimlab.dim_master_clock
   Dataset.xsimlab.snapshot_vars

**Methods**

.. autosummary::
   :toctree: _api_generated/
   :template: autosummary/accessor_method.rst

   Dataset.xsimlab.use_model
   Dataset.xsimlab.update_clocks
   Dataset.xsimlab.update_vars
   Dataset.xsimlab.set_master_clock
   Dataset.xsimlab.set_snapshot_clock
   Dataset.xsimlab.set_input_vars
   Dataset.xsimlab.set_snapshot_vars
   Dataset.xsimlab.run

Model
=====

Creating a model
----------------

.. currentmodule:: xsimlab
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
For running simulations, it is preferable to use the
``Dataset.xsimlab`` accessor instead. These methods might be useful
though, e.g., for debugging or for using ``Model`` objects with other
interfaces.

.. autosummary::
   :toctree: _api_generated/

   Model.initialize
   Model.run_step
   Model.finalize_step
   Model.finalize

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

Although it has the same name, this class is different from
:py:class:`xarray.Variable`.

.. autosummary::
   :toctree: _api_generated/

   Variable

**Attributes**

.. autosummary::
   :toctree: _api_generated/

   Variable.value
   Variable.state
   Variable.rate
   Variable.change

**Methods**

.. autosummary::
   :toctree: _api_generated/

   Variable.to_xarray_variable

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

**Attributes**

.. autosummary::
   :toctree: _api_generated/

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
