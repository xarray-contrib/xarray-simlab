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

.. _api_xarray_accessor:

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

   Dataset.xsimlab.clock_coords
   Dataset.xsimlab.clock_sizes
   Dataset.xsimlab.master_clock_dim
   Dataset.xsimlab.master_clock_coord
   Dataset.xsimlab.nsteps
   Dataset.xsimlab.output_vars
   Dataset.xsimlab.output_vars_by_clock

**Methods**

.. autosummary::
   :toctree: _api_generated/
   :template: autosummary/accessor_method.rst

   Dataset.xsimlab.update_clocks
   Dataset.xsimlab.update_vars
   Dataset.xsimlab.reset_vars
   Dataset.xsimlab.filter_vars
   Dataset.xsimlab.run
   Dataset.xsimlab.get_output_save_steps

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
process names and values are objects of ``Process`` subclasses
(attribute-style access is also supported).

.. autosummary::
   :toctree: _api_generated/

   Model.all_vars
   Model.all_vars_dict
   Model.index_vars
   Model.index_vars_dict
   Model.input_vars
   Model.input_vars_dict
   Model.dependent_processes
   Model.visualize

Running a model
---------------

In most cases, the methods and properties listed below should not be used
directly. For running simulations, it is preferable to use the xarray extension
instead, see :ref:`api_xarray_accessor`. These methods might be useful though,
e.g., for using ``Model`` objects with other interfaces.

.. autosummary::
   :toctree: _api_generated/

   Model.state
   Model.update_state
   Model.cache
   Model.update_cache
   Model.execute
   Model.validate

Process
=======

Creating a process
------------------

.. autosummary::
   :toctree: _api_generated/

   process

Process introspection and variables
-----------------------------------

.. autosummary::
   :toctree: _api_generated/

   process_info
   variable_info
   filter_variables

Process runtime methods
-----------------------

.. autosummary::
   :toctree: _api_generated/

   runtime

Variable
========

.. autosummary::
   :toctree: _api_generated/

   variable
   index
   any_object
   foreign
   group
   on_demand

Validators
==========

See also `attrs' validators`_. Some additional validators for common usage are
listed below. These are defined in ``xsimlab.validators``.

.. currentmodule:: xsimlab.validators
.. autosummary::
   :toctree: _api_generated/

   in_bounds
   is_subdtype

.. _`attrs' validators`: https://www.attrs.org/en/stable/examples.html#validators

Model runtime monitoring
========================

.. currentmodule:: xsimlab
.. autosummary::
   :toctree: _api_generated/

   monitoring.ProgressBar
   runtime_hook
   RuntimeHook
