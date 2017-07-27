.. _framework:

Modeling Framework
==================

This section briefly explains the design of the xarray-simlab's modeling
framework. For more practical details on how to create, inspect and run models,
see the relevant sections in this user guide.

Main concepts
-------------

The xarray-simlab's framework is built on only a few concepts that are layered
onto three levels, here listed from top to bottom:

- **models**, i.e., instances of :class:`~xsimlab.Model`
- **processes**, i.e., subclasses of :class:`~xsimlab.Process`
- **variables**, i.e., :class:`~xsimlab.Variable` objects or objects of
  derived or related classes.

A **model** is an ordered collection of processes. The role of a **process** is
twofold: declare a given subset of the **variables** used in a model and
define a specific set of instructions that uses, sets or modifies the values of
these variables.

Conceptually, a process is a logical component of a model. It may for example
represent a particular physical mechanism that is described in terms of one or
more state variables (e.g., scalar or vector fields) and one or more processes
-- with or without parameters -- that modify those state variables through time.
Some processes may be time-independent.

.. note::

   xarray-simlab does not explicitly distinguish between model parameters and
   state variables, both are declared as variables inside their own process.

.. note::

   xarray-simlab does not provide any built-in logic for tasks like generating
   computational meshes or setting boundary conditions. This should be
   implemented in 3rd-party libraries as time-independent processes. The reason
   is that theses tasks may vary from one model / domain to another while
   xarray-simlab aims to be a general purpose framework.

Processes are usually inter-dependent, i.e., a process often declares
references to variables that are declared in other processes (i.e., "foreign
variables") and sometimes even computes values for those variables. As a
consequence, processes may not be re-usable without their dependencies.
However, it allows declaring variables -- including useful metadata such as
description, default values, units, etc. -- at a unique place.

Simulation workflow
-------------------

A model run is divided into four stages:

1. initialization
2. run step
3. finalize step
4. finalization

During a simulation, stages 1 and 4 are only run once while steps 2 and 3 are
repeated for a given number of (time) steps.

Each process in a model usually implements some computation for at least one of
these stages. It is not always required for a process to perform computation at
every stage. For example, time-independent processes don't do nothing during
stages 2 and 3. Processes that depend on time, however, must supply an
implementation for at least stage 2 (run step).

Computation order and process dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The order in which computations are executed during a simulation is important.
For example, a process that sets or updates a variable must be executed before
the execution of all other processes that need to access this variable for their
own computation.

The same order is used at every stage of a model run. It is automatically
inferred -- as well as process dependencies -- when creating a new
``Model`` object, using the information contained in the variables and/or
foreign variables that are declared in each process.

Process dependencies together form a Directed Acyclic Graph (DAG), which will
eventually allow us running some processes in parallel at each stage of a model
run (this is not yet implemented).

Variable values
---------------

A single variable may accept up to 3 different values:

- ``state``, i.e., the value of the variable at a given time
- ``rate``, i.e., the value of the time-derivative at a given time
- ``change``, i.e., the value of the time-derivative integrated for a given
  time step.

These correspond to separate properties of ``Variable`` objects.

.. note::

   We allow multiple values mainly to avoid creating different Variable
   objects for the same state variable. The names and meanings given here above
   are just conventions, though.

   Model developers are free to get/set any of these properties at any stage of
   a model run. However, it is common practice to compute the ``rate`` or
   ``change`` values in the "run step" stage and to update the ``state`` value
   in the "finalize step" stage.

An additional property, ``value``, is also defined as an alias of ``state``.
It should be used primarily for variables that are time-invariant. Here again it
is just a convention as using the term "value" is a bit more accurate in this
case.

.. todo_

   input variable section

.. move_this_foreign_variable

   ForeignVariable.state return the same object (usually a numpy array) than
   Variable.state (replace class names by variable names in processes).
   ForeignVariable.state is actually a shortcut to ForeignVariable.ref_var.state.
