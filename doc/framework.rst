.. _framework:

Modeling Framework
==================

This section briefly explains the design of the xarray-simlab's modeling
framework. For more practical details on how to create, inspect and run models,
see the relevant sections in this user guide.

Main concepts
-------------

The xarray-simlab's framework is built on only a few concepts that are layered
onto three levels here listed from top to bottom:

- **models**, i.e., instances of :class:`~xsimlab.Model`
- **processes**, i.e., subclasses of :class:`~xsimlab.Process`
- **variables**, i.e., :class:`~xsimlab.Variable` objects or objects of
  derived or related classes.

A **model** is an ordered collection of processes. The role of a **process** is
twofold: declare a given subset of the **variables** used in a model and
define a specific set of instructions that uses, sets or modifies the values of
these variables.

Conceptually, a process is a logical component of a computational model. It may
for example represent a particular physical mechanism that is described in terms
of one or more state variables (e.g., scalar or vector fields) and one or more
processes -- with or without parameters -- that modify those state variables
through time. Some processes may be time-independent.

.. note::

   xarray-simlab does not explicitly distinguish between model parameters and
   state variables, both are declared as variables within their own process.

.. note::

   xarray-simlab does not provide any built-in logic for tasks like generating
   computational meshes or setting boundary conditions. It should rather be
   implemented in 3rd-party libraries as time-independent processes. The reason
   is that even such tasks may vary from one model / domain to another, while
   xarray-simlab aims to provide a general purpose framework.

Process dependencies
--------------------

Processes are components that are generally not totally independent. To compute
its own variables, a process often needs variables that are declared externally
in other processes.

In order to properly use those external variables, a process has to declare them
as "foreign variables". A :class:`~xsimlab.ForeignVariable` object is a
reference to an external variable. It can be used to get or compute values
as if it was the original variable.

A downside of this approach is that one cannot re-use a process without
its dependencies. But it presents the great advantage of declaring variables
-- including useful metadata such as description, default values, units, etc. --
at a single place.

Simulation workflow
-------------------

A model run is divided into four successive stages:

1. initialization
2. run step
3. finalize step
4. finalization

During a simulation, stages 1 and 4 are run only once while steps 2 and 3 are
repeated for a given number of (time) steps.

Each process provides its own computation instructions for those stages. This
is optional, except for time-dependent processes that must provide some
instructions at least for stage 2 (run step). For time-independent processes
stages 2 and 3 are ignored.

Process ordering
----------------

The order in which processes are executed during a simulation is important.
For example, a process that sets or updates a variable value must be executed
before the execution of all other processes that need this variable in their
own computation.

The same ordering is used at every stage of a model run. It is automatically
inferred -- as well as process dependencies -- when creating a new
``Model`` object, using the information contained in variables and/or foreign
variables that are declared in each process.

Process dependencies together form a Directed Acyclic Graph (DAG), which will
eventually allow us running some processes in parallel at each stage of a model
run (this is not yet implemented).

Variable values and dimensions
------------------------------

A single variable may accept up to 3 different values:

- a state, i.e., the value of the variable at a given time
- a rate, i.e., the value of the time-derivative at a given time
- a change, i.e., the value of the time-derivative integrated for a given
  time step.

These are accessible as properties of ``Variable`` objects, respectively named
``state``, ``rate`` and ``change``. An additional property ``value`` is defined
as an alias of ``state``.

.. note::

   These properties are for convenience only, it avoids duplicating
   Variable objects (including their metadata) representing state variables.

   The names and meanings given here above are just conventions. Model
   developers are free to get/set values for any of these properties at any
   stage of a model run.

   However, it is recommended to follow these conventions. Another common
   practice is to compute ``rate`` or ``change`` values during the "run step"
   stage and update ``state`` values during the "finalize step" stage.

   For time-invariant variables, ``rate`` or ``change`` values should not be
   used. Instead, it is preferable to use the property ``value`` ("state" is
   quite meaningless in this case).

.. todo_

   variable dimensions paragraph

.. todo_

   input variable section

.. move_this_foreign_variable

   ForeignVariable.state return the same object (usually a numpy array) than
   Variable.state (replace class names by variable names in processes).
   ForeignVariable.state is actually a shortcut to ForeignVariable.ref_var.state.
