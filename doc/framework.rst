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
twofold: (1) declare a given subset of the **variables** used in a model
and (2) define a specific set of instructions that use, initialize or update
the values of these variables.

Conceptually, a process is a logical component of a computational model. It may
for example represent a particular physical mechanism that is described in terms
of one or more state variables (e.g., scalar or vector fields) and one or more
operations -- with or without parameters -- that modify those state variables
through time. Note that some processes may be time-independent.

.. note::

   xarray-simlab does not explicitly distinguish between model parameters and
   state variables, both are declared as variables within their own process.

.. note::

   xarray-simlab does not provide any built-in logic for tasks like generating
   computational meshes or setting boundary conditions. It should rather be
   implemented in 3rd-party libraries as time-independent processes. The reason
   is that even such tasks may vary from one model / domain to another, whereas
   xarray-simlab aims to provide a general purpose framework.

Foreign variables
-----------------

Like different physical mechanisms involve some common state variables
(e.g., temperature or pressure), different processes may need to operate on
common variables.

In xarray-simlab, this common case is handled by declaring variables only once
within their own process but with the possibility of also declaring foreign
variables. :class:`~xsimlab.ForeignVariable` objects are references to
variables that are declared in other processes. It allows getting or setting
values just as if these references were the original variables.

A downside of this approach is that a lot of process (one-way) connections may
be hard-coded through the declaration of foreign variables. Therefore a process
cannot be re-used alone if it has links to other processes. However,
the great advantage of declaring variables at unique places is that
all their metadata (e.g., description, default value, units, etc.) are also
defined once.

Simulation workflow
-------------------

A model run is divided into four successive stages:

1. initialization
2. run step
3. finalize step
4. finalization

During a simulation, stages 1 and 4 are run only once while steps 2 and 3 are
repeated for a given number of (time) steps.

Each process provides its own computation instructions for those stages. Note
that this is optional, except for time-dependent processes that must provide
some instructions at least for stage 2 (run step). For time-independent
processes stages 2 and 3 are ignored.

Process dependencies and ordering
---------------------------------

The order in which processes are executed during a simulation is critical.
For example, if the role of a process is to provide a value for a given
variable, then the execution of this process must happen before the execution
of all other processes that use the same variable in their computation.

This role can be defined using the ``provided`` attribute of ``Variable``
and ``ForeignVariable`` objects, either set to True or False (note that
a process may still update a variable value even if ``provided`` is set to
False, see Model inputs section below).

In a model, the processes and their dependencies together form the nodes and
the edges of a Directed Acyclic Graph (DAG). The graph topology is fully
determined by the role set for each variable or foreign variable declared in
each process. An ordering that is computationally consistent can then be
obtained using topological sorting. This is done when creating a new
``Model`` object. The same ordering is used at every stage of a model run.

In theory, The DAG structure would also allow running the processes in parallel
at every stage of a model run. This is not yet implemented, though.

Variable values and dimensions
------------------------------

A single variable may accept up to 3 different values:

- a state, i.e., the value of the variable at a given time
- a rate, i.e., the value of the time-derivative at a given time
- a change, i.e., the value of the time-derivative integrated for a given
  time step.

These are accessible as properties of ``Variable`` and ``ForeignVariable``
objects, respectively named ``state``, ``rate`` and ``change``. An additional
property ``value`` is defined as an alias of ``state``.

.. note::

   These properties are for convenience only, it avoids duplicating
   Variable objects representing state variables.
   The names and descriptions of these properties are only conventions. There is
   actually no restriction in getting or setting values for any of these
   properties at any stage of a model run (it is let to the responsibility of
   model developers). However, it is recommended to follow these conventions as
   well as some good practice.

.. note::

   For state variables, a common practice is to compute ``rate`` or ``change``
   values during the "run step" stage and update ``state`` values during the
   "finalize step" stage.

   For time-invariant variables, ``rate`` or ``change`` properties should never
   be used. Moreover, it is preferable to use the property ``value`` instead of
   ``state`` as the latter is quite meaningless in this case.

.. todo_

   variable dimensions paragraph

.. todo_

   input variable section

.. move_this_foreign_variable

   ForeignVariable.state return the same object (usually a numpy array) than
   Variable.state (replace class names by variable names in processes).
   ForeignVariable.state is actually a shortcut to ForeignVariable.ref_var.state.
