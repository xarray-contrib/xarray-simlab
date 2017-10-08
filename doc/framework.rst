.. _framework:

Modeling Framework
==================

This section explains the design of the xarray-simlab modeling
framework. It is useful mostly for users who want to create new models
from scratch or customize existing models. Users who only want to run
simulations from existing models may skip this section.

For more practical details on how using the API to create, inspect and
run models, see the relevant sections of this user guide.

Main concepts
-------------

The xarray-simlab framework is built on a very few concepts that
allow great flexibility in model customization:

- models
- processes
- variables

These are detailed here below.

Models
------

Models are instances of the :class:`~xsimlab.Model` class. They
consist of ordered, immutable collections of processes. The
ordering is inferred automatically from the given processes (see below).

The Model class also implements specific methods for:

- introspection,
- running simulations,
- easy creation of new Model objects from existing ones by dropping,
  adding or replacing one or more processes.

Processes
---------

Processes are defined as custom Python classes that inherit from the
base class :class:`~xsimlab.Process`. The role of a process is twofold:

- declare a given subset of the variables used in a model,
- define a specific set of instructions that use or compute values for
  these variables during a model run.

Conceptually, a process is a logical component of a computational
model. It may for example represent a particular physical mechanism
that is described in terms of one or more state variables (e.g.,
scalar or vector fields) and one or more operations -- with or without
parameters -- that modify those state variables through time. Note
that some processes may be time-independent.

.. note::

   xarray-simlab does not provide any built-in logic for tasks like
   generating computational meshes or setting boundary conditions,
   which should rather be implemented in 3rd-party libraries as
   time-independent processes. Even those tasks may be too specialized
   to justify including them in this framework, which aims to be as
   general as possible.

Variables
---------

Variables are the most basic elements of a model. They consist of
:class:`~xsimlab.Variable` [*]_ objects that are declared in processes as class
attributes. They have the following properties:

- data values (state, rate or change -- see below),
- validators, i.e., callables for checking supplied data values,
- labeled dimensions (or no dimension for scalars),
- predefined meta-data attributes like description or default value,
- user-defined meta-data attributes (e.g., units, math symbol).

.. note::

   xarray-simlab does not distinguish between model parameters
   and state variables. Both are declared as Variable objects.

.. [*] usually variables consist of objects of derived classes like,
   e.g., ``FloatVariable`` or ``IntegerVariable`` depending on their
   expected value type.

Foreign variables
-----------------

Like different physical mechanisms involve some common state variables
(e.g., temperature or pressure), different processes may operate on
common variables.

In xarray-simlab, a variable is declared at a unique place, i.e.,
within one and only one process. Using common variables across
processes is achieved by declaring foreign variables.
:class:`~xsimlab.ForeignVariable` objects are references to
variables that are declared in other processes ; it allows getting or
setting values just as if these references were the original
variables.

The great advantage of declaring variables at unique places is that
all their meta-data are defined once. However, a downside of this
approach is that foreign variables may potentially add many hard-coded
links between processes, which makes harder reusing these processes
independently of each other.

Variable groups
---------------

In some cases, using variables groups may provide an elegant
alternative to hard-coded links between processes.

The membership of Variable objects to a group is defined via their
``group`` attribute. If we want to use in a separate process all the
variables of a group, instead of explicitly declaring foreign
variables we can declare a :class:`~xsimlab.VariableGroup` object
which behaves like an iterable of ForeignVariable objects pointing to
each of the variables of the group.

Variable groups are useful particularly in cases where we want to
combine different processes that act on the same variable, e.g. in
landscape evolution modeling combine the effect of different erosion
processes on the evolution of the surface elevation. This way we can
easily add or remove processes to/from a model and avoid missing or
broken links between processes.

Variable state, rate and change
-------------------------------

A single variable may accept up to 3 concurrent values that each
have a particular meaning:

- a state, i.e., the value of the variable at a given time,
- a rate, i.e., the value of the time-derivative at a given time,
- a change, i.e., the value of the time-derivative integrated for a given
  time step.

These are accessible as properties of Variable and ForeignVariable
objects: ``state``, ``rate`` and ``change``. An additional property
``value`` is defined as an alias of ``state``.

.. note::

   These properties are for convenience only, it avoids having to
   duplicate Variable objects for state variables. The properties are
   independent of each other and their given meanings serve only as
   conventions. Although there is no restriction in using any of these
   properties anywhere in a process, it is good practice to follow
   these conventions.

.. note::

   The ``rate`` and ``change`` properties should never be used for
   variables other than state variables.  Moreover, it is preferable
   to use the alias ``value`` instead of ``state`` as the latter is
   quite meaningless in this case.

.. todo_move_this_elsewhere

   For state variables, a common practice is to compute ``rate`` or
   ``change`` values during the "run step" stage and update ``state``
   values during the "finalize step" stage.

Simulation workflow
-------------------

A model run is divided into four successive stages:

1. initialization
2. run step
3. finalize step
4. finalization

During a simulation, stages 1 and 4 are run only once while steps 2
and 3 are repeated for a given number of (time) steps.

Each process provides its own computation instructions for those
stages. Note that this is optional, except for time-dependent
processes that must provide some instructions at least for stage 2
(run step). For time-independent processes stages 2 and 3 are ignored.

Process dependencies and ordering
---------------------------------

The order in which processes are executed during a simulation is
critical. For example, if the role of a process is to provide a value
for a given variable, then the execution of this process must happen
before the execution of all other processes that use the same variable
in their computation.

Such role can be defined using the ``provided`` attribute of Variable
and ForeignVariable objects, which is either set to True or False
(note that a process may still update a variable value even if
``provided`` is set to False, see Model inputs section below).

In a model, the processes and their dependencies together form the
nodes and the edges of a Directed Acyclic Graph (DAG). The graph
topology is fully determined by the role set for each variable or
foreign variable declared in each process. An ordering that is
computationally consistent can then be obtained using topological
sorting. This is done at Model object creation. The same ordering
is used at every stage of a model run.

In principle, the DAG structure would also allow running the processes
in parallel at every stage of a model run. This is not yet
implemented, though.

Model inputs
------------

In a model, inputs are variables that need a value to be set by the
user before running a simulation.

Like process ordering, inputs are automatically retrieved at Model
object creation, using the ``provided`` attribute of Variable and
ForeignVariable objects. Inputs are Variable objects for which
``provided`` is set to False and which don't have any linked
ForeignVariable object with ``provided`` set to True.

.. note::

   Any value required as model input relates to the ``state`` property
   (or its alias ``value``) of a Variable object. The ``rate`` and
   ``change`` properties should never be set by model users, like any
   property of ForeignVariable objects.

.. move_this_foreign_variable

   ForeignVariable.state return the same object (usually a numpy array) than
   Variable.state (replace class names by variable names in processes).
   ForeignVariable.state is actually a shortcut to ForeignVariable.ref_var.state.
