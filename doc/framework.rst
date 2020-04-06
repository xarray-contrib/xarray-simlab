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

Processes are defined as Python classes that are decorated by
:func:`~xsimlab.process`. The role of a process is twofold:

- declare a given subset of the variables used in a model,
- define a specific set of instructions that use or compute values for
  these variables during a model run.

Conceptually, a process is a logical component of a computational
model. It may for example represent a particular physical mechanism
that is described in terms of one or more state variables (e.g.,
scalar or vector fields) and one or more operations -- with or without
parameters -- that modify those state variables through time. Note
that some processes may be time-independent or may even be used to
declare variables without implementing any computation.

.. note::

   xarray-simlab does not provide any built-in logic for tasks like
   generating computational meshes or setting boundary conditions,
   which should rather be implemented in 3rd-party libraries as
   processes. Even those tasks may be too specialized to justify
   including them in this framework, which aims to be as general as
   possible.

A process-ified class behaves mostly like any other regular Python
class, i.e., there is a-priori nothing that prevents you from using
the common object-oriented features as you like. The only difference
is that you can here create classes in a very succinct way without
boilerplate, i.e., you don't need to implement dunder methods like
``__init__`` or ``__repr__`` as this is handled by the framework. In
fact, this framework uses and extends the attrs_ package:
:func:`~xsimlab.process` is a wrapper around :func:`attr.s` and the
functions used to create variables (see below) are thin wrappers
around :func:`attr.ib`.

.. _attrs: http://www.attrs.org

Variables
---------

Variables are the most basic elements of a model. They are declared in
processes as class attributes, using :func:`~xsimlab.variable`.
Declaring variables mainly consists of defining useful metadata such
as:

- labeled dimensions (or no dimension for scalars),
- predefined meta-data attributes, e.g., a short description,
- user-defined meta-data attributes, e.g., units or math symbol,
- the intent for a variable, i.e., whether the process
  needs (``intent='in'``), updates (``intent='inout'``) or computes
  (``intent='out'``) a value for that variable.

It is also possible to set a default value as well as value validator(s).
See `attrs' validators`_ for more details.

.. note::

   xarray-simlab does not distinguish between model parameters, input
   and output variables. All can be declared using
   :func:`~xsimlab.variable`.

.. _`attrs' validators`: https://www.attrs.org/en/stable/examples.html#validators

Foreign variables
~~~~~~~~~~~~~~~~~

Like different physical mechanisms involve some common state variables
(e.g., temperature or pressure), different processes may operate on
common variables.

In xarray-simlab, a variable is declared at a unique place, i.e.,
within one and only one process. Using common variables across
processes is achieved by declaring :func:`~xsimlab.foreign`
variables. These are simply references to variables that are declared
in other processes.

You can use foreign variables for almost any computation inside a
process just like original variables. The only difference is that
``intent='inout'`` is not supported for a foreign variable, i.e., a
process may either need or compute a value of a foreign variable but
may not update it (otherwise it would not be possible to unambiguously
determine process dependencies -- see below). For the same reason,
only one process in a model may compute a value of a variable (i.e.,
``intent='out'``).

The great advantage of declaring variables at unique places is that
all their meta-data are defined once. However, a downside of this
approach is that foreign variables may potentially add many hard-coded
links between processes, which makes harder reusing these processes
independently of each other.

Group variables
~~~~~~~~~~~~~~~

In some cases, using group variables may provide an elegant alternative to
hard-coded links between processes.

The membership of variables to one or several groups is defined via their
``groups`` attribute. If you want to reuse in a separate process all the
variables of a given group, instead of explicitly declaring each of them as
foreign variables you can simply declare a :func:`~xsimlab.group` variable. The
latter behaves like an iterable of foreign variables pointing to each of the
variables (model-wise) that are members of the same group.

Note that group variables implicitly have ``intent='in'``, i.e, they could only
be used to get the values of multiple foreign variables, not set their values.

Group variables are useful particularly in cases where you want to combine
(aggregate) different processes that act on the same variable, e.g. in landscape
evolution modeling combine the effect of different erosion processes on the
evolution of the surface elevation. This way you can easily add or remove
processes to/from a model and avoid missing or broken links between processes.

On-demand variables
~~~~~~~~~~~~~~~~~~~

On-demand variables are like regular variables, except that their
value is not intended to be computed systematically, e.g., at the
beginning or at each time step of a simulation, but instead only at a
given few times (or not at all). These are declared using
:func:`~xsimlab.on_demand` and must implement in the same
process-ified class a dedicated method -- i.e., decorated with
``@foo.compute`` where ``foo`` is the name of the variable -- that
returns their value. They implicitly have ``intent='out'``.

On-demand variables are useful, e.g., for optional model diagnostics.

Index variables
~~~~~~~~~~~~~~~

Index variables are intended for indexing data of other variables in a model
like, e.g., coordinate labels of grid nodes. They are declared using
:func:`~xsimlab.index`. They implicitly have ``intent='out'``, although their
values could be computed from other input variables.

'Object' variables
~~~~~~~~~~~~~~~~~~

Sometimes we need to share between processes one or more arbitrary objects,
e.g., callables or instances of custom classes that have no array-like
interface. Those objects should be declared in process-decorated classes using
:func:`~xsimlab.any_object`.

Within a model, those 'object' variables are reserved for internal use only,
i.e., they never require an input value (they implicitly have ``intent='out'``)
and they can't be saved as outputs as their value may not be compatible with the
xarray data model. Of course, it is still possible to create those objects using
data from other (input) variables declared in the process. Likewise, their data
could still be coerced into a scalar or an array and be saved as output via
another variable.

Simulation workflow
-------------------

A model run is divided into four successive stages:

1. initialization
2. run step
3. finalize step
4. finalization

During a simulation, stages 1 and 4 are run only once while stages 2
and 3 are repeated for a given number of (time) steps.

Each process-ified class may provide its own computation instructions
by implementing specific methods named ``.initialize()``,
``.run_step()``, ``.finalize_step()`` and ``.finalize()`` for each
stage above, respectively. Note that this is entirely optional. For
example, time-independent processes (e.g., for setting model grids)
usually implement stage 1 only. In a few cases, the role of a process
may even consist of just declaring some variables that are used
elsewhere.

Get / set variable values inside a process
------------------------------------------

Once you have declared a variable as a class attribute in a process, you can
further get and/or set its value like a regular instance attribute. For example,
if you declare a variable ``foo`` you can just use ``self.foo`` to get/set its
value inside one method of that class.

Additionally, the :func:`~xsimlab.process` decorator takes all variables
declared as class attributes and turns them into properties, which may be
read-only depending on the ``intent`` set for the variables. For all variables
except on-demand variables, the getter/setter methods of those properties
read/write values via a simple dictionary that is common to a simulation. Note
that those properties are created only for the case where a ``process``
decorated class is used within a ``Model`` object.

Process dependencies and ordering
---------------------------------

The order in which processes are executed during a simulation is
critical. For example, if the role of a process is to compute a value
for a given variable, then the execution of this process must happen
before the execution of all other processes that use the same variable
in their computation.

In a model, the processes and their dependencies together form the
nodes and the edges of a Directed Acyclic Graph (DAG). The graph
topology is fully determined by the ``intent`` set for each variable
or foreign variable declared in each process. An ordering that is
computationally consistent can then be obtained using topological
sorting. This is done at Model object creation. The same ordering is
used at every stage of a model run.

The DAG structure also allows running the processes in parallel at every stage
of a model run, see Section :ref:`run_parallel_single`.

Model inputs
------------

In a model, inputs are variables that need a value to be set by the
user before running a simulation.

Like process ordering, inputs are automatically retrieved at Model
object creation by looking at the ``intent`` set for all variables and
foreign variables in the model. A variable is a model input if it has
``intent`` set to ``'in'`` or ``'inout'`` and if it has no linked
foreign variable with ``intent='out'``.
