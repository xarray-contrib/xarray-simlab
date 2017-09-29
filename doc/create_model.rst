.. _create_model:

Create and modify models
========================

Like the previous :doc:`framework` section, this section is useful
mostly for users who want to create new models from scratch or
customize existing models. Users who only want to run simulations from
existing models may skip this section.

As a simple example, we will start here from a model which numerically
solves the 1-d advection equation using the `Lax method`_. The equation
may be written as:

.. math::

   \frac{\partial u}{\partial t} + \nu \frac{\partial u}{\partial x} = 0

with :math:`u(x, t)` as the quantity of interest and where :math:`\nu`
is the velocity. The discretized form of this equation may be written
as:

.. math::

   u^{n+1}_i = \frac{1}{2} (u^n_{i+1} + u^n_{i-1}) - \nu \frac{\Delta t}{2 \Delta x} (u^n_{i+1} - u^n_{i-1})

Where :math:`\Delta x` is the fixed spacing between the nodes
:math:`i` of a uniform grid and :math:`\Delta t` is the step duration
between times :math:`n` and :math:`n+1`.

We could just implement this numerical model with a few lines of
Python / Numpy code, e.g., here below assuming periodic boundary
conditions and a gaussian pulse as initial profile. We will show,
however, that it is very easy to refactor this code for using it with
xarray-simlab. We will also show that, while enabling useful features,
the refactoring still results in a short amount of readable code that
can be easily maintained.

.. literalinclude:: scripts/advection_lax_numpy.py

.. _`Lax method`: https://en.wikipedia.org/wiki/Lax%E2%80%93Friedrichs_method

Anatomy of a Process subclass
-----------------------------

Let's first wrap the code above into a single subclass of
:class:`~xsimlab.Process` named ``AdvectionLax1D``. Next we'll explain in
detail the content of this class.

.. literalinclude:: scripts/advection_lax_xsimlab.py
   :lines: 7-33

Process interface
~~~~~~~~~~~~~~~~~

``AdvectionLax1D`` has some class attributes declared at the top,
which together form the process' "public" interface, i.e., all the
variables that we want to be publicly exposed by this process. These
attributes usually correspond to instances of
:class:`~xsimlab.Variable` or derived classes, depending on their
expected value type, like :class:`~xsimlab.FloatVariable` in this case
(see section :doc:`api` for a full list of available classes).

The creation of Variable objects requires to explicitly provide
dimension label(s) for arrays or an empty tuple for scalars. In this
case, variables ``spacing``, ``length``, ``loc`` and ``scale`` are all
scalars, whereas ``x`` and ``u`` are both arrays defined on the
1-dimensional :math:`x` grid. Multiple choices can also be given as a
list, like variable ``v`` which represents a velocity field that can
be either constant (scalar) or variable (array) in space.

.. note::

   All variable objects also implicitly allow a time dimension as
   well as their own dimension (coordinate). See section :doc:`run_model`.

There is also a set of common arguments available to all Variable
types. All are optional. In the example above, ``description`` and
``attrs`` are used to define some (custom) metadata.

Variables ``x`` and ``u`` have also an option ``provided`` set to
``True``. It means that the process ``AdvectionLax1D`` itself provides
a value for these variables. ``provided=False`` (default) means
that a value should be provided elsewhere, either by another process
or as model input.

.. note::

   A process which updates the value (i.e., state) of a variable
   during a simulation does not necessarily imply setting
   ``provide=True`` for that variable, e.g., when it still requires an
   initial value.

Other options are available, see :class:`~xsimlab.Variable` for full
reference.

Process "runtime" methods
~~~~~~~~~~~~~~~~~~~~~~~~~

Beside its interface, the process ``AdvectionLax1D`` also implements
methods that will be called during simulation runtime:

- ``initialize`` will be called once at the beginning of a
  simulation. Here it is used to set the x-coordinate values of the
  grid and the initial values of ``u`` along the grid (gaussian
  pulse).
- ``run_step`` will be called at each time step iteration and have the
  current time step duration as required argument. This is where the
  Lax method is implemented.
- ``finalize_step`` will be called at each time step iteration too but
  after having called ``run_step`` for all other processes (if
  any). Its intended use is mainly to ensure that state variables like
  ``u`` are updated consistently and after having taken snapshots.

A fourth method ``finalize`` could also be implemented, but it is not
needed in this case. This method is called once at the end of the
simulation, e.g., for cleaning purposes.

Accessing process variables and values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Variable objects declared in ``AdvectionLax1D`` can be accessed
elsewhere in the class like normal attributes, e.g., using ``self.u``
for variable ``u``.

.. note::

   Like the other variables, ``self.u`` actually returns a copy of the
   corresponding ``FloatVariable`` object that is originally declared
   as a class attribute. Some internal magic happens in xarray-simlab
   in order to avoid value conflicts when using the same process in
   different contexts.

Variable objects may hold multiple, independent values that we set/get
via specific properties (see section :doc:`framework`), e.g.,
``self.u.state`` for :math:`u` values and ``self.x.value`` for
x-coordinate values on the grid. Note that we use here the property
``value`` for all time-independent variables, which is just an alias
of ``state`` (this is purely conventional).

Beside Variable object attributes, we can of course use normal
attributes in Process subclasses too, like ``self.u1`` in
``AdvectionLax1D``.

Creating a Model instance
-------------------------

Creating a new :class:`~xsimlab.Model` instance is very easy. We just
need to provide a dictionary with the process(es) that we want to
include in the model, e.g., with only the process created above:

.. literalinclude:: scripts/advection_lax_xsimlab.py
   :lines: 36

That's it! Now we can use that model with the xarray extension provided
by xarray-simlab to create new setups, run the model, take snapshots
for one or more variables on a given frequency, etc. (see section
:doc:`run_model`).

Fine-grained process refactoring
--------------------------------

The model created above isn't very flexible. What if we want to change
the initial conditions? Use a grid with variable spacing? Add another
physical process impacting :math:`u` such as a source or sink term?
In all cases we would need to modify the class ``AdvectionLax1D``.

This framework works best if instead we first split the problem into
small pieces, i.e., small Process subclasses that we can easily
combine and replace in models.

The ``AdvectionLax1D`` process may for example be refactored into 4
separate processes:

- ``UniformGrid1D`` : grid creation
- ``ProfileU`` : update :math:`u` values along the grid at each time
  iteration
- ``AdvectionLax`` : perform advection at each time iteration
- ``InitUGauss`` : create initial :math:`u` values along the grid.

**UniformGrid1D**

This process declares all grid-related variables and computes
x-coordinate values.

.. literalinclude:: scripts/advection_lax_xsimlab.py
   :lines: 39-50

``class Meta`` is used here to specify that this process is not time
dependent (by default processes are considered as
time-dependent). Grid x-coordinate values only need to be set once at
the beginning of the simulation ; there is no need to implement
``run_step`` here.

**ProfileU**

.. literalinclude:: scripts/advection_lax_xsimlab.py
   :lines: 53-64

``u_vars`` is declared as a :class:`~xsimlab.VariableGroup`, i.e., an
iterable of all variables declared elsewhere that belong the same
group ('u_vars' in this case). In this case, it allows to further add
one or more processes that would also impact :math:`u` in addition to
advection.

**AdvectionLax**

.. literalinclude:: scripts/advection_lax_xsimlab.py
   :lines: 67-84

``u_advected`` represents the effect of advection on the evolution of
:math:`u` and therefore belongs to the group 'u_vars'. By convention
we use the property ``change`` to store values for that variable.

Computing values of ``u_advected`` requires access to the values of
variables ``spacing`` and ``u`` that are already declared in the
``UniformGrid1D`` and ``ProfileU`` classes, respectively.
:class:`~xsimlab.ForeignVariable` allows to declare references to
these external variables and handle them just as if these were the
original variables. For example, ``self.grid_spacing.value`` in this
class will return the same value than ``self.spacing`` in
``UniformGrid1D``.

**InitUGauss**

.. literalinclude:: scripts/advection_lax_xsimlab.py
   :lines: 87-101

Note that ForeignVariable can also be used to set values for variables
that are declared in other processes, as for ``u`` here.

**Refactored model**

We now have all the building blocks to create a more flexible model:

.. literalinclude:: scripts/advection_lax_xsimlab.py
   :lines: 104-107

The order in which processes are given doesn't matter (it is a
dictionary). A computationally consistent order, as well as model
inputs among all declared variables, are both automatically figured
out when creating the Model instance.

In terms of computation and inputs, ``model2`` is equivalent to the
``model1`` instance created above ; it is just organized
differently.

Update existing models
----------------------

Between the two Model instances created so far, the advantage of
``model2`` over ``model1`` is that we can easily update the model --
change its behavior and/or add many new features -- without
sacrificing readability or losing the ability to get back to the
original, simple version.

**Example: adding a source term at a specific location**

For this we create a new process:

.. literalinclude:: scripts/advection_lax_xsimlab.py
   :lines: 110-134

A couple of comments about this class:

- ``u_source`` belongs to the group 'u_vars' and therefore will be
  added to ``u_advected`` in ``ProfileU`` process.
- Methods and/or properties other than the reserved "runtime" methods
  may be added in a Process subclass, just like in any other Python
  class.
- Nearest node index and source rate array will be recomputed at each
  time iteration because variables ``loc`` and ``flux`` may both have
  a time dimension (variable source location and intensity), i.e.,
  ``self.loc.value`` and ``self.flux.value`` may both change at each
  time iteration.

In this example we also want to start with a flat, zero :math:`u`
profile instead of a gaussian pulse. We create another (minimal)
process for that:

.. literalinclude:: scripts/advection_lax_xsimlab.py
   :lines: 137-147

Using one command, we can then update the model with these new
features:

.. literalinclude:: scripts/advection_lax_xsimlab.py
   :lines: 150-151

Compared to ``model2``, this new ``model3`` have a new process named
'source' and a replaced process 'init'.

**Removing one or more processes**

It is also possible to create new models by removing one or more
processes from existing Model instances, e.g.,

.. literalinclude:: scripts/advection_lax_xsimlab.py
   :lines: 154

In this latter case, users will have to provide initial values of
:math:`u` along the grid directly as an input array.

.. note::

   Model instances are immutable, i.e., once created it is not
   possible to modify these instances by adding, updating or removing
   processes. Both methods ``.update_processes`` and
   ``.drop_processes`` always return new Model instances.
