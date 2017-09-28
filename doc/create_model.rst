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

.. code-block:: python

    import numpy as np


    # grid
    spacing = 0.01
    length = 1.5
    x = np.arange(0, length, spacing)

    # velocity
    v = 1.

    # time
    start = 0.
    end = 1.
    step = 0.01

    # initial gauss profile
    loc = 0.3
    scale = 0.1
    u = np.exp(-1 / scale**2 * (x - loc)**2)
    u0 = u.copy()

    # time loop - Lax method
    factor = (v * step) / (2 * spacing)

    for t in np.arange(start, end, step):
        u_left = np.roll(u, 1)
        u_right = np.roll(u, -1)
        u1 = 0.5 * (u_right + u_left) - factor * (u_right - u_left)
        u = u1.copy()

.. _`Lax method`: https://en.wikipedia.org/wiki/Lax%E2%80%93Friedrichs_method

Anatomy of a Process subclass
-----------------------------

Let's first wrap the code above into a single subclass of
:class:`~xsimlab.Process` named ``AdvectionLax1D``. Next we'll explain in
detail the content of this class.

.. hidden_todo

   move all code blocks of this section in one or more Python modules
   so that it can be imported in other sections. Then use the
   literalinclude directive to show relevant code blocks in this
   sections.

.. code-block:: python

    class AdvectionLax1D(Process):
        """Wrap 1-dimensional advection in a single Process."""

        spacing = FloatVariable((), description='grid spacing')
        length = FloatVariable((), description='grid total length')
        x = FloatVariable('x', provided=True)

        v = FloatVariable([(), 'x'], description='velocity')

        loc = FloatVariable((), description='location of initial profile')
        scale = FloatVariable((), description='scale of initial profile')
        u = FloatVariable('x', description='quantity u',
                          attrs={'units': 'm'}, provided=True)

        def initialize(self):
            self.x.value = np.arange(0, self.length.value, self.spacing.value)
            self.u.state = np.exp(-1 / self.scale.value**2 *
                                  (self.x.value - self.loc.value)**2)

        def run_step(self, dt):
            factor = (self.v.value * dt) / (2 * self.spacing.value)
            u_left = np.roll(self.u.state, 1)
            u_right = np.roll(self.u.state, -1)
            self.u1 = 0.5 * (u_right + u_left) - factor * (u_right - u_left)

        def finalize_step(self):
            self.u.state = self.u1

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

.. code-block:: python

    model1 = Model({'advect': AdvectionLax1D})

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

.. code-block:: python

   class UniformGrid1D(Process):
       """Create a 1-dimensional, equally spaced grid."""
       spacing = FloatVariable((), description='uniform spacing')
       length = FloatVariable((), description='total length')
       x = FloatVariable('x', provided=True)

       class Meta:
           time_dependent = False

       def initialize(self):
           self.x.value = np.arange(0, self.length.value, self.spacing.value)

``class Meta`` is used here to specify that this process is not time
dependent (by default processes are considered as
time-dependent). Grid x-coordinate values only need to be set once at
the beginning of the simulation ; there is no need to implement
``run_step`` here.

**ProfileU**

.. code-block:: python

    class ProfileU(Process):
        """Compute the evolution of the profile of quantity `u`."""
        u_vars = VariableGroup('u_vars')
        u = FloatVariable('x', description='quantity u',
                          attrs={'units': 'm'})

        def run_step(self, *args):
            self.u.change = sum((var.change for var in self.u_vars))

        def finalize_step(self):
            self.u.state += self.u.change

``u_vars`` is declared as a :class:`~xsimlab.VariableGroup`, i.e., an
iterable of all variables declared elsewhere that belong the same
group ('u_vars' in this case). In this case, it allows to further add
one or more processes that would also impact :math:`u` in addition to
advection.

**AdvectionLax**

.. code-block:: python

    class AdvectionLax(Process):
        """Advection using finite difference (Lax method) on
        a fixed grid with periodic boundary conditions.
        """
        v = FloatVariable([(), 'x'], description='velocity')
        grid_spacing = ForeignVariable(UniformGrid1D, 'spacing')
        u = ForeignVariable(ProfileU, 'u')
        u_advected = FloatVariable('x', provided=True, group='u_vars')

        def run_step(self, dt):
            factor = self.v.value / (2 * self.grid_spacing.value)

            u_left = np.roll(self.u.state, 1)
            u_right = np.roll(self.u.state, -1)
            u_1 = 0.5 * (u_right + u_left) - factor * dt * (u_right - u_left)

            self.u_advected.change = u_1 - self.u.state

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

.. code-block:: python

    class InitUGauss(Process):
        """Initialize `u` profile using a gaussian pulse."""
        loc = FloatVariable((), description='location of initial pulse')
        scale = FloatVariable((), description='scale of initial pulse')
        x = ForeignVariable(UniformGrid1D, 'x')
        u = ForeignVariable(ProfileU, 'u', provided=True)

        class Meta:
            time_dependent = False

        def initialize(self):
            self.u.state = np.exp(
                -1 / self.scale.value**2 * (self.x.value - self.loc.value)**2
            )

Note that ForeignVariable can also be used to set values for variables
that are declared in other processes, as for ``u`` here.

**Refactored model**

We now have all the building blocks to create a more flexible model:

.. code-block:: python

    model2 = Model({'grid': UniformGrid1D,
                    'profile': ProfileU,
                    'init': InitUGauss,
                    'advect': Advection})

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

.. code-block:: python

    class SourcePoint(Process):
        """Source point for quantity `u`.

        The location of the source point is adjusted to coincide with
        the nearest node the grid.
        """
        loc = FloatVariable((), description='source location')
        flux = FloatVariable((), description='source flux')
        x = ForeignVariable(UniformGrid1D, 'x')
        u_source = FloatVariable('x', provided=True, group='u_vars')

        @property
        def nearest_node(self):
            idx = np.abs(self.x.value - self.loc.value).argmin()
            return idx

        @property
        def source_rate(self):
            src_array = np.zeros_like(self.x.value)
            src_array[self.nearest_node] = self.flux.value
            return src_array

        def run_step(self, dt):
            self.u_source.change = self.source_rate * dt

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

.. code-block:: python

    class InitUFlat(Process):
        """Flat initial profile of `u`."""
        x = ForeignVariable(UniformGrid1D, 'x')
        u = ForeignVariable(ProfileU, 'u', provided=True)

        class Meta:
            time_dependent = False

        def initialize(self):
            self.u.state = np.zeros_like(x)

Using one command, we can then update the model with these new
features:

.. code-block:: python

   model3 = model2.update_processes({'source': SourcePoint,
                                     'init': InitUFlat})

Compared to ``model2``, this new ``model3`` have a new process named
'source' and a replaced process 'init'.

**Removing one or more processes**

It is also possible to create new models by removing one or more
processes from existing Model instances, e.g.,

.. code-block:: python

   model4 = model2.drop_processes('init')

In this latter case, users will have to provide initial values of
:math:`u` along the grid directly as an input array.

.. note::

   Model instances are immutable, i.e., once created it is not
   possible to modify these instances by adding, updating or removing
   processes. Both methods ``.update_processes`` and
   ``.drop_processes`` always return new Model instances.
