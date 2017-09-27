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
the refactoring still results in a short amount of readable code.

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

    from xsimlab import Process, FloatVariable


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
elsewhere in the class like if these were regular attributes, e.g.,
using ``self.u`` for variable ``u``.

.. note::

   Like the other variables, ``self.u`` actually returns a copy of the
   corresponding ``FloatVariable`` object that is originally declared
   as a class attribute. Some internal magic happens in xarray-simlab
   in order to avoid conflicts when using the same process in
   different contexts.

Variable objects may hold multiple, independent values that we set/get
via specific properties (see section :doc:`framework`), e.g.,
``self.u.state`` for vector :math:`u` and ``self.x.value`` for
x-coordinate values. Note that we use here the property ``value`` for
all time-independent variables, but it is just an alias of ``state``.

Beside variable attributes, we can use normal attributes in process
subclasses too, like ``self.u1`` in ``AdvectionLax1D``, as long as
they have different names.

Creating a Model instance
-------------------------

Creating a new :class:`~xsimlab.Model` instance is very easy. We just
need to provide a dictionary with the processes that we want to
include in the model, e.g., with only the process created above:

.. code-block:: python

    from xsimlab import Model

    model = Model({'advect': AdvectionLax1D})

That's it! Now we can use that model with the xarray extension provided
by xarray-simlab to create new setups, run the model, take snapshots
for one or more variables on a given frequency, etc. (see section
:doc:`run_model`).

Fine-grained process refactoring
--------------------------------
