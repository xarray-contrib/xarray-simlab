.. _create_model:

Create and modify models
========================

Like :doc:`framework` section, this section is useful mostly for
users who want to create new models from scratch or customize existing
models. Users who only want to run simulations from existing models
may skip this section.

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
:math:`i` of a uniform grid and :math:`\Delta t` is the time step
duration between :math:`n` and :math:`n+1`.

We could just implement this numerical model with a few lines of
Python / Numpy code (assuming below periodic boundary conditions and
a gaussian pulse as initial conditions):

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

In this section we will show, however, that it is very easy to
refactor this Python / Numpy code using xarray-simlab. We will also
show that, while enabling useful features, the refactoring still
results in a short amount of readable code.

.. _`Lax method`: https://en.wikipedia.org/wiki/Lax%E2%80%93Friedrichs_method

Anatomy of a Process subclass
-----------------------------

Let's first wrap the code above into a single subclass of
:class:`~xsimlab.Process` that we name ``Advection1D``. Next we'll explain in
detail the content of the class.

.. hidden_todo

   move all code blocks of this section in a proper Python module to import in
   other sections. Use literalinclude directive to show relevant code blocks
   in this sections.

.. code-block:: python

    from xsimlab import Process, FloatVariable


    class Advection1D(Process):
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

``Advection1D`` has some class attributes declared at the top, which
together form the process' "public" interface, i.e., all the variables
that we want to be publicly exposed by this process. These attributes
usually correspond to instances of :class:`~xsimlab.Variable` class or
derived classes, depending on their expected value type, like
:class:`~xsimlab.FloatVariable` in this case (see section :doc:`api`
for a list of all available classes).

The creation of Variable objects requires to explicitly provide
dimension names. In this case, the variables ``spacing``, ``length``,
``loc`` and ``scale`` are all scalars (i.e., an empty tuple is given
as the 1st argument of the FloatVariable constructor), whereas ``x``
and ``u`` are both defined on the 1-dimensional :math:`x` grid. A list
of dimension names can also be given, like ``v`` which represents a
velocity field that can be either constant or variable in space.

.. note::

   All variable objects also implicitly allow a time dimension as
   well as their own dimension (coordinate). See section :doc:`run_model`.

Some metadata may also be assigned to each variable

warning: a process which updates the value (i.e., state) of a variable
does not necessarily provide a value for that variable. An initial value
might still be required

Process methods
~~~~~~~~~~~~~~~

``Advection1D`` also implements methods that are specific to a process:

- ``initialize``, which is called once at the beginning of a
  simulation. Here it is used to set the x-coordinates of the grid and
  the initial values of ``u`` along the grid (gaussian pulse).
- ``run_step``, which is called at each time step iteration and which
  requires current time step duration as argument.
- ``finalize_step``, which is also called at each time step iteration
  but after having called ``run_step`` for all other processes. Its main
  use is to ensure that state variables like ``u`` are updated consistently
  and after having taken snapshots.


Create a Model instance
-----------------------


Fine-grained process refactoring
--------------------------------
