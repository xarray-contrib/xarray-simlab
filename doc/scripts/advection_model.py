import numpy as np

from xsimlab import FloatVariable, VariableGroup, ForeignVariable
from xsimlab import Process, Model


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


model1 = Model({'advect': AdvectionLax1D})


class UniformGrid1D(Process):
    """Create a 1-dimensional, equally spaced grid."""

    spacing = FloatVariable((), description='uniform spacing')
    length = FloatVariable((), description='total length')
    x = FloatVariable('x', provided=True)

    class Meta:
        time_dependent = False

    def initialize(self):
        self.x.value = np.arange(0, self.length.value, self.spacing.value)


class ProfileU(Process):
    """Compute the evolution of the profile of quantity `u`."""

    u_vars = VariableGroup('u_vars')
    u = FloatVariable('x', description='quantity u',
                      attrs={'units': 'm'})

    def run_step(self, *args):
        self.u.change = sum((var.change for var in self.u_vars))

    def finalize_step(self):
        self.u.state += self.u.change


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


model2 = Model({'grid': UniformGrid1D,
                'profile': ProfileU,
                'init': InitUGauss,
                'advect': AdvectionLax})


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


class InitUFlat(Process):
    """Flat initial profile of `u`."""

    x = ForeignVariable(UniformGrid1D, 'x')
    u = ForeignVariable(ProfileU, 'u', provided=True)

    class Meta:
        time_dependent = False

    def initialize(self):
        self.u.state = np.zeros_like(self.x.value)


model3 = model2.update_processes({'source': SourcePoint,
                                  'init': InitUFlat})


model4 = model2.drop_processes('init')
