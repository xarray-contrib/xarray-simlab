import numpy as np

import xsimlab as xs


@xs.process
class AdvectionLax1D:
    """Wrap 1-dimensional advection in a single Process."""

    spacing = xs.variable(description="grid spacing", static=True)
    length = xs.variable(description="grid total length", static=True)
    x = xs.variable(dims="x", intent="out")

    v = xs.variable(dims=[(), "x"], description="velocity")

    loc = xs.variable(description="location of initial profile", static=True)
    scale = xs.variable(description="scale of initial profile", static=True)
    u = xs.variable(
        dims="x", intent="out", description="quantity u", attrs={"units": "m"}
    )

    def initialize(self):
        self.x = np.arange(0, self.length, self.spacing)
        self.u = np.exp(-1 / self.scale ** 2 * (self.x - self.loc) ** 2)

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        factor = (self.v * dt) / (2 * self.spacing)
        u_left = np.roll(self.u, 1)
        u_right = np.roll(self.u, -1)
        self.u1 = 0.5 * (u_right + u_left) - factor * (u_right - u_left)

    def finalize_step(self):
        self.u = self.u1


advect_model_raw = xs.Model({"advect": AdvectionLax1D})


@xs.process
class UniformGrid1D:
    """Create a 1-dimensional, equally spaced grid."""

    spacing = xs.variable(description="uniform spacing", static=True)
    length = xs.variable(description="total length", static=True)
    x = xs.index(dims="x")

    def initialize(self):
        self.x = np.arange(0, self.length, self.spacing)


@xs.process
class ProfileU:
    """Compute the evolution of the profile of quantity `u`."""

    u_vars = xs.group("u_vars")
    u = xs.variable(
        dims="x", intent="inout", description="quantity u", attrs={"units": "m"}
    )

    def run_step(self):
        self._delta_u = sum((v for v in self.u_vars))

    def finalize_step(self):
        self.u += self._delta_u


@xs.process
class AdvectionLax:
    """Advection using finite difference (Lax method) on
    a fixed grid with periodic boundary conditions.

    """

    v = xs.variable(dims=[(), "x"], description="velocity")
    grid_spacing = xs.foreign(UniformGrid1D, "spacing")
    u = xs.foreign(ProfileU, "u")
    u_advected = xs.variable(dims="x", intent="out", groups="u_vars")

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        factor = self.v / (2 * self.grid_spacing)

        u_left = np.roll(self.u, 1)
        u_right = np.roll(self.u, -1)
        u_1 = 0.5 * (u_right + u_left) - factor * dt * (u_right - u_left)

        self.u_advected = u_1 - self.u


@xs.process
class InitUGauss:
    """Initialize `u` profile using a Gaussian pulse."""

    loc = xs.variable(description="location of initial pulse", static=True)
    scale = xs.variable(description="scale of initial pulse", static=True)
    x = xs.foreign(UniformGrid1D, "x")
    u = xs.foreign(ProfileU, "u", intent="out")

    def initialize(self):
        self.u = np.exp(-1 / self.scale ** 2 * (self.x - self.loc) ** 2)


advect_model = xs.Model(
    {
        "grid": UniformGrid1D,
        "profile": ProfileU,
        "init": InitUGauss,
        "advect": AdvectionLax,
    }
)


@xs.process
class SourcePoint:
    """Source point for quantity `u`.

    The location of the source point is adjusted to coincide with
    the nearest node the grid.

    """

    loc = xs.variable(description="source location")
    flux = xs.variable(description="source flux")
    x = xs.foreign(UniformGrid1D, "x")
    u_source = xs.variable(dims="x", intent="out", groups="u_vars")

    @property
    def nearest_node(self):
        idx = np.abs(self.x - self.loc).argmin()
        return idx

    @property
    def source_rate(self):
        src_array = np.zeros_like(self.x)
        src_array[self.nearest_node] = self.flux
        return src_array

    @xs.runtime(args="step_delta")
    def run_step(self, dt):
        self.u_source = self.source_rate * dt


@xs.process
class InitUFlat:
    """Flat initial profile of `u`."""

    x = xs.foreign(UniformGrid1D, "x")
    u = xs.foreign(ProfileU, "u", intent="out")

    def initialize(self):
        self.u = np.zeros_like(self.x)


advect_model_src = advect_model.update_processes(
    {"source": SourcePoint, "init": InitUFlat}
)


advect_model_no_init = advect_model.drop_processes("init")


@xs.process
class FixedGridParams:
    spacing = xs.foreign(UniformGrid1D, "spacing", intent="out")
    length = xs.foreign(UniformGrid1D, "length", intent="out")

    def initialize(self):
        self.spacing = 0.01
        self.length = 1.0


advect_model_fgrid = advect_model.update_processes(
    {"fixed_grid_params": FixedGridParams}
)


@xs.process
class FixedGrid(UniformGrid1D):
    spacing = xs.variable(description="uniform spacing", intent="out")
    length = xs.variable(description="total length", intent="out")

    def initialize(self):
        self.spacing = 0.01
        self.length = 1.0
        super(FixedGrid, self).initialize()


advect_model_fgrid2 = advect_model.update_processes({"grid": FixedGrid})
