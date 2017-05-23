"""
Model and processes related to block uplift and stream-power channel erosion.
"""
import numpy as np

from .. import (Model, Process, Variable, ForeignVariable, UndefinedVariable,
                VariableList, diagnostic)
from ... import algos


class StackedGridXY(Process):
    """A 2-dimensional regular grid with grid nodes stacked in
    1-dimension.

    The grid is stacked along the `y` dimension.

    """
    x_size = Variable((), optional=True,
                      description='nb. of nodes in x')
    y_size = Variable((), optional=True,
                      description='nb. of nodes in y')
    x_length = Variable((), optional=True,
                        description='total grid length in x')
    y_length = Variable((), optional=True,
                        description='total grid length in y')
    x_spacing = Variable((), optional=True,
                         description='node spacing in x')
    y_spacing = Variable((), optional=True,
                         description='node spacing in y')
    x_origin = Variable((), optional=True, default_value=0.,
                        description='grid x-origin')
    y_origin = Variable((), optional=True, default_value=0.,
                        description='grid y-origin')
    x = Variable('node', provided=True)
    y = Variable('node', provided=True)

    class Meta:
        time_dependent = False

    def _validate_grid_params(self, size, length, spacing):
        params = {'size': size, 'length': length, 'spacing': spacing}
        provided_params = {k for k, v in params.items()
                           if np.asscalar(v.value) is not None}

        if provided_params == {'size', 'spacing', 'length'}:
            if (size.value - 1) * spacing.value == length.value:
                provided_params = {'size', 'length'}

        if provided_params == {'size', 'length'}:
            spacing.value = length.value / (size.value - 1)
        elif provided_params == {'spacing', 'length'}:
            size.value = int(length.value / spacing.value)
        elif provided_params == {'size', 'spacing'}:
            length.value = spacing.value * (size.value - 1)
        else:
            raise ValueError("Invalid combination of size (%d), spacing (%s) "
                             "and length (%s)"
                             % (size.value, spacing.value, length.value))

    def validate(self):
        self._validate_grid_params(self.x_size, self.x_length, self.x_spacing)
        self._validate_grid_params(self.y_size, self.y_length, self.y_spacing)

    def initialize(self):
        x = np.linspace(self.x_origin.value,
                        self.x_origin.value + self.x_length.value,
                        self.x_size.value)
        y = np.linspace(self.y_origin.value,
                        self.y_origin.value + self.y_length.value,
                        self.y_size.value)

        X, Y = np.meshgrid(x, y)

        self.x.value = X.flatten()
        self.y.value = Y.flatten()


class BoundaryFacesXY(Process):
    """Boundary conditions where each face of the grid in
    both x and y is considered as a boundary.

    """
    x_size = ForeignVariable(StackedGridXY, 'x_size')
    y_size = ForeignVariable(StackedGridXY, 'y_size')
    active_nodes = Variable('node', provided=True)

    class Meta:
        time_dependent = False

    def initialize(self):
        mask = np.ones((self.y_size.value, self.x_size.value), dtype=bool)
        bound_indexers = [0, -1, (slice(None), 0), (slice(None), -1)]

        for idx in bound_indexers:
            mask[idx] = False

        self.active_nodes.value = mask.flatten()


class TotalErosion(Process):
    """Sum of all erosion processes."""

    erosion = Variable('node', provided=True)
    erosion_vars = UndefinedVariable()

    def run_step(self, *args):
        var = self.erosion_vars[0]
        self.erosion.change = var.change

        for var in self.erosion_vars[1:]:
            self.erosion.change += var.change


class TotalExhumation(Process):
    """Sum of all exhumation processes."""

    exhumation = Variable('node', provided=True)
    exhumation_vars = UndefinedVariable()

    def run_step(self, *args):
        var = self.exhumation_vars[0]
        self.exhumation.change = var.change

        for var in self.exhumation_vars[1:]:
            self.exhumation.change += var.change


class Topography(Process):
    """Topography evolution resulting from the balance between
    total exhumation and total erosion.

    This process has also two diagnostics available:
    topographic slope and curvature.

    """
    elevation = Variable('node', description='topographic elevation')
    total_erosion = ForeignVariable(TotalErosion, 'erosion')
    total_exhumation = ForeignVariable(TotalExhumation, 'exhumation')

    def run_step(self, *args):
        self.elevation.change = (
            self.total_exhumation.change - self.total_erosion.change)

    def finalize_step(self):
        self.elevation.state += self.elevation.change

    @diagnostic
    def slope(self):
        """topographic slope"""
        raise NotImplementedError()

    @diagnostic({'units': '1/m'})
    def curvature(self):
        """topographic curvature"""
        raise NotImplementedError()


class SingleFlowRouterD8(Process):
    """Compute flow receivers using D8 and also compute the node
    ordering stack following Braun and Willet method.

    """
    x_size = ForeignVariable(StackedGridXY, 'x_size')
    y_size = ForeignVariable(StackedGridXY, 'y_size')
    x_spacing = ForeignVariable(StackedGridXY, 'x_spacing')
    y_spacing = ForeignVariable(StackedGridXY, 'y_spacing')
    elevation = ForeignVariable(Topography, 'elevation')
    flow_receiver = Variable('node', provided=True)
    distance_to_receiver = Variable('node', provided=True)
    stack = Variable('node', provided=True)

    def initialize(self):
        nn = self.x_size.value * self.y_size.value
        self.nn = nn

        self.flow_receiver.state = np.arange(nn, dtype=np.int)
        self.distance_to_receiver.state = np.zeros(nn)

        self.ndonor = np.zeros(nn, dtype=np.int)
        self.donor = np.empty((nn, 8), dtype=np.int)
        self.stack.state = np.empty_like(self.ndonor)

    def run_step(self, *args):
        algos.compute_flow_receiver_d8(
            self.flow_receiver.state, self.distance_to_receiver.state,
            self.elevation.state, self.x_size.value, self.y_size.value,
            self.x_spacing.value, self.y_spacing.value)
        algos.compute_stack(self.stack.state,
                            self.flow_receiver.state,
                            self.donor, self.ndonor, self.nn)


class PropagateArea(Process):
    """Compute drainage area."""

    dx = ForeignVariable(StackedGridXY, 'x_spacing')
    dy = ForeignVariable(StackedGridXY, 'y_spacing')
    flow_receiver = ForeignVariable(SingleFlowRouterD8, 'flow_receiver')
    stack = ForeignVariable(SingleFlowRouterD8, 'stack')
    area = Variable('node', provided=True)

    def initialize(self):
        self.grid_cell_area = self.dx.value * self.dy.value
        self.area.state = np.empty(self.flow_receiver.state.size)

    def run_step(self, *args):
        self.area.state[:] = self.grid_cell_area
        algos.propagate_area(self.area.state, self.stack.state,
                             self.flow_receiver.state)


class StreamPower(Process):
    """Compute channel erosion using the stream power law."""

    k_coef = Variable((), description='stream-power constant')
    m_exp = Variable((), description='stream-power drainage area exponent')
    n_exp = Variable((), description='stream-power slope exponent')
    erosion = Variable('node', provided=True)

    flow_receiver = ForeignVariable(SingleFlowRouterD8, 'flow_receiver')
    distance_to_receiver = ForeignVariable(SingleFlowRouterD8,
                                           'distance_to_receiver')
    stack = ForeignVariable(SingleFlowRouterD8, 'stack')
    area = ForeignVariable(PropagateArea, 'area')
    elevation = ForeignVariable(Topography, 'elevation')

    def initialize(self):
        self.tolerance = 1e-3
        self.nn = self.elevation.value.size
        self.erosion.change = np.zeros_like(self.elevation.state)

    def run_step(self, dt):
        # note: numba significant speed-up when array -> scalar
        algos.compute_stream_power(
            self.erosion.change, self.elevation.state,
            self.stack.state, self.flow_receiver.state,
            self.distance_to_receiver.state, self.area.state,
            np.asscalar(self.k_coef.value),
            np.asscalar(self.m_exp.value),
            np.asscalar(self.n_exp.value),
            dt, self.tolerance, self.nn)


class Uplift(Process):
    """Compute uplift."""

    u_coef = Variable((), description='uplift rate')
    active_nodes = ForeignVariable(BoundaryFacesXY, 'active_nodes')
    uplift = Variable((), provided=True)

    def initialize(self):
        self.uplift.change = np.zeros(self.active_nodes.value.size)

    def run_step(self, dt):
        self.uplift.change[self.active_nodes.value] = self.u_coef.value * dt


ers_vars = VariableList([ForeignVariable(StreamPower, 'erosion')])
exh_vars = VariableList([ForeignVariable(Uplift, 'uplift')])


stream_power_model = Model(
    {'grid': StackedGridXY(),
     'boundaries': BoundaryFacesXY(),
     'uplift': Uplift(),
     'flow_routing': SingleFlowRouterD8(),
     'area': PropagateArea(),
     'spower': StreamPower(),
     'erosion': TotalErosion(erosion_vars=ers_vars),
     'exhumation': TotalExhumation(exhumation_vars=exh_vars),
     'topography': Topography()}
)
