import numpy as np
from xarray import Variable

from ... import algos


class Model(object):
    def __init__py(self):
        pass


class StreamPowerModel(Model):
    _model_name = 'stream_power'
    _param_vars = {'k': 'spower__k',
                   'm': 'spower__m',
                   'n': 'spower__n'}

    def __init__(self, grid_dims=('x', 'y'), clock_dim='clock'):
        self.grid_dims = grid_dims
        self.clock_dim = clock_dim

    def run(self, dataset):
        xdim, ydim = self.grid_dims
        x = dataset[xdim]
        y = dataset[ydim]
        nx, ny = x.size, y.size
        x_length, y_length = np.ptp(x.values), np.ptp(y.values)

        dt = float(np.unique(dataset[self.clock_dim].diff(self.clock_dim)))
        nstep = dataset.dims[self.clock_dim]

        uplift_rate = dataset['uplift__u'].values

        k = dataset['spower__k'].values
        n = dataset['spower__n'].values
        m = dataset['spower__m'].values

        tolerance = 1e-3

        nn = nx * ny
        dx, dy = x_length / (nx - 1), y_length / (ny - 1)

        active_nodes = algos.set_active_nodes_mask(nx, ny)

        receiver = np.arange(nn, dtype=np.int)
        dist2receiver = np.zeros(nn)
        elevation = dataset.stack(grid=('y', 'x'))['topographic__elevation'].values.copy()
        out_elevation = np.empty((nstep, nx, ny))
        out_elevation[0] = elevation.reshape(nx, ny)

        ndonor = np.zeros(nn, dtype=np.int)
        donor = np.empty((nn, 8), dtype=np.int)
        stack = np.empty_like(ndonor)

        area = np.empty(nn)

        for istep in range(nstep):
            algos.compute_flow_receiver_d8(receiver, dist2receiver, elevation,
                                           nx, ny, dx, dy)
            algos.compute_stack(stack, receiver, donor, ndonor, nn)

            area[:] = dx * dy
            algos.propagate_area(area, stack, receiver)

            algos.compute_uplift(elevation, active_nodes, uplift_rate, dt)

            algos.compute_stream_power(elevation, stack, receiver,
                                       dist2receiver, area, k, m, n, dt,
                                       tolerance, nn)

            out_elevation[istep] = elevation.reshape(nx, ny)

        out_elevation_var = Variable((self.clock_dim, *self.grid_dims),
                                     out_elevation)
        dataset['topographic__elevation'] = out_elevation_var
