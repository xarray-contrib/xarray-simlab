import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from xsimlab import xr_accessor


def test_filter_accessor():
    ds = xr.Dataset(data_vars={'var1': ('x', [1, 2]), 'var2': ('y', [3, 4])},
                    coords={'x': [1, 2], 'y': [3, 4]})
    filtered = ds.filter(lambda var: 'x' in var.dims)
    assert 'var1' in filtered and 'var2' not in filtered
    assert 'x' in filtered.coords and 'y' not in filtered.coords


class TestSimlabAccessor(object):

    _master_clock_key = xr_accessor.SimlabAccessor._master_clock_key
    _snapshot_clock_key = xr_accessor.SimlabAccessor._snapshot_clock_key

    def test_dim_master_clock(self):
        attrs = {self._master_clock_key: 1}
        ds = xr.Dataset(coords={'clock': ('clock', [1, 2], attrs)})

        assert ds.xsimlab.dim_master_clock == 'clock'
        assert ds.xsimlab._dim_master_clock == 'clock'  # cache
        assert ds.xsimlab.dim_master_clock == 'clock'   # get cached value

        ds = xr.Dataset()
        assert ds.xsimlab.dim_master_clock is None

    def test_dim_master_clock_setter(self):
        ds = xr.Dataset(coords={'clock': [1, 2], 'clock2': [3, 4]})

        ds.xsimlab.dim_master_clock = 'clock'
        assert self._master_clock_key in ds.clock.attrs

        ds.xsimlab.dim_master_clock = 'clock2'
        assert self._master_clock_key not in ds.clock.attrs
        assert self._master_clock_key in ds.clock2.attrs

        with pytest.raises(KeyError):
            ds.xsimlab.dim_master_clock = 'invalid_clock'

    def test_set_master_clock(self):
        data = [0, 2, 4, 6, 8]

        valid_kwargs = [
            {'data': data},
            # data provided -> ignore other arguments even if invalid
            {'data': data, 'end': 0, 'nsteps': -1, 'step': 3},
            {'nsteps': 4, 'end': 8, 'step': 2},
            {'nsteps': 4, 'end': 8},
            {'nsteps': 4, 'step': 2},
            {'step': 2, 'end': 8}
        ]
        for kwargs in valid_kwargs:
            ds = xr.Dataset()
            ds.xsimlab.set_master_clock('clock', **kwargs)
            assert_array_equal(ds.clock.values, data)

        invalid_kwargs = [
            {'nsteps': 4, 'end': 8, 'step': 3},
            {'start': 1, 'nsteps': 4, 'end': 8, 'step': 2},
            {'nsteps': 4}
        ]
        for kwargs in invalid_kwargs:
            with pytest.raises(ValueError) as excinfo:
                ds = xr.Dataset()
                ds.xsimlab.set_master_clock('clock', **kwargs)
            assert "Invalid combination" in str(excinfo.value)

        ds = xr.Dataset()
        ds.xsimlab.set_master_clock('clock', data=data)
        assert self._master_clock_key in ds.clock.attrs

        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab.set_master_clock('clock', data=data)
        assert "already exists" in str(excinfo.value)

    def test_set_snapshot_clock(self):
        with pytest.raises(ValueError) as excinfo:
            ds = xr.Dataset()
            ds.xsimlab.set_snapshot_clock('snap_clock', data=[1, 2])
        assert "no master clock" in str(excinfo.value)

        ds = xr.Dataset()
        ds.xsimlab.set_master_clock('clock', data=[0, 2, 4, 6, 8])

        ds.xsimlab.set_snapshot_clock('snap_clock', end=8, step=4)
        assert_array_equal(ds['snap_clock'], [0, 4, 8])
        assert self._snapshot_clock_key in ds['snap_clock'].attrs

        ds.xsimlab.set_snapshot_clock('snap_clock', data=[0, 3, 8])
        assert_array_equal(ds['snap_clock'], [0, 4, 8])

        with pytest.raises(KeyError):
            ds.xsimlab.set_snapshot_clock('snap_clock', data=[0, 3, 8],
                                          auto_adjust=False)
