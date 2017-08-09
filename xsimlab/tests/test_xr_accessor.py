import pytest
import xarray as xr
import numpy as np

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
    _snapshot_vars_key = xr_accessor.SimlabAccessor._snapshot_vars_key

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
            np.testing.assert_array_equal(ds.clock.values, data)

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
        ds.xsimlab.set_master_clock('clock', data=data,
                                    units='years since 1-1-1 0:0:0',
                                    calendar='365_day')
        assert self._master_clock_key in ds.clock.attrs
        assert ds.clock.attrs['units'] == 'years since 1-1-1 0:0:0'
        assert ds.clock.attrs['calendar'] == '365_day'

        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab.set_master_clock('clock', data=data)
        assert "already exists" in str(excinfo.value)

        ds = xr.Dataset()
        da = xr.DataArray(data, dims='other_dim')
        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab.set_master_clock('clock', data=da)
        assert "expected dimension" in str(excinfo.value)

    def test_set_snapshot_clock(self):
        with pytest.raises(ValueError) as excinfo:
            ds = xr.Dataset()
            ds.xsimlab.set_snapshot_clock('snap_clock', data=[1, 2])
        assert "no master clock" in str(excinfo.value)

        ds = xr.Dataset()
        ds.xsimlab.set_master_clock('clock', data=[0, 2, 4, 6, 8],
                                    units='years since 1-1-1 0:0:0',
                                    calendar='365_day')

        ds.xsimlab.set_snapshot_clock('snap_clock', end=8, step=4)
        np.testing.assert_array_equal(ds['snap_clock'], [0, 4, 8])
        assert self._snapshot_clock_key in ds['snap_clock'].attrs
        assert 'units' in ds['snap_clock'].attrs
        assert 'calendar' in ds['snap_clock'].attrs

        ds.xsimlab.set_snapshot_clock('snap_clock', data=[0, 3, 8])
        np.testing.assert_array_equal(ds['snap_clock'], [0, 4, 8])

        with pytest.raises(KeyError):
            ds.xsimlab.set_snapshot_clock('snap_clock', data=[0, 3, 8],
                                          auto_adjust=False)

    def test_set_input_vars(self, model):
        ds = xr.Dataset()

        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab.set_input_vars('process', var=1)
        assert "no model attached" in str(excinfo.value)

        ds.xsimlab.use_model(model)
        with pytest.raises(KeyError) as excinfo:
            ds.xsimlab.set_input_vars('invalid_process', var=1)
        assert "no process named" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab.set_input_vars('some_process', some_param=0,
                                      invalid_var=1)
        assert "not valid input variables" in str(excinfo.value)

        ds.xsimlab.set_input_vars('quantity', quantity=('x', np.zeros(10)))
        expected = xr.DataArray(data=np.zeros(10), dims='x')
        assert "quantity__quantity" in ds
        xr.testing.assert_equal(ds['quantity__quantity'], expected)

        # test time and parameter dimensions
        ds.xsimlab.set_input_vars(model.some_process, some_param=[1, 2])
        expected = xr.DataArray(data=[1, 2], dims='some_process__some_param',
                                coords={'some_process__some_param': [1, 2]})
        xr.testing.assert_equal(ds['some_process__some_param'], expected)
        del ds['some_process__some_param']

        ds['clock'] = ('clock', [0, 1], {self._master_clock_key: 1})
        ds.xsimlab.set_input_vars('some_process', some_param=('clock', [1, 2]))
        expected = xr.DataArray(data=[1, 2], dims='clock',
                                coords={'clock': [0, 1]})
        xr.testing.assert_equal(ds['some_process__some_param'], expected)

        # test optional
        ds.xsimlab.set_input_vars('grid')
        expected = xr.DataArray(data=5)
        xr.testing.assert_equal(ds['grid__x_size'], expected)

    def test_set_snapshot_vars(self, model):
        ds = xr.Dataset()
        ds['clock'] = ('clock', [0, 2, 4, 6, 8], {self._master_clock_key: 1})
        ds['snap_clock'] = ('snap_clock', [0, 4, 8],
                            {self._snapshot_clock_key: 1})
        ds['not_a_clock'] = ('not_a_clock', [0, 1])

        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab.set_snapshot_vars(None, process='var')
        assert "no model attached" in str(excinfo.value)

        ds.xsimlab.use_model(model)
        with pytest.raises(KeyError) as excinfo:
            ds.xsimlab.set_snapshot_vars(None, invalid_process='var')
        assert "no process named" in str(excinfo.value)

        with pytest.raises(KeyError) as excinfo:
            ds.xsimlab.set_snapshot_vars(None, quantity='invalid_var')
        assert "has no variable" in str(excinfo.value)

        ds.xsimlab.set_snapshot_vars(None, grid='x')
        assert ds.attrs[self._snapshot_vars_key] == 'grid__x'

        ds.xsimlab.set_snapshot_vars('clock', some_process='some_effect',
                                     quantity='quantity')
        expected = {'some_process__some_effect', 'quantity__quantity'}
        actual = set(ds['clock'].attrs[self._snapshot_vars_key].split(','))
        assert actual == expected

        ds.xsimlab.set_snapshot_vars('snap_clock',
                                     other_process=('other_effect', 'x2'))
        expected = {'other_process__other_effect', 'other_process__x2'}
        actual = set(ds['snap_clock'].attrs[self._snapshot_vars_key].split(','))
        assert actual == expected

        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab.set_snapshot_vars('not_a_clock', quantity='quantity')
        assert "not a valid clock" in str(excinfo.value)

    def test_snapshot_vars(self, model):
        ds = xr.Dataset()
        ds['clock'] = ('clock', [0, 2, 4, 6, 8], {self._master_clock_key: 1})
        ds['snap_clock'] = ('snap_clock', [0, 4, 8],
                            {self._snapshot_clock_key: 1})
        # snapshot clock with no snapshot variable (attribute) set
        ds['snap_clock2'] = ('snap_clock2', [0, 8],
                             {self._snapshot_clock_key: 1})

        ds.xsimlab.use_model(model)
        ds.xsimlab.set_snapshot_vars(None, grid='x')
        ds.xsimlab.set_snapshot_vars('clock', quantity='quantity')
        ds.xsimlab.set_snapshot_vars('snap_clock',
                                     other_process=('other_effect', 'x2'))

        expected = {None: set([('grid', 'x')]),
                    'clock': set([('quantity', 'quantity')]),
                    'snap_clock': set([('other_process', 'other_effect'),
                                       ('other_process', 'x2')])}
        actual = {k: set(v) for k, v in ds.xsimlab.snapshot_vars.items()}
        assert actual == expected

    def test_run(self, model, input_dataset):
        input_dataset.xsimlab.use_model(model)

        # safe mode True: model cloned -> values not set in original model
        _ = input_dataset.xsimlab.run()
        assert model.quantity.quantity.value is None

        # safe mode False: model not cloned -> values set in original model
        _ = input_dataset.xsimlab.run(safe_mode=False)
        assert model.quantity.quantity.value is not None

    def test_run_multi(self):
        ds = xr.Dataset()

        with pytest.raises(NotImplementedError):
            ds.xsimlab.run_multi()
