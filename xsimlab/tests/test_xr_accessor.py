import pytest
import xarray as xr
import numpy as np

from xsimlab import xr_accessor, create_setup


def test_filter_accessor():
    ds = xr.Dataset(data_vars={'var1': ('x', [1, 2]), 'var2': ('y', [3, 4])},
                    coords={'x': [1, 2], 'y': [3, 4]})
    filtered = ds.filter(lambda var: 'x' in var.dims)
    assert 'var1' in filtered and 'var2' not in filtered
    assert 'x' in filtered.coords and 'y' not in filtered.coords


class TestSimlabAccessor(object):

    _clock_key = xr_accessor.SimlabAccessor._clock_key
    _master_clock_key = xr_accessor.SimlabAccessor._master_clock_key
    _snapshot_vars_key = xr_accessor.SimlabAccessor._snapshot_vars_key

    def test_clock_coords(self):
        ds = xr.Dataset(
            coords={
                'mclock': ('mclock', [0, 1, 2],
                           {self._clock_key: 1, self._master_clock_key: 1}),
                'sclock': ('sclock', [0, 2], {self._clock_key: 1}),
                'no_clock': ('no_clock', [3, 4])
            }
        )
        assert set(ds.xsimlab.clock_coords) == {'mclock', 'sclock'}

    def test_master_clock_dim(self):
        attrs = {self._clock_key: 1, self._master_clock_key: 1}
        ds = xr.Dataset(coords={'clock': ('clock', [1, 2], attrs)})

        assert ds.xsimlab.master_clock_dim == 'clock'
        assert ds.xsimlab._master_clock_dim == 'clock'  # cache
        assert ds.xsimlab.master_clock_dim == 'clock'   # get cached value

        ds = xr.Dataset()
        assert ds.xsimlab.master_clock_dim is None

    def test_set_master_clock_dim(self):
        ds = xr.Dataset(coords={'clock': [1, 2], 'clock2': [3, 4]})

        ds.xsimlab._set_master_clock_dim('clock')
        assert self._master_clock_key in ds.clock.attrs

        ds.xsimlab._set_master_clock_dim('clock2')
        assert self._master_clock_key not in ds.clock.attrs
        assert self._master_clock_key in ds.clock2.attrs

        with pytest.raises(KeyError):
            ds.xsimlab._set_master_clock_dim('invalid_clock')

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
            ds.xsimlab._set_master_clock('clock', **kwargs)
            np.testing.assert_array_equal(ds.clock.values, data)

        invalid_kwargs = [
            {'nsteps': 4, 'end': 8, 'step': 3},
            {'start': 1, 'nsteps': 4, 'end': 8, 'step': 2},
            {'nsteps': 4}
        ]
        for kwargs in invalid_kwargs:
            with pytest.raises(ValueError) as excinfo:
                ds = xr.Dataset()
                ds.xsimlab._set_master_clock('clock', **kwargs)
            assert "Invalid combination" in str(excinfo.value)

        ds = xr.Dataset()
        ds.xsimlab._set_master_clock('clock', data=data,
                                     units='years since 1-1-1 0:0:0',
                                     calendar='365_day')
        assert self._master_clock_key in ds.clock.attrs
        assert ds.clock.attrs['units'] == 'years since 1-1-1 0:0:0'
        assert ds.clock.attrs['calendar'] == '365_day'

        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab._set_master_clock('clock', data=data)
        assert "already exists" in str(excinfo.value)

        ds = xr.Dataset()
        da = xr.DataArray(data, dims='other_dim')
        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab._set_master_clock('clock', data=da)
        assert "expected dimension" in str(excinfo.value)

    def test_set_snapshot_clock(self):
        with pytest.raises(ValueError) as excinfo:
            ds = xr.Dataset()
            ds.xsimlab._set_snapshot_clock('snap_clock', data=[1, 2])
        assert "no master clock" in str(excinfo.value)

        ds = xr.Dataset()
        ds.xsimlab._set_master_clock('clock', data=[0, 2, 4, 6, 8],
                                     units='years since 1-1-1 0:0:0',
                                     calendar='365_day')

        ds.xsimlab._set_snapshot_clock('snap_clock', end=8, step=4)
        np.testing.assert_array_equal(ds['snap_clock'], [0, 4, 8])
        assert self._clock_key in ds['snap_clock'].attrs
        assert 'units' in ds['snap_clock'].attrs
        assert 'calendar' in ds['snap_clock'].attrs

        ds.xsimlab._set_snapshot_clock('snap_clock', data=[0, 3, 8])
        np.testing.assert_array_equal(ds['snap_clock'], [0, 4, 8])

        with pytest.raises(KeyError):
            ds.xsimlab._set_snapshot_clock('snap_clock', data=[0, 3, 8],
                                           auto_adjust=False)

    def test_set_input_vars(self, model):
        ds = xr.Dataset()

        ds.xsimlab.use_model(model)
        with pytest.raises(KeyError) as excinfo:
            ds.xsimlab._set_input_vars(model, 'invalid_process', var=1)
        assert "no process named" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab._set_input_vars(model, 'some_process', some_param=0,
                                       invalid_var=1)
        assert "not valid input variables" in str(excinfo.value)

        ds.xsimlab._set_input_vars(model, 'quantity',
                                   quantity=('x', np.zeros(10)))
        expected = xr.DataArray(data=np.zeros(10), dims='x')
        assert "quantity__quantity" in ds
        xr.testing.assert_equal(ds['quantity__quantity'], expected)

        # test time and parameter dimensions
        ds.xsimlab._set_input_vars(model, model.some_process, some_param=[1, 2])
        expected = xr.DataArray(data=[1, 2], dims='some_process__some_param',
                                coords={'some_process__some_param': [1, 2]})
        xr.testing.assert_equal(ds['some_process__some_param'], expected)
        del ds['some_process__some_param']

        ds['clock'] = ('clock', [0, 1], {self._master_clock_key: 1})
        ds.xsimlab._set_input_vars(model, 'some_process',
                                   some_param=('clock', [1, 2]))
        expected = xr.DataArray(data=[1, 2], dims='clock',
                                coords={'clock': [0, 1]})
        xr.testing.assert_equal(ds['some_process__some_param'], expected)

        # test optional
        ds.xsimlab._set_input_vars(model, 'grid')
        expected = xr.DataArray(data=5)
        xr.testing.assert_equal(ds['grid__x_size'], expected)

    def test_set_snapshot_vars(self, model):
        ds = xr.Dataset()
        ds['clock'] = ('clock', [0, 2, 4, 6, 8],
                       {self._clock_key: 1, self._master_clock_key: 1})
        ds['snap_clock'] = ('snap_clock', [0, 4, 8], {self._clock_key: 1})
        ds['not_a_clock'] = ('not_a_clock', [0, 1])

        ds.xsimlab.use_model(model)
        with pytest.raises(KeyError) as excinfo:
            ds.xsimlab._set_snapshot_vars(model, None, invalid_process='var')
        assert "no process named" in str(excinfo.value)

        with pytest.raises(KeyError) as excinfo:
            ds.xsimlab._set_snapshot_vars(model, None, quantity='invalid_var')
        assert "has no variable" in str(excinfo.value)

        ds.xsimlab._set_snapshot_vars(model, None, grid='x')
        assert ds.attrs[self._snapshot_vars_key] == 'grid__x'

        ds.xsimlab._set_snapshot_vars(model, 'clock',
                                      some_process='some_effect',
                                      quantity='quantity')
        expected = {'some_process__some_effect', 'quantity__quantity'}
        actual = set(ds['clock'].attrs[self._snapshot_vars_key].split(','))
        assert actual == expected

        ds.xsimlab._set_snapshot_vars(model, 'snap_clock',
                                      other_process=('other_effect', 'x2'))
        expected = {'other_process__other_effect', 'other_process__x2'}
        actual = set(ds['snap_clock'].attrs[self._snapshot_vars_key].split(','))
        assert actual == expected

        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab._set_snapshot_vars(model, 'not_a_clock',
                                          quantity='quantity')
        assert "not a valid clock" in str(excinfo.value)

    def test_snapshot_vars(self, model):
        ds = xr.Dataset()
        ds['clock'] = ('clock', [0, 2, 4, 6, 8],
                       {self._clock_key: 1, self._master_clock_key: 1})
        ds['snap_clock'] = ('snap_clock', [0, 4, 8], {self._clock_key: 1})
        # snapshot clock with no snapshot variable (attribute) set
        ds['snap_clock2'] = ('snap_clock2', [0, 8],
                             {self._clock_key: 1})

        ds.xsimlab.use_model(model)
        ds.xsimlab._set_snapshot_vars(model, None, grid='x')
        ds.xsimlab._set_snapshot_vars(model, 'clock', quantity='quantity')
        ds.xsimlab._set_snapshot_vars(model, 'snap_clock',
                                      other_process=('other_effect', 'x2'))

        expected = {None: set([('grid', 'x')]),
                    'clock': set([('quantity', 'quantity')]),
                    'snap_clock': set([('other_process', 'other_effect'),
                                       ('other_process', 'x2')])}
        actual = {k: set(v) for k, v in ds.xsimlab.snapshot_vars.items()}
        assert actual == expected

    def test_run(self, model, input_dataset):
        # safe mode True: model cloned -> values not set in original model
        _ = input_dataset.xsimlab.run(model=model)
        assert model.quantity.quantity.value is None

        # safe mode False: model not cloned -> values set in original model
        _ = input_dataset.xsimlab.run(model=model, safe_mode=False)
        assert model.quantity.quantity.value is not None

    def test_run_multi(self):
        ds = xr.Dataset()

        with pytest.raises(NotImplementedError):
            ds.xsimlab.run_multi()


def test_create_setup(model, input_dataset):
    with pytest.raises(TypeError) as excinfo:
        create_setup()
    assert "No context on context stack" in str(excinfo.value)

    expected = xr.Dataset()
    actual = create_setup(model=model)
    xr.testing.assert_identical(actual, expected)

    with pytest.raises(ValueError) as excinfo:
        create_setup(model=model, clocks={})
    assert "cannot determine which clock" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        create_setup(model=model,
                     clocks={
                         'clock': {'data': [0, 1, 2]},
                         'out': {'data': [0, 2]}
                     })
    assert "cannot determine which clock" in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:
        create_setup(model=model,
                     clocks={'clock': {'data': [0, 1, 2]}},
                     master_clock='non_existing_clock_dim')
    assert "master clock dimension name" in str(excinfo.value)

    ds = create_setup(model=model, clocks={'clock': {'data': [0, 1, 2]}})
    assert ds.xsimlab.master_clock_dim == 'clock'

    ds = create_setup(model=model,
                      clocks={
                          'clock': {'data': [0, 1, 2]}
                      },
                      master_clock={
                          'dim': 'clock',
                          'units': 'days since 1-1-1 0:0:0',
                          'calendar': '365_days'})
    assert 'units' in ds.clock.attrs
    assert 'calendar' in ds.clock.attrs

    ds = create_setup(
        model=model,
        input_vars={
            'grid': {'x_size': 10},
            'quantity': {'quantity': ('x', np.zeros(10))},
            'some_process': {'some_param': 1},
            'other_process': {'other_param': ('clock', [1, 2, 3, 4, 5])}
        },
        clocks={
            'clock': {'data': [0, 2, 4, 6, 8]},
            'out': {'data': [0, 4, 8]},
        },
        master_clock='clock',
        snapshot_vars={
            'clock': {'quantity': 'quantity'},
            'out': {'some_process': 'some_effect',
                    'other_process': 'other_effect'},
            None: {'grid': 'x'}
        })
    xr.testing.assert_identical(ds, input_dataset)
