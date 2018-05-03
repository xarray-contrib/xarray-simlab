import pytest
import xarray as xr
import numpy as np

from xsimlab import xr_accessor, create_setup
from xsimlab.xr_accessor import (as_variable_key,
                                 _flatten_inputs, _flatten_outputs,
                                 _maybe_get_model_from_context)


def test_filter_accessor():
    ds = xr.Dataset(data_vars={'var1': ('x', [1, 2]), 'var2': ('y', [3, 4])},
                    coords={'x': [1, 2], 'y': [3, 4]})
    filtered = ds.filter(lambda var: 'x' in var.dims)
    assert 'var1' in filtered and 'var2' not in filtered
    assert 'x' in filtered.coords and 'y' not in filtered.coords


def test_get_model_from_context(model):
    with pytest.raises(TypeError) as excinfo:
        _maybe_get_model_from_context(None)
    assert "No model found in context" in str(excinfo.value)

    with model as m:
        assert _maybe_get_model_from_context(None) is m

    with pytest.raises(TypeError) as excinfo:
        _maybe_get_model_from_context('not a model')
    assert "is not an instance of xsimlab.Model" in str(excinfo.value)


def test_as_variable_key():
    assert as_variable_key(('foo', 'bar')) == ('foo', 'bar')
    assert as_variable_key('foo__bar') == ('foo', 'bar')
    assert as_variable_key('foo_bar__baz') == ('foo_bar', 'baz')

    with pytest.raises(ValueError) as excinfo:
        as_variable_key('foo__bar__baz')
    assert "not a valid input variable" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        as_variable_key('foo__')
    assert "not a valid input variable" in str(excinfo.value)


@pytest.mark.parametrize('input_vars,expected', [
    ({('foo', 'bar'): 1}, {('foo', 'bar'): 1}),
    ({'foo__bar': 1}, {('foo', 'bar'): 1}),
    ({'foo': {'bar': 1}}, {('foo', 'bar'): 1})
])
def test_flatten_inputs(input_vars, expected):
    assert _flatten_inputs(input_vars) == expected


@pytest.mark.parametrize('output_vars,expected', [
    ({'clock': 'foo__bar'}, {'clock': [('foo', 'bar')]}),
    ({'clock': ('foo', 'bar')}, {'clock': [('foo', 'bar')]}),
    ({'clock': [('foo', 'bar')]}, {'clock': [('foo', 'bar')]}),
    ({'clock': [('foo__bar')]}, {'clock': [('foo', 'bar')]}),
    ({'clock': {'foo': 'bar'}}, {'clock': [('foo', 'bar')]}),
    ({'clock': {'foo': ['bar', 'baz']}},
     {'clock': [('foo', 'bar'), ('foo', 'baz')]})
])
def test_flatten_outputs(output_vars, expected):
    assert _flatten_outputs(output_vars) == expected


def test_flatten_outputs_error():
    with pytest.raises(ValueError) as excinfo:
        _flatten_outputs({'clock': 2})
    assert "Cannot interpret" in str(excinfo.value)


class TestSimlabAccessor(object):

    _clock_key = xr_accessor.SimlabAccessor._clock_key
    _master_clock_key = xr_accessor.SimlabAccessor._master_clock_key
    _output_vars_key = xr_accessor.SimlabAccessor._output_vars_key

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

    def test_set_input_vars(self, model, in_dataset):
        in_vars = {('init_profile', 'n_points'): 5,
                   ('roll', 'shift'): 1,
                   ('add', 'offset'): ('clock', [1, 2, 3, 4, 5])}

        ds = xr.Dataset(coords={'clock': [0, 2, 4, 6, 8]})
        ds.xsimlab._set_input_vars(model, in_vars)

        for vname in ('init_profile__n_points', 'roll__shift', 'add__offset'):
            # xr.testing.assert_identical also checks attrs of coordinates
            # (not needed here)
            xr.testing.assert_equal(ds[vname], in_dataset[vname])
            assert ds[vname].attrs == in_dataset[vname].attrs

        in_vars[('not_an', 'input_var')] = None

        with pytest.raises(KeyError) as excinfo:
            ds.xsimlab._set_input_vars(model, in_vars)
        assert "not valid key(s)" in str(excinfo.value)

    def test_update_clocks(self, model):
        ds = xr.Dataset()
        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab.update_clocks(model=model, clocks={})
        assert "cannot determine which clock" in str(excinfo.value)

        ds = xr.Dataset()
        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab.update_clocks(
                model=model,
                clocks={'clock': {'data': [0, 1, 2]},
                        'out': {'data': [0, 2]}}
            )
        assert "cannot determine which clock" in str(excinfo.value)

        ds = xr.Dataset()
        with pytest.raises(KeyError) as excinfo:
            ds.xsimlab.update_clocks(
                model=model,
                clocks={'clock': {'data': [0, 1, 2]}},
                master_clock='non_existing_clock_dim'
            )
        assert "master clock dimension name" in str(excinfo.value)

        ds = xr.Dataset()
        ds = ds.xsimlab.update_clocks(
            model=model,
            clocks={'clock': {'data': [0, 1, 2]}}
        )
        assert ds.xsimlab.master_clock_dim == 'clock'

        ds.clock.attrs[self._output_vars_key] = 'profile__u'

        ds = ds.xsimlab.update_clocks(
            model=model,
            clocks={'clock': {'data': [0, 1, 2]}},
            master_clock={'dim': 'clock',
                          'units': 'days since 1-1-1 0:0:0',
                          'calendar': '365_days'}
        )
        assert 'units' in ds.clock.attrs
        assert 'calendar' in ds.clock.attrs
        assert ds.clock.attrs[self._output_vars_key] == 'profile__u'

    def test_update_vars(self, model, in_dataset):
        ds = in_dataset.xsimlab.update_vars(
            model=model,
            input_vars={('roll', 'shift'): 2},
            output_vars={'out': ('profile', 'u')}
        )

        assert not ds['roll__shift'].equals(in_dataset['roll__shift'])
        assert not ds['out'].identical(in_dataset['out'])

    def test_filter_vars(self, simple_model, in_dataset):
        ds = in_dataset.xsimlab.filter_vars(model=simple_model)

        assert 'add__offset' not in ds
        assert sorted(ds.xsimlab.clock_coords) == ['clock', 'out']
        assert ds.out.attrs[self._output_vars_key] == 'roll__u_diff'

    def test_set_output_vars(self, model):
        ds = xr.Dataset()
        ds['clock'] = ('clock', [0, 2, 4, 6, 8],
                       {self._clock_key: 1, self._master_clock_key: 1})
        ds['out'] = ('out', [0, 4, 8], {self._clock_key: 1})
        ds['not_a_clock'] = ('not_a_clock', [0, 1])

        with pytest.raises(KeyError) as excinfo:
            ds.xsimlab._set_output_vars(model, None, [('invalid', 'var')])
        assert "not valid key(s)" in str(excinfo.value)

        ds.xsimlab._set_output_vars(model, None, [('profile', 'u_opp')])
        assert ds.attrs[self._output_vars_key] == 'profile__u_opp'

        ds.xsimlab._set_output_vars(model, 'out',
                                    [('roll', 'u_diff'), ('add', 'u_diff')])
        expected = 'roll__u_diff,add__u_diff'
        assert ds['out'].attrs[self._output_vars_key] == expected

        with pytest.raises(ValueError) as excinfo:
            ds.xsimlab._set_output_vars(model, 'not_a_clock',
                                        [('profile', 'u')])
        assert "not a valid clock" in str(excinfo.value)

    def test_output_vars(self, model):
        ds = xr.Dataset()
        ds['clock'] = ('clock', [0, 2, 4, 6, 8],
                       {self._clock_key: 1, self._master_clock_key: 1})
        ds['out'] = ('out', [0, 4, 8], {self._clock_key: 1})
        # snapshot clock with no output variable (attribute) set
        ds['out2'] = ('out2', [0, 8], {self._clock_key: 1})

        ds.xsimlab._set_output_vars(model, None, [('profile', 'u_opp')])
        ds.xsimlab._set_output_vars(model, 'clock', [('profile', 'u')])
        ds.xsimlab._set_output_vars(model, 'out',
                                    [('roll', 'u_diff'), ('add', 'u_diff')])

        expected = {None: [('profile', 'u_opp')],
                    'clock': [('profile', 'u')],
                    'out': [('roll', 'u_diff'), ('add', 'u_diff')]}
        assert ds.xsimlab.output_vars == expected

    def test_run(self, model, in_dataset):
        # safe mode False: model not cloned -> values set in original model
        _ = in_dataset.xsimlab.run(model=model, safe_mode=False)
        assert model.profile.u is not None

    def test_run_multi(self):
        ds = xr.Dataset()

        with pytest.raises(NotImplementedError):
            ds.xsimlab.run_multi()


def test_create_setup(model, in_dataset):
    expected = xr.Dataset()
    actual = create_setup(model=model)
    xr.testing.assert_identical(actual, expected)

    ds = create_setup(
        model=model,
        input_vars={
            'init_profile': {'n_points': 5},
            ('roll', 'shift'): 1,
            'add__offset': ('clock', [1, 2, 3, 4, 5])
        },
        clocks={
            'clock': {'data': [0, 2, 4, 6, 8]},
            'out': {'data': [0, 4, 8]},
        },
        master_clock='clock',
        output_vars={
            'clock': 'profile__u',
            'out': [('roll', 'u_diff'), ('add', 'u_diff')],
            None: {'profile': 'u_opp'}
        })
    xr.testing.assert_identical(ds, in_dataset)
