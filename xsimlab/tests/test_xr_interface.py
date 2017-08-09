import pytest

import numpy as np
import xarray as xr

from xsimlab.xr_accessor import SimlabAccessor
from xsimlab.xr_interface import DatasetModelInterface


class TestDatasetModelInterface(object):

    def test_constructor(self, model, input_dataset):
        ds = xr.Dataset()
        with pytest.raises(ValueError) as excinfo:
            DatasetModelInterface(model, ds)
        assert "missing master clock dimension" in str(excinfo.value)

        invalid_ds = input_dataset.drop('quantity__quantity')
        with pytest.raises(KeyError) as excinfo:
            DatasetModelInterface(model, invalid_ds)
        assert "missing data variables" in str(excinfo.value)

    def test_set_model_inputs(self, input_dataset, ds_model_interface):
        ds = input_dataset.drop('other_process__other_param')
        ds_model_interface.set_model_inputs(ds)
        model = ds_model_interface.model

        assert model.grid.x_size.value == 10
        np.testing.assert_array_equal(model.quantity.quantity.value,
                                      np.zeros(10))
        assert model.some_process.some_param.value == 1
        assert model.other_process.other_param.value is None

    def test_split_data_vars_clock(self, ds_model_interface):
        ds_clock, ds_no_clock = ds_model_interface.split_data_vars_clock()
        assert 'other_process__other_param' in ds_clock
        assert 'other_process__other_param' not in ds_no_clock

    def test_time_step_lengths(self, ds_model_interface):
        np.testing.assert_array_equal(ds_model_interface.time_step_lengths,
                                      [2, 2, 2, 2])

    def test_init_snapshots(self, ds_model_interface):
        ds_model_interface.init_snapshots()

        expected = {('quantity', 'quantity'), ('some_process', 'some_effect'),
                    ('other_process', 'other_effect'), ('grid', 'x')}
        assert set(ds_model_interface.snapshot_values) == expected

        expected = {'clock': np.array([True, True, True, True, True]),
                    'out': np.array([True, False, True, False, True])}
        assert ds_model_interface.snapshot_save.keys() == expected.keys()
        for k in expected:
            np.testing.assert_array_equal(ds_model_interface.snapshot_save[k],
                                          expected[k])

    def test_take_snapshot_var(self, ds_model_interface):
        ds_model_interface.init_snapshots()
        ds_model_interface.set_model_inputs(ds_model_interface.dataset)

        key = ('quantity', 'quantity')
        ds_model_interface.take_snapshot_var(key)
        expected = np.zeros(10)
        actual = ds_model_interface.snapshot_values[key][0]
        np.testing.assert_array_equal(actual, expected)

        # ensure snapshot array is a copy
        actual[0] = 1
        assert actual[0] != ds_model_interface.model.quantity.quantity.value[0]

    @pytest.mark.parametrize(
        'istep,expected_len',
        [(0, [1, 1, 1, 0]), (1, [1, 0, 0, 0]), (-1, [1, 1, 1, 1])]
    )
    def test_take_snapshots(self, ds_model_interface, istep, expected_len):
        ds_model_interface.init_snapshots()
        ds_model_interface.set_model_inputs(ds_model_interface.dataset)

        keys = [('quantity', 'quantity'), ('some_process', 'some_effect'),
                ('other_process', 'other_effect'), ('grid', 'x')]

        ds_model_interface.take_snapshots(istep)
        for k, length in zip(keys, expected_len):
            assert len(ds_model_interface.snapshot_values[k]) == length

    def test_snapshot_to_xarray_variable(self, ds_model_interface):
        ds_model_interface.init_snapshots()
        ds_model_interface.set_model_inputs(ds_model_interface.dataset)
        ds_model_interface.model.initialize()

        ds_model_interface.take_snapshots(0)

        expected = xr.Variable('x', np.zeros(10),
                               {'description': 'a quantity'})
        actual = ds_model_interface.snapshot_to_xarray_variable(
            ('quantity', 'quantity'), clock='clock')
        xr.testing.assert_identical(actual, expected)

        ds_model_interface.take_snapshots(-1)

        expected = xr.Variable(('clock', 'x'), np.zeros((2, 10)))
        actual = ds_model_interface.snapshot_to_xarray_variable(
            ('quantity', 'quantity'), clock='clock')
        xr.testing.assert_equal(actual, expected)

        expected = xr.Variable('x', np.arange(10))
        actual = ds_model_interface.snapshot_to_xarray_variable(('grid', 'x'))
        xr.testing.assert_equal(actual, expected)

    def test_run_model(self, input_dataset, ds_model_interface):
        out_ds = ds_model_interface.run_model()

        expected = input_dataset.copy()
        del expected.attrs[SimlabAccessor._snapshot_vars_key]
        del expected.clock.attrs[SimlabAccessor._snapshot_vars_key]
        del expected.out.attrs[SimlabAccessor._snapshot_vars_key]
        expected['grid__x'] = ('x', np.arange(10), {'description': ''})
        expected['quantity__quantity'] = (
            ('clock', 'x'),
            np.arange(0, 10, 2)[:, None] * np.arange(10) * 1.,
            {'description': 'a quantity'}
        )
        expected['some_process__some_effect'] = (
            ('out', 'x'), np.tile(np.arange(2, 12), 3).reshape(3, 10),
            {'description': ''}
        )
        expected['other_process__other_effect'] = (
            ('out', 'x'), np.tile(np.arange(-2, 8), 3).reshape(3, 10),
            {'description': ''}
        )

        xr.testing.assert_identical(out_ds, expected)
