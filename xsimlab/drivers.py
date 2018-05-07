import copy

import numpy as np
import xarray as xr

from .utils import variables_dict


class BaseSimulationDriver(object):
    """Base class that provides a minimal interface for creating
    simulation drivers (should be inherited).

    It also implements methods for binding a simulation data store to
    a model and for updating both this active data store and the
    simulation output store.

    """
    def __init__(self, model, store, output_store):
        self.model = model
        self.store = store
        self.output_store = output_store

        self._bind_store_to_model()

    def _bind_store_to_model(self):
        """Bind the simulation active data store to each process in the
        model.
        """
        for p_obj in self.model.values():
            p_obj.__xsimlab_store__ = self.store

    def update_store(self, input_vars):
        """Update the simulation active data store with input variable
        values.

        ``input_vars`` is a dictionary where keys are store keys, i.e.,
        ``(process_name, var_name)`` tuples, and values are the input
        values to set in the store.

        Values are first copied from ``input_vars`` before being put in
        the store to prevent weird behavior (as model processes might
        update in-place the values in the store).

        Entries of ``input_vars`` that doesn't correspond to model
        inputs are silently ignored.

        """
        for key in self.model.input_vars:
            value = input_vars.get(key)

            if value is not None:
                self.store[key] = copy.copy(value)

    def update_output_store(self, output_var_keys):
        """Update the simulation output store (i.e., append new values to the
        store) from snapshots of variables given in ``output_var_keys`` list.
        """
        for key in output_var_keys:
            p_name, var_name = key
            p_obj = self.model._processes[p_name]
            value = getattr(p_obj, var_name)

            self.output_store.append(key, value)

    def run_model(self):
        """Main function of the driver used to run a simulation (must be
        implemented in sub-classes).
        """
        raise NotImplementedError()


def _get_dims_from_variable(array, var, clock):
    """Given an array with numpy compatible interface and a
    (xarray-simlab) variable, return dimension labels for the
    array.

    """
    ndim = array.ndim
    if clock is not None:
        ndim -= 1      # ignore clock dimension

    for dims in var.metadata['dims']:
        if len(dims) == ndim:
            return dims

    return tuple()


class XarraySimulationDriver(BaseSimulationDriver):
    """Simulation driver using xarray.Dataset objects as I/O.

    - Perform some sanity checks on the content of the given input Dataset.
    - Set model inputs from data variables or coordinates in the input Dataset.
    - Save model outputs for given model variables, defined in specific
      attributes of the input Dataset, on time frequencies given by clocks
      defined as coordinates in the input Dataset.
    - Get simulation results as a new xarray.Dataset object.

    """
    def __init__(self, dataset, model, store, output_store):
        self.dataset = dataset
        self.model = model

        super(XarraySimulationDriver, self).__init__(
            model, store, output_store)

        self.master_clock_dim = dataset.xsimlab.master_clock_dim
        if self.master_clock_dim is None:
            raise ValueError("Missing master clock dimension / coordinate")

        self._check_missing_model_inputs()

        self.output_vars = dataset.xsimlab.output_vars
        self.output_save_steps = self._get_output_save_steps()

    def _check_missing_model_inputs(self):
        """Check if all model inputs have their corresponding variables
        in the input Dataset.
        """
        missing_xr_vars = []

        for p_name, var_name in self.model.input_vars:
            xr_var_name = p_name + '__' + var_name

            if xr_var_name not in self.dataset:
                missing_xr_vars.append(xr_var_name)

        if missing_xr_vars:
            raise KeyError("Missing variables %s in Dataset"
                           % missing_xr_vars)

    def _get_output_save_steps(self):
        """Returns a dictionary where keys are names of clock coordinates and
        values are numpy boolean arrays that specify whether or not to
        save outputs at every step of a simulation.
        """
        save_steps = {}

        for clock in self.output_vars:
            if clock is None:
                continue

            elif clock == self.master_clock_dim:
                save_steps[clock] = np.ones_like(
                    self.dataset[self.master_clock_dim].values, dtype=bool)

            else:
                save_steps[clock] = np.in1d(
                    self.dataset[self.master_clock_dim].values,
                    self.dataset[clock].values)

        return save_steps

    def _get_time_steps(self):
        """Return a xarray.DataArray of duration between two
        consecutive time steps."""
        mclock = self.dataset[self.master_clock_dim]
        return mclock.diff(self.master_clock_dim).values

    def _split_clock_inputs(self):
        """Return two datasets with time-independent and time-dependent
        inputs."""
        ds_in = self.dataset.filter(
            lambda var: self.master_clock_dim not in var.dims)
        ds_in_clock = self.dataset.filter(
            lambda var: self.master_clock_dim in var.dims)

        return ds_in, ds_in_clock

    def _set_input_vars(self, dataset):
        for p_name, var_name in self.model.input_vars:
            xr_var_name = p_name + '__' + var_name
            xr_var = dataset.get(xr_var_name)

            if xr_var is not None:
                data = xr_var.data.copy()

                if data.ndim == 0:
                    # convert array to scalar
                    data = data.item()

                self.store[(p_name, var_name)] = data

    def _maybe_save_output_vars(self, istep):
        # TODO: optimize this for performance
        for clock, var_keys in self.output_vars.items():
            save_output = (
                clock is None and istep == -1 or
                clock is not None and self.output_save_steps[clock][istep]
            )

            if save_output:
                self.update_output_store(var_keys)

    def _to_xr_variable(self, key, clock):
        """Convert an output variable to a xarray.Variable object."""
        p_name, var_name = key
        p_obj = self.model[p_name]
        var = variables_dict(type(p_obj))[var_name]

        data = self.output_store[key]
        if clock is None:
            data = data[0]

        dims = _get_dims_from_variable(data, var, clock)
        if clock is not None:
            dims = (clock,) + dims

        attrs = var.metadata['attrs'].copy()
        if var.metadata['description']:
            attrs['description'] = var.metadata['description']

        return xr.Variable(dims, data, attrs=attrs)

    def _get_output_dataset(self):
        """Return a new dataset as a copy of the input dataset updated with
        output variables.
        """
        from .xr_accessor import SimlabAccessor

        xr_vars = {}

        for clock, vars in self.output_vars.items():
            for key in vars:
                var_name = '__'.join(key)
                xr_vars[var_name] = self._to_xr_variable(key, clock)

        out_ds = self.dataset.update(xr_vars, inplace=False)

        # remove output_vars attributes in output dataset
        for clock in self.output_vars:
            if clock is None:
                attrs = out_ds.attrs
            else:
                attrs = out_ds[clock].attrs
            attrs.pop(SimlabAccessor._output_vars_key)

        return out_ds

    def run_model(self):
        """Run the model and return a new Dataset with all the simulation
        inputs and outputs.

        - Set model inputs from the input Dataset (update
          time-dependent model inputs -- if any -- before each time step).
        - Save outputs (snapshots) between the 'run_step' and the
          'finalize_step' stages or at the end of the simulation.

        """
        ds_in, ds_in_clock = self._split_clock_inputs()
        has_clock_inputs = bool(ds_in_clock.data_vars)

        dt_array = self._get_time_steps()

        self._set_input_vars(ds_in)
        self.model.initialize()

        for istep, dt in enumerate(dt_array):
            if has_clock_inputs:
                ds_in_step = ds_in_clock.isel(**{self.master_clock_dim: istep})
                self._set_input_vars(ds_in_step)

            self.model.run_step(dt)
            self._maybe_save_output_vars(istep)
            self.model.finalize_step()

        self._maybe_save_output_vars(-1)
        self.model.finalize()

        return self._get_output_dataset()
