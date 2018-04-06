import copy

import numpy as np
import xarray as xr

from .utils import attr_fields_dict


def _get_dims_from_variable(array, variable):
    """Given an array of values (snapshot) and a (xarray-simlab) Variable
    object, Return dimension labels for the array."""
    for dims in variable.allowed_dims:
        if len(dims) == array.ndim:
            return dims
    return tuple()


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
                self.store[key] = copy(value)

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
        implemented in subclasses).
        """
        raise NotImplementedError()


class XarraySimulationDriver(BaseSimulationDriver):
    """Simulation driver using xarray.Dataset objects as I/O.

    - Performs some sanity checks on the content of the given input Dataset.
    - Sets model inputs from the input Dataset.
    - Saves model outputs for given model variables (defined in attributes of
      Dataset) following one or several clocks (i.e., Dataset coordinates).
    - Gets simulation results as a new xarray.Dataset object.

    """
    def __init__(self, dataset, model, store, output_store):
        self.model = model

        super(XarraySimulationDriver, self).__init__(model, store,
                                                      output_store)

        self.output_vars = dataset.xsimlab.output_vars
        self.output_save_steps = self._get_output_save_steps()

        self.master_clock_dim = dataset.xsimlab.master_clock_dim
        if self.master_clock_dim is None:
            raise ValueError("Missing master clock dimension / coordinate")

        self._check_missing_model_inputs()

    def _check_missing_model_inputs(self):
        """Check if all model inputs have their corresponding data variables
        in the input Dataset.
        """
        missing_data_vars = []

        for p_name, var_name in self.model.input_vars:
            xr_var_name = p_name + '__' + var_name

            if xr_var_name not in self.dataset.data_vars:
                missing_data_vars.append(xr_var_name)

        if missing_data_vars:
            raise KeyError("Missing data variables %s in Dataset"
                           % missing_data_vars)

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

    def _set_input_vars(self, dataset):
        for p_name, var_name in self.model.input_vars:
            xr_var_name = p_name + '__' + var_name
            xr_var = dataset.get(xr_var_name)

            if xr_var is not None:
                self.store[(p_name, var_name)] = xr_var.data.copy()

    def _maybe_save_output_vars(self, istep):
        if istep == -1:
            var_keys = self.output_vars.get(None, [])
            self.update_output_store(var_keys)

        else:
            for clock, var_keys in self.output_vars.items():
                if clock is None and self.snapshot_save[clock][istep]:
                    self.update_output_store(var_keys)

    def snapshot_to_xarray_variable(self, key, clock=None):
        """Convert snapshots taken for a specific model variable to an
        xarray.Variable object.
        """
        p_name, var_name = key
        p_obj = self.model[p_name]
        variable = attr_fields_dict(p_obj)[var_name]

        array_list = self.output_store[key]
        first_array = array_list[0]

        if len(array_list) == 1:
            data = first_array
        else:
            data = np.stack(array_list)

        dims = _get_dims_from_variable(first_array, variable)
        if clock is not None and len(array_list) > 1:
            dims = (clock,) + dims

        attrs = variable.attrs.copy()
        attrs['description'] = variable.description

        return xr.Variable(dims, data, attrs=attrs)

    def create_output_dataset(self):
        """Build a new output Dataset from the input Dataset and
        all snapshots taken during a model run.
        """
        from .xr_accessor import SimlabAccessor

        xr_variables = {}

        for clock, vars in self.output_vars.items():
            for key in vars:
                var_name = '__'.join(key)
                xr_variables[var_name] = self.snapshot_to_xarray_variable(
                    key, clock=clock
                )

        out_ds = self.dataset.update(xr_variables, inplace=False)

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
        ds_in = self.dataset.filter(
            lambda var: self.master_clock_dim not in var.dims)
        ds_in_clock = self.dataset.filter(
            lambda var: self.master_clock_dim in var.dims)

        has_clock_inputs = bool(ds_in_clock.data_vars)

        mclock = self.dataset[self.master_clock_dim]
        da_dt = mclock.diff(self.master_clock_dim)

        self._set_input_vars(ds_in)
        self.model.initialize()

        for istep, dt in enumerate(da_dt):
            if has_clock_inputs:
                ds_in_step = ds_in_clock.isel(**{self.master_clock_dim: istep})
                self._set_input_vars(ds_in_step)

            self.model.run_step(dt)
            self._maybe_save_output_vars(istep)
            self.model.finalize_step()

        self._maybe_save_output_vars(-1)
        self.model.finalize()

        return self.create_output_dataset()
