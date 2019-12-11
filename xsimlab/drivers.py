from collections.abc import Mapping
import copy

import numpy as np
import xarray as xr

from .utils import variables_dict


class RuntimeContext(Mapping):
    """A mapping providing runtime information at the current time step."""

    _context_keys = ('sim_start',
                     'sim_end',
                     'step',
                     'step_start',
                     'step_end',
                     'step_delta')

    def __init__(self, **kwargs):
        self._context = {k: 0 for k in self._context_keys}

        self.update(**kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.__setitem__(k, v)

    def __getitem__(self, key):
        return self._context[key]

    def __setitem__(self, key, value):
        if key not in self._context_keys:
            raise KeyError("Invalid key {!r}, should be one of {!r}"
                           .format(key, self._context_keys))

        self._context[key] = value

    def __len__(self):
        return len(self._context)

    def __iter__(self):
        return iter(self._context)


class BaseSimulationDriver:
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

    def _set_in_store(self, input_vars, check_static=True):
        for key in self.model.input_vars:
            value = input_vars.get(key)

            if value is None:
                continue

            p_name, var_name = key
            var = variables_dict(self.model[p_name].__class__)[var_name]

            if check_static and var.metadata.get('static', False):
                raise RuntimeError("Cannot set value in store for "
                                   "static variable {!r} defined "
                                   "in process {!r}"
                                   .format(p_name, var_name))

            self.store[key] = copy.copy(value)

    def initialize_store(self, input_vars):
        """Pre-populate the simulation active data store with input
        variable values.

        This should be called before the simulation starts.

        ``input_vars`` is a dictionary where keys are store keys, i.e.,
        ``(process_name, var_name)`` tuples, and values are the input
        values to set in the store.

        Values are first copied from ``input_vars`` before being put in
        the store to prevent weird behavior (as model processes might
        update in-place the values in the store).

        Entries of ``input_vars`` that doesn't correspond to model
        inputs are silently ignored.

        """
        self._set_in_store(input_vars, check_static=False)

    def update_store(self, input_vars):
        """Update the simulation active data store with input variable
        values.

        Like ``initialize_store``, but here meant to be called during
        simulation runtime.

        """
        self._set_in_store(input_vars, check_static=True)

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

    def _get_input_vars(self, dataset):
        input_vars = {}

        for p_name, var_name in self.model.input_vars:
            xr_var_name = p_name + '__' + var_name
            xr_var = dataset.get(xr_var_name)

            if xr_var is not None:
                data = xr_var.data

                if data.ndim == 0:
                    # convert array to scalar
                    data = data.item()

                input_vars[(p_name, var_name)] = data

        return input_vars

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

        out_ds = self.dataset.copy()
        out_ds.update(xr_vars)

        # remove output_vars attributes in output dataset
        for clock in self.output_vars:
            if clock is None:
                attrs = out_ds.attrs
            else:
                attrs = out_ds[clock].attrs
            attrs.pop(SimlabAccessor._output_vars_key)

        return out_ds

    def _get_runtime_datasets(self):
        mclock_dim = self.master_clock_dim
        mclock_coord = self.dataset[mclock_dim]

        init_data_vars = {
            '_sim_start': mclock_coord[0],
            '_sim_end': mclock_coord[-1]
        }

        ds_init = (self.dataset.assign(init_data_vars)
                               .drop_dims(mclock_dim))

        step_data_vars = {
            '_clock_start': mclock_coord,
            '_clock_end': mclock_coord.shift({mclock_dim: 1}),
            '_clock_diff': mclock_coord.diff(mclock_dim, label='lower')
        }

        ds_all_steps = (self.dataset.drop(list(ds_init.data_vars.keys()),
                                          errors='ignore')
                                    .isel({mclock_dim: slice(0, -1)})
                                    .assign(step_data_vars))

        ds_gby_steps = ds_all_steps.groupby(mclock_dim)

        return ds_init, ds_gby_steps

    def run_model(self):
        """Run the model and return a new Dataset with all the simulation
        inputs and outputs.

        - Set model inputs from the input Dataset (update
          time-dependent model inputs -- if any -- before each time step).
        - Save outputs (snapshots) between the 'run_step' and the
          'finalize_step' stages or at the end of the simulation.

        """
        ds_init, ds_gby_steps = self._get_runtime_datasets()

        runtime_context = RuntimeContext(
            sim_start=ds_init['_sim_start'].values,
            sim_end=ds_init['_sim_end'].values
        )

        self.initialize_store(self._get_input_vars(ds_init))

        self.model.execute('initialize', runtime_context)

        for step, (_, ds_step) in enumerate(ds_gby_steps):

            runtime_context.update(step=step,
                                   step_start=ds_step['_clock_start'].values,
                                   step_end=ds_step['_clock_end'].values,
                                   step_delta=ds_step['_clock_diff'].values)

            self.update_store(self._get_input_vars(ds_step))

            self.model.execute('run_step', runtime_context)
            self._maybe_save_output_vars(step)
            self.model.execute('finalize_step', runtime_context)

        self._maybe_save_output_vars(-1)
        self.model.execute('finalize', runtime_context)

        return self._get_output_dataset()
