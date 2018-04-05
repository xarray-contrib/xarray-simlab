"""
xarray extensions (accessors).

"""
from collections import defaultdict

import numpy as np
from xarray import Dataset, register_dataset_accessor

from .model import Model
from .xr_interface import DatasetModelInterface


@register_dataset_accessor('filter')
def filter_accessor(dataset):
    """A temporary hack until ``filter`` is available in xarray (GH916)."""

    def filter(func=None, like=None, regex=None):
        variables = {k: v for k, v in dataset._variables.items() if func(v)}
        coord_names = [c for c in dataset._coord_names if c in variables]

        return dataset._replace_vars_and_dims(variables,
                                              coord_names=coord_names)

    return filter


def _maybe_get_model_from_context(model):
    """Return the given model or try to find it in the context if there was
    none supplied.
    """
    if model is None:
        try:
            return Model.get_context()
        except TypeError:
            raise TypeError("no model found in context")

    if not isinstance(model, Model):
        raise TypeError("%s is not an instance of xsimlab.Model" % model)

    return model


@register_dataset_accessor('xsimlab')
class SimlabAccessor(object):
    """simlab extension to :class:`xarray.Dataset`."""

    _clock_key = '_xsimlab_snapshot_clock'
    _master_clock_key = '_xsimlab_master_clock'
    _output_vars_key = '_xsimlab_output_vars'

    def __init__(self, ds):
        self._ds = ds
        self._master_clock_dim = None

    @property
    def clock_coords(self):
        """Dictionary of :class:`xarray.DataArray` objects corresponding to
        clock coordinates.
        """
        return {k: coord for k, coord in self._ds.coords.items()
                if self._clock_key in coord.attrs}

    @property
    def master_clock_dim(self):
        """Dimension used as master clock for model runs. Returns None
        if no dimension is set as master clock.

        See Also
        --------
        :meth:`Dataset.xsimlab.update_clocks`

        """
        if self._master_clock_dim is not None:
            return self._master_clock_dim
        else:
            for c in self._ds.coords.values():
                if c.attrs.get(self._master_clock_key, False):
                    dim = c.dims[0]
                    self._master_clock_dim = dim
                    return dim
            return None

    def _set_master_clock_dim(self, dim):
        if dim not in self._ds.coords:
            raise KeyError("Dataset has no %r dimension coordinate. "
                           "To create a new master clock dimension, "
                           "use Dataset.xsimlab.update_clock."
                           % dim)

        if self.master_clock_dim is not None:
            self._ds[self.master_clock_dim].attrs.pop(self._master_clock_key)

        self._ds[dim].attrs[self._clock_key] = np.uint8(True)
        self._ds[dim].attrs[self._master_clock_key] = np.uint8(True)
        self._master_clock_dim = dim

    def _set_clock_data(self, dim, data, start, end, step, nsteps):
        if data is not None:
            data_dims = getattr(data, 'dims', None)
            if data_dims is not None and data_dims != (dim,):
                raise ValueError("expected dimension %r for clock coordinate"
                                 "but found %r" % (dim, data_dims))
            return data

        args = {'step': step, 'nsteps': nsteps, 'end': end}
        provided_args = {k for k, v in args.items() if v is not None}

        if provided_args == {'nsteps', 'end', 'step'}:
            if end - start == nsteps * step:
                provided_args = {'nsteps', 'end'}
        if provided_args == {'nsteps', 'end'}:
            data = np.linspace(start, end, nsteps + 1)
        elif provided_args == {'step', 'nsteps'}:
            data = np.arange(start, start + (nsteps + 1) * step, step)
        elif provided_args == {'step', 'end'}:
            data = np.arange(start, end + step, step)
        else:
            raise ValueError("Invalid combination of nsteps (%s), step (%s) "
                             "and end (%s)" % (nsteps, step, end))

        return data

    def _set_master_clock(self, dim, data=None, start=0., end=None,
                          step=None, nsteps=None, units=None, calendar=None):
        if dim in self._ds.dims:
            raise ValueError("dimension %r already exists" % dim)

        self._ds[dim] = self._set_clock_data(dim, data, start, end,
                                              step, nsteps)
        if units is not None:
            self._ds[dim].attrs['units'] = units
        if calendar is not None:
            self._ds[dim].attrs['calendar'] = calendar

        self._set_master_clock_dim(dim)

    def _set_snapshot_clock(self, dim, data=None, start=0., end=None,
                            step=None, nsteps=None, auto_adjust=True):
        if self.master_clock_dim is None:
            raise ValueError("no master clock dimension/coordinate is defined "
                             "in Dataset. "
                             "Use `Dataset.xsimlab._set_master_clock` first")

        clock_data = self._set_clock_data(dim, data, start, end, step, nsteps)

        da_master_clock = self._ds[self.master_clock_dim]

        if auto_adjust:
            kwargs = {'method': 'nearest'}
        else:
            kwargs = {}

        indexer = {self.master_clock_dim: clock_data}
        kwargs.update(indexer)
        da_snapshot_clock = da_master_clock.sel(**kwargs)

        self._ds[dim] = da_snapshot_clock.rename({self.master_clock_dim: dim})
        # .sel copies variable attributes
        self._ds[dim].attrs.pop(self._master_clock_key)

        for attr_name in ('units', 'calendar'):
            attr_value = da_master_clock.attrs.get(attr_name)
            if attr_value is not None:
                self._ds[dim].attrs[attr_name] = attr_value

    def _set_input_vars(self, model, process, **inputs):
        if isinstance(process, Process):
            process = process.name
        if process not in model:
            raise KeyError("no process named %r found in current model"
                           % process)

        process_inputs = model.input_vars[process]

        invalid_inputs = set(inputs) - set(process_inputs)
        if invalid_inputs:
            raise ValueError("%s are not valid input variables of %r"
                             % (', '.join([name for name in invalid_inputs]),
                                process))

        # convert to xarray variables and validate the given dimensions
        xr_variables = {}
        for name, var in model.input_vars[process].items():
            xr_var = var.to_xarray_variable(inputs.get(name))
            var.validate_dimensions(xr_var.dims,
                                    ignore_dims=(self.master_clock_dim,
                                                 'this_variable'))
            xr_variables[name] = xr_var

        # validate at the process level
        # first assign values to a cloned process object to avoid conflicts
        process_obj = model._processes[process].clone()
        for name, xr_var in xr_variables.items():
            process_obj[name].value = xr_var.values
        process_obj.validate()

        # maybe set optional variables, and validate each variable
        for name, xr_var in xr_variables.items():
            var = process_obj[name]
            if var.value is not xr_var.values:
                xr_var = var.to_xarray_variable(var.value)
                xr_variables[name] = xr_var
            var.run_validators(xr_var)
            var.validate(xr_var)

        # add variables to dataset if all validation tests passed
        # also rename the 'this_variable' dimension if present
        for name, xr_var in xr_variables.items():
            xr_var_name = process + '__' + name
            rename_dict = {'this_variable': xr_var_name}
            dims = tuple(rename_dict.get(dim, dim) for dim in xr_var.dims)
            xr_var.dims = dims
            self._ds[xr_var_name] = xr_var

    def _set_output_vars(self, model, clock_dim, **process_vars):
        xr_vars_list = []

        for proc_name, vars in sorted(process_vars.items()):
            if proc_name not in model:
                raise KeyError("no process named %r found in current model"
                               % proc_name)
            process = model[proc_name]
            if isinstance(vars, str):
                vars = [vars]
            for var_name in vars:
                if process.variables.get(var_name, None) is None:
                    raise KeyError("process %r has no variable %r"
                                   % (proc_name, var_name))
                xr_vars_list.append(proc_name + '__' + var_name)

        output_vars = ','.join(xr_vars_list)

        if clock_dim is None:
            self._ds.attrs[self._output_vars_key] = output_vars
        else:
            if clock_dim not in self.clock_coords:
                raise ValueError("%r coordinate is not a valid clock "
                                 "coordinate. " % clock_dim)
            coord = self.clock_coords[clock_dim]
            coord.attrs[self._output_vars_key] = output_vars

    def _get_output_vars(self, name, obj):
        vars_str = obj.attrs.get(self._output_vars_key, '')
        if vars_str:
            return {name: [tuple(s.split('__'))
                           for s in vars_str.split(',')]}
        else:
            return {}

    @property
    def output_vars(self):
        """Returns a dictionary of snapshot clock dimension names as keys and
        output variable names - i.e. lists of (process name, variable name)
        tuples - as values.
        """
        output_vars = {}
        for cname, coord in self._ds.coords.items():
            output_vars.update(self._get_output_vars(cname, coord))
        output_vars.update(self._get_output_vars(None, self._ds))
        return output_vars

    def update_clocks(self, model=None, clocks=None, master_clock=None):
        """Update clock coordinates.

        Drop all clock coordinates (if any) and add a new set of master and
        snapshot clock coordinates.
        Also copy all snapshot-specific attributes of the replaced coordinates.

        More details about the values allowed for the parameters below can be
        found in the doc of :meth:`xsimlab.create_setup`.

        Parameters
        ----------
        model : :class:`xsimlab.Model` object, optional
            Reference model. If None, tries to get model from context.
        clocks : dict of dicts, optional
            Used to create one or several clock coordinates.
        master_clock : str or dict, optional
            Name (and units/calendar) of the clock coordinate (dimension) to
            use as master clock.

        Returns
        -------
        updated : Dataset
            Another Dataset with new or replaced coordinates.

        See Also
        --------
        :meth:`xsimlab.create_setup`

        """
        model = _maybe_get_model_from_context(model)

        ds = self._ds.drop(self.clock_coords)

        attrs_master_clock = {}

        if isinstance(master_clock, str):
            master_clock_dim = master_clock
        elif isinstance(master_clock, dict):
            master_clock_dim = master_clock.pop('dim')
            attrs_master_clock.update(master_clock)
        elif master_clock is None and clocks is not None and len(clocks) == 1:
            master_clock_dim = list(clocks.keys())[0]
        else:
            master_clock_dim = None

        if clocks is not None:
            if master_clock_dim is None:
                raise ValueError("cannot determine which clock coordinate is "
                                 "the master clock")
            elif master_clock_dim not in clocks:
                raise KeyError("master clock dimension name %r not found "
                               "in `clocks`" % master_clock_dim)

            master_clock_kwargs = clocks.pop(master_clock_dim)
            master_clock_kwargs.update(attrs_master_clock)
            ds.xsimlab._set_master_clock(master_clock_dim,
                                         **master_clock_kwargs)

            for dim, kwargs in clocks.items():
                ds.xsimlab._set_snapshot_clock(dim, **kwargs)

        for dim, var_list in self.output_vars.items():
            var_dict = defaultdict(list)
            for proc_name, var_name in var_list:
                var_dict[proc_name].append(var_name)

            if dim is None or dim in ds:
                ds.xsimlab._set_output_vars(model, dim, **var_dict)

        return ds

    def update_vars(self, model=None, input_vars=None, output_vars=None):
        """Update model input values and/or output variable names.

        Add or replace all input values (resp. output variable names) per
        given process (resp. clock coordinate).

        More details about the values allowed for the parameters below can be
        found in the doc of :meth:`xsimlab.create_setup`.

        Parameters
        ----------
        model : :class:`xsimlab.Model` object, optional
            Reference model. If None, tries to get model from context.
        input_vars : dict of dicts, optional
            Model input values given per process.
        output_vars : dict of dicts, optional
            Model variables to save as simulation output, given per
            clock coordinate.

        Returns
        -------
        updated : Dataset
            Another Dataset with new or replaced variables (inputs) and/or
            attributes (snaphots).

        See Also
        --------
        :meth:`xsimlab.create_setup`

        """
        model = _maybe_get_model_from_context(model)

        ds = self._ds.copy()

        if input_vars is not None:
            for proc_name, vars in input_vars.items():
                ds.xsimlab._set_input_vars(model, proc_name, **vars)

        if output_vars is not None:
            for dim, proc_vars in output_vars.items():
                ds.xsimlab._set_output_vars(model, dim, **proc_vars)

        return ds

    def filter_vars(self, model=None):
        """Filter Dataset content according to Model.

        Keep only data variables and coordinates that correspond to inputs of
        the model (keep clock coordinates too). Also update snapshot-specific
        attributes so that their values all correspond to processes and
        variables defined in the model.

        Parameters
        ----------
        model : :class:`xsimlab.Model` object, optional
            Reference model. If None, tries to get model from context.

        Returns
        -------
        filtered : Dataset
            Another Dataset with (maybe) dropped variables and updated
            attributes.

        See Also
        --------
        :meth:`Dataset.xsimlab.update_vars`

        """
        model = _maybe_get_model_from_context(model)

        drop_variables = []

        for xr_var_name in self._ds:
            if xr_var_name in self.clock_coords:
                continue
            try:
                proc_name, var_name = xr_var_name.split('__')
            except ValueError:
                continue

            if not model.is_input((proc_name, var_name)):
                drop_variables.append(xr_var_name)

        ds = self._ds.drop(drop_variables)

        for dim, var_list in self.output_vars.items():
            var_dict = defaultdict(list)
            for proc_name, var_name in var_list:
                if model.get(proc_name, {}).get(var_name, False):
                    var_dict[proc_name].append(var_name)

            ds.xsimlab._set_output_vars(model, dim, **var_dict)

        return ds

    def run(self, model=None, safe_mode=True):
        """Run the model.

        Parameters
        ----------
        model : :class:`xsimlab.Model` object, optional
            Reference model. If None, tries to get model from context.
        safe_mode : bool, optional
            If True (default), it is safe to run multiple simulations
            simultaneously. Generally safe mode shouldn't be disabled, except
            in a few cases (e.g., debugging).

        Returns
        -------
        output : Dataset
            Another Dataset with both model inputs and outputs.

        """
        model = _maybe_get_model_from_context(model)

        if safe_mode:
            model = model.clone()

        ds_model_interface = DatasetModelInterface(model, self._ds)
        out_ds = ds_model_interface.run_model()
        return out_ds

    def run_multi(self):
        """Run multiple models.

        Not yet implemented.

        See Also
        --------
        :meth:`xarray.Dataset.xsimlab.run`

        """
        # TODO:
        raise NotImplementedError()


def create_setup(model=None, clocks=None, master_clock=None,
                 input_vars=None, output_vars=None):
    """Create a specific setup for model runs.

    This convenient function creates a new :class:`xarray.Dataset` object with
    model input values, time steps and model output variables (including
    snapshot times) as data variables, coordinates and attributes.

    Parameters
    ----------
    model : :class:`xsimlab.Model` object, optional
        Create a simulation setup for this model. If None, tries to get model
        from context.
    clocks : dict of dicts, optional
        Used to create one or several clock coordinates. The structure of the
        dict of dicts looks like ``{'dim': {key: value, ...}, ...}``.
        See the "Notes" section below for allowed keys and values.
        If only one clock is provided, it will be used as master clock.
    master_clock : str or dict, optional
        Name of the clock coordinate (dimension) to use as master clock (i.e.,
        for time steps).
        A dictionary with at least a 'dim' key can be provided instead, it
        allows setting time units and calendar (CF-conventions) with
        'units' and 'calendar' keys.
    input_vars : dict of dicts, optional
        Model inputs values given per process. The structure of the dict of
        dicts looks like ``{'process_name': {'var_name': value, ...}, ...}``.
        Values are anything that can be easily converted to
        :class:`xarray.Variable` objects, e.g., single values, array-like,
        (dims, data, attrs) tuples or xarray objects.
    output_vars : dict of dicts, optional
        Model variables to save as simulation output, given per clock
        coordinate. The structure of the dict of dicts looks like
        ``{'dim': {'process_name': 'var_name', ...}, ...}``.
        If ``'dim'`` corresponds to the dimension of a clock coordinate,
        snapshot values will be recorded at each time given by the coordinate
        labels. if None is given, only one value will be recorded at the
        end of the simulation.
        Note that instead of ``'var_name'``, a tuple of multiple variable names
        (declared in the same process) can be given.

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        A new Dataset object with model inputs as data variables or coordinates
        (depending on their given value) and clock coordinates.
        The names of the input variables also include the name of their process
        (i.e., 'process_name__var_name').

    Notes
    -----
    Allowed parameters for creating clock coordinates:

    - data : array-like or :class:`pandas.Index`, optional
        Absolute time values for the master clock (must be 1-dimensional).
        If provided, all other parameters below will be ignored.
        A :py:class:`pandas.DatetimeIndex` object can be used, e.g.,
        when working with datetime-like values.
    - start : float, optional
        Start simulation time (default: 0).
    - end : float, optional
        End simulation time.
    - step : float, optional
        Time step duration.
    - nsteps : int, optional
        Number of time steps.
    - auto_adjust : bool, optional
        Only for snapshot clock coordinates. If True (default), the resulting
        coordinate labels are automatically adjusted so that they are consistent
        with the labels of the master clock coordinate. Otherwise raise a
        KeyError if labels are not valid. (DataArray.sel is used internally).

    Inputs of ``model`` for which no value is given are still added as variables
    in the returned Dataset, using their default value (if any). It requires
    that their process are provided as keys of ``input_vars``, though.

    Output variable names are added in Dataset as specific attributes
    (global and/or clock coordinate attributes).

    """
    model = _maybe_get_model_from_context(model)

    ds = (Dataset()
          .xsimlab.update_clocks(model=model, clocks=clocks,
                                 master_clock=master_clock)
          .xsimlab.update_vars(model=model, input_vars=input_vars,
                               output_vars=output_vars))

    return ds
