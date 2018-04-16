"""
xarray extensions (accessors).

"""
from collections import defaultdict

import numpy as np
from xarray import as_variable, Dataset, register_dataset_accessor

from .drivers import XarraySimulationDriver
from .model import Model
from .stores import InMemoryOutputStore
from .utils import attr_fields_dict


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
            raise TypeError("No model found in context")

    if not isinstance(model, Model):
        raise TypeError("%s is not an instance of xsimlab.Model" % model)

    return model


def as_variable_key(key):
    """Returns ``key`` as a tuple of the form
    ``('process_name', 'var_name')``.

    If ``key`` is given as a string, then process name and variable
    name must be separated unambiguously by '__' (double underscore)
    and must not be empty.

    """
    key_tuple = None

    if isinstance(key, tuple) and len(key) == 2:
        key_tuple = key

    elif isinstance(key, str):
        key_split = key.split('__')
        if len(key_split) == 2:
            p_name, var_name = key_split
            if p_name and var_name:
                key_tuple = (p_name, var_name)

    if key_tuple is None:
        raise ValueError("{!r} is not a valid input variable key".format(key))

    return key_tuple


def _flatten_inputs(input_vars):
    """Returns ``input_vars`` as a flat dictionary where keys are tuples in
    the form ``(process_name, var_name)``. Raises an error if the
    given format appears to be invalid.

    """
    flatten_vars = {}

    for key, val in input_vars.items():
        if isinstance(key, str) and isinstance(val, dict):
            for var_name, var_value in val.items():
                flatten_vars[(key, var_name)] = var_value

        else:
            flatten_vars[as_variable_key(key)] = val

    return flatten_vars


def _flatten_outputs(output_vars):
    """Returns ``output_vars`` as a flat dictionary where keys are clock
    names (or None) and values are lists of tuples in the form
    ``(process_name, var_name)``.

    """
    flatten_vars = {}

    for clock, out_vars in output_vars.items():
        if isinstance(out_vars, dict):
            var_list = []
            for p_name, var_names in out_vars.items():
                if isinstance(var_names, str):
                    var_list.append((p_name, var_names))
                else:
                    var_list += [(p_name, vname) for vname in var_names]

        elif isinstance(out_vars, [tuple, str]):
            var_list = [as_variable_key(out_vars)]

        elif isinstance(out_vars, list):
            var_list = [as_variable_key(k) for k in out_vars]

        else:
            raise ValueError("Cannot interpret {!r} as valid output "
                             "variable key(s)".format(out_vars))

        flatten_vars[clock] = var_list

    return flatten_vars


@register_dataset_accessor('xsimlab')
class SimlabAccessor(object):
    """simlab extension to :class:`xarray.Dataset`."""

    _clock_key = '__xsimlab_output_clock__'
    _master_clock_key = '__xsimlab_master_clock__'
    _output_vars_key = '__xsimlab_output_vars__'

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

    def _set_input_vars(self, model, input_vars):
        invalid_inputs = set(input_vars) - set(model.input_vars)
        if invalid_inputs:
            raise KeyError(
                "{} is/are not valid key(s) for input variables in model {}"
                .format(', '.join([k for k in invalid_inputs]), model)
            )

        missing_inputs = set(model.input_vars) - set(input_vars)
        if missing_inputs:
            raise KeyError(
                "Missing value for input variable(s) {}"
                .format(', '.join([k for k in missing_inputs]))
            )

        for (p_name, var_name), data in input_vars.items():
            p_obj = model[p_name]
            var = attr_fields_dict(type(p_obj))[var_name]

            xr_var_name = p_name + '__' + var_name
            xr_var = as_variable(data)

            xr_var.attrs.update(var.metadata['attrs'])
            if var.metadata['description']:
                xr_var.attrs['description'] = var.metadata['description']

            self._ds[xr_var_name] = xr_var

    def _set_output_vars(self, model, clock, output_vars):
        invalid_outputs = set(output_vars) - set(model.all_vars)
        if invalid_outputs:
            raise KeyError(
                "{} is/are not valid key(s) for variables in model {}"
                .format(', '.join([k for k in invalid_outputs]), model)
            )

        output_vars = ','.join([p_name + '__' + var_name
                                for (p_name, var_name) in output_vars])

        if clock is None:
            self._ds.attrs[self._output_vars_key] = output_vars

        else:
            if clock not in self.clock_coords:
                raise ValueError("{!r} coordinate is not a valid clock "
                                 "coordinate.".format(clock))
            coord = self.clock_coords[clock]
            coord.attrs[self._output_vars_key] = output_vars

    def _get_output_vars(self, clock, ds_or_coord):
        out_attr = ds_or_coord.attrs.get(self._output_vars_key, '')

        if out_attr:
            return {clock: [as_variable_key(k) for k in out_attr.split(',')]}
        else:
            return {}

    @property
    def output_vars(self):
        """Returns a dictionary of snapshot clock dimension names as keys and
        output variable names - i.e. lists of (process name, variable name)
        tuples - as values.
        """
        output_vars = {}

        for clock, clock_coord in self.clock_coords.items():
            output_vars.update(self._get_output_vars(clock, clock_coord))

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
            for p_name, var_name in var_list:
                var_dict[p_name].append(var_name)

            if dim is None or dim in ds:
                ds.xsimlab._set_output_vars(model, dim, **var_dict)

        return ds

    def update_vars(self, model=None, input_vars=None, output_vars=None):
        """Update model input values and/or output variable names.

        More details about the values allowed for the parameters below can be
        found in the doc of :meth:`xsimlab.create_setup`.

        Parameters
        ----------
        model : :class:`xsimlab.Model` object, optional
            Reference model. If None, tries to get model from context.
        input_vars : dict, optional
            Model input values (may be grouped per process name, as dict of
            dicts).
        output_vars : dict, optional
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
            ds.xsimlab._set_input_vars(model, _flatten_inputs(input_vars))

        if output_vars is not None:
            for clock, out_vars in output_vars.items():
                ds.xsimlab._set_output_vars(model, clock,
                                            _flatten_outputs(out_vars))

        return ds

    def filter_vars(self, model=None):
        """Filter Dataset content according to Model.

        Keep only data variables and coordinates that correspond to
        inputs of the model (keep clock coordinates too).

        Also update xsimlab-specific attributes so that output
        variables given per clock only refer to processes and
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

        # drop variables
        drop_variables = []

        for xr_var_name in self._ds:
            if xr_var_name in self.clock_coords:
                continue

            try:
                p_name, var_name = xr_var_name.split('__')
            except ValueError:
                continue

            if (p_name, var_name) not in model.input_vars:
                drop_variables.append(xr_var_name)

        ds = self._ds.drop(drop_variables)

        # update output variable attributes
        for clock, out_vars in self.output_vars.items():
            new_out_vars = [key for key in out_vars if key in model.all_vars]
            ds.xsimlab._set_output_vars(model, clock, new_out_vars)

        return ds

    def _clean_output_dataset(self, ds):
        """Return a new dataset after having removed unnecessary attributes."""
        clean_ds = ds.copy()

        for clock in clean_ds.output_vars:
            if clock is None:
                attrs = clean_ds.attrs
            else:
                attrs = clean_ds[clock].attrs

            attrs.pop(self._output_vars_key)

        return clean_ds

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

        store = {}
        output_store = InMemoryOutputStore()

        driver = XarraySimulationDriver(model, self._ds, store, output_store)

        out_ds = driver.run_model().pipe(self._clean_output_dataset)

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

    This convenient function creates a new :class:`xarray.Dataset`
    object with everything needed to run a model (i.e., input values,
    time steps, output variables to save at given times) as data
    variables, coordinates and attributes.

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
    input_vars : dict, optional
        Dictionary with values given for model inputs. Entries of the
        dictionary may look like:

        - ``'foo': {'bar': value, ...}`` or
        - ``('foo', 'bar'): value`` or
        - ``'foo__bar': value``

        where ``foo`` is the name of a existing process in the model and
        ``bar`` is the name of an (input) variable declared in that process.

        Values are anything that can be easily converted to
        :class:`xarray.Variable` objects, e.g., single values, array-like,
        ``(dims, data, attrs)`` tuples or xarray objects.
    output_vars : dict, optional
        Dictionary with model variable names to save as simulation output,
        given per clock coordinate. Entries of the given dictionary may
        look like:

        - ``'dim': {'foo': 'bar'}`` or
        - ``'dim': {'foo': ('bar', 'baz')}`` or
        - ``'dim': ('foo', 'bar')`` or
        - ``'dim': [('foo', 'bar'), ('foo', 'baz')]`` or
        - ``'dim': 'foo__bar'`` or
        - ``'dim': ['foo__bar', 'foo__baz']``

        where ``foo`` is the name of a existing process in the model and
        ``bar``, ``baz`` are the names of variables declared in that process.

        If ``'dim'`` corresponds to the dimension of a clock coordinate,
        new output values will be saved at each time given by the coordinate
        labels. if None is given instead, only one value will be saved at the
        end of the simulation.

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        A new Dataset object with model inputs as data variables or coordinates
        (depending on their given value) and clock coordinates.
        The names of the input variables also include the name of their process
        (i.e., 'foo__bar').

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
