"""
xarray extensions (accessors).

"""
import numpy as np
from xarray import as_variable, Dataset, register_dataset_accessor

from .drivers import XarraySimulationDriver
from .model import Model
from .stores import InMemoryOutputStore
from .utils import variables_dict


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

        elif isinstance(out_vars, (tuple, str)):
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
    """Simlab extension to :class:`xarray.Dataset`."""

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

    def _set_clock_coord(self, dim, data):
        xr_var = as_variable(data, name=dim)

        if xr_var.dims != (dim,):
            raise ValueError("Invalid dimension(s) given for clock coordinate "
                             "{dim!r}: found {invalid_dims!r}, "
                             "expected {dim!r}"
                             .format(dim=dim, invalid_dims=xr_var.dims))

        xr_var.attrs[self._clock_key] = np.uint8(True)

        self._ds.coords[dim] = xr_var

    def _uniformize_clock_coords(self, dim=None, units=None, calendar=None):
        """Ensure consistency across all clock coordinates.

        - maybe update master clock dimension
        - maybe set or update the same units and/or calendar for all
          coordinates as attributes
        - check that all clocks are synchronized with master clock, i.e.,
          there is no coordinate label that is not present in master clock

        """
        if dim is not None:
            if self.master_clock_dim is not None:
                old_mclock_coord = self._ds[self.master_clock_dim]
                old_mclock_coord.attrs.pop(self._master_clock_key)

            if dim not in self._ds.coords:
                raise KeyError("Master clock dimension name {} as no "
                               "defined coordinate in Dataset"
                               .format(dim))

            self._ds[dim].attrs[self._master_clock_key] = np.uint8(True)
            self._master_clock_dim = dim

        if units is not None:
            for coord in self.clock_coords.values():
                coord.attrs['units'] = units

        if calendar is not None:
            for coord in self.clock_coords.values():
                coord.attrs['calendar'] = calendar

        master_clock_idx = self._ds.indexes.get(self.master_clock_dim)

        for clock_dim in self.clock_coords:
            if clock_dim == self.master_clock_dim:
                continue

            clock_idx = self._ds.indexes[clock_dim]
            diff_idx = clock_idx.difference(master_clock_idx)

            if diff_idx.size:
                raise ValueError("Clock coordinate {} is not synchronized "
                                 "with master clock coordinate {}. "
                                 "The following coordinate labels are "
                                 "absent in master clock: {}"
                                 .format(clock_dim, self.master_clock_dim,
                                         diff_idx.values))

    def _set_input_vars(self, model, input_vars):
        invalid_inputs = set(input_vars) - set(model.input_vars)
        if invalid_inputs:
            raise KeyError(
                "{} is/are not valid key(s) for input variables in model {}"
                .format(', '.join([str(k) for k in invalid_inputs]), model)
            )

        for (p_name, var_name), data in input_vars.items():
            p_obj = model[p_name]
            var = variables_dict(type(p_obj))[var_name]

            xr_var_name = p_name + '__' + var_name
            xr_var = as_variable(data)

            if var.metadata['description']:
                xr_var.attrs['description'] = var.metadata['description']
            xr_var.attrs.update(var.metadata['attrs'])

            self._ds[xr_var_name] = xr_var

    def _set_output_vars(self, model, clock, output_vars):
        invalid_outputs = set(output_vars) - set(model.all_vars)
        if invalid_outputs:
            raise KeyError(
                "{} is/are not valid key(s) for variables in model {}"
                .format(', '.join([str(k) for k in invalid_outputs]), model)
            )

        output_vars_str = ','.join([p_name + '__' + var_name
                                    for (p_name, var_name) in output_vars])

        if clock is None:
            self._ds.attrs[self._output_vars_key] = output_vars_str

        else:
            if clock not in self.clock_coords:
                raise ValueError("{!r} coordinate is not a valid clock "
                                 "coordinate.".format(clock))
            coord = self.clock_coords[clock]
            coord.attrs[self._output_vars_key] = output_vars_str

    def _maybe_update_output_vars(self, clock, ds_or_coord, output_vars):
        out_attr = ds_or_coord.attrs.get(self._output_vars_key)

        if out_attr is not None:
            output_vars[clock] = [as_variable_key(k)
                                  for k in out_attr.split(',')]

    @property
    def output_vars(self):
        """Returns a dictionary of clock dimension names (or None) as keys and
        output variable names - i.e. lists of ``('p_name', 'var_name')``
        tuples - as values.

        """
        output_vars = {}

        for clock, clock_coord in self.clock_coords.items():
            self._maybe_update_output_vars(clock, clock_coord, output_vars)

        self._maybe_update_output_vars(None, self._ds, output_vars)

        return output_vars

    def update_clocks(self, model=None, clocks=None, master_clock=None):
        """Set or update clock coordinates.

        Also copy from the replaced coordinates any attribute that is
        specific to model output variables.

        Parameters
        ----------
        model : :class:`xsimlab.Model` object, optional
            Reference model. If None, tries to get model from context.
        clocks : dict, optional
            Used to create one or several clock coordinates. Dictionary
            values are anything that can be easily converted to
            :class:`xarray.IndexVariable` objects (e.g., a 1-d
            :class:`numpy.ndarray` or a :class:`pandas.Index`).
        master_clock : str or dict, optional
            Name of the clock coordinate (dimension) to use as master clock.
            If not set, the name is inferred from ``clocks`` (only if
            one coordinate is given and if Dataset has no master clock
            defined yet).
            A dictionary can also be given with one of several of these keys:

            - ``dim`` : name of the master clock dimension/coordinate
            - ``units`` : units of all clock coordinate labels
            - ``calendar`` : a unique calendar for all (time) clock coordinates

        Returns
        -------
        updated : Dataset
            Another Dataset with new or replaced coordinates.

        See Also
        --------
        :meth:`xsimlab.create_setup`

        """
        model = _maybe_get_model_from_context(model)

        ds = self._ds.copy()

        if isinstance(master_clock, str):
            master_clock_dict = {'dim': master_clock}

        elif master_clock is None:
            if (clocks is not None and len(clocks) == 1 and
                    self.master_clock_dim is None):
                master_clock_dict = {'dim': list(clocks.keys())[0]}
            else:
                master_clock_dict = {}

        else:
            master_clock_dict = master_clock

        master_clock_dim = master_clock_dict.get('dim', self.master_clock_dim)

        if clocks is not None:
            if master_clock_dim is None:
                raise ValueError("Cannot determine which clock coordinate is "
                                 "the master clock")
            elif (master_clock_dim not in clocks and
                  master_clock_dim not in self.clock_coords):
                raise KeyError("Master clock dimension name {!r} not found "
                               "in `clocks` nor in Dataset"
                               .format(master_clock_dim))

            for dim, data in clocks.items():
                ds.xsimlab._set_clock_coord(dim, data)

        ds.xsimlab._uniformize_clock_coords(**master_clock_dict)

        for clock, var_keys in self.output_vars.items():
            if clock is None or clock in ds:
                ds.xsimlab._set_output_vars(model, clock, var_keys)

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
            for clock, out_vars in _flatten_outputs(output_vars).items():
                ds.xsimlab._set_output_vars(model, clock, out_vars)

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

        for xr_var_name in self._ds.variables:
            if xr_var_name in self.clock_coords:
                continue

            try:
                p_name, var_name = xr_var_name.split('__')
            except ValueError:
                # not a xsimlab model input: make sure to remove it
                p_name, var_name = ('', xr_var_name)

            if (p_name, var_name) not in model.input_vars:
                drop_variables.append(xr_var_name)

        ds = self._ds.drop(drop_variables)

        # update output variable attributes
        for clock, out_vars in self.output_vars.items():
            new_out_vars = [key for key in out_vars if key in model.all_vars]
            ds.xsimlab._set_output_vars(model, clock, new_out_vars)

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

        store = {}
        output_store = InMemoryOutputStore()

        driver = XarraySimulationDriver(self._ds, model, store, output_store)

        return driver.run_model()

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
    clocks : dict, optional
        Used to create one or several clock coordinates. Dictionary
        values are anything that can be easily converted to
        :class:`xarray.IndexVariable` objects (e.g., a 1-d
        :class:`numpy.ndarray` or a :class:`pandas.Index`).
    master_clock : str or dict, optional
        Name of the clock coordinate (dimension) to use as master clock.
        If not set, the name is inferred from ``clocks`` (only if
        one coordinate is given and if Dataset has no master clock
        defined yet).
        A dictionary can also be given with one of several of these keys:

        - ``dim`` : name of the master clock dimension/coordinate
        - ``units`` : units of all clock coordinate labels
        - ``calendar`` : a unique calendar for all (time) clock coordinates
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
