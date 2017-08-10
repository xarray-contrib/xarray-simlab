"""
xarray extensions (accessors).

"""

import numpy as np
from xarray import Dataset, register_dataset_accessor

from .process import Process
from .model import Model
from .xr_interface import DatasetModelInterface


@register_dataset_accessor('filter')
def filter_accessor(dataset):
    """A temporary hack until `filter` is available in xarray (GH916)."""

    def filter(func=None, like=None, regex=None):
        variables = {k: v for k, v in dataset._variables.items() if func(v)}
        coord_names = [c for c in dataset._coord_names if c in variables]

        return dataset._replace_vars_and_dims(variables,
                                              coord_names=coord_names)

    return filter


@register_dataset_accessor('xsimlab')
class SimlabAccessor(object):
    """simlab extension to `xarray.Dataset`."""

    _master_clock_key = '_xsimlab_master_clock'
    _snapshot_clock_key = '_xsimlab_snapshot_clock'
    _snapshot_vars_key = '_xsimlab_snapshot_vars'

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._model = None
        self._dim_master_clock = None

    @property
    def dim_master_clock(self):
        """Dimension used as master clock for model runs. Returns None
        if no dimension is set as master clock.

        See Also
        --------
        :meth:`Dataset.xsimlab.set_master_clock`

        """
        if self._dim_master_clock is not None:
            return self._dim_master_clock
        else:
            for c in self._obj.coords.values():
                if c.attrs.get(self._master_clock_key, False):
                    dim = c.dims[0]
                    self._dim_master_clock = dim
                    return dim
            return None

    @dim_master_clock.setter
    def dim_master_clock(self, dim):
        if dim not in self._obj.coords:
            raise KeyError("Dataset has no %r dimension coordinate. "
                           "To create a new master clock dimension, "
                           "use Dataset.xsimlab.set_master_clock instead."
                           % dim)

        if self.dim_master_clock is not None:
            self._obj[self.dim_master_clock].attrs.pop(self._master_clock_key)

        self._obj[dim].attrs[self._master_clock_key] = np.uint8(True)
        self._dim_master_clock = dim

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

    def set_master_clock(self, dim, data=None, start=0., end=None,
                         step=None, nsteps=None, units=None, calendar=None):
        """Add a dimension coordinate as master clock for model runs.

        Parameters
        ----------
        dim : str
            Name of the dimension / coordinate to add.
        data : array-like or xarray object or pandas Index, optional
            Absolute time values for the master clock (must be 1-dimensional).
            If provided, all other parameters below will be ignored.
        start : float, optional
            Start simulation time (default: 0).
        end : float, optional
            End simulation time.
        step : float, optional
            Time step duration.
        nsteps : int, optional
            Number of time steps.
        units : str, optional
            Time units (CF-convention style) that will be added as attribute of
            the coordinate. Mostly useful when `data` is not provided or when
            it doesn't have datetime-like values.
        calendar : str, optional
            Calendar (CF-convention style). It will also be added in coordinate
            attributes.

        Raises
        ------
        ValueError
            If Dataset has already a dimension named `dim`
            or in case of ambiguous combination of `nsteps`, `step` and `end`
            or if any xarray object provided for `data` has other dimensions
            than `dim`.

        Notes
        -----
        Arguments other than `dim` and `data` are quite irrelevant when
        working with datetime-like values. In this case, it is better to provide
        a value for `data` using, e.g., a `pandas.DatetimeIndex` object.

        """
        if dim in self._obj.dims:
            raise ValueError("dimension %r already exists" % dim)

        self._obj[dim] = self._set_clock_data(dim, data, start, end,
                                              step, nsteps)
        if units is not None:
            self._obj[dim].attrs['units'] = units
        if calendar is not None:
            self._obj[dim].attrs['calendar'] = calendar

        self.dim_master_clock = dim

    def set_snapshot_clock(self, dim, data=None, start=0., end=None,
                           step=None, nsteps=None, auto_adjust=True):
        """Set or add a dimension coordinate used by model snapshots.

        The coordinate labels must be also labels of the master clock
        coordinate. By default labels are adjusted automatically.

        Parameters
        ----------
        dim : str
            Name of the dimension / coordinate.
        data : array-like, optional
            Absolute time values for the master clock (must be 1-dimensional).
            If provided, all other parameters below will be ignored.
        start : float, optional
            Start simulation time (default: 0).
        end : float, optional
            End simulation time.
        step : float, optional
            Time step duration.
        nsteps : int, optional
            Number of time steps.
        auto_adjust : bool, optional
            If True (default), the resulting coordinate labels are
            automatically adjusted so that they are consistent with the
            labels of the master clock coordinate.
            Otherwise raise a KeyError if labels are not valid.
            (DataArray.sel is used internally).

        Raises
        ------
        ValueError
            If no master clock dimension / coordinate is found in Dataset.

        Notes
        -----
        If set, 'units' and 'calendar' attributes will be copied from the master
        clock coordinate.

        See Also
        --------
        :meth:`Dataset.xsimlab.set_master_clock`
        :meth:`Dataset.xsimlab.dim_master_clock`

        """
        if self.dim_master_clock is None:
            raise ValueError("no master clock dimension/coordinate is defined "
                             "in Dataset. "
                             "Use `Dataset.xsimlab.set_master_clock` first")

        clock_data = self._set_clock_data(dim, data, start, end, step, nsteps)

        da_master_clock = self._obj[self.dim_master_clock]

        if auto_adjust:
            kwargs = {'method': 'nearest'}
        else:
            kwargs = {}

        indexer = {self.dim_master_clock: clock_data}
        kwargs.update(indexer)
        da_snapshot_clock = da_master_clock.sel(**kwargs)

        self._obj[dim] = da_snapshot_clock.rename({self.dim_master_clock: dim})
        # _xsimlab_master_clock attribute has propagated with .sel
        self._obj[dim].attrs.pop(self._master_clock_key)
        self._obj[dim].attrs[self._snapshot_clock_key] = np.uint8(True)

        for attr_name in ('units', 'calendar'):
            attr_value = da_master_clock.attrs.get(attr_name)
            if attr_value is not None:
                self._obj[dim].attrs[attr_name] = attr_value

    def use_model(self, obj):
        """Set a Model to use with this Dataset.

        Parameters
        ----------
        obj : object
            The `Model` instance to use.

        Raises
        ------
        TypeError
            If `obj` is not a Model object.

        """
        if not isinstance(obj, Model):
            raise TypeError("%r is not a Model object" % obj)
        self._model = obj

    @property
    def model(self):
        """Model instance to use with this dataset (read-only).

        See Also
        --------
        :meth:`xarray.Dataset.xsimlab.use_model`

        """
        return self._model

    @model.setter
    def model(self, value):
        raise AttributeError("can't set 'model' attribute, "
                             "use `Dataset.xsimlab.use_model` instead")

    def set_input_vars(self, process, **inputs):
        """Set or add Dataset variables that correspond to model
        inputs variables (for a given process), with given (or default)
        values.

        The names of the Dataset variables also include the name of the process
        (i.e., Dataset['<process_name>__<variable_name>']).

        Parameters
        ----------
        process : str or Process object
            (Name of) the process in which the model input variables are
            defined.
        **inputs : {var_name: value, ...}
            Inputs with variable names as keys and variable values as values.
            Values can be any object that can be easily converted to a
            xarray.Variable.
            Dataset variables with default values will be added for
            all other model inputs defined in the process that are not
            provided here.

        See Also
        --------
        :py:meth:`xarray.as_variable`

        """
        if self._model is None:
            raise ValueError("no model attached to this Dataset. Use "
                             "`Dataset.xsimlab.use_model` first.")

        if isinstance(process, Process):
            process = process.name
        if process not in self._model:
            raise KeyError("the model attached to this Dataset has no "
                           "process named %r" % process)

        process_inputs = self._model.input_vars[process]

        invalid_inputs = set(inputs) - set(process_inputs)
        if invalid_inputs:
            raise ValueError("%s are not valid input variables of %r"
                             % (', '.join([name for name in invalid_inputs]),
                                process))

        # convert to xarray variables and validate the given dimensions
        xr_variables = {}
        for name, var in self._model.input_vars[process].items():
            xr_var = var.to_xarray_variable(inputs.get(name))
            var.validate_dimensions(xr_var.dims,
                                    ignore_dims=(self.dim_master_clock,
                                                 'this_variable'))
            xr_variables[name] = xr_var

        # validate at the process level
        # first assign values to a cloned process object to avoid conflicts
        process_obj = self._model._processes[process].clone()
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
            self._obj[xr_var_name] = xr_var

    def set_snapshot_vars(self, clock_dim, **process_vars):
        """Set model variables to save as snapshots during a model run.

        Parameters
        ----------
        clock_dim : str or None
            Name of dimension corresponding to the snapshot clock to use for
            these variables. If None, only one snapshot is done at the end of
            the simulation and the variables won't have a clock dimension
            (also useful for getting variables in processes that are not time
            dependent).
        **process_vars : {process: variables, ...}
            Keyword arguments where keys are names of processes and variables
            (str or list of str) are one or more names of variables defined
            in these processes.

        Notes
        -----
        Specific attributes are added to the `clock_dim` coordinate or to the
        Dataset if `clock_dim` is None.

        """
        if self._model is None:
            raise ValueError("no model attached to this Dataset. Use "
                             "`Dataset.xsimlab.use_model` first.")

        xr_vars_list = []

        for proc_name, vars in process_vars.items():
            if proc_name not in self._model:
                raise KeyError("the model attached to this Dataset has no "
                               "process named %r" % proc_name)
            process = self._model[proc_name]
            if isinstance(vars, str):
                vars = [vars]
            for var_name in vars:
                if process.variables.get(var_name, None) is None:
                    raise KeyError("process %r has no variable %r"
                                   % (proc_name, var_name))
                xr_vars_list.append(proc_name + '__' + var_name)

        snapshot_vars = ','.join(xr_vars_list)

        if clock_dim is None:
            self._obj.attrs[self._snapshot_vars_key] = snapshot_vars
        else:
            coord = self._obj.coords[clock_dim]
            is_clock = (coord.attrs.get(self._master_clock_key, False) or
                        coord.attrs.get(self._snapshot_clock_key, False))
            if not is_clock:
                raise ValueError("%r coordinate is not a valid clock "
                                 "coordinate. " % clock_dim)
            coord.attrs[self._snapshot_vars_key] = snapshot_vars

    def _get_snapshot_vars(self, name, obj):
        vars_str = obj.attrs.get(self._snapshot_vars_key, '')
        if vars_str:
            return {name: [tuple(s.split('__'))
                           for s in vars_str.split(',')]}
        else:
            return {}

    @property
    def snapshot_vars(self):
        """Returns a dictionary of snapshot clock dimension names as keys and
        snapshot variable names - i.e. lists of (process name, variable name)
        tuples - as values.
        """
        snapshot_vars = {}
        for cname, coord in self._obj.coords.items():
            snapshot_vars.update(self._get_snapshot_vars(cname, coord))
        snapshot_vars.update(self._get_snapshot_vars(None, self._obj))
        return snapshot_vars

    def run(self, safe_mode=True):
        """Run the model.

        Parameters
        ----------
        safe_mode : bool, optional
            If True (default), it is safe to run multiple simulations
            simultaneously. Generally safe mode shouldn't be disabled, except
            in a few cases (e.g., debugging).

        Returns
        -------
        out_ds : Dataset
            Another Dataset with model inputs and outputs.

        """
        if self._model is None:
            raise ValueError("No model attached to this Dataset")

        if safe_mode:
            model = self._model.clone()
        else:
            model = self._model

        ds_model_interface = DatasetModelInterface(model, self._obj)
        out_ds = ds_model_interface.run_model()
        return out_ds

    def run_multi(self):
        """Run multiple models.

        Parameters
        ----------

        See Also
        --------
        :meth:`xarray.Dataset.xsimlab.run`

        """
        # TODO:
        raise NotImplementedError()


def create_setup(model=None, input_vars=None, clocks=None, master_clock=None,
                 snapshot_vars=None):
    """Create a specific setup for model runs.

    This is a convenient function that creates a new xarray.Dataset object
    with model input values, time steps and model output variables (including
    snapshot times) as data variables, coordinates and attributes.

    Parameters
    ----------
    model : Model object, optional
        Create a simulation setup for this model.
    input_vars : dict of dicts, optional
        Used to set values for model inputs. The structure of the dict of
        dicts looks like {'process_name': {'var_name': value, ...}, ...}.
        The given values are anything that can be easily converted to
        xarray.Variable objects, e.g., single values, array-like,
        (dims, data, attrs) tuples or xarray objects.
    clocks : dict of dicts, optional
        Used to set master or snapshot clocks. The structure of the dict of
        dicts looks like {'dim': {kwarg: value, ...}, ...}.
        kwarg is any keyword argument of `Dataset.xsimlab.set_master_clock` or
        `Dataset.xsimlab.set_snapshot_clock`. If only one clock is provided,
        it will be used as master clock.
    master_clock : str or dict, optional
        Name of the clock coordinate (dimension) that will be used as master
        clock. A dictionary with at least a 'dim' key can also be provided.
        Time units and calendar (CF-conventions) can be set manually using
        'units' and 'calendar' keys, respectively.
    snapshot_vars : dict of dicts, optional
        Model variables to save as simulation snapshots. The structure of the
        dict of dicts looks like
        {'dim': {'process_name': 'var_name', ...}, ...}.
        'dim' correspond to the dimension of a clock coordinate. None can be
        used instead to take snapshots only at the end of a simulation.
        To take snapshots for multiple variables that belong to the same
        process, a tuple can be given instead of a string.

    Returns
    -------
    input_dataset : xarray.Dataset object
        A new Dataset object that may be used for running simulations.

    Notes
    -----
    All inputs of `model` that are not provided in `input_vars` and that have a
    default value will also be added as data variables in the returned Dataset.

    """
    if model is None:
        # TODO: try get model object from context
        raise ValueError("no model provided")

    attrs_master_clock = {}

    if isinstance(master_clock, str):
        dim_master_clock = master_clock
    elif isinstance(master_clock, dict):
        dim_master_clock = master_clock.pop('dim')
        attrs_master_clock.update(master_clock)
    elif master_clock is None and clocks is not None and len(clocks) == 1:
        dim_master_clock = list(clocks.keys())[0]
    else:
        dim_master_clock = None

    ds = Dataset()
    ds.xsimlab.use_model(model)

    if clocks is not None:
        if dim_master_clock is None:
            raise ValueError("cannot determine which clock coordinate is "
                             "the master clock")
        elif dim_master_clock not in clocks:
            raise KeyError("master clock dimension name %r not found "
                           "in `clocks`" % dim_master_clock)

        ds.xsimlab.set_master_clock(dim_master_clock,
                                    **clocks.pop(dim_master_clock),
                                    **attrs_master_clock)
        for dim, kwargs in clocks.items():
            ds.xsimlab.set_snapshot_clock(dim, **kwargs)

    if input_vars is not None:
        for proc_name in model.input_vars:
            if proc_name not in input_vars:
                input_vars[proc_name] = {}
        for proc_name, vars in input_vars.items():
            ds.xsimlab.set_input_vars(proc_name, **vars)

    if snapshot_vars is not None:
        for dim, proc_vars in snapshot_vars.items():
            ds.xsimlab.set_snapshot_vars(dim, **proc_vars)

    return ds
