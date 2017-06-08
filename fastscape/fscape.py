"""
Fastscape extension to xarray.

"""
from collections import OrderedDict

import numpy as np
from xarray import Variable, register_dataset_accessor

from .core.utils import Frozen, SortedKeysDict
from .models import Model, Process


@register_dataset_accessor('filter')
def filter_accessor(dataset):
    """A temporary hack until `filter` is available in xarray (GH916)."""

    def filter(func=None, like=None, regex=None):
        variables = {k: v for k, v in dataset._variables.items() if func(v)}
        coord_names = [c for c in dataset._coord_names if c in variables]

        return dataset._replace_vars_and_dims(variables,
                                              coord_names=coord_names)

    return filter


@register_dataset_accessor('fscape')
class FastscapeAccessor(object):
    """Fastscape extension to `xarray.Dataset`."""

    _context_attr = '__fscape_context__'
    _master_clock_key = '_fscape_master_clock'
    _snapshot_clock_key = '_fscape_snapshot_clock'
    _snapshot_vars_key = '_fscape_snapshot_vars'

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
        Dataset.fscape.set_master_clock

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
                           "use Dataset.fscape.set_master_clock instead."
                           % dim)

        if self.dim_master_clock is not None:
            self[dim].attrs.pop(self._master_clock_key)

        self._obj[dim].attrs[self._master_clock_key] = True
        self._dim_master_clock = dim

    def _set_clock_data(self, data, start, end, step, nsteps):
        if data is not None:
            return data

        args = {'step': step, 'nsteps': nsteps, 'end': end}
        provided_args = {k for k, v in args.items() if v is not None}

        if provided_args == {'nsteps', 'end', 'step'}:
            if end - start == nsteps * step:
                provided_args = {'nsteps', 'end'}
        if provided_args == {'nsteps', 'end'}:
            data = np.linspace(start, end, nsteps)
        elif provided_args == {'step', 'nsteps'}:
            data = np.arange(start, start + (nsteps + 1) * step, step)
        elif provided_args == {'step', 'end'}:
            data = np.arange(start, end + step, step)
        else:
            raise ValueError("Invalid combination of nsteps (%s), step (%s) "
                             "and end (%s)" % (nsteps, step, end))

        return data

    def set_master_clock(self, dim, data=None, start=0., end=None,
                         step=None, nsteps=None):
        """Add a dimension coordinate as master clock for model runs.

        Parameters
        ----------
        dim : str
            Name of the dimension / coordinate to add.
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

        Raises
        ------
        ValueError
            - If Dataset has already a dimension named `dim`.
            - In case of ambiguous combination of `nsteps`, `step` and `end`.

        """
        if dim in self._obj.dims:
            raise ValueError("dimension %r already exists" % dim)

        self._obj[dim] = self._set_clock_data(data, start, end, step, nsteps)
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

        See Also
        --------
        Dataset.fscape.set_master_clock
        Dataset.fscape.dim_master_clock

        """
        if self.dim_master_clock is None:
            raise ValueError("no master clock dimension/coordinate is defined "
                             "in Dataset. "
                             "Use `Dataset.fscape.set_master_clock` first")

        clock_data = self._set_clock_data(data, start, end, step, nsteps)

        da_master_clock = self._obj[self.dim_master_clock]

        if auto_adjust:
            kwargs = {'method': 'nearest'}
        else:
            kwargs = {}

        indexer = {self.dim_master_clock: clock_data}
        da_snapshot_clock = da_master_clock.sel(**indexer, **kwargs)

        self._obj[dim] = da_snapshot_clock.rename({self.dim_master_clock: dim})
        # _fscape_master_clock attribute has propagated with .sel
        self._obj[dim].attrs.pop(self._master_clock_key)
        self._obj[dim].attrs[self._snapshot_clock_key] = True

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
        """Model instance to use with this dataset."""
        return self._model

    @model.setter
    def model(self, value):
        raise AttributeError("can't set 'model' attribute, "
                             "use `Dataset.fscape.use_model` instead")

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
        xarray.as_variable

        """
        if self._model is None:
            raise ValueError("no model attached to this Dataset. Use "
                             "`Dataset.fscape.use_model` first.")

        if isinstance(process, Process):
            process = process.name
        if process not in self._model:
            raise ValueError("the model attached to this Dataset has no "
                             "process named %r" % process)

        process_inputs = self._model.input_vars[process]

        invalid_inputs = set(inputs) - set(process_inputs)
        if invalid_inputs:
            raise ValueError("%s are not valid input variables of %r"
                             % (', '.join([name for name in invalid_inputs]),
                                process))

        # convert to xarray variables and validate at the variable level
        variables = {}
        for name, var in self._model.input_vars[process].items():
            variables[name] = var.to_xarray_variable(inputs.get(name))

        # validate at the process level
        # assign values to a cloned process -> validate -> get updated values
        process_obj = self._model._processes[process].clone()

        for name, xr_var in variables.items():
            process_obj[name].value = xr_var.values

        process_obj.validate()

        for name, xr_var in variables.items():
            var = process_obj[name]
            if var.value is not xr_var.values:
                variables[name] = var.to_xarray_variable(var.value)

        # add variables to dataset if all validation tests passed
        for k, v in variables.items():
            self._obj[process + '__' + k] = v

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
                             "`Dataset.fscape.use_model` first.")

        xr_vars_list = []

        for proc_name, vars in process_vars.items():
            if proc_name not in self._model:
                raise ValueError("the model attached to this Dataset has no "
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
            clock_var = self._obj[clock_dim]
            if not clock_var.attrs.get(self._snapshot_clock_key, False):
                raise ValueError("%r coordinate is not a snapshot clock "
                                 "coordinate. "
                                 "Use Dataset.fscape.set_snapshot_clock first"
                                 % clock_dim)
            clock_var.attrs[self._snapshot_vars_key] = snapshot_vars

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

    def set_params(self, component, **kwargs):
        """Set one or several model parameters and their values.

        Parameters
        ----------
        component : str
           Name of the model component to which the parameters given by
           **kwargs are related.
        **kwargs : key=value
            key : str
                Name of the parameter. If not present, it will be added to this
                Dataset object with the name '<component>__<parameter_name>'.
            value : float or array-like
                Parameter value. It will be added either as a data variable or
                a coordinate depending on whether the value is a scalar or a >1
                length array.

        """
        for k, v in kwargs.items():
            p_name = '__'.join([component, k])
            p_value = np.asarray(v)
            if p_value.size == 1:
                p_dim = tuple()
            else:
                p_dim = p_name
            p_attrs = {self._context_attr: 'param'}
            self._obj[p_name] = Variable(p_dim, p_value, attrs=p_attrs)

    def _var_in_context(self, variable, context):
        return variable.attrs.get(self._context_attr) == context

    @property
    def param_dims(self):
        """Return model parameter dimensions."""
        coords = [v for k, v in self._obj.coords.items()
                  if self._var_in_context(v, 'param')]
        dims = OrderedDict()
        for var in coords:
            for dim, size in zip(var.dims, var.shape):
                dims[dim] = size
        return Frozen(SortedKeysDict(dims))

    def reduce_param_dims(self, new_dim, method='product'):
        """Reduce the parameter dimensions down to a single dimension.

        Parameters
        ----------
        new_dim : str
            Name of the dimension to create.
        method : {'product', 'align'}
            Used to set the parameter value combinations along the new
            dimension. 'product' generates all possible combinations (i.e.,
            cartesian product) using `Dataset.stack()`, while 'align' uses only
            the provided combinations using `Dataset.set_index()` and assuming
            that all parameter dimensions have the same size.

        Returns
        -------
        reduced : Dataset
            A new dataset with the same data but with only one dimension for
            all model parameters.

        See Also
        --------
        Dataset.stack
        Dataset.set_index

        """
        if method == 'product':
            new_dataset = self._obj.stack(**{new_dim: self.param_dims})
        elif method == 'align':
            new_dataset = self._obj.set_index(**{new_dim: self.param_dims})

        new_dataset[new_dim].attrs[self._context_attr] = 'param'

        return new_dataset

    def expand_param_dim(self, dim):
        """Expand a dimension of parameter values combinations into a
        n-dimensional parameter space.

        Only work for a dimension resulting from a reduction using
        cartesian product.

        Parameters
        ----------
        dim : str
            Name of the dimension to expand.

        Raises
        ------
        ValueError
            If the given dimension doesn't correspond to model parameter
            values.

        See Also
        --------
        Dataset.fscape.reduce_param_dims

        """
        if not self._var_in_context(self._obj[dim], 'param'):
            raise ValueError("dimension %r doesn't correspond to model "
                             "parameter values" % dim)

        new_dataset = self._obj.unstack(dim)

        param_dim_names = self._obj[dim].to_index().names
        for k in param_dim_names:
            new_dataset[k].attrs[self._context_attr] = 'param'

        return new_dataset

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

        return self._model.run(self._obj, safe_mode=safe_mode)

    def run_multi(self):
        """Run multiple models.

        Parameters
        ----------

        See Also
        --------
        Dataset.fscape.run

        """
        # TODO:
        raise NotImplementedError()
