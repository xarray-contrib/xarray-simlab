"""
Fastscape extension to xarray.

"""
from collections import OrderedDict

import numpy as np
from xarray import Variable, register_dataset_accessor

from .core.utils import Frozen, SortedKeysDict, _get_args_not_none
from .core.nputils import _expand_value
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

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._model = None
        self._master_clock_dim = None

    def set_regular_grid(self, dims, nnodes=None, spacing=None, length=None,
                         origin=0.):
        """Create a n-dimensional regular grid and add its coordinates
        to this `Dataset` object.

        Parameters
        ----------
        dims : str or array-like
            Name of the grid dimension(s) / coordinate(s).
        nnodes : int or array-like or None, optional
            Number of grid nodes in each dimension. If a single
            value is given, it will be applied to every dimension.
        spacing : float or array-like or None, optional
            Distance between two grid points in each dimension.
        length : float or array-like or None, optional
            Total length of the grid in each dimension.
        origin : float or array-like, optional
            Coordinate(s) of the grid origin.

        Raises
        ------
        ValueError
            In case of ambiguous combination of `nnodes`, `spacing`
            and `length`.

        """
        if isinstance(dims, str):
            dims = [dims],
        ndim = len(dims)

        nnodes = _expand_value(nnodes, ndim)
        spacing = _expand_value(spacing, ndim)
        length = _expand_value(length, ndim)
        origin = _expand_value(origin, ndim)

        coords = OrderedDict()
        args = _get_args_not_none(('nnodes', 'spacing', 'length'),
                                  (nnodes, spacing, length))

        if args == ('nnodes', 'spacing', 'length'):
            eq_length = ((nnodes - 1) * spacing == length).astype(bool)
            if np.all(eq_length):
                args = ('nnodes', 'length')

        if args == ('nnodes', 'length'):
            for c, o, n, l in zip(dims, origin, nnodes, length):
                coords[c] = np.linspace(o, o + l, n)
        elif args == ('spacing', 'length'):
            for c, o, s, l in zip(dims, origin, spacing, length):
                coords[c] = np.arange(o, o + l + s, s)
        elif args == ('nnodes', 'spacing'):
            for c, o, n, s in zip(dims, origin, nnodes, spacing):
                coords[c] = np.arange(o, o + (n * s), s)
        else:
            raise ValueError("Invalid combination of number of grid nodes, "
                             "node spacing and grid length: (%r, %r, %r)"
                             % (nnodes, spacing, length))

        self._obj.coords.update(coords)

    def set_clock(self, dim='time', data=None, nsteps=None, step=None,
                  duration=None, start=0.):
        """Set a clock (model or output time steps) and add the
        corresponding dimension / coordinate to this Dataset.

        Parameters
        ----------
        dim : str, optional
            Name of the clock dimension (default: 'time').
        data : array-like or None, optional
            Clock values. If not None, the other parameters below will be
            ignored.
        nsteps : int or None, optional
            Total number of time steps.
        step : float or None, optional
            Time step duration.
        duration : float or None, optional
            Total duration.
        start : float, optional
            Start time (default: 0.).

        Raises
        ------
        ValueError
            In case of ambiguous combination of `nsteps`, `step` and
            `duration`.

        """
        if data is not None:
            self._obj.coords[dim] = data
            return

        args = _get_args_not_none(('nsteps', 'step', 'duration'),
                                  (nsteps, step, duration))

        if args == ('nsteps', 'step', 'duration'):
            if (nsteps - 1) * step == duration:
                args = ('nsteps', 'duration')

        if args == ('nsteps', 'duration'):
            data = np.linspace(start, start + duration, nsteps)
        elif args == ('nsteps', 'step'):
            data = np.arange(start, start + nsteps * step, step)
        elif args == ('step', 'duration'):
            data = np.arange(start, start + duration, step)
        else:
            raise ValueError("Invalid combination of number of time steps, "
                             "time step duration and total duration "
                             "(%r, %r, %r)" % (nsteps, step, duration))

        self._obj.coords[dim] = data

    add_time = set_clock

    @property
    def master_clock_dim(self):
        """Dimension used as master clock when running model simulations.

        See Also
        --------
        Dataset.fscape.set_master_clock

        """
        return self._master_clock_dim

    @master_clock_dim.setter
    def master_clock_dim(self, dim):
        if dim not in self._obj.dims and dim not in self._obj:
            raise KeyError("Dataset has no %r dimension coordinate. "
                           "To create a new master clock dimension, "
                           "use Dataset.fscape.set_master_clock instead."
                           % dim)

        self._master_clock_dim = dim

    def set_master_clock(self, dim, data=None, start=0., end=None, step=None,
                         nsteps=None):
        """Add or reset the dimension and/or coordinate used as master clock for
        running model simulations.

        Parameters
        ----------
        dim : str
            Name of the dimension to reset or create as model master clock.
        data : array-like, optional
            Absolute time values for the master clock. If provided, all
            other parameters below will be ignored.
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
            In case of ambiguous combination of `nsteps`, `step` and
            `end`.

        """
        if data is not None:
            self._obj[dim] = data
            self._master_clock_dim = dim
            return

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

        self._obj[dim] = data
        self._master_clock_dim = dim

    def set_snapshot_clocks(self, **dim_indexers):
        """Set or add one or more dimensions (with coordinates) used for model
        snapshots.

        The resulting coordinates are always aligned with the master clock
        coordinate (internally, this function calls `DataArray.sel` on the
        master clock dimension).

        Parameters
        ----------
        **dim_indexers : {dim: indexer, ...}
            Keyword arguments where keys are the name of the snapshots
            clocks dimensions and values are indexers of the master clock
            coordinate. Note that when indexers are arrays, nearest neighbor
            lookup is activated.

        Raises
        ------
        ValueError or KeyError
            If no master clock dimension / coordinate is defined or found in
            the Dataset.

        See Also
        --------
        Dataset.fscape.set_master_clock

        """
        mclock_dim = self._master_clock_dim

        if mclock_dim is None:
            raise ValueError("no master clock dimension/coordinate is defined "
                             "in Dataset. "
                             "Use `Dataset.fscape.set_master_clock` first")
        if mclock_dim not in self._obj:
            raise KeyError("no master clock %r coordinate found. "
                           "Use `Dataset.fscape.set_master_clock` first"
                           % mclock_dim)

        da_master_clock = self._obj[mclock_dim]

        for dim, indexer in dim_indexers.items():
            if isinstance(indexer, slice):
                kwargs = {}
            else:
                kwargs = {'method': 'nearest'}
            da_snapshot_clock = da_master_clock.sel(**{mclock_dim: indexer},
                                                    **kwargs)
            self._obj.coords[dim] = da_snapshot_clock.rename({mclock_dim: dim})

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
            raise ValueError("No model attached to this Dataset")

        if isinstance(process, Process):
            process = process.name
        if process not in self._model:
            raise ValueError("The model attached to this Dataset has no "
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

    def run(self):
        """Run the model.

        Parameters
        ----------

        Returns
        -------
        out_ds : Dataset
            Another Dataset with model inputs and outputs.

        """
        if self._model is None:
            raise ValueError("No model attached to this Dataset")

        return self._model.run(self._obj,
                               master_clock_dim=self._master_clock_dim)

    def run_multi(self):
        """Run multiple models."""
        # TODO:
        raise NotImplementedError()

    def run_model(self, model):
        model.run(self._obj)
