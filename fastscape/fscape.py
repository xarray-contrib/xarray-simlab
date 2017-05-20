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
            raise ValueError("Invalid combination of number of time steeps, "
                             "time step duration and total duration "
                             "(%r, %r, %r)" % (nsteps, step, duration))

        self._obj.coords[dim] = data

    add_time = set_clock

    @property
    def model(self):
        """Model instance to use with this dataset."""
        return self._model

    @model.setter
    def model(self, obj):
        if not isinstance(obj, Model):
            raise TypeError("%r is not a Model object" % obj)
        self._model = obj

    def add_inputs(self, process, **inputs):
        """Add to the Dataset new variables from model input variables defined
        in a given process, with given (or default) values.

        The names of the new variables are like
        '<process_name>__<variable_name>'.

        Parameters
        ----------
        process : str or Process object
            (Name of) the process.
        **inputs
            Inputs with variable names as keys and variable values as values.
            Variables with default values will be added to the Dataset for
            all other variables of the process that are inputs of the model
            but that are not provided here.

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

        variables = {}
        for name, var in self._model.input_vars[process].items():
            vname = process + '__' + name
            variables[vname] = var.to_xarray_variable(inputs.get(name))

        for k, v in variables.items():
            self._obj[k] = v

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

    def run(self, time_dim='time'):
        """Run the model.

        Parameters
        ----------
        time_dim : str, optional
            Name of the dimension in the Dataset that is used to set
            the time steps (default: 'time'). The dimension must have absolute
            time coordinates.

        Returns
        -------
        out_ds : Dataset
            Another Dataset with model inputs and outputs.

        """
        if self._model is None:
            raise ValueError("No model attached to this Dataset")

        return self._model.run(self._obj, time_dim=time_dim)

    def run_model(self, model):
        model.run(self._obj)
