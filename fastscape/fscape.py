"""
Fastscape extension to xarray.

"""
import numpy as np
from xarray import Dataset, Variable, register_dataset_accessor

from .core.nputils import _expand_value


@register_dataset_accessor('filter')
def filter_accessor(dataset):
    """A temporary hack until `filter` is available in xarray (GH916)."""
    def filter(func=None, like=None, regex=None):
        # TODO
        pass
    return filter


@register_dataset_accessor('fscape')
class FastscapeAccessor(object):
    """Fastscape extension to `xarray.Dataset`."""

    _context_attr = '__fscape_context__'

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def create_regular_grid(self, coord_names=('x', 'y'), nnodes=None,
                            spacing=None, length=None, origin=0.):
        """Create a n-dimensional regular grid and add its coordinates
        to this `Dataset` object.

        Parameters
        ----------
        coord_names : str or tuple
            Name of the grid coordinate(s).
        nnodes : int or tuple or None, optional
            Number of grid nodes in each dimension. If a single
            value is given, it will be applied to every dimension.
        spacing : float or tuple or None, optional
            Distance between two grid points in each dimension.
        length : float or tuple or None, optional
            Total length of the grid in each dimension.
        origin : float or tuple, optional
            Coordinate(s) of the grid origin.

        Raises
        ------
        ValueError
            In case of ambiguous combination of `nnodes`,
            `spacing` and `length`.

        """
        if isinstance(coord_names, str):
            coord_names = [coord_names],
        ndim = len(coord_names)

        nnodes = _expand_value(nnodes, ndim)
        spacing = _expand_value(spacing, ndim)
        length = _expand_value(length, ndim)
        origin = _expand_value(origin, ndim)

        grid_coord = {}
        test_none = [a is None for a in (nnodes, spacing, length)]

        if test_none == [False, False, False]:
            test_length = ((nnodes - 1) * spacing == length).astype(bool)
            if np.all(test_length):
                test_none = [False, True, False]

        if test_none == [False, True, False]:
            for c, o, n, l in zip(coord_names, origin, nnodes, length):
                grid_coord[c] = np.linspace(o, o + l, n)
        elif test_none == [True, False, False]:
            for c, o, s, l in zip(coord_names, origin, spacing, length):
                grid_coord[c] = np.arange(o, o + l + s, s)
        elif test_none == [False, False, True]:
            for c, o, n, s in zip(coord_names, origin, nnodes, spacing):
                grid_coord[c] = np.arange(o, o + (n * s), s)
        else:
            raise ValueError("Invalid combination of number of grid nodes, "
                             "node spacing and grid length")

        self._obj.coords.update(grid_coord)

    def set_params(self, component, **kwargs):
        """Set one or several model parameters and their values.

        Parameters
        ----------
        component : str
           Name of the model component to which the parameters
           given by **kwargs are related.
        **kwargs : key=value
            key : str
                Name of the parameter. If not present, it will be added to this
                Dataset object with the name '<component>__<parameter_name>'.
            value : float or array-like
                Parameter value. It will be added either as a data variable or a
                coordinate depending on whether the value is a scalar or a >1
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
        coords = {k: v for k, v in self._obj.coords.items()
                  if self._var_in_context(v, 'param')}
        return Dataset(coords=coords).dims

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
            A new dataset with the same data but with only one dimension for all
            model parameters.

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
            If the given dimension doesn't correspond to model parameter values.

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
