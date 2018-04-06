from collections import defaultdict

import numpy as np
import xarray as xr

from .utils import attr_fields_dict
from .xr_accessor import SimlabAccessor


def _get_dims_from_variable(array, variable):
    """Given an array of values (snapshot) and a (xarray-simlab) Variable
    object, Return dimension labels for the array."""
    for dims in variable.allowed_dims:
        if len(dims) == array.ndim:
            return dims
    return tuple()


class InMemoryOutputStore(object):

    def __init__(self):
        self._store = defaultdict(list)

    def append(self):
        pass

    def _snapshot_to_xarray_variable(self, key, clock=None):
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

    def to_dataset(self):
        xr_variables = {}

        for clock, vars in self.output_vars.items():
            for key in vars:
                var_name = '__'.join(key)
                xr_variables[var_name] = self._snapshot_to_xarray_variable(
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
