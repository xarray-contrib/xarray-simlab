from collections.abc import MutableMapping
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr
import zarr

from . import Model
from .process import variables_dict
from .utils import normalize_encoding


VarKey = Tuple[str, str]
EncodingDict = Dict[str, Dict[str, Any]]

_DIMENSION_KEY = "_ARRAY_DIMENSIONS"


def _get_vars_to_store(
    dataset: xr.Dataset, model: Model
) -> Dict[VarKey, Optional[str]]:
    """Get all model variables to save in the store (i.e., the
    output variables and and in the index variables) and their
    clock dimension.
    """
    var_clocks = {k: v for k, v in dataset.xsimlab.output_vars.items()}
    var_clocks.update({vk: None for vk in model.index_vars})

    return var_clocks


def _variable_value_getter(model: Model, var_key: VarKey) -> Callable:
    p_name, v_name = var_key
    p_obj = model[p_name]

    def value_getter():
        return getattr(p_obj, v_name)

    return value_getter


def _get_var_info(
    dataset: xr.Dataset, model: Model, encoding: EncodingDict
) -> Dict[VarKey, Dict]:

    var_info = {}

    var_clocks = {k: v for k, v in dataset.xsimlab.output_vars.items()}
    var_clocks.update({vk: None for vk in model.index_vars})

    for var_key, clock in _get_vars_to_store(dataset, model).items():
        p_name, v_name = var_key
        v_name_str = f"{p_name}__{v_name}"
        p_obj = model[p_name]
        v_obj = variables_dict(type(p_obj))[v_name]

        v_encoding = v_obj.metadata["encoding"]
        v_encoding.update(normalize_encoding(encoding.get(v_name_str)))

        var_info[var_key] = {
            "clock": clock,
            "name": v_name_str,
            "metadata": v_obj.metadata,
            "value_getter": _variable_value_getter(model, var_key),
            "value": None,
            "shape": None,
            "encoding": v_encoding,
        }

    return var_info


def default_fill_value_from_dtype(dtype):
    if dtype.kind == "f":
        return np.nan
    elif dtype.kind in "c":
        return (
            default_fill_value_from_dtype(dtype.type().real.dtype),
            default_fill_value_from_dtype(dtype.type().imag.dtype),
        )
    else:
        return 0


class ZarrSimulationStore:
    def __init__(
        self,
        dataset: xr.Dataset,
        model: Model,
        zobject: Optional[Union[zarr.Group, MutableMapping, str]] = None,
        encoding: Optional[EncodingDict] = None,
        batch_dim: Optional[str] = None,
    ):
        self.dataset = dataset
        self.model = model

        self.in_memory = False
        self.consolidated = False

        if isinstance(zobject, zarr.Group):
            self.zgroup = zobject
        elif zobject is None:
            self.zgroup = zarr.group(store=zarr.MemoryStore())
            self.in_memory = True
        else:
            self.zgroup = zarr.group(store=zobject)

        self.output_vars = dataset.xsimlab.output_vars_by_clock
        self.output_save_steps = dataset.xsimlab.get_output_save_steps()

        if encoding is None:
            encoding = {}

        self.var_info = _get_var_info(dataset, model, encoding)
        self.var_cache = {k: {} for k in _get_vars_to_store(dataset, model)}

        self.batch_dim = batch_dim
        if batch_dim is not None:
            self.batch_size = dataset.dims[batch_dim]
        else:
            self.batch_size = -1

        self.mclock_dim = dataset.xsimlab.master_clock_dim
        self.clock_sizes = dataset.xsimlab.clock_sizes

        # initialize clock incrementers
        self.clock_incs = {clock: 0 for clock in dataset.xsimlab.clock_coords}
        self.clock_incs[None] = 0

    def write_input_xr_dataset(self):
        # output/index variables already in input dataset will be replaced
        drop_vars = [vi["name"] for vi in self.var_info.values()]
        ds = self.dataset.drop(drop_vars, errors="ignore")

        ds.xsimlab._reset_output_vars(self.model, {})
        ds.to_zarr(self.zgroup.store, group=self.zgroup.path, mode="a")

    def init_var_cache(self, batch: int, model: Model):
        for var_key in self.var_cache:
            self.var_cache[var_key][batch] = {
                "value_getter": _variable_value_getter(model, var_key),
                "value": None,
                "shape": None
            }

    def _cache_value_as_array(self, var_key: VarKey, batch: int):
        print(self.var_cache)
        value = self.var_cache[var_key][batch]["value_getter"]()

        if np.isscalar(value) or isinstance(value, (list, tuple)):
            value = np.asarray(value)

        self.var_cache[var_key][batch]["value"] = value

    def _create_zarr_dataset(self, var_key: VarKey, batch: int, name: Optional[str] = None):
        var_info = self.var_info[var_key]
        var_cache = self.var_cache[var_key]

        if name is None:
            name = var_info["name"]

        array = var_cache[batch]["value"]
        clock = var_info["clock"]

        shape = list(array.shape)

        if clock is not None:
            shape.insert(0, self.clock_sizes[clock])
        if self.batch_dim is not None:
            shape.insert(0, self.batch_size)

        # init shape for dynamically sized arrays
        var_cache[batch]["shape"] = np.asarray(shape)

        zkwargs = {
            "shape": tuple(shape),
            "chunks": True,
            "dtype": array.dtype,
            "compressor": "default",
            "fill_value": default_fill_value_from_dtype(array.dtype),
        }

        zkwargs.update(var_info["encoding"])

        # TODO: more performance assessment
        # if self.in_memory:
        #     chunks = False
        #     compressor = None

        zdataset = self.zgroup.create_dataset(name, **zkwargs)

        # add dimension labels and variable attributes as metadata
        dim_labels = None

        for dims in var_info["metadata"]["dims"]:
            if len(dims) == array.ndim:
                dim_labels = list(dims)

        if dim_labels is None:
            raise ValueError(
                f"Output array of {array.ndim} dimension(s) "
                f"for variable '{name}' doesn't match any of "
                f"its accepted dimension(s): {var_info['metadata']['dims']}"
            )

        if clock is not None:
            dim_labels.insert(0, clock)
        if self.batch_dim is not None:
            dim_labels.insert(0, self.batch_dim)

        zdataset.attrs[_DIMENSION_KEY] = tuple(dim_labels)
        if var_info["metadata"]["description"]:
            zdataset.attrs["description"] = var_info["metadata"]["description"]
        zdataset.attrs.update(var_info["metadata"]["attrs"])

        # reset consolidated since metadata has just been updated
        self.consolidated = False

    def _maybe_resize_zarr_dataset(self, var_key: VarKey, batch: int):
        # Maybe increases the length of one or more dimensions of
        # the zarr array (only increases, never shrinks dimensions).
        var_info = self.var_info[var_key]
        var_cache = self.var_cache[var_key]

        zkey = var_info["name"]
        zshape = var_cache[batch]["shape"]
        array = var_cache[batch]["value"]
        array_shape = list(array.shape)

        # maybe prepend clock dim (do not resize this dim)
        if var_info["clock"] is not None:
            array_shape.insert(0, 0)

        # maybe preprend batch dim (do not resize this dim)
        if self.batch_dim is not None:
            array_shape.insert(0, 0)

        new_shape = np.maximum(zshape, array_shape)

        if np.any(new_shape > zshape):
            var_cache[batch]["shape"] = new_shape
            self.zgroup[zkey].resize(new_shape)

    def write_output_vars(self, batch: int, step: int):
        save_istep = self.output_save_steps.isel(**{self.mclock_dim: step})

        for clock, var_keys in self.output_vars.items():
            if clock is None and step != -1:
                continue
            if not save_istep.data_vars.get(clock, True):
                continue

            clock_inc = self.clock_incs[clock]

            for vk in var_keys:
                self._cache_value_as_array(vk, batch)

            if clock_inc == 0:
                for vk in var_keys:
                    self._create_zarr_dataset(vk, batch)

            for vk in var_keys:
                zkey = self.var_info[vk]["name"]
                array = self.var_cache[vk][batch]["value"]

                self._maybe_resize_zarr_dataset(vk, batch)

                if clock is None:
                    if batch == -1:
                        idx = slice(None)
                    else:
                        idx = batch

                else:
                    idx_dims = [clock_inc] + [slice(0, n) for n in array.shape]

                    if batch != -1:
                        idx_dims.insert(0, batch)

                    idx = tuple(idx_dims)

                self.zgroup[zkey][idx] = array

            self.clock_incs[clock] += 1

    def write_index_vars(self):
        for var_key in self.model.index_vars:
            var_cache = self.var_cache[var_key]

            # index variable values must be invariant accross batch runs!
            # pick the 1st one found
            batch = next(iter(var_cache))

            _, vname = var_key
            self._cache_value_as_array(var_key, batch)
            self._create_zarr_dataset(var_key, batch, name=vname)

            self.zgroup[vname][:] = var_cache[batch]["value"]

    def consolidate(self):
        zarr.consolidate_metadata(self.zgroup.store)
        self.consolidated = True

    def open_as_xr_dataset(self) -> xr.Dataset:
        if self.in_memory:
            chunks = None
        else:
            chunks = "auto"

        ds = xr.open_zarr(
            self.zgroup.store,
            group=self.zgroup.path,
            chunks=chunks,
            consolidated=self.consolidated,
            # disable mask (not nice with zarr default fill_value=0)
            mask_and_scale=False,
        )

        if self.in_memory:
            # lazy loading may be confusing for the default, in-memory option
            ds.load()
        else:
            # load scalar data vars (there might be many of them: model params)
            for da in ds.data_vars.values():
                if not da.dims:
                    da.load()

        return ds
