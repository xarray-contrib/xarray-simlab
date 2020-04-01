from collections.abc import MutableMapping
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr
import zarr

from . import Model
from .utils import get_batch_size, normalize_encoding
from .variable import VarType


VarKey = Tuple[str, str]
EncodingDict = Dict[str, Dict[str, Any]]

_DIMENSION_KEY = "_ARRAY_DIMENSIONS"


def _get_var_info(
    dataset: xr.Dataset, model: Model, encoding: EncodingDict
) -> Dict[VarKey, Dict]:

    var_info = {}

    var_clocks = {k: v for k, v in dataset.xsimlab.output_vars.items()}
    var_clocks.update({vk: None for vk in model.index_vars})

    for var_key, clock in var_clocks.items():
        var_cache = model._var_cache[var_key]

        # encoding defined at model run
        run_encoding = normalize_encoding(
            encoding.get(var_cache["name"]), extra_keys=["chunks", "synchronizer"]
        )

        # encoding defined in model variable + update
        v_encoding = var_cache["metadata"]["encoding"]
        v_encoding.update(run_encoding)

        var_info[var_key] = {
            "clock": clock,
            "name": var_cache["name"],
            "metadata": var_cache["metadata"],
            "shape": None,  # not yet known
            "encoding": v_encoding,
        }

    return var_info


def default_fill_value_from_dtype(dtype=None):
    if dtype is None:
        return 0
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

        self.batch_dim = batch_dim
        self.batch_size = get_batch_size(dataset, batch_dim)

        self.mclock_dim = dataset.xsimlab.master_clock_dim
        self.clock_sizes = dataset.xsimlab.clock_sizes

        # initialize clock incrementers
        self.clock_incs = self._init_clock_incrementers()

    def _init_clock_incrementers(self):
        clock_incs = {}

        clock_keys = list(self.dataset.xsimlab.clock_coords) + [None]

        for clock in clock_keys:
            clock_incs[clock] = {}

            batch_keys = range(self.batch_size) if self.batch_dim else [-1]

            for batch in batch_keys:
                clock_incs[clock][batch] = 0

        return clock_incs

    def write_input_xr_dataset(self):
        # remove output/index variables already present (if any)
        drop_vars = [vi["name"] for vi in self.var_info.values()]
        ds = self.dataset.drop(drop_vars, errors="ignore")

        # remove xarray-simlab reserved attributes for output variables
        ds.xsimlab._reset_output_vars(self.model, {})

        ds.to_zarr(self.zgroup.store, group=self.zgroup.path, mode="a")

    def _create_zarr_dataset(
        self, model: Model, var_key: VarKey, name: Optional[str] = None
    ):
        var_info = self.var_info[var_key]

        if name is None:
            name = var_info["name"]

        value = model._var_cache[var_key]["value"]
        clock = var_info["clock"]

        dtype = getattr(value, "dtype", np.asarray(value).dtype)
        shape = list(np.shape(value))

        add_batch_dim = (
            self.batch_dim is not None
            and var_info["metadata"]["var_type"] != VarType.INDEX
        )

        if clock is not None:
            shape.insert(0, self.clock_sizes[clock])
        if add_batch_dim:
            shape.insert(0, self.batch_size)

        zkwargs = {
            "shape": tuple(shape),
            "chunks": True,
            "dtype": dtype,
            "compressor": "default",
            "fill_value": default_fill_value_from_dtype(dtype),
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
            if len(dims) == len(np.shape(value)):
                dim_labels = list(dims)

        if dim_labels is None:
            raise ValueError(
                f"Output array of {value.ndim} dimension(s) "
                f"for variable '{name}' doesn't match any of "
                f"its accepted dimension(s): {var_info['metadata']['dims']}"
            )

        if clock is not None:
            dim_labels.insert(0, clock)
        if add_batch_dim:
            dim_labels.insert(0, self.batch_dim)

        zdataset.attrs[_DIMENSION_KEY] = tuple(dim_labels)
        if var_info["metadata"]["description"]:
            zdataset.attrs["description"] = var_info["metadata"]["description"]
        zdataset.attrs.update(var_info["metadata"]["attrs"])

        # init shape for dynamically sized arrays
        var_info["shape"] = shape

        # reset consolidated since metadata has just been updated
        self.consolidated = False

    def _maybe_create_zarr_dataset(
        self, model: Model, var_key: VarKey, name: Optional[str] = None,
    ):
        # do not create if already exists (only for batches of simulation)
        try:
            self._create_zarr_dataset(model, var_key, name=name)
        except ValueError as err:
            if self.batch_dim:
                pass
            else:
                raise err

    def _maybe_resize_zarr_dataset(
        self, model: Model, var_key: VarKey,
    ):
        # Maybe increases the length of one or more dimensions of
        # the zarr array (only increases, never shrinks dimensions).
        var_info = self.var_info[var_key]

        zkey = var_info["name"]
        zshape = var_info["shape"]
        value = model._var_cache[var_key]["value"]
        value_shape = list(np.shape(value))

        # maybe prepend clock dim (do not resize this dim)
        if var_info["clock"] is not None:
            value_shape.insert(0, 0)

        # maybe preprend batch dim (do not resize this dim)
        if self.batch_dim is not None:
            value_shape.insert(0, 0)

        new_shape = np.maximum(zshape, value_shape)

        if np.any(new_shape > zshape):
            var_info["shape"] = new_shape
            self.zgroup[zkey].resize(new_shape)

    def write_output_vars(self, batch: int, step: int, model: Optional[Model] = None):
        if model is None:
            model = self.model

        save_istep = self.output_save_steps.isel(**{self.mclock_dim: step})

        for clock, var_keys in self.output_vars.items():
            if clock is None and step != -1:
                continue
            if not save_istep.data_vars.get(clock, True):
                continue

            clock_inc = self.clock_incs[clock][batch]

            for vk in var_keys:
                model.cache_state(vk)

            if clock_inc == 0:
                for vk in var_keys:
                    self._maybe_create_zarr_dataset(model, vk)

            for vk in var_keys:
                zkey = self.var_info[vk]["name"]
                value = model._var_cache[vk]["value"]

                self._maybe_resize_zarr_dataset(model, vk)

                if clock is None:
                    if batch != -1:
                        idx = batch
                    elif np.isscalar(value):
                        idx = tuple()
                    else:
                        idx = slice(None)

                else:
                    idx_dims = [clock_inc] + [slice(0, n) for n in np.shape(value)]

                    if batch != -1:
                        idx_dims.insert(0, batch)

                    idx = tuple(idx_dims)

                self.zgroup[zkey][idx] = value

            self.clock_incs[clock][batch] += 1

    def write_index_vars(self, model: Optional[Model] = None):
        if model is None:
            model = self.model

        for var_key in model.index_vars:
            _, vname = var_key
            model.cache_state(var_key)

            self._maybe_create_zarr_dataset(model, var_key, name=vname)
            self.zgroup[vname][:] = model._var_cache[var_key]["value"]

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
