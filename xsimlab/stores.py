from collections import defaultdict
from collections.abc import MutableMapping
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import xarray as xr
import zarr

from xsimlab import Model
from xsimlab.process import variables_dict


_DIMENSION_KEY = "_ARRAY_DIMENSIONS"


def _get_output_vars_by_clock(dataset: xr.Dataset) -> Dict[str, Tuple[str, str]]:
    out_vars = defaultdict(list)

    for k, clock in dataset.xsimlab.output_vars.items():
        out_vars[clock].append(k)

    return out_vars


def _variable_value_getter(process_obj: Any, var_name: str) -> Callable:
    def value_getter():
        return getattr(process_obj, var_name)

    return value_getter


def _get_var_info(dataset: xr.Dataset, model: Model) -> Dict[Tuple[str, str], Dict]:
    var_info = {}

    var_clocks = dataset.xsimlab.output_vars.copy()
    var_clocks.update({vk: None for vk in model.index_vars})

    for var_key, clock in var_clocks.items():
        p_name, v_name = var_key
        p_obj = model[p_name]
        v_obj = variables_dict(type(p_obj))[v_name]

        var_info[var_key] = {
            "clock": clock,
            "name": f"{p_name}__{v_name}",
            "obj": v_obj,
            "value_getter": _variable_value_getter(p_obj, v_name),
        }

    return var_info


def _get_output_steps_by_clock(dataset: xr.Dataset) -> Dict[str, np.ndarray]:
    """Returns a dictionary where keys are names of clock coordinates and
    values are numpy boolean arrays that specify whether or not to
    save outputs at every step of a simulation.

    """
    output_steps = {}

    mclock_dim = dataset.xsimlab.master_clock_dim
    master_coord = dataset[mclock_dim]

    for clock, coord in dataset.xsimlab.clock_coords.items():
        if clock == mclock_dim:
            output_steps[clock] = np.ones_like(coord.values, dtype=bool)
        else:
            output_steps[clock] = np.in1d(master_coord.values, coord.values)

    output_steps[None] = np.zeros_like(master_coord.values, dtype=bool)
    output_steps[None][-1] = True

    return output_steps


def _get_clock_sizes(dataset: xr.Dataset) -> Dict[str, int]:
    return {clock: coord.size for clock, coord in dataset.xsimlab.clock_coords.items()}


def _init_clock_incrementers(dataset: xr.Dataset) -> Dict[str, int]:
    incs = {clock: 0 for clock in dataset.xsimlab.clock_coords}
    incs[None] = 0

    return incs


def _default_fill_value_from_dtype(dtype):
    if dtype.kind == 'f':
        return np.nan
    elif dtype.kind in 'c':
        return (
            _default_fill_value_from_dtype(dtype.type().real.dtype),
            _default_fill_value_from_dtype(dtype.type().imag.dtype)
        )
    else:
        return 0


class ZarrOutputStore:
    def __init__(
        self,
        dataset: xr.Dataset,
        model: Model,
        zobject: Union[zarr.Group, MutableMapping, str, None],
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

        self.output_vars = _get_output_vars_by_clock(dataset)
        self.output_steps = _get_output_steps_by_clock(dataset)

        self.var_info = _get_var_info(dataset, model)

        self.clock_sizes = _get_clock_sizes(dataset)
        self.clock_incs = _init_clock_incrementers(dataset)

    def write_input_xr_dataset(self):
        # output/index variables already in input dataset will be replaced
        drop_vars = [vi["name"] for vi in self.var_info.values()]
        ds = self.dataset.drop(drop_vars, errors="ignore")

        ds.xsimlab._reset_output_vars(self.model, {})
        ds.to_zarr(self.zgroup.store, group=self.zgroup.path, mode="a")

    def _create_zarr_dataset(self, var_key: Tuple[str, str], name=None):
        var = self.var_info[var_key]["obj"]

        if name is None:
            name = self.var_info[var_key]["name"]

        value = self.var_info[var_key]["value_getter"]()
        if np.isscalar(value):
            array = np.asarray(value)
        else:
            array = value

        clock = self.var_info[var_key]["clock"]

        if clock is None:
            shape = array.shape
            chunks = True  # auto chunks
        else:
            shape = (self.clock_sizes[clock],) + tuple(array.shape)
            chunks = (1,) + tuple(array.shape)

        compressor = "default"

        if self.in_memory:
            # chunks = False    # assess performance impact first?
            compressor = None

        zdataset = self.zgroup.create_dataset(
            name,
            shape=shape,
            chunks=chunks,
            dtype=array.dtype,
            compressor=compressor,
            # TODO: smarter fill_value based on array dtype
            #       (0 may be non-missing value)
            fill_value=_default_fill_value_from_dtype(array.dtype),
        )

        # add dimension labels and variable attributes as metadata
        dim_labels = None

        for dims in var.metadata["dims"]:
            if len(dims) == array.ndim:
                dim_labels = list(dims)

        if dim_labels is None:
            raise ValueError(
                f"Output array of {array.ndim} dimension(s) "
                f"for variable '{name}' doesn't match any of"
                f"its accepted dimension(s): {var.metadata['dims']}"
            )

        if clock is not None:
            dim_labels.insert(0, clock)

        zdataset.attrs[_DIMENSION_KEY] = tuple(dim_labels)
        if var.metadata["description"]:
            zdataset.attrs["description"] = var.metadata["description"]
        zdataset.attrs.update(var.metadata["attrs"])

        # reset consolidated since metadata has just been updated
        self.consolidated = False

    def write_output_vars(self, istep: int):
        for clock, var_keys in self.output_vars.items():
            if not self.output_steps[clock][istep]:
                continue

            clock_inc = self.clock_incs[clock]

            if clock_inc == 0:
                for vk in var_keys:
                    self._create_zarr_dataset(vk)

            for vk in var_keys:
                zkey = self.var_info[vk]["name"]
                array = self.var_info[vk]["value_getter"]()

                if clock is None:
                    self.zgroup[zkey][:] = array
                else:
                    self.zgroup[zkey][clock_inc] = array

            self.clock_incs[clock] += 1

    def write_index_vars(self):
        for var_key in self.model.index_vars:
            _, vname = var_key
            self._create_zarr_dataset(var_key, name=vname)

            array = self.var_info[var_key]["value_getter"]()
            self.zgroup[vname][:] = array

    def consolidate(self):
        zarr.consolidate_metadata(self.zgroup.store)
        self.consolidated = True

    def open_as_xr_dataset(self) -> xr.Dataset:
        if self.in_memory:
            chunks = None
        else:
            chunks = "auto"

        return xr.open_zarr(
            self.zgroup.store,
            group=self.zgroup.path,
            chunks=chunks,
            consolidated=self.consolidated,
            # disable mask (not nice with zarr default fill_value=0)
            mask_and_scale=False,
        )
