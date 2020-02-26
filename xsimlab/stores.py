from collections import defaultdict
from collections.abc import MutableMapping
from typing import Any, Dict, Tuple, Union

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


def _get_output_vars_str(dataset: xr.Dataset) -> Dict[Tuple[str, str], str]:
    return {
        (pname, vname): pname + "__" + vname
        for (pname, vname) in dataset.xsimlab.output_vars
    }


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
    return {clock: 0 for clock in dataset.xsimlab.clock_coords}


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
        self.output_vars_str = _get_output_vars_str(dataset)
        self.output_steps = _get_output_steps_by_clock(dataset)

        self.clock_sizes = _get_clock_sizes(dataset)
        self.clock_incs = _init_clock_incrementers(dataset)

    def write_input_xr_dataset(self):
        # output variables already in input dataset will be replaced
        ovars = self.dataset.xsimlab.output_vars.keys()
        ds = self.dataset.drop(list(ovars), errors="ignore")

        ds.xsimlab._reset_output_vars(self.model, {})
        ds.to_zarr(self.zgroup.store, group=self.zgroup.path, mode="a")

    def _create_zarr_dataset(
        self, var_key: Tuple[str, str], name: str, array: Any, clock: str
    ):
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
            fill_value=0,
        )

        # add dimension labels and variable attributes as metadata
        p_name, var_name = var_key
        p_obj = self.model[p_name]
        var = variables_dict(type(p_obj))[var_name]

        dim_labels = None

        for dims in var.metadata["dims"]:
            if len(dims) == array.ndim:
                dim_labels = list(dims)

        if dim_labels is None:
            raise ValueError(
                f"Output array of {array.ndim} dimension(s) "
                f"for variable '{p_name}__{var_name}' doesn't match any of"
                f"its accepted dimension(s): {var.metadata['dims']}"
            )

        if clock is not None:
            dim_labels.insert(0, clock)

        zdataset.attrs[_DIMENSION_KEY] = tuple(dim_labels)
        zdataset.attrs["description"] = var.metadata["description"]
        zdataset.attrs.update(var.metadata["attrs"])

        # reset consolidated since metadata has just been updated
        self.consolidated = False

    def write_output_vars(self, state: MutableMapping, istep: int):
        for clock, var_keys in self.output_vars.items():
            if not self.output_steps[clock][istep]:
                continue

            clock_inc = self.clock_incs[clock]

            if clock_inc == 0:
                for vk in var_keys:
                    name = self.output_vars_str[vk]
                    self._create_zarr_dataset(vk, name, state[vk], clock)

            for vk in var_keys:
                vk_str = self.output_vars_str[vk]
                self.zgroup[vk_str][clock_inc] = state[vk]

            self.clock_incs[clock] += 1

    def write_index_vars(self, state: MutableMapping):
        for var_key in self.model.index_vars:
            _, vname = var_key
            self._create_zarr_dataset(var_key, vname, state[var_key], None)
            self.zgroup[vname][:] = state[var_key]

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
