from enum import Enum
from typing import Any, Iterator, Mapping

import dask
import pandas as pd

from .hook import flatten_hooks, group_hooks, RuntimeHook
from .stores import ZarrSimulationStore
from .utils import get_batch_size


class ValidateOption(Enum):
    INPUTS = "inputs"
    ALL = "all"


class CheckDimsOption(Enum):
    STRICT = "strict"
    TRANSPOSE = "transpose"


class RuntimeContext(Mapping[str, Any]):
    """A mapping providing runtime information at the current time step."""

    _context_keys = (
        "batch_size",
        "batch",
        "sim_start",
        "sim_end",
        "step",
        "nsteps",
        "step_start",
        "step_end",
        "step_delta",
    )

    def __init__(self, **kwargs):
        self._context = {k: 0 for k in self._context_keys}

        self.update(**kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.__setitem__(k, v)

    def __getitem__(self, key: str) -> Any:
        return self._context[key]

    def __setitem__(self, key: str, value: Any):
        if key not in self._context_keys:
            raise KeyError(
                f"Invalid key {key!r}, should be one of {self._context_keys!r}"
            )

        self._context[key] = value

    def __len__(self) -> int:
        return len(self._context)

    def __iter__(self) -> Iterator[str]:
        return iter(self._context)

    def __contains__(self, key: object) -> bool:
        return key in self._context

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._context!r})"


class BaseSimulationDriver:
    """Base class that provides a minimal interface for creating
    simulation drivers (should be inherited).

    """

    def __init__(self, model):
        self.model = model

    def run_model(self):
        """Main function of the driver used to run a simulation (must be
        implemented in sub-classes).
        """
        raise NotImplementedError()

    def get_results(self):
        """Function of the driver used to return results of a simulation
        (must be implemented in sub-classes).
        """
        raise NotImplementedError()


def _reset_multi_indexes(dataset):
    """Reset all multi-indexes and return them so that they can be rebuilt later.

    Currently multi-index coordinates can't be serialized by zarr.
    Also, Multi-index levels may correspond to model input variables.

    """
    multi_indexes = {}
    dims = []

    for cname in dataset.coords:
        idx = dataset.indexes.get(cname)

        if isinstance(idx, pd.MultiIndex):
            multi_indexes[cname] = idx.names
            dims.append(cname)

    return dataset.reset_index(dims), multi_indexes


def _check_missing_master_clock(dataset):
    if dataset.xsimlab.master_clock_dim is None:
        raise ValueError("Missing master clock dimension / coordinate")


def _check_missing_inputs(dataset, model):
    """Check if all model inputs have their corresponding variables
    in the input Dataset.
    """
    missing_xr_vars = []

    for p_name, var_name in model.input_vars:
        xr_var_name = p_name + "__" + var_name

        if xr_var_name not in dataset:
            missing_xr_vars.append(xr_var_name)

    if missing_xr_vars:
        raise KeyError(f"Missing variables {missing_xr_vars} in Dataset")


def _get_all_active_hooks(hooks):
    """Get all active runtime hooks (i.e, provided as argument, activated from
    context manager or glabally registered) and return them grouped by runtime
    event.

    """
    active_hooks = set(hooks) | RuntimeHook.active

    return group_hooks(flatten_hooks(active_hooks))


def _generate_runtime_datasets(dataset):
    """Create xarray Dataset objects that will be used during runtime of one
    simulation.

    Return a 2-length tuple where the 1st item is a Dataset used
    at the initialize stage and the 2st item is a DatasetGroupBy for
    iteration through run steps.

    Runtime data is added to those datasets.

    """
    mclock_dim = dataset.xsimlab.master_clock_dim

    # prevent non-index coordinates be included
    mclock_coord = dataset[mclock_dim].reset_coords(drop=True)

    init_data_vars = {
        "_sim_start": mclock_coord[0],
        "_nsteps": dataset.xsimlab.nsteps,
        "_sim_end": mclock_coord[-1],
    }

    ds_init = dataset.assign(init_data_vars).drop_dims(mclock_dim)

    step_data_vars = {
        "_clock_start": mclock_coord,
        "_clock_end": mclock_coord.shift({mclock_dim: 1}),
        "_clock_diff": mclock_coord.diff(mclock_dim, label="lower"),
    }

    ds_all_steps = (
        dataset.drop(list(ds_init.variables), errors="ignore")
        .isel({mclock_dim: slice(0, -1)})
        .assign(step_data_vars)
    )

    ds_gby_steps = ds_all_steps.groupby(mclock_dim)

    return ds_init, ds_gby_steps


def _maybe_transpose(dataset, model, check_dims, batch_dim):
    """Check and maybe re-order the dimensions of model input variables in the input
    dataset.

    Dimensions are re-ordered like this: (<batch dim>, <master clock dim>, *model var dims)

    Raise an error if dimensions found in the dataset are not valid or could not
    be transposed.

    """
    strict = check_dims is CheckDimsOption.STRICT
    transpose = check_dims is CheckDimsOption.TRANSPOSE

    ds_transposed = dataset.copy()

    for var_key in model.input_vars:
        xr_var_name = model.cache[var_key]["name"]
        xr_var = dataset.get(xr_var_name)

        if xr_var is None:
            continue

        # all valid dimensions in the right order
        dims = [list(d) for d in model.cache[var_key]["metadata"]["dims"]]
        dims += [[dataset.xsimlab.master_clock_dim] + d for d in dims]
        if batch_dim is not None:
            dims += [[batch_dim] + d for d in dims]

        dims = [tuple(d) for d in dims]

        # unordered -> ordered mapping for all valid dimensions
        valid_dims = {frozenset(d): d for d in dims}

        # actual dimensions (not ordered)
        actual_dims = frozenset(xr_var.dims)

        if transpose and actual_dims in valid_dims:
            reordered_dims = valid_dims[actual_dims]
            ds_transposed[xr_var_name] = xr_var.transpose(*reordered_dims)

        if (strict or transpose) and ds_transposed[xr_var_name].dims not in dims:
            raise ValueError(
                f"Invalid dimension(s) for variable '{xr_var_name}': "
                f"found {xr_var.dims!r}, "
                f"must be one of {','.join([str(d) for d in dims])}"
            )

    return ds_transposed


def _maybe_transpose_back(dataset_out, dataset_in, check_dims):
    """Maybe re-order the dimensions of variables in the output dataset so that it
    matches the order in the input dataset.

    Additional dimensions in the output dataset (e.g., batch and/or clock
    dimensions) are moved upfront.

    """
    if check_dims is not CheckDimsOption.TRANSPOSE:
        return dataset_out

    ds_transposed = dataset_out.copy()

    for xr_var_name, xr_var_out in dataset_out.variables.items():
        xr_var_in = dataset_in.variables.get(xr_var_name)

        if xr_var_in is None:
            continue

        dims_in = xr_var_in.dims
        dims_out = xr_var_out.dims

        dims_reordered = [d for d in dims_out if d not in dims_in]
        dims_reordered += dims_in

        if not xr_var_out.chunks:
            # TODO: transpose does not work on lazily loaded zarr datasets with no chunks
            xr_var_out.load()

        if dims_reordered != dims_out:
            ds_transposed[xr_var_name] = xr_var_out.transpose(*dims_reordered)

    return ds_transposed


def _get_input_vars(dataset, model):
    input_vars = {}

    for p_name, var_name in model.input_vars:
        xr_var_name = p_name + "__" + var_name
        xr_var = dataset.get(xr_var_name)

        if xr_var is None:
            continue

        data = xr_var.data

        if data.ndim == 0:
            # convert array to scalar
            data = data.item()

        input_vars[(p_name, var_name)] = data

    return input_vars


def _run(
    dataset,
    model,
    store,
    hooks,
    validate,
    batch=-1,
    batch_size=-1,
    parallel=False,
    scheduler=None,
):
    """Run one simulation.

    - initialize and update runtime context
    - Set model inputs from the input Dataset (update
      time-dependent model inputs -- if any -- before each time step).
    - Save outputs (snapshots) between the 'run_step' and the
      'finalize_step' stages or at the end of the simulation.

    """
    ds_init, ds_gby_steps = _generate_runtime_datasets(dataset)

    validate_all = validate is ValidateOption.ALL
    validate_inputs = validate_all or validate is ValidateOption.INPUTS

    execute_kwargs = {
        "hooks": hooks,
        "validate": validate_all,
        "parallel": parallel,
        "scheduler": scheduler,
    }

    rt_context = RuntimeContext(
        batch_size=batch_size,
        batch=batch,
        sim_start=ds_init["_sim_start"].values,
        nsteps=ds_init["_nsteps"].values,
        sim_end=ds_init["_sim_end"].values,
    )

    in_vars = _get_input_vars(ds_init, model)
    model.update_state(in_vars, validate=validate_inputs, ignore_static=True)
    model.execute("initialize", rt_context, **execute_kwargs)

    for step, (_, ds_step) in enumerate(ds_gby_steps):

        rt_context.update(
            step=step,
            step_start=ds_step["_clock_start"].values,
            step_end=ds_step["_clock_end"].values,
            step_delta=ds_step["_clock_diff"].values,
        )

        in_vars = _get_input_vars(ds_step, model)
        model.update_state(in_vars, validate=validate_inputs, ignore_static=False)
        model.execute("run_step", rt_context, **execute_kwargs)

        store.write_output_vars(batch, step, model=model)

        model.execute("finalize_step", rt_context, **execute_kwargs)

    store.write_output_vars(batch, -1, model=model)

    model.execute("finalize", rt_context, **execute_kwargs)

    store.write_index_vars(model=model)


class XarraySimulationDriver(BaseSimulationDriver):
    """Simulation driver using xarray.Dataset objects as I/O.

    - Perform some sanity checks on the content of the given input Dataset.
    - Set (and maybe validate) model inputs from data variables or coordinates
      in the input Dataset.
    - Save model outputs for given model variables, defined in specific
      attributes of the input Dataset, on time frequencies given by clocks
      defined as coordinates in the input Dataset.
    - Get simulation results as a new xarray.Dataset object.

    """

    def __init__(
        self,
        dataset,
        model,
        batch_dim=None,
        store=None,
        encoding=None,
        check_dims=CheckDimsOption.STRICT,
        validate=ValidateOption.INPUTS,
        hooks=None,
        parallel=False,
        scheduler=None,
    ):
        self.model = model

        super(XarraySimulationDriver, self).__init__(model)

        self.dataset, self.multi_indexes = _reset_multi_indexes(dataset)

        _check_missing_master_clock(self.dataset)
        _check_missing_inputs(self.dataset, model)

        self.batch_dim = batch_dim
        self.batch_size = get_batch_size(dataset, batch_dim)

        if check_dims is not None:
            check_dims = CheckDimsOption(check_dims)
        self._check_dims_option = check_dims

        if validate is not None:
            validate = ValidateOption(validate)
        self._validate_option = validate

        if hooks is None:
            hooks = []
        self.hooks = _get_all_active_hooks(hooks)

        self.parallel = parallel
        self.scheduler = scheduler

        if parallel:
            lock = dask.utils.get_scheduler_lock(scheduler=scheduler)
        else:
            lock = None

        self.store = ZarrSimulationStore(
            self.dataset,
            model,
            zobject=store,
            encoding=encoding,
            batch_dim=batch_dim,
            lock=lock,
        )

    def get_results(self):
        """Get simulation results as a xarray.Dataset loaded from
        the zarr store.
        """
        self.store.consolidate()

        # TODO: replace index variables data with simulation data
        # (could be advanced Index objects that don't support serialization)

        ds_out = (
            self.store.open_as_xr_dataset()
            # rebuild multi-indexes
            .set_index(self.multi_indexes)
            # transpose back
            .pipe(_maybe_transpose_back, self.dataset, self._check_dims_option)
        )

        return ds_out

    def run_model(self):
        """Run one or multiple simulation(s)."""
        self.store.write_input_xr_dataset()

        ds_in = _maybe_transpose(
            self.dataset, self.model, self._check_dims_option, self.batch_dim
        )
        args = (self.store, self.hooks, self._validate_option)

        if self.batch_dim is None:
            _run(
                ds_in,
                self.model,
                *args,
                parallel=self.parallel,
                scheduler=self.scheduler,
            )

        else:
            ds_gby_batch = ds_in.groupby(self.batch_dim)
            futures = []

            for batch, (_, ds_batch) in enumerate(ds_gby_batch):
                model = self.model.clone()

                if self.parallel:
                    futures.append(
                        dask.delayed(_run)(
                            ds_batch,
                            model,
                            *args,
                            batch=batch,
                            batch_size=self.batch_size,
                        )
                    )
                else:
                    _run(ds_batch, model, *args, batch=batch)

            if self.parallel:
                dask.compute(futures, scheduler=self.scheduler)
