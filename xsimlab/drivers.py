from enum import Enum
from typing import Any, Iterator, Mapping

import dask
import pandas as pd

from .hook import flatten_hooks, group_hooks, RuntimeHook
from .stores import ZarrSimulationStore
from .utils import get_batch_size, variables_dict


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


def _reset_multi_indexes(dataset):
    multi_indexes = {}
    dims = []

    for cname in dataset.coords:
        idx = dataset.indexes.get(cname)

        if isinstance(idx, pd.MultiIndex):
            multi_indexes[cname] = idx.names
            dims.append(cname)

    return dataset.reset_index(dims), multi_indexes


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
        # these are not yet supported with zarr
        self.dataset, self.multi_indexes = _reset_multi_indexes(dataset)

        _check_missing_master_clock(self.dataset)
        _check_missing_inputs(self.dataset, model)

        self.model = model

        super(XarraySimulationDriver, self).__init__(model)

        self.batch_dim = batch_dim
        self.batch_size = get_batch_size(dataset, batch_dim)

        if check_dims is not None:
            check_dims = CheckDimsOption(check_dims)
        self._check_dims_option = check_dims

        self._original_dims = {}

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

    def _maybe_transpose(self, xr_var, p_name, var_name):
        var = variables_dict(self.model[p_name].__class__)[var_name]

        dims = var.metadata["dims"]
        dims_set = {frozenset(d): d for d in dims}
        xr_dims_set = frozenset(xr_var.dims)

        strict = self._check_dims_option is CheckDimsOption.STRICT
        transpose = self._check_dims_option is CheckDimsOption.TRANSPOSE

        if transpose and xr_dims_set in dims_set:
            self._original_dims[xr_var.name] = xr_var.dims
            xr_var = xr_var.transpose(*dims_set[xr_dims_set])

        if (strict or transpose) and xr_var.dims not in dims:
            raise ValueError(
                f"Invalid dimension(s) for variable '{p_name}__{var_name}': "
                f"found {xr_var.dims!r}, "
                f"must be one of {','.join([str(d) for d in dims])}"
            )

        return xr_var

    def _get_input_vars(self, dataset):
        input_vars = {}

        for p_name, var_name in self.model.input_vars:
            xr_var_name = p_name + "__" + var_name
            xr_var = dataset.get(xr_var_name)

            if xr_var is None:
                continue

            xr_var = self._maybe_transpose(xr_var, p_name, var_name)

            data = xr_var.data

            if data.ndim == 0:
                # convert array to scalar
                data = data.item()

            input_vars[(p_name, var_name)] = data

        return input_vars

    def get_results(self):
        """Get simulation results as a xarray.Dataset loaded from
        the zarr store.
        """
        self.store.consolidate()

        out_ds = self.store.open_as_xr_dataset()

        # TODO: replace index variables data with simulation data
        # (could be advanced Index objects that don't support serialization)
        # for key in self.model.index_vars:
        #     _, var_name = key
        #     out_ds[var_name] = (out_ds[var_name].dims, self.model.state[key])

        # rebuild multi-indexes
        out_ds = out_ds.set_index(self.multi_indexes)

        # transpose back
        for xr_var_name, dims in self._original_dims.items():
            xr_var = out_ds[xr_var_name]

            reordered_dims = [d for d in xr_var.dims if d not in dims]
            reordered_dims += dims

            if not xr_var.chunks:
                # TODO: transpose does not work on lazily loaded zarr datasets with no chunks
                xr_var.load()

            out_ds[xr_var_name] = xr_var.transpose(*reordered_dims)

        return out_ds

    def run_model(self):
        """Run one or multiple simulation(s)."""
        self.store.write_input_xr_dataset()

        if self.batch_dim is None:
            model = self.model
            self._run_one_model(self.dataset, model, parallel=self.parallel)

        else:
            ds_gby_batch = self.dataset.groupby(self.batch_dim)
            futures = []

            for batch, (_, ds_batch) in enumerate(ds_gby_batch):
                model = self.model.clone()

                if self.parallel:
                    futures.append(
                        dask.delayed(self._run_one_model)(ds_batch, model, batch=batch)
                    )
                else:
                    self._run_one_model(ds_batch, model, batch=batch)

            if self.parallel:
                dask.compute(futures, scheduler=self.scheduler)

    def _run_one_model(self, dataset, model, batch=-1, parallel=False):
        """Run one simulation.

        - Set model inputs from the input Dataset (update
          time-dependent model inputs -- if any -- before each time step).
        - Save outputs (snapshots) between the 'run_step' and the
          'finalize_step' stages or at the end of the simulation.

        """
        ds_init, ds_gby_steps = _generate_runtime_datasets(dataset)

        validate_all = self._validate_option is ValidateOption.ALL
        validate_inputs = validate_all or self._validate_option is ValidateOption.INPUTS

        execute_kwargs = {
            "hooks": self.hooks,
            "validate": validate_all,
            "parallel": parallel,
            "scheduler": self.scheduler,
        }

        rt_context = RuntimeContext(
            batch_size=self.batch_size,
            batch=batch,
            sim_start=ds_init["_sim_start"].values,
            nsteps=ds_init["_nsteps"].values,
            sim_end=ds_init["_sim_end"].values,
        )

        model.update_state(
            self._get_input_vars(ds_init), validate=validate_inputs, ignore_static=True
        )
        model.execute("initialize", rt_context, **execute_kwargs)

        for step, (_, ds_step) in enumerate(ds_gby_steps):

            rt_context.update(
                step=step,
                step_start=ds_step["_clock_start"].values,
                step_end=ds_step["_clock_end"].values,
                step_delta=ds_step["_clock_diff"].values,
            )
            model.update_state(
                self._get_input_vars(ds_step),
                validate=validate_inputs,
                ignore_static=False,
            )
            model.execute("run_step", rt_context, **execute_kwargs)

            self.store.write_output_vars(batch, step, model=model)

            model.execute("finalize_step", rt_context, **execute_kwargs)

        self.store.write_output_vars(batch, -1, model=model)

        model.execute("finalize", rt_context, **execute_kwargs)

        self.store.write_index_vars(model=model)
