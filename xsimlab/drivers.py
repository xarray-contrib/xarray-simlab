import copy
from enum import Enum
from typing import Any, Iterator, Mapping

import attr

from .hook import flatten_hooks, group_hooks, RuntimeHook
from .stores import ZarrOutputStore
from .utils import variables_dict


class ValidateOption(Enum):
    INPUTS = "inputs"
    ALL = "all"


class CheckDimsOption(Enum):
    STRICT = "strict"
    TRANSPOSE = "transpose"


class RuntimeContext(Mapping[str, Any]):
    """A mapping providing runtime information at the current time step."""

    _context_keys = (
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

    It implements methods for binding an active simulation
    data store (i.e., state) to a model and for feeding/updating this
    this active data from outside of the process classes (e.g., from
    inputs).

    """

    def __init__(self, model, state):
        self.model = model
        self.state = state

        self._bind_state_to_model()

    def _bind_state_to_model(self):
        """Bind the simulation active data store to each process in the
        model.
        """
        self.model.store = self.state

        for p_obj in self.model.values():
            p_obj.__xsimlab_store__ = self.state

    def _set_state(self, input_vars, check_static=True):
        for key in self.model.input_vars:
            value = input_vars.get(key)

            if value is None:
                continue

            p_name, var_name = key
            var = variables_dict(self.model[p_name].__class__)[var_name]

            if check_static and var.metadata.get("static", False):
                raise RuntimeError(
                    "Cannot set value in store for "
                    f"static variable {var_name!r} defined "
                    f"in process {p_name!r}"
                )

            if var.converter is not None:
                self.state[key] = var.converter(value)
            else:
                self.state[key] = copy.copy(value)

    def initialize_state(self, input_vars):
        """Pre-populate the simulation active data store (state)
        with input variable values.

        This should be called before the simulation starts.

        ``input_vars`` is a dictionary where keys are state keys, i.e.,
        ``(process_name, var_name)`` tuples, and values are the input
        values to set in the active store.

        Values are first copied from ``input_vars`` before being put in
        the store to prevent weird behavior (as model processes might
        update in-place the values in the store).

        Entries of ``input_vars`` that doesn't correspond to model
        inputs are silently ignored.

        """
        self._set_state(input_vars, check_static=False)

    def update_state(self, input_vars):
        """Update the simulation active data store with input variable
        values.

        Like ``initialize_state``, but here meant to be called during
        simulation runtime.

        """
        self._set_state(input_vars, check_static=True)

    def validate(self, p_names):
        """Run validators for all processes given in `p_names`."""

        for pn in p_names:
            attr.validate(self.model[pn])

    def run_model(self):
        """Main function of the driver used to run a simulation (must be
        implemented in sub-classes).
        """
        raise NotImplementedError()


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
        state,
        zobject,
        check_dims=CheckDimsOption.STRICT,
        validate=ValidateOption.INPUTS,
        hooks=None,
    ):
        self.dataset = dataset
        self.model = model

        super(XarraySimulationDriver, self).__init__(model, state)

        if self.dataset.xsimlab.master_clock_dim is None:
            raise ValueError("Missing master clock dimension / coordinate")

        self._check_missing_model_inputs()

        if check_dims is not None:
            check_dims = CheckDimsOption(check_dims)
        self._check_dims_option = check_dims

        self._original_dims = {}

        if validate is not None:
            validate = ValidateOption(validate)
        self._validate_option = validate

        if hooks is None:
            hooks = set()
        hooks = set(hooks) | RuntimeHook.active
        self._hooks = group_hooks(flatten_hooks(hooks))

        self.output_store = ZarrOutputStore(dataset, model, zobject)

    def _check_missing_model_inputs(self):
        """Check if all model inputs have their corresponding variables
        in the input Dataset.
        """
        missing_xr_vars = []

        for p_name, var_name in self.model.input_vars:
            xr_var_name = p_name + "__" + var_name

            if xr_var_name not in self.dataset:
                missing_xr_vars.append(xr_var_name)

        if missing_xr_vars:
            raise KeyError(f"Missing variables {missing_xr_vars} in Dataset")

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

    def _get_runtime_datasets(self):
        mclock_dim = self.dataset.xsimlab.master_clock_dim
        mclock_coord = self.dataset[mclock_dim]

        init_data_vars = {
            "_sim_start": mclock_coord[0],
            "_nsteps": self.dataset.xsimlab.nsteps,
            "_sim_end": mclock_coord[-1],
        }

        ds_init = self.dataset.assign(init_data_vars).drop_dims(mclock_dim)

        step_data_vars = {
            "_clock_start": mclock_coord,
            "_clock_end": mclock_coord.shift({mclock_dim: 1}),
            "_clock_diff": mclock_coord.diff(mclock_dim, label="lower"),
        }

        ds_all_steps = (
            self.dataset.drop(list(ds_init.data_vars.keys()), errors="ignore")
            .isel({mclock_dim: slice(0, -1)})
            .assign(step_data_vars)
        )

        ds_gby_steps = ds_all_steps.groupby(mclock_dim)

        return ds_init, ds_gby_steps

    def _maybe_validate_inputs(self, input_vars):
        p_names = set([v[0] for v in input_vars])

        if self._validate_option is not None:
            self.validate(p_names)

    def _get_output_dataset(self):
        self.output_store.consolidate()

        out_ds = self.output_store.open_as_xr_dataset()

        # replace index variables data with simulation data
        # (could be advanced Index objects that don't support serialization)
        for key in self.model.index_vars:
            _, var_name = key
            out_ds[var_name].data = self.state[key]

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
        """Run the model and return a new Dataset with all the simulation
        inputs and outputs.

        - Set model inputs from the input Dataset (update
          time-dependent model inputs -- if any -- before each time step).
        - Save outputs (snapshots) between the 'run_step' and the
          'finalize_step' stages or at the end of the simulation.

        """
        self.output_store.write_input_xr_dataset()
        ds_init, ds_gby_steps = self._get_runtime_datasets()

        validate_all = self._validate_option is ValidateOption.ALL

        runtime_context = RuntimeContext(
            sim_start=ds_init["_sim_start"].values,
            nsteps=ds_init["_nsteps"].values,
            sim_end=ds_init["_sim_end"].values,
        )

        in_vars = self._get_input_vars(ds_init)
        self.initialize_state(in_vars)
        self._maybe_validate_inputs(in_vars)

        self.model.execute(
            "initialize", runtime_context, hooks=self._hooks, validate=validate_all,
        )

        for step, (_, ds_step) in enumerate(ds_gby_steps):

            runtime_context.update(
                step=step,
                step_start=ds_step["_clock_start"].values,
                step_end=ds_step["_clock_end"].values,
                step_delta=ds_step["_clock_diff"].values,
            )

            in_vars = self._get_input_vars(ds_step)
            self.update_state(in_vars)
            self._maybe_validate_inputs(in_vars)

            self.model.execute(
                "run_step", runtime_context, hooks=self._hooks, validate=validate_all,
            )

            self.output_store.write_output_vars(step)

            self.model.execute(
                "finalize_step",
                runtime_context,
                hooks=self._hooks,
                validate=validate_all,
            )

        self.output_store.write_output_vars(-1)
        self.output_store.write_index_vars()

        self.model.execute("finalize", runtime_context, hooks=self._hooks)

        return self._get_output_dataset()
