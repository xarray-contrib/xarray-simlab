"""
xarray extensions (accessors).

"""
from collections import defaultdict
import warnings

import attr
import numpy as np
from xarray import as_variable, Dataset, register_dataset_accessor

from .drivers import XarraySimulationDriver
from .model import get_model_variables, Model
from .utils import Frozen, variables_dict
from .variable import VarType


@register_dataset_accessor("filter")
def filter_accessor(dataset):
    """A temporary hack until ``filter`` is available in xarray (GH916)."""

    def filter(func=None, like=None, regex=None):
        variables = {k: v for k, v in dataset._variables.items() if func(v)}
        coord_names = [c for c in dataset._coord_names if c in variables]

        return dataset._replace_vars_and_dims(variables, coord_names=coord_names)

    return filter


def _maybe_get_model_from_context(model):
    """Return the given model or try to find it in the context if there was
    none supplied.
    """
    if model is None:
        if not Model.active:
            raise ValueError("No model found in context")

        model = Model.active[0]

    if not isinstance(model, Model):
        raise TypeError(f"{model} is not an instance of xsimlab.Model")

    return model


def as_variable_key(key):
    """Returns ``key`` as a tuple of the form
    ``('process_name', 'var_name')``.

    If ``key`` is given as a string, then process name and variable
    name must be separated unambiguously by '__' (double underscore)
    and must not be empty.

    """
    key_tuple = None

    if isinstance(key, tuple) and len(key) == 2:
        key_tuple = key

    elif isinstance(key, str):
        key_split = key.split("__")
        if len(key_split) == 2:
            p_name, var_name = key_split
            if p_name and var_name:
                key_tuple = (p_name, var_name)

    if key_tuple is None:
        raise ValueError(f"{key!r} is not a valid input variable key")

    return key_tuple


def _flatten_inputs(input_vars):
    """Returns ``input_vars`` as a flat dictionary where keys are tuples in
    the form ``(process_name, var_name)``. Raises an error if the
    given format appears to be invalid.

    """
    flatten_vars = {}

    for key, val in input_vars.items():
        if isinstance(key, str) and isinstance(val, dict):
            for var_name, var_value in val.items():
                flatten_vars[(key, var_name)] = var_value

        else:
            flatten_vars[as_variable_key(key)] = val

    return flatten_vars


def _flatten_outputs(output_vars):
    """Returns ``output_vars`` as a flat dictionary where keys are clock
    names (or None) and values are lists of tuples in the form
    ``(process_name, var_name)``.

    """
    flatten_vars = {}

    for clock, out_vars in output_vars.items():
        if isinstance(out_vars, dict):
            var_list = []
            for p_name, var_names in out_vars.items():
                if isinstance(var_names, str):
                    var_list.append((p_name, var_names))
                else:
                    var_list += [(p_name, vname) for vname in var_names]

        elif isinstance(out_vars, (tuple, str)):
            var_list = [as_variable_key(out_vars)]

        elif isinstance(out_vars, list):
            var_list = [as_variable_key(k) for k in out_vars]

        else:
            raise ValueError(
                f"Cannot interpret {out_vars!r} as valid output variable key(s)"
            )

        flatten_vars[clock] = var_list

    return flatten_vars


@register_dataset_accessor("xsimlab")
class SimlabAccessor:
    """Simlab extension to :class:`xarray.Dataset`."""

    _clock_key = "__xsimlab_output_clock__"
    _master_clock_key = "__xsimlab_master_clock__"
    _output_vars_key = "__xsimlab_output_vars__"

    def __init__(self, ds):
        self._ds = ds
        self._master_clock_dim = None
        self._clock_coords = None

    @property
    def clock_coords(self):
        """Mapping from clock dimensions to :class:`xarray.DataArray` objects
        corresponding to their coordinates.

        Cannot be modified directly.
        """
        if self._clock_coords is None:
            self._clock_coords = {
                k: coord
                for k, coord in self._ds.coords.items()
                if self._clock_key in coord.attrs
            }

        return Frozen(self._clock_coords)

    @property
    def clock_sizes(self):
        """Mapping from clock dimensions to lengths.

        Cannot be modified directly.
        """
        return Frozen({k: coord.size for k, coord in self.clock_coords.items()})

    @property
    def master_clock_dim(self):
        """Dimension used as master clock for model runs. Returns None
        if no dimension is set as master clock.

        See Also
        --------
        :meth:`Dataset.xsimlab.update_clocks`

        """
        # it is fine to cache the value here as inconsistency may appear
        # only when deleting the master clock coordinate from the dataset,
        # which would raise early anyway

        if self._master_clock_dim is not None:
            return self._master_clock_dim
        else:
            for c in self._ds.coords.values():
                if c.attrs.get(self._master_clock_key, False):
                    dim = c.dims[0]
                    self._master_clock_dim = dim
                    return dim
            return None

    @property
    def master_clock_coord(self):
        """Master clock coordinate (as a :class:`xarray.DataArray` object).

        Returns None if no master clock is defined in the dataset.
        """
        return self._ds.get(self.master_clock_dim)

    @property
    def nsteps(self):
        """Number of simulation steps, computed from the master
        clock coordinate.

        Returns 0 if no master clock is defined in the dataset.

        """
        if self.master_clock_dim is None:
            return 0
        else:
            return self._ds[self.master_clock_dim].size - 1

    def get_output_save_steps(self):
        """Returns save steps for each clock as boolean values.

        Returns
        -------
        save_steps : :class:`xarray.Dataset`
            A new Dataset with boolean data variables for each clock
            dimension other than the master clock, where values specify
            whether or not to save outputs at every step of a simulation.

        """
        ds = Dataset(coords={self.master_clock_dim: self.master_clock_coord})

        for clock, coord in self.clock_coords.items():
            if clock != self.master_clock_dim:
                save_steps = np.in1d(self.master_clock_coord.values, coord.values)
                ds[clock] = (self.master_clock_dim, save_steps)

        return ds

    def _set_clock_coord(self, dim, data):
        xr_var = as_variable(data, name=dim)

        if xr_var.dims != (dim,):
            raise ValueError(
                "Invalid dimension(s) given for clock coordinate "
                f"{dim!r}: found {xr_var.dims!r}, "
                f"expected {dim!r}"
            )

        xr_var.attrs[self._clock_key] = np.uint8(True)

        self._ds.coords[dim] = xr_var

    def _uniformize_clock_coords(self, dim=None, units=None, calendar=None):
        """Ensure consistency across all clock coordinates.

        - maybe update master clock dimension
        - maybe set or update the same units and/or calendar for all
          coordinates as attributes
        - check that all clocks are synchronized with master clock, i.e.,
          there is no coordinate label that is not present in master clock

        """
        if dim is not None:
            if self.master_clock_dim is not None:
                old_mclock_coord = self._ds[self.master_clock_dim]
                old_mclock_coord.attrs.pop(self._master_clock_key)

            if dim not in self._ds.coords:
                raise KeyError(
                    f"Master clock dimension name {dim} as no "
                    "defined coordinate in Dataset"
                )

            self._ds[dim].attrs[self._master_clock_key] = np.uint8(True)
            self._master_clock_dim = dim

        if units is not None:
            for coord in self.clock_coords.values():
                coord.attrs["units"] = units

        if calendar is not None:
            for coord in self.clock_coords.values():
                coord.attrs["calendar"] = calendar

        master_clock_idx = self._ds.indexes.get(self.master_clock_dim)

        for clock_dim in self.clock_coords:
            if clock_dim == self.master_clock_dim:
                continue

            clock_idx = self._ds.indexes[clock_dim]
            diff_idx = clock_idx.difference(master_clock_idx)

            if diff_idx.size:
                raise ValueError(
                    f"Clock coordinate {clock_dim} is not synchronized "
                    f"with master clock coordinate {self.master_clock_dim}. "
                    "The following coordinate labels are "
                    f"absent in master clock: {diff_idx.values}"
                )

    def _set_input_vars(self, model, input_vars):
        invalid_inputs = set(input_vars) - set(model.input_vars)

        if invalid_inputs:
            raise KeyError(
                ", ".join([str(k) for k in invalid_inputs])
                + f" is/are not valid key(s) for input variables in model {model}",
            )

        for var_key, data in input_vars.items():
            var_metadata = model.cache[var_key]["metadata"]
            xr_var_name = model.cache[var_key]["name"]

            try:
                xr_var = as_variable(data)
            except TypeError:
                # try retrieve dimension labels from model variable's
                # dimension labels that match the number of dimensions
                ndims = len(np.shape(data))
                dim_labels = {len(d): d for d in var_metadata["dims"]}
                dims = dim_labels.get(ndims)

                if dims is None:
                    raise TypeError(
                        "Could not get dimension labels from model "
                        f"for variable {xr_var_name!r} with value {data}"
                    )

                xr_var = as_variable((dims, data))

            if var_metadata["description"]:
                xr_var.attrs["description"] = var_metadata["description"]
            xr_var.attrs.update(var_metadata["attrs"])

            # maybe delete first to avoid merge conflicts
            # (we just want to replace here)
            if xr_var_name in self._ds:
                del self._ds[xr_var_name]

            self._ds[xr_var_name] = xr_var

    def _set_output_vars_attr(self, clock, value):
        # avoid update attrs in original dataset

        if clock is None:
            attrs = self._ds.attrs.copy()
        else:
            attrs = self._ds[clock].attrs.copy()

        if value is None:
            attrs.pop(self._output_vars_key, None)
        else:
            attrs[self._output_vars_key] = value

        if clock is None:
            self._ds.attrs = attrs
        else:
            new_coord = self._ds.coords[clock].copy()
            new_coord.attrs = attrs
            self._ds[clock] = new_coord

    def _set_output_vars(self, model, output_vars, clear=False):
        # TODO: remove this ugly code (depreciated output_vars format)
        o_vars = {}

        for k, v in output_vars.items():
            if k is None or k in self.clock_coords:
                warnings.warn(
                    "Setting clock dimensions or `None` as keys for `output_vars`"
                    " is depreciated; use variable names instead (and clock "
                    "dimensions or `None` as values, see docs).",
                    FutureWarning,
                    stacklevel=2,
                )
                o_vars.update({vn: k for vn in _flatten_outputs({k: v})[k]})

            else:
                o_vars[k] = v

        output_vars = _flatten_inputs(o_vars)

        # end of depreciated code block

        if not clear:
            _output_vars = {k: v for k, v in self.output_vars.items()}
            _output_vars.update(output_vars)
            output_vars = _output_vars

        invalid_outputs = set(output_vars) - set(model.all_vars)
        if invalid_outputs:
            raise KeyError(
                ", ".join([f"{pn}__{vn}" for pn, vn in invalid_outputs])
                + f" is/are not valid key(s) for variables in model {model}",
            )

        object_outputs = set(output_vars) & set(
            get_model_variables(model, var_type=VarType.OBJECT)
        )
        if object_outputs:
            raise ValueError(
                f"Object variables can't be set as model outputs: "
                + ", ".join([f"{pn}__{vn}" for pn, vn in object_outputs])
            )

        clock_vars = defaultdict(list)

        for (p_name, var_name), clock in output_vars.items():
            if clock is not None and clock not in self.clock_coords:
                raise ValueError(
                    f"{clock!r} coordinate is not a valid clock coordinate."
                )

            xr_var_name = p_name + "__" + var_name
            clock_vars[clock].append(xr_var_name)

        for clock, var_list in clock_vars.items():
            var_str = ",".join(var_list)
            self._set_output_vars_attr(clock, var_str)

        # reset clock_coords cache as attributes of those coords
        # may have been updated
        self._clock_coords = None

    def _reset_output_vars(self, model, output_vars):
        self._set_output_vars_attr(None, None)

        for clock in self.clock_coords:
            self._set_output_vars_attr(clock, None)

        self._set_output_vars(model, output_vars, clear=True)

    @property
    def output_vars(self):
        """Returns a dictionary of output variable names - in the form of
        ``('p_name', 'var_name')`` tuples - as keys and the clock dimension
        names (or None) on which to save snapshots as values.

        Cannot be modified directly.
        """

        def xr_attr_to_dict(attrs, clock):
            var_str = attrs.get(self._output_vars_key)

            if var_str is None:
                return {}
            else:
                return {as_variable_key(k): clock for k in var_str.split(",")}

        o_vars = {}

        for clock, coord in self.clock_coords.items():
            o_vars.update(xr_attr_to_dict(coord.attrs, clock))

        o_vars.update(xr_attr_to_dict(self._ds.attrs, None))

        return Frozen(o_vars)

    @property
    def output_vars_by_clock(self):
        """Returns a dictionary of output variables grouped by clock (keys).

        Cannot be modified directly.
        """
        o_vars = defaultdict(list)

        for k, clock in self.output_vars.items():
            o_vars[clock].append(k)

        return Frozen(dict(o_vars))

    def update_clocks(self, model=None, clocks=None, master_clock=None):
        """Set or update clock coordinates.

        Also copy from the replaced coordinates any attribute that is
        specific to model output variables.

        Parameters
        ----------
        model : :class:`xsimlab.Model` object, optional
            Reference model. If None, tries to get model from context.
        clocks : dict, optional
            Used to create one or several clock coordinates. Dictionary
            values are anything that can be easily converted to
            :class:`xarray.IndexVariable` objects (e.g., a 1-d
            :class:`numpy.ndarray` or a :class:`pandas.Index`).
        master_clock : str or dict, optional
            Name of the clock coordinate (dimension) to use as master clock.
            If not set, the name is inferred from ``clocks`` (only if
            one coordinate is given and if Dataset has no master clock
            defined yet).
            A dictionary can also be given with one of several of these keys:

            - ``dim`` : name of the master clock dimension/coordinate
            - ``units`` : units of all clock coordinate labels
            - ``calendar`` : a unique calendar for all (time) clock coordinates

        Returns
        -------
        updated : Dataset
            Another Dataset with new or replaced coordinates.

        See Also
        --------
        :meth:`xsimlab.create_setup`

        """
        model = _maybe_get_model_from_context(model)

        ds = self._ds.copy()

        if isinstance(master_clock, str):
            master_clock_dict = {"dim": master_clock}

        elif master_clock is None:
            if (
                clocks is not None
                and len(clocks) == 1
                and self.master_clock_dim is None
            ):
                master_clock_dict = {"dim": list(clocks.keys())[0]}
            else:
                master_clock_dict = {}

        else:
            master_clock_dict = master_clock

        master_clock_dim = master_clock_dict.get("dim", self.master_clock_dim)

        if clocks is not None:
            if master_clock_dim is None:
                raise ValueError(
                    "Cannot determine which clock coordinate is the master clock"
                )
            elif (
                master_clock_dim not in clocks
                and master_clock_dim not in self.clock_coords
            ):
                raise KeyError(
                    f"Master clock dimension name {master_clock_dim!r} not found "
                    "in `clocks` nor in Dataset"
                )

            for dim, data in clocks.items():
                ds.xsimlab._set_clock_coord(dim, data)

        ds.xsimlab._uniformize_clock_coords(**master_clock_dict)

        # operations on clock coords may have discarded coord attributes
        o_vars = {k: v for k, v in self.output_vars.items() if v is None or v in ds}
        ds.xsimlab._set_output_vars(model, o_vars)

        return ds

    def update_vars(self, model=None, input_vars=None, output_vars=None):
        """Update model input values and/or output variable names.

        More details about the values allowed for the parameters below can be
        found in the doc of :meth:`xsimlab.create_setup`.

        Parameters
        ----------
        model : :class:`xsimlab.Model` object, optional
            Reference model. If None, tries to get model from context.
        input_vars : dict, optional
            Model input values (may be grouped per process name, as dict of
            dicts).
        output_vars : dict, optional
            Model variables to save as simulation output (time-dependent or
            time-independent).

        Returns
        -------
        updated : Dataset
            Another Dataset with new or replaced variables (inputs) and/or
            attributes (snapshots).

        See Also
        --------
        :meth:`xsimlab.create_setup`

        """
        model = _maybe_get_model_from_context(model)

        ds = self._ds.copy()

        if input_vars is not None:
            ds.xsimlab._set_input_vars(model, _flatten_inputs(input_vars))

        if output_vars is not None:
            ds.xsimlab._set_output_vars(model, output_vars)

        return ds

    def reset_vars(self, model=None):
        """Set or reset Dataset variables with model input default
        values (if any).

        Parameters
        ----------
        model : :class:`xsimlab.Model` object, optional
            Reference model. If None, tries to get model from context.

        Returns
        -------
        updated : Dataset
            Another Dataset with new and/or replaced variables.

        See Also
        --------
        :meth:`Dataset.xsimlab.update_vars`

        """
        model = _maybe_get_model_from_context(model)

        ds = self._ds.copy()

        input_vars_default = {}

        for p_name, var_name in model.input_vars:
            p_obj = model[p_name]
            var = variables_dict(type(p_obj))[var_name]

            if var.default is not attr.NOTHING:
                input_vars_default[(p_name, var_name)] = var.default

        ds.xsimlab._set_input_vars(model, input_vars_default)

        return ds

    def filter_vars(self, model=None):
        """Filter Dataset content according to Model.

        Keep only data variables and coordinates that correspond to
        inputs of the model (keep clock coordinates too).

        Also update xsimlab-specific attributes so that output
        variables given per clock only refer to processes and
        variables defined in the model.

        Parameters
        ----------
        model : :class:`xsimlab.Model` object, optional
            Reference model. If None, tries to get model from context.

        Returns
        -------
        filtered : Dataset
            Another Dataset with (maybe) dropped variables and updated
            attributes.

        See Also
        --------
        :meth:`Dataset.xsimlab.update_vars`

        """
        model = _maybe_get_model_from_context(model)

        # drop variables
        drop_variables = []

        for xr_var_name in self._ds.variables:
            if xr_var_name in self.clock_coords:
                continue

            try:
                p_name, var_name = xr_var_name.split("__")
            except ValueError:
                # not a xsimlab model input: make sure to remove it
                p_name, var_name = ("", xr_var_name)

            if (p_name, var_name) not in model.input_vars:
                drop_variables.append(xr_var_name)

        ds = self._ds.drop(drop_variables)

        # update output variable attributes
        o_vars = {k: v for k, v in self.output_vars.items() if k in model.all_vars}
        ds.xsimlab._reset_output_vars(model, o_vars)

        return ds

    def run(
        self,
        model=None,
        batch_dim=None,
        check_dims="strict",
        validate="inputs",
        store=None,
        encoding=None,
        hooks=None,
        parallel=False,
        scheduler=None,
        safe_mode=True,
    ):
        """Run the model.

        Parameters
        ----------
        model : :class:`xsimlab.Model` object, optional
            Reference model. If None, tries to get model from context.
        batch_dim : str, optional
            Dimension label in the input dataset used to run a batch of
            simulations.
        check_dims : {'strict', 'transpose'}, optional
            Check the dimension(s) of each input variable given in Dataset.
            It may be one of the following options:

            - 'strict': the dimension labels must exactly correspond to
              (one of) the label sequences defined by their respective model
              variables (default)
            - 'transpose': input variables might be transposed in order to
              match (one of) the label sequences defined by their respective
              model variables

            Note that ``batch_dim`` (if any) and clock dimensions are excluded
            from this check. If None is given, no check is performed.
        validate : {'inputs', 'all'}, optional
            Define what will be validated using the variable's validators
            defined in ``model``'s processes (if any). It may be one of the
            following options:

            - 'inputs': validate only values given as inputs (default)
            - 'all': validate both input values and values set through foreign
              variables in process classes

            The latter may significantly impact performance, but it may be
            useful for debugging. If None is given, no validation is performed.
        store : str or :class:`collections.abc.MutableMapping` or :class:`zarr.Group` object, optional
            If a string (path) is given, simulation I/O data
            will be saved in that specified directory in the file
            system. If None is given (default), all data will be saved in
            memory. This parameter also directly accepts a zarr group object
            or (most of) zarr store objects for more storage options
            (see notes below).
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{'my_variable': {'dtype': 'int16', 'fill_value': -9999,}, ...}``.
            Encoding options provided here override encoding options defined in
            model variables (see :func:`~xsimlab.variable` for a full list of
            of options available). Additionally, 'chunks' and 'synchronizer'
            options are supported here.
        hooks : list, optional
            One or more runtime hooks, i.e., functions decorated with
            :func:`~xsimlab.runtime_hook` or instances of
            :class:`~xsimlab.RuntimeHook`. The latter can also be used using
            the ``with`` statement or using their ``register()`` method.
        parallel : bool, optional
            If True, run the simulation(s) in parallel using Dask (default: False).
            If a dimension label is set for ``batch_dim``, each simulation in
            the batch will be run in parallel. Otherwise, the processes in
            ``model`` will be executed in parallel for each simulation stage.
        scheduler : str, optional
            Dask's scheduler used to run the simulation(s) in parallel. See
            :func:`dask.compute`. It also accepts any instance of
            ``distributed.Client``.
        safe_mode : bool, optional
            If True (default), a clone of ``model`` will be used to run each
            simulation so that it is safe to run multiple simulations
            simultaneously (provided that the code executed in ``model`` is
            thread-safe too). Generally safe mode shouldn't be disabled, except
            in a few cases (e.g., debugging).

        Returns
        -------
        output : Dataset
            Another Dataset with both model inputs and outputs. The data is lazily
            loaded from the zarr store used to save inputs and outputs.

        Notes
        -----
        xarray-simlab uses the zarr library (https://zarr.readthedocs.io) to
        save model inputs and outputs during a simulation. zarr provides a
        common interface to multiple storage solutions (e.g., in memory, on
        disk, cloud-based storage, databases, etc.). Some stores may not work
        well with xarray-simlab, though. For example
        :class:`zarr.storage.ZipStore` is not supported because it is not
        possible to write data to a dataset after it has been created.

        xarray-simlab uses the dask library (https://docs.dask.org) to run the
        simulation(s) in parallel. Dask is a powerful library that allows
        running tasks (either simulations or model processes) on a single
        machine (multi-threads or multi-processes) or even on a distributed
        architecture (HPC, Cloud). Even though xarray-simlab includes some
        safeguards against race conditions, those might still occur under some
        circumstances and thus require extra care. In particular:

        - The code implemented in the process classes of ``model`` must be
          thread-safe if a dask multi-threaded scheduler is used, and must be
          serializable if a multi-process or distributed scheduler is used.
        - Multi-process or distributed schedulers are not well supported or
          may have poor performance when running the ``model`` processes in
          parallel (i.e., single-model parallelism), depending on the amount
          of data shared between the processes. See :meth:`xsimlab.Model.execute`
          for more details.
        - Not all zarr stores are safe to write in multiple threads or processes.
          For example, :class:`zarr.storage.MemoryStore` used by default is
          safe to write in multiple threads but not in multiple processes.
        - If chunks are specified in ``encoding`` with chunk size > 1
          for ``batch_dim``, then one of the zarr synchronizers should be used
          too, otherwise model output values will not be saved properly.
          Pick :class:`zarr.sync.ThreadSynchronizer` or
          :class:`zarr.sync.ProcessSynchronizer` depending on which dask scheduler
          is used. Also, check that the (distributed) scheduler doesn't use
          both multiple processes and multiple threads.

        """
        model = _maybe_get_model_from_context(model)

        if safe_mode:
            model = model.clone()

        driver = XarraySimulationDriver(
            self._ds,
            model,
            batch_dim=batch_dim,
            store=store,
            encoding=encoding,
            check_dims=check_dims,
            validate=validate,
            hooks=hooks,
            parallel=parallel,
            scheduler=scheduler,
        )

        driver.run_model()

        return driver.get_results()


def create_setup(
    model=None,
    clocks=None,
    master_clock=None,
    input_vars=None,
    output_vars=None,
    fill_default=True,
):
    """Create a specific setup for model runs.

    This convenient function creates a new :class:`xarray.Dataset`
    object with everything needed to run a model (i.e., input values,
    time steps, output variables to save at given times) as data
    variables, coordinates and attributes.

    Parameters
    ----------
    model : :class:`xsimlab.Model` object, optional
        Create a simulation setup for this model. If None, tries to get model
        from context.
    clocks : dict, optional
        Used to create one or several clock coordinates. Dictionary
        values are anything that can be easily converted to
        :class:`xarray.IndexVariable` objects (e.g., a 1-d
        :class:`numpy.ndarray` or a :class:`pandas.Index`).
    master_clock : str or dict, optional
        Name of the clock coordinate (dimension) to use as master clock.
        If not set, the name is inferred from ``clocks`` (only if
        one coordinate is given and if Dataset has no master clock
        defined yet).
        A dictionary can also be given with one of several of these keys:

        - ``dim`` : name of the master clock dimension/coordinate
        - ``units`` : units of all clock coordinate labels
        - ``calendar`` : a unique calendar for all (time) clock coordinates
    input_vars : dict, optional
        Dictionary with values given for model inputs. Entries of the
        dictionary may look like:

        - ``'foo': {'bar': value, ...}`` or
        - ``('foo', 'bar'): value`` or
        - ``'foo__bar': value``

        where ``foo`` is the name of a existing process in the model and
        ``bar`` is the name of an (input) variable declared in that process.

        Values are anything that can be easily converted to
        :class:`xarray.Variable` objects, e.g., single values, array-like,
        ``(dims, data, attrs)`` tuples or xarray objects.
        For array-like values with no dimension labels, xarray-simlab will look
        in ``model`` variables metadata for labels matching the number
        of dimensions of those arrays.
    output_vars : dict, optional
        Dictionary with model variable names to save as simulation output
        (time-dependent or time-independent). Entries of the dictionary look
        similar than for ``input_vars`` (see here above), except that here
        ``value`` must correspond to the dimension of a clock coordinate
        (i.e., new output values will be saved at each time given by the
        coordinate labels) or ``None`` (i.e., only one value will be saved
        at the end of the simulation).
    fill_default : bool, optional
        If True (default), automatically fill the dataset with all model
        inputs missing in ``input_vars`` and their default value (if any).

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        A new Dataset object with model inputs as data variables or coordinates
        (depending on their given value) and clock coordinates.
        The names of the input variables also include the name of their process
        (i.e., 'foo__bar').

    Notes
    -----
    Output variable names are added in Dataset as specific attributes
    (global and/or clock coordinate attributes).

    """
    model = _maybe_get_model_from_context(model)

    def maybe_fill_default(ds):
        if fill_default:
            return ds.xsimlab.reset_vars(model=model)
        else:
            return ds

    ds = (
        Dataset()
        .xsimlab.update_clocks(model=model, clocks=clocks, master_clock=master_clock)
        .pipe(maybe_fill_default)
        .xsimlab.update_vars(
            model=model, input_vars=input_vars, output_vars=output_vars
        )
    )

    return ds
