from collections import OrderedDict

import numpy as np
import xarray as xr

from .variable.base import (AbstractVariable, Variable, ForeignVariable,
                            VariableList, VariableGroup)
from .process import Process
from . import xr_accessor
from .utils import AttrMapping
from .formatting import _calculate_col_width, pretty_print, maybe_truncate


class ModelRunSnapshots(object):
    """Interface that allows taking snapshots for given `Variable` objects
    during a model run.

    Snaphots are only for values given by `Variable.state` (or equivalently
    `Variable.value`).

    """

    def __init__(self, model, dataset):
        self.model = model
        self.ds = dataset

        self.snapshot_clocks_vars = dataset.xsimlab.snapshot_vars

        self.snapshot_arrays = {}
        for vars in self.snapshot_clocks_vars.values():
            self.snapshot_arrays.update({v: [] for v in vars})

        master_clock_values = dataset[dataset.xsimlab.dim_master_clock].values
        self.snapshot_clocks_steps = {
            clock: np.in1d(master_clock_values, dataset[clock].values)
            for clock in self.snapshot_clocks_vars if clock is not None
        }

    def _take_snapshot_var(self, key):
        proc_name, var_name = key
        model_var = self.model._processes[proc_name]._variables[var_name]
        self.snapshot_arrays[key].append(np.array(model_var.state))

    def take_snapshots(self, step):
        for clock, vars in self.snapshot_clocks_vars.items():
            if clock is None:
                if step == -1:
                    for key in vars:
                        self._take_snapshot_var(key)
            elif self.snapshot_clocks_steps[clock][step]:
                for key in vars:
                    self._take_snapshot_var(key)

    def _get_dims(self, array, variable):
        for dims in variable.allowed_dims:
            if len(dims) == array.ndim:
                return dims

        return tuple()

    def _to_xarray_variable(self, key, clock=None):
        proc_name, var_name = key
        variable = self.model._processes[proc_name]._variables[var_name]

        array_list = self.snapshot_arrays[key]
        first_array = array_list[0]

        if len(array_list) == 1:
            data = first_array
        else:
            data = np.stack(array_list)

        dims = self._get_dims(first_array, variable)
        if clock is not None:
            dims = (clock,) + dims

        attrs = variable.attrs.copy()
        attrs['description'] = variable.description

        return xr.Variable(dims, data, attrs=attrs)

    def to_dataset(self):
        xr_variables = {}

        for clock, vars in self.snapshot_clocks_vars.items():
            for key in vars:
                var_name = '__'.join(key)
                xr_variables[var_name] = self._to_xarray_variable(
                    key, clock=clock
                )

        out_ds = self.ds.update(xr_variables, inplace=False)

        for clock in self.snapshot_clocks_vars:
            if clock is None:
                attrs = out_ds.attrs
            else:
                attrs = out_ds[clock].attrs
            attrs.pop(xr_accessor.SimlabAccessor._snapshot_vars_key)

        return out_ds


def _set_process_names(processes):
    # type: Dict[str, Process]
    """Set process names using keys of the mapping."""
    for k, p in processes.items():
        p._name = k


def _set_group_vars(processes):
    """Assign variables that belong to variable groups."""
    for proc in processes.values():
        for var in proc._variables.values():
            if isinstance(var, VariableGroup):
                var._set_variables(processes)


def _get_foreign_vars(processes):
    # type: Dict[str, Process] -> Dict[str, List[ForeignVariable]]

    foreign_vars = {}

    for proc_name, proc in processes.items():
        foreign_vars[proc_name] = []

        for var in proc._variables.values():
            if isinstance(var, (tuple, list, VariableGroup)):
                foreign_vars[proc_name] += [
                    v for v in var if isinstance(v, ForeignVariable)
                ]
            elif isinstance(var, ForeignVariable):
                foreign_vars[proc_name].append(var)

    return foreign_vars


def _link_foreign_vars(processes):
    """Assign process instances to foreign variables."""
    proc_lookup = {v.__class__: v for v in processes.values()}

    for variables in _get_foreign_vars(processes).values():
        for var in variables:
            var._other_process_obj = proc_lookup[var._other_process_cls]


def _get_input_vars(processes):
    # type: Dict[str, Process] -> Dict[str, Dict[str, Variable]]

    input_vars = {}

    for proc_name, proc in processes.items():
        input_vars[proc_name] = {}

        for k, var in proc._variables.items():
            if isinstance(var, Variable) and not var.provided:
                input_vars[proc_name][k] = var

    # case of variables provided by other processes
    foreign_vars = _get_foreign_vars(processes)
    for variables in foreign_vars.values():
        for var in variables:
            if input_vars[var.ref_process.name].get(var.var_name, False):
                if var.provided:
                    del input_vars[var.ref_process.name][var.var_name]

    return {k: v for k, v in input_vars.items() if v}


def _get_process_dependencies(processes):
    # type: Dict[str, Process] -> Dict[str, List[Process]]

    dep_processes = {k: set() for k in processes}
    foreign_vars = _get_foreign_vars(processes)

    for proc_name, variables in foreign_vars.items():
        for var in variables:
            if var.provided:
                dep_processes[var.ref_process.name].add(proc_name)
            else:
                ref_var = var.ref_var
                if ref_var.provided or getattr(ref_var, 'optional', False):
                    dep_processes[proc_name].add(var.ref_process.name)

    return {k: list(v) for k, v in dep_processes.items()}


def _sort_processes(dep_processes):
    # type: Dict[str, List[Process]] -> List[str]
    """Stack-based depth-first search traversal.

    This is based on Tarjan's method for topological sorting.

    Part of the code below is copied and modified from:

    - dask 0.14.3 (Copyright (c) 2014-2015, Continuum Analytics, Inc.
      and contributors)
      Licensed under the BSD 3 License
      http://dask.pydata.org

    """
    ordered = []

    # Nodes whose descendents have been completely explored.
    # These nodes are guaranteed to not be part of a cycle.
    completed = set()

    # All nodes that have been visited in the current traversal.  Because
    # we are doing depth-first search, going "deeper" should never result
    # in visiting a node that has already been seen.  The `seen` and
    # `completed` sets are mutually exclusive; it is okay to visit a node
    # that has already been added to `completed`.
    seen = set()

    for key in dep_processes:
        if key in completed:
            continue
        nodes = [key]
        while nodes:
            # Keep current node on the stack until all descendants are visited
            cur = nodes[-1]
            if cur in completed:
                # Already fully traversed descendants of cur
                nodes.pop()
                continue
            seen.add(cur)

            # Add direct descendants of cur to nodes stack
            next_nodes = []
            for nxt in dep_processes[cur]:
                if nxt not in completed:
                    if nxt in seen:
                        # Cycle detected!
                        cycle = [nxt]
                        while nodes[-1] != nxt:
                            cycle.append(nodes.pop())
                        cycle.append(nodes.pop())
                        cycle.reverse()
                        cycle = '->'.join(cycle)
                        raise RuntimeError(
                            "Cycle detected in process graph: %s" % cycle
                        )
                    next_nodes.append(nxt)

            if next_nodes:
                nodes.extend(next_nodes)
            else:
                # cur has no more descendants to explore, so we're done with it
                ordered.append(cur)
                completed.add(cur)
                seen.remove(cur)
                nodes.pop()
    return ordered


class Model(AttrMapping):
    """An immutable collection (mapping) of process units that together
    form a computational model.

    This collection is ordered such that the computational flow is
    consistent with process inter-dependencies.

    Ordering doesn't need to be explicitly provided ; it is dynamically
    computed using the processes interfaces.

    Processes interfaces are also used for automatically retrieving
    the model inputs, i.e., all the variables which require setting a
    value before running the model.

    """
    def __init__(self, processes):
        """
        Parameters
        ----------
        processes : dict
            Dictionnary with process names as keys and subclasses of
            `Process` as values.

        """
        processes_obj = {}

        for k, cls in processes.items():
            if not issubclass(cls, Process):
                raise TypeError("%s is not a subclass of Process object" % cls)
            processes_obj[k] = cls()

        _set_process_names(processes_obj)
        _set_group_vars(processes_obj)
        _link_foreign_vars(processes_obj)

        self._input_vars = _get_input_vars(processes_obj)
        self._dep_processes = _get_process_dependencies(processes_obj)
        self._processes = OrderedDict(
            [(k, processes_obj[k])
             for k in _sort_processes(self._dep_processes)]
        )
        self._time_processes = OrderedDict(
            [(k, proc) for k, proc in self._processes.items()
             if proc.meta['time_dependent']]
        )

        super(Model, self).__init__(self._processes)
        self._initialized = True

    def _get_proc_var_name(self, variable):
        # type: AbstractVariable -> Union[tuple[str, str], None]
        for proc_name, variables in self._processes.items():
            for var_name, var in variables.items():
                if var is variable:
                    return proc_name, var_name
        return None

    @property
    def input_vars(self):
        """Returns all variables that require setting a
        value before running the model.

        These variables are grouped by process name (dict of dicts).

        """
        return self._input_vars

    def is_input(self, variable):
        """Returns True if the variable is a model input.

        Parameters
        ----------
        variable : object or tuple
            Either a Variable object or a (str, str) tuple
            corresponding to process name and variable name.

        """
        if isinstance(variable, AbstractVariable):
            proc_name, var_name = self._get_proc_var_name(variable)
        elif isinstance(variable, (VariableList, VariableGroup)):
            # VariableList and VariableGroup objects are never model inputs
            return False
        else:
            proc_name, var_name = variable

        return self._input_vars.get(proc_name, {}).get(var_name, False)

    def visualize(self, show_only_variable=None, show_inputs=False,
                  show_variables=False):
        """Render the model as a graph using dot (require graphviz).

        Parameters
        ----------
        show_only_variable : object or tuple, optional
            Show only a variable (and all other linked variables) given either
            as a Variable object or a tuple corresponding to process name and
            variable name. Deactivated by default.
        show_inputs : bool, optional
            If True, show all input variables in the graph (default: False).
            Ignored if `show_only_variable` is not None.
        show_variabless : bool, optional
            If True, show also the other variables (default: False).
            Ignored if `show_only_variable` is not None.

        See Also
        --------
        dot.dot_graph

        """
        from .dot import dot_graph
        return dot_graph(self, show_only_variable=show_only_variable,
                         show_inputs=show_inputs,
                         show_variables=show_variables)

    def _get_dsk(self, func='run_step'):
        return {k: (getattr(self._processes[k], func), v)
                for k, v in self._dep_processes.items()}

    def _set_inputs_values(self, ds):
        """Set model inputs values from xarray.Dataset."""
        for name, var in ds.data_vars.items():
            proc_name, var_name = name.split('__')

            if self.is_input((proc_name, var_name)):
                self[proc_name][var_name].value = var.values.copy()

    def initialize(self):
        """Run `.initialize()` for each processes in the model."""
        for proc in self._processes.values():
            proc.initialize()

    def run_step(self, step):
        """Run `.run_step()` for each time dependent processes in the model.
        """
        for proc in self._time_processes.values():
            proc.run_step(step)

    def finalize_step(self):
        """Run `.finalize_step()` for each time dependent processes
        in the model.
        """
        for proc in self._time_processes.values():
            proc.finalize_step()

    def finalize(self):
        """Run `.finalize()` for each processes in the model."""
        for proc in self._processes.values():
            proc.finalize()

    def run(self, ds, safe_mode=True):
        """Run the model.

        Parameters
        ----------
        ds : xarray.Dataset object
            Dataset to use as model input.
        safe_mode : bool, optional
            If True (default), it is safe to run multiple simulations
            simultaneously. Generally safe mode shouldn't be disabled, except
            in a few cases (e.g., debugging).

        Returns
        -------
        out_ds : xarray.Dataset object
            Another Dataset with model inputs and outputs.

        """
        dim_master_clock = ds.xsimlab.dim_master_clock
        if dim_master_clock is None:
            raise ValueError("missing master clock dimension / coordinate ")

        if safe_mode:
            obj = self.clone()
        else:
            obj = self

        ds_no_time = ds.filter(lambda v: dim_master_clock not in v.dims)
        obj._set_inputs_values(ds_no_time)

        obj.initialize()

        ds_time = ds.filter(lambda v: dim_master_clock in v.dims)
        has_time_var = bool(ds_time.data_vars)

        time_steps = ds[dim_master_clock].diff(dim_master_clock).values

        snapshots = ModelRunSnapshots(obj, ds)

        for i, dt in enumerate(time_steps):
            if has_time_var:
                ds_step = ds_time.isel(**{dim_master_clock: i})
                obj._set_inputs_values(ds_step)

            obj.run_step(dt)

            snapshots.take_snapshots(i)

            obj.finalize_step()

        snapshots.take_snapshots(-1)

        obj.finalize()

        return snapshots.to_dataset()

    def clone(self):
        """Clone the Model.

        This is equivalent to a deep copy, except that variable data
        (i.e., `state`, `value`, `change` or `rate` properties) in all
        processes are not copied.
        """
        processes_cls = {k: type(obj) for k, obj in self._processes.items()}
        return type(self)(processes_cls)

    def update_processes(self, processes):
        """Add or replace processe(s) in this model.

        Parameters
        ----------
        processes : dict
            Dictionnary with process names as keys and subclasses of
            `Process` as values.

        Returns
        -------
        updated : Model
            New Model instance with updated processes.

        """
        processes_cls = {k: type(obj) for k, obj in self._processes.items()}
        processes_cls.update(processes)
        return type(self)(processes_cls)

    def drop_processes(self, keys):
        """Drop processe(s) from this model.

        Parameters
        ----------
        keys : str or list of str
            Name(s) of the processes to drop.

        Returns
        -------
        dropped : Model
            New Model instance with dropped processes.

        """
        if isinstance(keys, str):
            keys = [keys]

        processes_cls = {k: type(obj) for k, obj in self._processes.items()
                         if k not in keys}
        return type(self)(processes_cls)

    def __repr__(self):
        n_inputs = sum([len(v) for v in self._input_vars.values()])

        hdr = ("<xsimlab.Model (%d processes, %d inputs)>\n"
               % (len(self._processes), n_inputs))

        max_line_length = 70
        col_width = max([_calculate_col_width(var)
                         for var in self._input_vars.values()])

        blocks = []
        for proc_name in self._processes:
            proc_str = "%s" % proc_name

            inputs = self._input_vars.get(proc_name, {})
            lines = []
            for name, var in inputs.items():
                line = pretty_print("    %s " % name, col_width)
                line += maybe_truncate("(in) %s" % var.description,
                                       max_line_length - col_width)
                lines.append(line)

            if lines:
                proc_str += '\n' + '\n'.join(lines)
            blocks.append(proc_str)

        return hdr + '\n'.join(blocks)
