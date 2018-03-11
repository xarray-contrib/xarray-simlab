from collections import OrderedDict

from .variable import VarType
from .process import filter_variables, get_target_variable
from .utils import AttrMapping, ContextMixin
from .formatting import _calculate_col_width, pretty_print, maybe_truncate


class _ModelBuilder(object):
    """Used to iteratively build a new model.

    - Reconstruct process/variable dependencies
    - Sort processes DAG
    - Retrieve model inputs
    - Split time dependent vs. independent processes

    """
    def __init__(self, processes_cls):
        self._processes_cls = processes_cls
        self._processes_obj = {k: cls() for k, cls in processes_cls.items()}

        self._reverse_lookup = {cls: k for k, cls in processes_cls.items()}

        # a cache for group keys
        self._group_keys = {}

    def set_process_names(self):
        for p_name, p_obj in self.processes_obj.items():
            p_obj.__xsimlab_name__ = p_name

    def _get_var_key(self, p_name, var):
        """Get store and on-demand keys for variable `var` declared in
        process `p_name`.

        Returned keys are either None (if no key), a tuple or a list
        of tuples (for group variables).

        A store key tuple looks like `('foo', 'bar')` where 'foo' is
        the name of any process in the model and 'bar' is the name of
        a variable declared in that process.

        Similarly, an on-demand key tuple looks like `(foo_obj, 'bar')`,
        but where `foo_obj` is a process object rather than its name.

        """
        store_key = None
        od_key = None

        var_type = var.metadata['var_type']

        if var_type == VarType.VARIABLE:
            store_key = (p_name, var.name)

        elif var_type == VarType.FOREIGN:
            target_p_cls, target_var = get_target_variable(var)

            target_p_name = self._reverse_lookup(target_p_cls)
            target_p_obj = self._processes_obj[target_p_name]

            if target_var.metadata['var_type'] == VarType.ON_DEMAND:
                od_key = (target_p_obj, target_var.name)
            else:
                store_key = (target_p_name, target_var.name)

        elif var_type == VarType.GROUP:
            group = var.metadata['group']

            store_key, od_key = self._group_keys.get(
                group, self._get_group_var_keys(group)
            )

        return store_key, od_key

    def _get_group_var_keys(self, group):
        """Get store and on-demand keys for a group variable."""
        store_keys = []
        od_keys = []

        for p_name, p_obj in self.processes_obj.items():
            var = filter_variables(p_obj, group=group)
            store_key, od_key = self._get_var_key(p_name, var)

            if store_key is not None:
                store_keys.append(store_key)
            if od_key is not None:
                od_keys.append(od_key)

        group_keys = (store_keys, od_keys)
        self._group_keys[group] = group_keys

        return group_keys

    def set_process_keys(self):
        """Get store and on-demand keys for all variables in a model
        and add them in their respective process using the following
        attributes:

        __xsimlab_store_keys__
        __xsimlab_od_keys__

        """
        for p_name, p_obj in self.processes_obj.items():
            for var in filter_variables(p_obj):
                store_key, od_key = self._get_var_key(p_name, var)

                if store_key is not None:
                    p_obj.__xsimlab_store_keys__[var.name] = store_key
                if od_key is not None:
                    p_obj.__xsimlab_od_keys__[var.name] = od_key

    def get_input_variables(self):
        """Get all input variables in the model as a list of
        `(process_name, var_name)` tuples.

        Model input variables meet the following condition:

        - model-wise (i.e., all processes), there is no variable with
          intent='out' that targets those variables (in store keys).
        - group variables always have intent='in' but are never model
          inputs.

        """
        filter_in = lambda var: (
            var.metadata['var_type'] != VarType.GROUP and
            var.metadata['intent'] in ('in', 'inout')
        )
        filter_out = lambda var: (
            var.metadata['var_type'] != VarType.ON_DEMAND and
            var.metadata['intent'] == 'out'
        )

        in_keys = []
        out_keys = []

        for p_name, p_obj in self.processes_obj.items():
            in_keys += [p_obj.__xsimlab_store_keys__[var.name]
                        for var in filter_variables(p_obj, func=filter_in)]

            out_keys += [p_obj.__xsimlab_store_keys__[var.name]
                         for var in filter_variables(p_obj, func=filter_out)]

        return list(set(in_keys) - set(out_keys))


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
                            "cycle detected in process graph: %s" % cycle
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


class Model(AttrMapping, ContextMixin):
    """An immutable collection of process units that together form a
    computational model.

    This collection is ordered such that the computational flow is
    consistent with process inter-dependencies.

    Ordering doesn't need to be explicitly provided ; it is dynamically
    computed using the processes interfaces.

    Processes interfaces are also used for automatically retrieving
    the model inputs, i.e., all the variables that require setting a
    value before running the model.

    """
    def __init__(self, processes):
        """
        Parameters
        ----------
        processes : dict
            Dictionnary with process names as keys and classes (decorated with
            :func:`process`) as values.

        """
        processes_obj = {}

        for k, cls in processes.items():
            if getattr(cls, '__xsimlab_name__') is None:
                raise TypeError("class {} is not a process compatible class: "
                                "you might want decorate it first using "
                                "@process".format(cls.__name__))

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
        return None, None

    @property
    def input_vars(self):
        """Returns all variables that require setting a
        value before running the model.

        These variables are grouped by process name (dict of dicts).

        """
        return self._input_vars

    def is_input(self, variable):
        """Test if a variable is an input of Model.

        Parameters
        ----------
        variable : object or tuple
            Either a Variable object or a (str, str) tuple
            corresponding to process name and variable name.

        Returns
        -------
        is_input : bool
            True if the variable is a input of Model (otherwise False).

        """
        if isinstance(variable, AbstractVariable):
            proc_name, var_name = self._get_proc_var_name(variable)
        elif isinstance(variable, (VariableList, VariableGroup)):
            proc_name, var_name = None, None   # prevent unpack iterable below
        else:
            proc_name, var_name = variable

        if self._input_vars.get(proc_name, {}).get(var_name, False):
            return True
        return False

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
        show_variables : bool, optional
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

        hdr = ("<xsimlab.Model (%d processes, %d inputs)>"
               % (len(self._processes), n_inputs))

        if not len(self._processes):
            return hdr

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

        return hdr + '\n' + '\n'.join(blocks)
