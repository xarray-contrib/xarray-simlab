"""
Part of the code below is copied and modified from:

- dask 0.14.3 (Copyright (c) 2014-2015, Continuum Analytics, Inc.
  and contributors)
  Licensed under the BSD 3 License
  http://dask.pydata.org

"""
from collections import OrderedDict

from .variable import AbstractVariable, Variable, ForeignVariable
from .process import Process
from ..core.utils import AttrMapping
from ..core.formatting import (_calculate_col_width, pretty_print,
                               maybe_truncate)


def _reverse_processes_dict(processes):
    return {v.__class__: k for k, v in processes.items()}


def _get_foreign_vars(processes):
    # type: Dict[str, Process] -> Dict[str, List[ForeignVariable]]

    foreign_vars = {}

    for proc_name, proc in processes.items():
        foreign_vars[proc_name] = []

        for var in proc._variables.values():
            if isinstance(var, (tuple, list)):
                foreign_vars[proc_name] += [
                    v for v in var if isinstance(v, ForeignVariable)
                ]
            elif isinstance(var, ForeignVariable):
                foreign_vars[proc_name].append(var)

    return foreign_vars


def _get_input_vars(processes):
    # type: Dict[str, Process] -> Dict[str, Dict[str, Variable]]

    input_vars = {}
    proc_cls_name = _reverse_processes_dict(processes)

    for proc_name, proc in processes.items():
        input_vars[proc_name] = {}

        for k, var in proc._variables.items():
            if isinstance(var, Variable) and not var.provided:
                input_vars[proc_name][k] = var

    # case of variables provided by other processes
    foreign_vars = _get_foreign_vars(processes)
    for variables in foreign_vars.values():
        for var in variables:
            other_proc_name = proc_cls_name[var.other_process]
            if input_vars[other_proc_name].get(var.var_name, False):
                if var.provided:
                    del input_vars[other_proc_name][var.var_name]

    return {k: v for k, v in input_vars.items() if v}


def _get_process_dependencies(processes):
    # type: Dict[str, Process] -> Dict[str, List[Process]]

    dep_processes = {k: set() for k in processes}
    proc_cls_name = _reverse_processes_dict(processes)
    foreign_vars = _get_foreign_vars(processes)

    for proc_name, variables in foreign_vars.items():
        for var in variables:
            other_proc_name = proc_cls_name[var.other_process]
            if var.provided:
                dep_processes[other_proc_name].add(proc_name)
            else:
                pvar = var.other_process._variables[var.var_name]
                if pvar.provided or getattr(pvar, 'optional', False):
                    dep_processes[proc_name].add(other_proc_name)

    return {k: list(v) for k, v in dep_processes.items()}


def _sort_processes(dep_processes):
    # type: Dict[str, List[Process]] -> List[str]
    """Stack-based depth-first search traversal.

    This is based on Tarjan's method for topological sorting.
    Code taken and modified from Dask (New BSD License,
    https://github.com/dask/dask).

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


def _link_foreign_vars(processes):
    """Assign process instances to foreign variables."""
    proc_cls_name = _reverse_processes_dict(processes)

    for variables in _get_foreign_vars(processes).values():
        for var in variables:
            proc_obj = processes[proc_cls_name[var.other_process]]
            var._assign_other_process_obj(proc_obj)


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
            Dictionnary with process names as keys and `Process`
            objects as values. The order doesn't matter.

        """
        for k, p in processes.items():
            if not isinstance(p, Process):
                raise TypeError("%s is not a Process object" % p)

        self._hash = None
        self._input_vars = _get_input_vars(processes)
        self._dep_processes = _get_process_dependencies(processes)
        _link_foreign_vars(processes)

        self._processes = OrderedDict(
            [(k, processes[k]) for k in _sort_processes(self._dep_processes)]
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
        else:
            proc_name, var_name = variable

        return self._input_vars.get(proc_name, {}).get(var_name, False)

    def _get_dsk(self, func='run_step'):
        return {k: (getattr(self._processes[k], func), v)
                for k, v in self._dep_processes.items()}

    def visualize(self, show_only_variable=None, show_inputs=True,
                  show_variables=False):
        """Render the model as a graph using dot (require graphviz).

        Parameters
        ----------
        show_only_variable : object or tuple, optional
            Show only a variable (and all other linked variables) given either
            as a Variable object or a tuple corresponding to process name and
            variable name. Deactivated by default.
        show_inputs : bool, optional
            If True (default), show all input variables in the graph.
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

    def update_processes(self, processes):
        """Add or replace processe(s) in this model.

        Parameters
        ----------
        processes : dict
            Dictionnary with process names as keys and
            `Process` objects as values.

        Returns
        -------
        updated : Model
            New Model instance with updated processes.

        """
        # TODO: also copy process instances (deep)?
        new_processes = self._processes.copy()
        new_processes.update(processes)
        return type(self)(new_processes)

    def drop_processes(self, names):
        """Drop processe(s) from this model.

        Parameters
        ----------
        names : str or list of str
            Name(s) of the processes to drop.

        Returns
        -------
        dropped : Model
            New Model instance with dropped processes.

        """
        if isinstance(names, str):
            names = [names]

        # TODO: also automatically remove dependent ForeingVariables
        #       if they are defined within lists?
        # TODO: also copy process instances (deep)?
        new_processes = self._processes.copy()
        for n in names:
            del new_processes[n]
        return type(self)(new_processes)

    def __repr__(self):
        n_inputs = sum([len(v) for v in self._input_vars.values()])

        hdr = ("<fastscape.models.Model (%d processes, %d inputs)>\n"
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
