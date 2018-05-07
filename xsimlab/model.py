from collections import OrderedDict, defaultdict
from inspect import isclass

from .variable import VarIntent, VarType
from .process import (ensure_process_decorated, filter_variables,
                      get_target_variable)
from .utils import AttrMapping, ContextMixin, has_method, variables_dict
from .formatting import repr_model


def _flatten_keys(key_seq):
    """returns a flat list of keys, i.e., ``('foo', 'bar')`` tuples, from
    a nested sequence.

    """
    flat_keys = []

    for key in key_seq:
        if not isinstance(key, tuple):
            flat_keys += _flatten_keys(key)
        else:
            flat_keys.append(key)

    return flat_keys


class _ModelBuilder(object):
    """Used to iteratively build a new model.

    This builder implements the following tasks:

    - Attach the model instance to each process and assign their given
      name in model.
    - Define for each variable of the model its corresponding key
      (in store or on-demand)
    - Find variables that are model inputs
    - Find process dependencies and sort processes (DAG)
    - Find the processes that implement the method relative to each
      step of a simulation

    """
    def __init__(self, processes_cls):
        self._processes_cls = processes_cls
        self._processes_obj = {k: cls() for k, cls in processes_cls.items()}

        self._reverse_lookup = {cls: k for k, cls in processes_cls.items()}

        self._input_vars = None

        self._dep_processes = None
        self._sorted_processes = None

        # a cache for group keys
        self._group_keys = {}

    def bind_processes(self, model_obj):
        for p_name, p_obj in self._processes_obj.items():
            p_obj.__xsimlab_model__ = model_obj
            p_obj.__xsimlab_name__ = p_name

    def _get_var_key(self, p_name, var):
        """Get store and/or on-demand keys for variable `var` declared in
        process `p_name`.

        Returned key(s) are either None (if no key), a tuple or a list
        of tuples (for group variables).

        A key tuple looks like ``('foo', 'bar')`` where 'foo' is the
        name of any process in the model and 'bar' is the name of a
        variable declared in that process.

        """
        store_key = None
        od_key = None

        var_type = var.metadata['var_type']

        if var_type == VarType.VARIABLE:
            store_key = (p_name, var.name)

        elif var_type == VarType.ON_DEMAND:
            od_key = (p_name, var.name)

        elif var_type == VarType.FOREIGN:
            target_p_cls, target_var = get_target_variable(var)
            target_p_name = self._reverse_lookup.get(target_p_cls, None)

            if target_p_name is None:
                raise KeyError(
                    "Process class '{}' missing in Model but required "
                    "by foreign variable '{}' declared in process '{}'"
                    .format(target_p_cls.__name__, var.name, p_name)
                )

            store_key, od_key = self._get_var_key(target_p_name, target_var)

        elif var_type == VarType.GROUP:
            var_group = var.metadata['group']
            store_key, od_key = self._get_group_var_keys(var_group)

        return store_key, od_key

    def _get_group_var_keys(self, group):
        """Get from cache or find model-wise store and on-demand keys
        for all variables related to a group (except group variables).

        """
        if group in self._group_keys:
            return self._group_keys[group]

        store_keys = []
        od_keys = []

        filter_group = lambda var: (var.metadata.get('group') == group and
                                    var.metadata['var_type'] != VarType.GROUP)

        for p_name, p_obj in self._processes_obj.items():
            for var in filter_variables(p_obj, func=filter_group).values():
                store_key, od_key = self._get_var_key(p_name, var)

                if store_key is not None:
                    store_keys.append(store_key)
                if od_key is not None:
                    od_keys.append(od_key)

        self._group_keys[group] = store_keys, od_keys

        return store_keys, od_keys

    def set_process_keys(self):
        """Find store and/or on-demand keys for all variables in a model and
        store them in their respective process, i.e., the following
        attributes:

        __xsimlab_store_keys__  (store keys)
        __xsimlab_od_keys__     (on-demand keys)

        """
        for p_name, p_obj in self._processes_obj.items():
            for var in filter_variables(p_obj).values():
                store_key, od_key = self._get_var_key(p_name, var)

                if store_key is not None:
                    p_obj.__xsimlab_store_keys__[var.name] = store_key
                if od_key is not None:
                    p_obj.__xsimlab_od_keys__[var.name] = od_key

    def get_all_variables(self):
        """Get all variables in the model as a list of
        ``(process_name, var_name)`` tuples.

        """
        all_keys = []

        for p_name, p_cls in self._processes_cls.items():
            all_keys += [(p_name, var_name)
                         for var_name in variables_dict(p_cls)]

        return all_keys

    def get_input_variables(self):
        """Get all input variables in the model as a list of
        ``(process_name, var_name)`` tuples.

        Model input variables meet the following conditions:

        - model-wise (i.e., in all processes), there is no variable with
          intent='out' targeting those variables (in store keys).
        - although group variables always have intent='in', they are not
          model inputs.

        """
        filter_in = lambda var: (
            var.metadata['var_type'] != VarType.GROUP and
            var.metadata['intent'] != VarIntent.OUT
        )
        filter_out = lambda var: var.metadata['intent'] == VarIntent.OUT

        in_keys = []
        out_keys = []

        for p_name, p_obj in self._processes_obj.items():
            in_keys += [
                p_obj.__xsimlab_store_keys__.get(var.name)
                for var in filter_variables(p_obj, func=filter_in).values()
            ]
            out_keys += [
                p_obj.__xsimlab_store_keys__.get(var.name)
                for var in filter_variables(p_obj, func=filter_out).values()
            ]

        self._input_vars = [k for k in set(in_keys) - set(out_keys)
                            if k is not None]

        return self._input_vars

    def get_process_dependencies(self):
        """Return a dictionary where keys are each process of the model and
        values are lists of the names of dependent processes (or empty
        lists for processes that have no dependencies).

        Process 1 depends on process 2 if the later declares a
        variable (resp. a foreign variable) with intent='out' that
        itself (resp. its target variable) is needed in process 1.

        """
        self._dep_processes = {k: set() for k in self._processes_obj}

        d_keys = {}   # all store/on-demand keys for each process

        for p_name, p_obj in self._processes_obj.items():
            d_keys[p_name] = _flatten_keys([
                p_obj.__xsimlab_store_keys__.values(),
                p_obj.__xsimlab_od_keys__.values()
            ])

        for p_name, p_obj in self._processes_obj.items():
            for var in filter_variables(p_obj, intent=VarIntent.OUT).values():
                if var.metadata['var_type'] == VarType.ON_DEMAND:
                    key = p_obj.__xsimlab_od_keys__[var.name]
                else:
                    key = p_obj.__xsimlab_store_keys__[var.name]

                for pn in self._processes_obj:
                    if pn != p_name and key in d_keys[pn]:
                        self._dep_processes[pn].add(p_name)

        self._dep_processes = {k: list(v)
                               for k, v in self._dep_processes.items()}

        return self._dep_processes

    def _sort_processes(self):
        """Sort processes based on their dependencies (return a list of sorted
        process names).

        Stack-based depth-first search traversal.

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

        for key in self._dep_processes:
            if key in completed:
                continue
            nodes = [key]
            while nodes:
                # Keep current node on the stack until all descendants are
                # visited
                cur = nodes[-1]
                if cur in completed:
                    # Already fully traversed descendants of cur
                    nodes.pop()
                    continue
                seen.add(cur)

                # Add direct descendants of cur to nodes stack
                next_nodes = []
                for nxt in self._dep_processes[cur]:
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
                    # cur has no more descendants to explore,
                    # so we're done with it
                    ordered.append(cur)
                    completed.add(cur)
                    seen.remove(cur)
                    nodes.pop()
        return ordered

    def get_sorted_processes(self):
        self._sorted_processes = OrderedDict(
            [(p_name, self._processes_obj[p_name])
             for p_name in self._sort_processes()]
        )
        return self._sorted_processes

    def get_stage_processes(self, stage):
        """Return a (sorted) list of all process objects that implement the
        method relative to a given simulation stage {'initialize', 'run_step',
        'finalize_step', 'finalize'}.

        """
        return [p_obj for p_obj in self._sorted_processes.values()
                if has_method(p_obj, stage)]


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

        Raises
        ------
        :exc:`TypeError`
            If values in ``processes`` are not classes.
        :exc:`NoteAProcessClassError`
            If values in ``processes`` are not classes decorated with
            :func:`process`.

        """
        for cls in processes.values():
            if not isclass(cls):
                raise TypeError("Dictionary values must be classes, "
                                "found {}".format(cls))
            ensure_process_decorated(cls)

        builder = _ModelBuilder(processes)

        builder.bind_processes(self)
        builder.set_process_keys()

        self._all_vars = builder.get_all_variables()
        self._all_vars_dict = None

        self._input_vars = builder.get_input_variables()
        self._input_vars_dict = None

        self._dep_processes = builder.get_process_dependencies()
        self._processes = builder.get_sorted_processes()

        self._p_initialize = builder.get_stage_processes('initialize')
        self._p_run_step = builder.get_stage_processes('run_step')
        self._p_finalize_step = builder.get_stage_processes('finalize_step')
        self._p_finalize = builder.get_stage_processes('finalize')

        super(Model, self).__init__(self._processes)
        self._initialized = True

    @property
    def all_vars(self):
        """Returns all variables in the model as a list of
        ``(process_name, var_name)`` tuples (or an empty list).

        """
        return self._all_vars

    @property
    def all_vars_dict(self):
        """Returns all variables in the model as a dictionary of lists of
        variable names grouped by process.

        """
        if self._all_vars_dict is None:
            all_vars = defaultdict(list)

            for p_name, var_name in self._all_vars:
                all_vars[p_name].append(var_name)

            self._all_vars_dict = dict(all_vars)

        return self._all_vars_dict

    @property
    def input_vars(self):
        """Returns all variables that require setting a value before running
        the model.

        A list of ``(process_name, var_name)`` tuples (or an empty list)
        is returned.

        """
        return self._input_vars

    @property
    def input_vars_dict(self):
        """Returns all variables that require setting a value before running
        the model.

        Unlike :attr:`Model.input_vars`, a dictionary of lists of
        variable names grouped by process is returned.

        """
        if self._input_vars_dict is None:
            inputs = defaultdict(list)

            for p_name, var_name in self._input_vars:
                inputs[p_name].append(var_name)

            self._input_vars_dict = dict(inputs)

        return self._input_vars_dict

    @property
    def dependent_processes(self):
        """Returns a dictionary where keys are process names and values are
        lists of the names of dependent processes.

        """
        return self._dep_processes

    def visualize(self, show_only_variable=None, show_inputs=False,
                  show_variables=False):
        """Render the model as a graph using dot (require graphviz).

        Parameters
        ----------
        show_only_variable : tuple, optional
            Show only a variable (and all other variables sharing the
            same value) given as a tuple ``(process_name, variable_name)``.
            Deactivated by default.
        show_inputs : bool, optional
            If True, show all input variables in the graph (default: False).
            Ignored if `show_only_variable` is not None.
        show_variables : bool, optional
            If True, show also the other variables (default: False).
            Ignored if ``show_only_variable`` is not None.

        See Also
        --------
        :func:`dot.dot_graph`

        """
        from .dot import dot_graph
        return dot_graph(self, show_only_variable=show_only_variable,
                         show_inputs=show_inputs,
                         show_variables=show_variables)

    def initialize(self):
        """Run the 'initialize' stage of a simulation."""
        for p in self._p_initialize:
            p.initialize()

    def run_step(self, step):
        """Run a single 'run_step()' stage of a simulation."""
        for p in self._p_run_step:
            p.run_step(step)

    def finalize_step(self):
        """Run a single 'finalize_step' stage of a simulation."""
        for p in self._p_finalize_step:
            p.finalize_step()

    def finalize(self):
        """Run the 'finalize' stage of a simulation."""
        for p in self._p_finalize:
            p.finalize()

    def clone(self):
        """Clone the Model, i.e., create a new Model instance with the same
        process classes but different instances.

        """
        processes_cls = {k: type(obj) for k, obj in self._processes.items()}
        return type(self)(processes_cls)

    def update_processes(self, processes):
        """Add or replace processe(s) in this model.

        Parameters
        ----------
        processes : dict
            Dictionnary with process names as keys and process classes
            as values.

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
        return repr_model(self)
