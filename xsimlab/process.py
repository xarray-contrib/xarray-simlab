from collections import OrderedDict
from enum import Enum
import inspect
import sys
import warnings

import attr

from .variable import VarIntent, VarType
from .formatting import add_attribute_section, repr_process, var_details
from .utils import has_method, variables_dict


class NotAProcessClassError(ValueError):
    """
    A non-``xsimlab.process`` class has been passed into a ``xsimlab``
    function.

    """

    pass


def _get_embedded_process_cls(cls):
    if getattr(cls, "__xsimlab_process__", False):
        return cls

    else:
        try:
            return cls.__xsimlab_cls__
        except AttributeError:
            raise NotAProcessClassError(f"{cls!r} is not a process-decorated class.")


def get_process_cls(obj_or_cls):
    if not inspect.isclass(obj_or_cls):
        cls = type(obj_or_cls)
    else:
        cls = obj_or_cls

    return _get_embedded_process_cls(cls)


def get_process_obj(obj_or_cls):
    if inspect.isclass(obj_or_cls):
        cls = obj_or_cls
    else:
        cls = type(obj_or_cls)

    return _get_embedded_process_cls(cls)()


def filter_variables(process, var_type=None, intent=None, group=None, func=None):
    """Filter the variables declared in a process.

    Parameters
    ----------
    process : object or class
        Process class or object.
    var_type : {'variable', 'on_demand', 'foreign', 'group'}, optional
        Return only variables of a specified type.
    intent : {'in', 'out', 'inout'}, optional
        Return only input, output or input/output variables.
    group : str, optional
        Return only variables that belong to a given group.
    func : callable, optional
        A callable that takes a variable (i.e., a :class:`attr.Attribute`
        object) as input and return True or False. Useful for more advanced
        filtering.

    Returns
    -------
    attributes : dict
        A dictionary of variable names as keys and :class:`attr.Attribute`
        objects as values.

    """
    process_cls = get_process_cls(process)

    # be consistent and always return a dict (not OrderedDict) when no filter
    vars = dict(variables_dict(process_cls))

    if var_type is not None:
        vars = {
            k: v
            for k, v in vars.items()
            if v.metadata.get("var_type") == VarType(var_type)
        }

    if intent is not None:
        vars = {
            k: v
            for k, v in vars.items()
            if v.metadata.get("intent") == VarIntent(intent)
        }

    if group is not None:
        vars = {k: v for k, v in vars.items() if group in v.metadata.get("groups", [])}

    if func is not None:
        vars = {k: v for k, v in vars.items() if func(v)}

    return vars


def get_target_variable(var):
    """Return the target (original) variable of a given variable and
    the process class in which the target variable is declared.

    If `var` is not a foreign variable, return itself and None instead
    of a process class.

    If the target of foreign variable is another foreign variable (and
    so on...), this function follow the links until the original
    variable is found. An error is thrown if a cyclic pattern is detected.

    """
    target_process_cls = None
    target_var = var

    visited = []

    while target_var.metadata["var_type"] == VarType.FOREIGN:
        visited.append((target_process_cls, target_var))

        target_process_cls = target_var.metadata["other_process_cls"]
        var_name = target_var.metadata["var_name"]
        target_var = filter_variables(target_process_cls)[var_name]

        # TODO: maybe remove this? not even sure such a cycle may happen
        # unless we allow later providing other values than classes as first
        # argument of `foreign`
        if (target_process_cls, target_var) in visited:  # pragma: no cover
            cycle = "->".join(
                [
                    "{}.{}".format(cls.__name__, var.name)
                    if cls is not None
                    else var.name
                    for cls, var in visited
                ]
            )

            raise RuntimeError(f"Cycle detected in process dependencies: {cycle}")

    return target_process_cls, target_var


def _dummy_converter(value):
    return value


def _make_property_variable(var):
    """Create a property for a variable or a foreign variable (after
    some sanity checks).

    The property get/set functions either read/write values from/to the active
    simulation data store (i.e. state) or get (and trigger computation of) the
    value of an on-demand variable.

    The property is read-only if `var` is declared as input.

    """
    var_name = var.name
    var_converter = var.converter or _dummy_converter

    def get_from_state(self):
        key = self.__xsimlab_state_keys__[var_name]
        return self.__xsimlab_state__[key]

    def get_on_demand(self):
        p_name, v_name = self.__xsimlab_od_keys__[var_name]
        p_obj = self.__xsimlab_model__._processes[p_name]
        return getattr(p_obj, v_name)

    def put_in_state(self, value):
        key = self.__xsimlab_state_keys__[var_name]
        self.__xsimlab_state__[key] = var_converter(value)

    target_process_cls, target_var = get_target_variable(var)

    var_type = var.metadata["var_type"]
    target_type = target_var.metadata["var_type"]
    var_intent = var.metadata["intent"]
    target_intent = target_var.metadata["intent"]

    var_doc = var_details(var)

    if target_process_cls is not None:
        target_str = ".".join([target_process_cls.__name__, target_var.name])
    else:
        target_str = target_var.name

    if target_type == VarType.GROUP:
        raise ValueError(
            f"Variable {var.name!r} links to group variable "
            f"{target_str!r}, which is not supported. Declare {var.name!r} "
            "as a group variable instead."
        )

    elif (
        var_type == VarType.FOREIGN
        and var_intent == VarIntent.OUT
        and target_intent == VarIntent.OUT
    ):
        raise ValueError(
            f"Conflict between foreign variable {var.name!r} and its "
            f"target variable {target_str!r}, both have intent='out'."
        )

    elif target_type == VarType.ON_DEMAND:
        return property(fget=get_on_demand, doc=var_doc)

    elif var_intent == VarIntent.IN:
        return property(fget=get_from_state, doc=var_doc)

    else:
        return property(fget=get_from_state, fset=put_in_state, doc=var_doc)


def _make_property_on_demand(var):
    """Create a read-only property for an on-demand variable.

    This property is a simple wrapper around the variable's compute method.

    """
    if "compute" not in var.metadata:
        raise KeyError(
            "No compute method found for on_demand variable "
            f"'{var.name}'. A method decorated with '@{var.name}.compute' "
            "is required in the class definition."
        )

    get_method = var.metadata["compute"]

    return property(fget=get_method, doc=var_details(var))


def _make_property_group(var):
    """Create a read-only property for a group variable."""

    var_name = var.name

    def getter_state_or_on_demand(self):
        model = self.__xsimlab_model__
        state_keys = self.__xsimlab_state_keys__.get(var_name, [])
        od_keys = self.__xsimlab_od_keys__.get(var_name, [])

        for key in state_keys:
            yield self.__xsimlab_state__[key]

        for key in od_keys:
            p_name, v_name = key
            p_obj = model._processes[p_name]
            yield getattr(p_obj, v_name)

    return property(fget=getter_state_or_on_demand, doc=var_details(var))


class _RuntimeMethodExecutor:
    """Used to execute a process 'runtime' method in the context of a
    simulation.

    """

    def __init__(self, meth, args=None):
        self.meth = meth

        if args is None:
            args = []
        elif isinstance(args, str):
            args = [k.strip() for k in args.split(",") if k]
        elif isinstance(args, (list, tuple)):
            args = tuple(args)
        else:
            raise ValueError("args must be either a string, a list or a tuple")

        self.args = tuple(args)

    def execute(self, obj, runtime_context, state=None):
        if state is not None:
            obj.__xsimlab_state__ = state

        args = [runtime_context[k] for k in self.args]

        self.meth(obj, *args)


def runtime(meth=None, args=None):
    """Function decorator applied to a method of a process class that is
    called during simulation runtime.

    Parameters
    ----------
    meth : callable, optional
        The method to wrap (leave it to None if you use this function
        as a decorator).
    args : str or list or tuple, optional
        One or several labels of values that will be passed as
        positional argument(s) of the method during simulation runtime.
        The following labels are defined:

        - ``batch_size`` : total number of simulations run in the batch
        - ``batch`` : current simulation number in the batch
        - ``sim_start`` : simulation start (date)time
        - ``sim_end`` : simulation end (date)time
        - ``step`` : current step number
        - ``step_start`` : current step start (date)time
        - ``step_end``: current step end (date)time
        - ``step_delta``: current step duration

    Returns
    -------
    runtime_method
       The same method that can be called during a simulation
       with runtime data.

    """

    def wrapper(func):
        func.__xsimlab_executor__ = _RuntimeMethodExecutor(func, args)
        return func

    if meth is not None:
        return wrapper(meth)
    else:
        return wrapper


class SimulationStage(Enum):
    INITIALIZE = "initialize"
    RUN_STEP = "run_step"
    FINALIZE_STEP = "finalize_step"
    FINALIZE = "finalize"


def _create_runtime_executors(cls):
    runtime_executors = OrderedDict()

    for stage in SimulationStage:
        if not has_method(cls, stage.value):
            continue

        meth = getattr(cls, stage.value)
        executor = getattr(meth, "__xsimlab_executor__", None)

        if executor is None:
            nparams = len(inspect.signature(meth).parameters)

            if stage == SimulationStage.RUN_STEP and nparams == 2:
                # TODO: remove (depreciated)
                warnings.warn(
                    "`run_step(self, dt)` accepting by default "
                    "one positional argument is depreciated and "
                    "will be removed in a future version of "
                    "xarray-simlab. Use the `@runtime` "
                    "decorator.",
                    FutureWarning,
                )
                args = ["step_delta"]

            elif nparams > 1:
                raise TypeError(
                    "Process runtime methods with positional "
                    "parameters should be decorated with "
                    "`@runtime`"
                )

            else:
                args = None

            executor = _RuntimeMethodExecutor(meth, args=args)

        runtime_executors[stage] = executor

    return runtime_executors


def _get_out_variables(cls):
    def filter_out(var):
        var_type = var.metadata["var_type"]
        var_intent = var.metadata["intent"]

        if var_type != VarType.ON_DEMAND and var_intent != VarIntent.IN:
            return True
        else:
            return False

    return filter_variables(cls, func=filter_out)


class _ProcessExecutor:
    """Used to execute a process during simulation runtime."""

    def __init__(self, cls):
        self.cls = cls
        self.runtime_executors = _create_runtime_executors(cls)
        self.out_vars = _get_out_variables(cls)

    @property
    def stages(self):
        return [k.value for k in self.runtime_executors]

    def execute(self, obj, stage, runtime_context, state=None):
        executor = self.runtime_executors.get(stage)

        if executor is None:
            return {}
        else:
            executor.execute(obj, runtime_context, state=state)

            skeys = [obj.__xsimlab_state_keys__[k] for k in self.out_vars]
            sobj = obj.__xsimlab_state__
            return {k: sobj[k] for k in skeys if k in sobj}


def _process_cls_init(obj):
    """Set the following instance attributes with None or empty values
    (proper values will be set later at model creation):

    __xsimlab_model__ : obj
        :class:`Model` instance to which the process instance is attached.
    __xsimlab_name__ : str
        Name given for this process in the model.
    __xsimlab_state__ : dict or object
        Simulation active data store.
    __xsimlab_state_keys__ : dict
        Dictionary that maps variable names to their corresponding key
        (or list of keys for group variables) in the active store.
        Such keys consist of pairs like `('foo', 'bar')` where
        'foo' is the name of any process in the same model and 'bar' is
        the name of a variable declared in that process.
    __xsimlab_od_keys__ : dict
        Dictionary that maps variable names to the location of their target
        on-demand variable (or a list of locations for group variables).
        Locations are tuples like state keys.

    """
    obj.__xsimlab_model__ = None
    obj.__xsimlab_name__ = None
    obj.__xsimlab_state__ = None
    obj.__xsimlab_state_keys__ = {}
    obj.__xsimlab_od_keys__ = {}


class _ProcessBuilder:
    """Used to iteratively create a new process class from an existing
    "dataclass", i.e., a class decorated with ``attr.attrs``.

    The process class is a direct child of the given dataclass, with
    attributes (fields) redefined and properties created so that it
    can be used within a model.

    """

    _make_prop_funcs = {
        VarType.VARIABLE: _make_property_variable,
        VarType.INDEX: _make_property_variable,
        VarType.OBJECT: _make_property_variable,
        VarType.ON_DEMAND: _make_property_on_demand,
        VarType.FOREIGN: _make_property_variable,
        VarType.GROUP: _make_property_group,
    }

    def __init__(self, attr_cls):
        self._base_cls = attr_cls
        self._p_cls_dict = {}

    def _reset_attributes(self):
        new_attributes = OrderedDict()

        for k, attrib in attr.fields_dict(self._base_cls).items():
            new_attributes[k] = attr.attrib(
                metadata=attrib.metadata,
                validator=attrib.validator,
                converter=attrib.converter,
                default=attrib.default,
                init=False,
                repr=False,
            )

        return new_attributes

    def _make_process_subclass(self):
        p_cls = attr.make_class(
            self._base_cls.__name__,
            self._reset_attributes(),
            bases=(self._base_cls,),
            init=False,
            repr=False,
        )

        setattr(p_cls, "__init__", _process_cls_init)
        setattr(p_cls, "__repr__", repr_process)
        setattr(p_cls, "__xsimlab_process__", True)
        setattr(p_cls, "__xsimlab_executor__", _ProcessExecutor(p_cls))

        return p_cls

    def add_properties(self):
        for var_name, var in attr.fields_dict(self._base_cls).items():
            var_type = var.metadata.get("var_type")

            if var_type is not None:
                make_prop_func = self._make_prop_funcs[var_type]

                self._p_cls_dict[var_name] = make_prop_func(var)

    def render_docstrings(self):
        new_doc = add_attribute_section(self._base_cls)

        self._base_cls.__doc__ = new_doc

    def build_class(self):
        p_cls = self._make_process_subclass()

        # Attach properties (and docstrings)
        for name, value in self._p_cls_dict.items():
            setattr(p_cls, name, value)

        return p_cls


def process(maybe_cls=None, autodoc=True):
    """A class decorator that adds everything needed to use the class
    as a process.

    A process represents a logical unit in a computational model.

    A process class usually implements:

    - An interface as a set of variables defined as class attributes (see
      :func:`variable`, :func:`on_demand`, :func:`foreign` and :func:`group`).
      When the class is used within a :class:`Model` object, this decorator
      automatically adds properties to get/set values for these variables.

    - One or more methods among ``initialize()``, ``run_step()``,
      ``finalize_step()`` and ``finalize()``, which are called at different
      stages of a simulation and perform some computation based on the
      variables defined in the process interface.

    - Decorated methods to compute, validate or set a default value for one or
      more variables.

    Parameters
    ----------
    maybe_cls : class, optional
        Allows to apply this decorator to a class either as ``@process`` or
        ``@process(*args)``.
    autodoc : bool, optional
        (default: True) Automatically adds an attributes section to the
        docstring of the class to which the decorator is applied, using the
        metadata of each variable declared in the class.

    """

    def wrap(cls):
        attr_cls = attr.attrs(cls, repr=False)

        builder = _ProcessBuilder(attr_cls)

        builder.add_properties()

        if autodoc:
            builder.render_docstrings()

        setattr(attr_cls, "__xsimlab_cls__", builder.build_class())

        return attr_cls

    if maybe_cls is None:
        return wrap
    else:
        return wrap(maybe_cls)


def process_info(process, buf=None):
    """Concise summary of process variables and simulation stages
    implemented.

    Equivalent to __repr__ of a process but accepts either an instance
    or a class.

    Parameters
    ----------
    process : object or class
        Process class or object.
    buf : object, optional
        Writable buffer (default: sys.stdout).

    """
    if buf is None:  # pragma: no cover
        buf = sys.stdout

    process = get_process_obj(process)

    buf.write(repr_process(process))


def variable_info(process, var_name, buf=None):
    """Get detailed information about a variable.

    Parameters
    ----------
    process : object or class
        Process class or object.
    var_name : str
        Variable name.
    buf : object, optional
        Writable buffer (default: sys.stdout).

    """
    if buf is None:  # pragma: no cover
        buf = sys.stdout

    process = get_process_cls(process)
    var = variables_dict(process)[var_name]

    buf.write(var_details(var))
