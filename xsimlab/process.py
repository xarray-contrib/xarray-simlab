from inspect import isclass
import sys

import attr

from .variable import VarIntent, VarType
from .formatting import repr_process, var_details
from .utils import variables_dict


class NotAProcessClassError(ValueError):
    """
    A non-``xsimlab.process`` class has been passed into a ``xsimlab``
    function.

    """
    pass


def ensure_process_decorated(cls):
    if not getattr(cls, "__xsimlab_process__", False):
        raise NotAProcessClassError("{cls!r} is not a "
                                    "process-decorated class.".format(cls=cls))


def get_process_cls(obj_or_cls):
    if not isclass(obj_or_cls):
        cls = type(obj_or_cls)
    else:
        cls = obj_or_cls

    ensure_process_decorated(cls)

    return cls


def get_process_obj(obj_or_cls):
    if isclass(obj_or_cls):
        cls = obj_or_cls
        obj = cls()
    else:
        cls = type(obj_or_cls)
        obj = obj_or_cls

    ensure_process_decorated(cls)

    return obj


def filter_variables(process, var_type=None, intent=None, group=None,
                     func=None):
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
        vars = {k: v for k, v in vars.items()
                if v.metadata.get('var_type') == VarType(var_type)}

    if intent is not None:
        vars = {k: v for k, v in vars.items()
                if v.metadata.get('intent') == VarIntent(intent)}

    if group is not None:
        vars = {k: v for k, v in vars.items()
                if v.metadata.get('group') == group}

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

    while target_var.metadata['var_type'] == VarType.FOREIGN:
        visited.append((target_process_cls, target_var))

        target_process_cls = target_var.metadata['other_process_cls']
        var_name = target_var.metadata['var_name']
        target_var = filter_variables(target_process_cls)[var_name]

        # TODO: maybe remove this? not even sure such a cycle may happen
        # unless we allow later providing other values than classes as first
        # argument of `foreign`
        if (target_process_cls, target_var) in visited:  # pragma: no cover
            cycle = '->'.join(['{}.{}'.format(cls.__name__, var.name)
                               if cls is not None else var.name
                               for cls, var in visited])

            raise RuntimeError("Cycle detected in process dependencies: {}"
                               .format(cycle))

    return target_process_cls, target_var


def _attrify_class(cls):
    """Return a `cls` after having passed through :func:`attr.attrs`.

    This pulls out and converts `attr.ib` declared as class attributes
    into :class:`attr.Attribute` objects and it also adds
    dunder-methods such as `__init__`.

    The following instance attributes are also defined with None or
    empty values (proper values will be set later at model creation):

    __xsimlab_model__ : obj
        :class:`Model` instance to which the process instance is attached.
    __xsimlab_name__ : str
        Name given for this process in the model.
    __xsimlab_store__ : dict or object
        Simulation data store.
    __xsimlab_store_keys__ : dict
        Dictionary that maps variable names to their corresponding key
        (or list of keys for group variables) in the store.
        Such keys consist of pairs like `('foo', 'bar')` where
        'foo' is the name of any process in the same model and 'bar' is
        the name of a variable declared in that process.
    __xsimlab_od_keys__ : dict
        Dictionary that maps variable names to the location of their target
        on-demand variable (or a list of locations for group variables).
        Locations are tuples like store keys.

    """
    def init_process(self):
        self.__xsimlab_model__ = None
        self.__xsimlab_name__ = None
        self.__xsimlab_store__ = None
        self.__xsimlab_store_keys__ = {}
        self.__xsimlab_od_keys__ = {}

    setattr(cls, '__attrs_post_init__', init_process)

    return attr.attrs(cls)


def _make_property_variable(var):
    """Create a property for a variable or a foreign variable (after
    some sanity checks).

    The property get/set functions either read/write values from/to
    the simulation data store or get (and trigger computation of) the
    value of an on-demand variable.

    The property is read-only if `var` is declared as input.

    """
    var_name = var.name

    def get_from_store(self):
        key = self.__xsimlab_store_keys__[var_name]
        return self.__xsimlab_store__[key]

    def get_on_demand(self):
        p_name, v_name = self.__xsimlab_od_keys__[var_name]
        p_obj = self.__xsimlab_model__._processes[p_name]
        return getattr(p_obj, v_name)

    def put_in_store(self, value):
        key = self.__xsimlab_store_keys__[var_name]
        self.__xsimlab_store__[key] = value

    target_process_cls, target_var = get_target_variable(var)

    var_type = var.metadata['var_type']
    target_type = target_var.metadata['var_type']
    var_intent = var.metadata['intent']
    target_intent = target_var.metadata['intent']

    var_doc = var_details(var)

    if target_process_cls is not None:
        target_str = '.'.join([target_process_cls.__name__, target_var.name])
    else:
        target_str = target_var.name

    if target_type == VarType.GROUP:
        raise ValueError("Variable {var!r} links to group variable {target!r}, "
                         "which is not supported. Declare {var!r} as a group "
                         "variable instead."
                         .format(var=var.name, target=target_str))

    elif (var_type == VarType.FOREIGN and
          var_intent == VarIntent.OUT and target_intent == VarIntent.OUT):
        raise ValueError("Conflict between foreign variable {!r} and its "
                         "target variable {!r}, both have intent='out'."
                         .format(var.name, target_str))

    elif target_type == VarType.ON_DEMAND:
        return property(fget=get_on_demand, doc=var_doc)

    elif var_intent == VarIntent.IN:
        return property(fget=get_from_store, doc=var_doc)

    else:
        return property(fget=get_from_store, fset=put_in_store, doc=var_doc)


def _make_property_on_demand(var):
    """Create a read-only property for an on-demand variable.

    This property is a simple wrapper around the variable's compute method.

    """
    if 'compute' not in var.metadata:
        raise KeyError("No compute method found for on_demand variable "
                       "'{name}'. A method decorated with '@{name}.compute' "
                       "is required in the class definition."
                       .format(name=var.name))

    get_method = var.metadata['compute']

    return property(fget=get_method, doc=var_details(var))


def _make_property_group(var):
    """Create a read-only property for a group variable."""

    var_name = var.name

    def getter_store_or_on_demand(self):
        model = self.__xsimlab_model__
        store_keys = self.__xsimlab_store_keys__.get(var_name, [])
        od_keys = self.__xsimlab_od_keys__.get(var_name, [])

        for key in store_keys:
            yield self.__xsimlab_store__[key]

        for key in od_keys:
            p_name, v_name = key
            p_obj = model._processes[p_name]
            yield getattr(p_obj, v_name)

    return property(fget=getter_store_or_on_demand, doc=var_details(var))


class _ProcessBuilder(object):
    """Used to iteratively create a new process class.

    The original class must be already "attr-yfied", i.e., it must
    correspond to a class returned by `attr.attrs`.

    """
    _make_prop_funcs = {
        VarType.VARIABLE: _make_property_variable,
        VarType.ON_DEMAND: _make_property_on_demand,
        VarType.FOREIGN: _make_property_variable,
        VarType.GROUP: _make_property_group
    }

    def __init__(self, attr_cls):
        self._cls = attr_cls
        self._cls.__xsimlab_process__ = True
        self._cls_dict = {}

    def add_properties(self, var_type):
        make_prop_func = self._make_prop_funcs[var_type]

        for var_name, var in filter_variables(self._cls, var_type).items():
            self._cls_dict[var_name] = make_prop_func(var)

    def add_repr(self):
        self._cls_dict['__repr__'] = repr_process

    def render_docstrings(self):
        # self._cls_dict['__doc__'] = "Process-ified class."
        raise NotImplementedError("autodoc is not yet implemented.")

    def build_class(self):
        cls = self._cls

        # Attach properties (and docstrings)
        for name, value in self._cls_dict.items():
            setattr(cls, name, value)

        return cls


def process(maybe_cls=None, autodoc=False):
    """A class decorator that adds everything needed to use the class
    as a process.

    A process represents a logical unit in a computational model.

    A process class usually implements:

    - An interface as a set of variables defined as class attributes
      (see :func:`variable`, :func:`on_demand`, :func:`foreign` and
      :func:`group`). This decorator automatically adds properties to
      get/set values for these variables.

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
        If True, render the docstrings template and fill the
        corresponding sections with variable metadata (default: False).

    """
    def wrap(cls):
        attr_cls = _attrify_class(cls)

        builder = _ProcessBuilder(attr_cls)

        for var_type in VarType:
            builder.add_properties(var_type)

        if autodoc:
            builder.render_docstrings()

        builder.add_repr()

        return builder.build_class()

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
