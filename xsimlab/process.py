from inspect import isclass

import attr

from .variable import AttrType


def get_variables(process, attr_type=None):
    """Helper function that returns variables declared in a process.

    Useful when one wants to access the variables metadata.

    Parameters
    ----------
    process : object or class
        Process class or object.
    attr_type : {'variable', 'on_demand', 'foreign', 'group'}, optional
        Only return variables of a given kind (by default, return all
        variables).

    Returns
    -------
    attributes : list
        A list of :class:`attr.Attribute` objects.

    """
    if not isclass(process):
        process = process.__class__

    if attr_type is None:
        return [field for field in attr.fields(process)]

    else:
        return [field for field in attr.fields(process)
                if field.metadata.get('attr_type') == AttrType(attr_type)]


def _attrify_class(cls):
    """Return a class as if the input class `cls` was decorated with
    `attr.s`.

    `attr.s` turns `attr.ib` (or derived) class attributes into fields
    and adds dunder-methods such as `__init__`.

    """
    def add_obj_attrs(self):
        """Add instance attributes that are needed later during model creation
        or simulation runtime.

        """
        self.__xsimlab_name__ = None
        self.__xsimlab_model__ = None
        self.__xsimlab_store__ = None
        self.__xsimlab_foreign__ = {}

    setattr(cls, '__attrs_post_init__', add_obj_attrs)

    return attr.attrs(cls)


def _make_property_variable(var):
    """Create a property for a variable."""

    var_name = var.name

    def getter(self):
        return self.__xsimlab_store__[(self.__xsimlab_name__, var_name)]

    def setter(self, value):
        self.__xsimlab_store__[(self.__xsimlab_name__, var_name)] = value

    return property(fget=getter, fset=setter)


def _make_property_on_demand(var):
    """Create a read-only property for an on-demand variable."""

    if 'compute' not in var.metadata:
        raise KeyError("no compute method found for on_demand variable "
                       "'{name}': a method decorated with '@{name}.compute' "
                       "is required in the class definition."
                       .format(name=var.name))

    getter = var.metadata['compute']

    return property(fget=getter)


def _make_property_foreign(var):
    """Create a property for a foreign variable."""

    var_name = var.name

    def getter(self):
        o_proc_name, o_var_name = self.__xsimlab_foreign__[var_name]
        try:
            return self.__xsimlab_store__[(o_proc_name, o_var_name)]
        except KeyError:
            # values of on_demand variables are not in the store
            model = self.__xsimlab_model__
            return getattr(model._processes[o_proc_name], o_var_name)

    def setter(self, value):
        # no fastpath access (prevent setting read-only variables in store)
        o_proc_name, o_var_name = self.__xsimlab_foreign__[var_name]
        model = self.__xsimlab_model__
        return setattr(model._processes[o_proc_name], o_var_name, value)

    if var.metadata['intent'] == 'input':
        return property(fget=getter)
    else:
        return property(fget=getter, fset=setter)


def _make_property_group(var):
    """Create a read-only property for a group variable."""

    var_name = var.name

    def getter(self):
        for o_proc_name, o_var_name in self.__xsimlab_foreign__[var_name]:
            try:
                yield self.__xsimlab_store__[(o_proc_name, o_var_name)]
            except KeyError:
                # values of on_demand variables are not in the store
                model = self.__xsimlab_model__
                return getattr(model._processes[o_proc_name], o_var_name)

    return property(fget=getter)


class _ProcessBuilder(object):
    """Used to iteratively create a new process class.

    The original class must be already "attr-yfied", i.e., it must
    correspond to a class returned by `attr.attrs`.

    """
    _make_prop_funcs = {
        AttrType.VARIABLE: _make_property_variable,
        AttrType.ON_DEMAND: _make_property_on_demand,
        AttrType.FOREIGN: _make_property_foreign,
        AttrType.GROUP: _make_property_group
    }

    def __init__(self, attr_cls):
        self._cls = attr_cls
        self._cls_dict = {}

    def add_properties(self, attr_type):
        make_prop_func = self._make_prop_funcs[attr_type]

        for var in get_variables(self._cls, attr_type):
            self._cls_dict[var.name] = make_prop_func(var)

    def render_docstrings(self):
        self._cls_dict['__doc__'] = "Process-ified class."

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

    - One or more methods among `initialize()`, `run_step()`,
      `finalize_step()` and `finalize()`, which are called at different
      stages of a simulation and perform some computation based on the
      variables defined in the process interface.

    - Decorated methods to compute, validate or set a default value for one or
      more variables.

    Parameters
    ----------
    maybe_cls : class, optional
        Allows to apply this decorator to a class either as `@process` or
        `@process(*args)`.
    autodoc : bool, optional
        If True, render the docstrings given as a template and fill the
        corresponding sections with metadata found in the class
        (default: False).

    """
    def wrap(cls):
        attr_cls = _attrify_class(cls)

        builder = _ProcessBuilder(attr_cls)

        for attr_type in AttrType:
            builder.add_properties(attr_type)

        if autodoc:
            builder.render_docstrings()

        return builder.build_class()

    if maybe_cls is None:
        return wrap
    else:
        return wrap(maybe_cls)
