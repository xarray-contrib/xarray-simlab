from inspect import isclass

import attr

from .variable import AttrType


def get_variables(process, attr_type=None):
    if not isclass(process):
        process = process.__class__

    if attr_type is None:
        return [field for field in attr.fields(process)]

    else:
        return [field for field in attr.fields(process)
                if field.metadata['attr_type'] == AttrType(attr_type)]


def _attrify_class(cls):
    """Return a class as if the input class `cls` was
    decorated with `attr.s`.

    `attr.s` turns `attr.ib` (or derived) class attributes
    into fields and adds dunder-methods such as `__init__`.

    """
    def post_init(self):
        """Init instance attributes that will be used
        during model creation or simulation runtime.
        """
        self.name = None
        self.__xsimlab_store__ = None
        self.__xsimlab_store_keys__ = {}

    setattr(cls, '__attrs_post_init__', post_init)

    return attr.attrs(cls)


def _make_property_variable(var):
    """Create a property for a variable."""

    var_name = var.name

    def getter(self):
        return self.__xsimlab_store__[(self.name, var_name)]

    def setter(self, value):
        self.__xsimlab_store__[(self.name, var_name)] = value

    return property(fget=getter, fset=setter)


def _make_property_derived(var):
    """Create a read-only property for a derived variable."""

    var_name = var.name

    if 'compute' not in var.metadata:
        raise KeyError("no compute method found for derived variable '{name}': "
                       "a method decorated with '@{name}.compute' is required "
                       "in the class definition.".format(name=var.name))

    func_compute_value = var.metadata['compute']

    def getter(self):
        value = func_compute_value(self)
        self.__xsimlab_store__[(self.name, var_name)] = value
        return value

    return property(fget=getter)


def _make_property_foreign(var):
    """Create a property for a foreign variable."""

    var_name = var.name

    def getter(self):
        key = self.__xsimlab_store_keys__[var_name]
        return self.__xsimlab_store__[key]

    def setter(self, value):
        key = self.__xsimlab_store_keys__[var_name]
        self.__xsimlab_store__[key] = value

    return property(fget=getter, fset=setter)


def _make_property_group(var):
    """Create a read-only property for a group variable."""

    var_name = var.name

    def getter(self):
        for key in self.__xsimlab_store_keys__[var_name]:
            yield self.__xsimlab_store__[key]

    return property(fget=getter)


class _ProcessBuilder(object):
    """Used to iteratively create a new process class.

    The original class must be already "attr-yfied", i.e.,
    it must correspond to a class returned by `attr.attrs`.

    """
    _make_prop_funcs = {
        AttrType.VARIABLE: _make_property_variable,
        AttrType.DERIVED: _make_property_derived,
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
    """Decorator to define a class as a process."""

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
