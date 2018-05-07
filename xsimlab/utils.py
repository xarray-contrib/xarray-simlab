"""
Internal utilties; not for external use.

"""
import threading
from collections import (Mapping, KeysView, ItemsView, ValuesView,
                         OrderedDict)
from contextlib import suppress
from importlib import import_module

from attr import fields_dict


def variables_dict(process_cls):
    """Get all xsimlab variables declared in a process.

    Exclude attr.Attribute objects that are not xsimlab-specific.
    """
    return OrderedDict((k, v)
                       for k, v in fields_dict(process_cls).items()
                       if 'var_type' in v.metadata)


def has_method(obj, meth):
    return callable(getattr(obj, meth, False))


def maybe_to_list(obj):
    return obj if isinstance(obj, list) else [obj]


def import_required(mod_name, error_msg):
    """Attempt to import a required dependency.
    Raises a RuntimeError if the requested module is not available.
    """
    try:
        return import_module(mod_name)
    except ImportError:
        raise RuntimeError(error_msg)


class AttrMapping(object):
    """A class similar to `collections.abc.Mapping`,
    which also allows getting keys with attribute access.

    This class doesn't use `abc.ABCMeta` so it can be
    inherited in classes that use other metaclasses.

    Part of the code below is copied and modified from:

    - xarray 0.9.3 (Copyright 2014-2017, xarray Developers)
      Licensed under the Apache License, Version 2.0
      https://github.com/pydata/xarray

    - python standard library (abc.collections module)
      Copyright 2001-2017 Python Software Foundation; All Rights Reserved
      PSF License
      https://www.python.org/

    """
    # TODO: use abc.ABCMeta now that metaclasses are not used anymore?
    _initialized = False

    def __init__(self, mapping=None):
        self._mapping = mapping if mapping is not None else {}

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def __getitem__(self, key):
        return self._mapping[key]

    def get(self, key, default=None):
        'D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.'
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def keys(self):
        "D.keys() -> a set-like object providing a view on D's keys"
        return KeysView(self)

    def items(self):
        "D.items() -> a set-like object providing a view on D's items"
        return ItemsView(self)

    def values(self):
        "D.values() -> an object providing a view on D's values"
        return ValuesView(self)

    def __eq__(self, other):
        if not isinstance(other, (Mapping, AttrMapping)):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    __reversed__ = None

    def __hash__(self):
        return hash(frozenset(self._mapping.items()))

    def __getattr__(self, name):
        if name != '__setstate__':
            # this avoids an infinite loop when pickle looks for the
            # __setstate__ attribute before the object is initialized
            with suppress(KeyError):
                return self._mapping[name]
        raise AttributeError("%r object has no attribute %r" %
                             (type(self).__name__, name))

    def __setattr__(self, name, value):
        if self._initialized and name in self._mapping:
            raise AttributeError(
                "cannot override attribute %r of this %r object"
                % (name, type(self).__name__)
            )
        object.__setattr__(self, name, value)

    def __dir__(self):
        """Provide method name lookup and completion. Only provide 'public'
        methods.
        """
        extra_attrs = list(self._mapping)
        return sorted(set(dir(type(self)) + extra_attrs))


class ContextMixin(object):
    """Functionality for objects that put themselves in a context using
    the `with` statement.

    Part of the code below is copied and modified from:

    - pymc3 3.1 (Copyright (c) 2009-2013 The PyMC developers)
      Licensed under the Apache License, Version 2.0
      https://github.com/pymc-devs/pymc3

    """
    contexts = threading.local()

    def __enter__(self):
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        # no race-condition here, cls.contexts is a thread-local object
        # be sure not to override contexts in a subclass however!
        if not hasattr(cls.contexts, 'stack'):
            cls.contexts.stack = []
        return cls.contexts.stack

    @classmethod
    def get_context(cls):
        """Return the deepest context on the stack."""
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            raise TypeError("No context on context stack")
