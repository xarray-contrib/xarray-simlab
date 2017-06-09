"""
Internal utilties; not for external use.

"""
from collections import (Mapping, MutableMapping, OrderedDict,
                         KeysView, ItemsView, ValuesView)
from functools import wraps
from contextlib import suppress
from importlib import import_module


def _get_args_not_none(arg_names, arg_vals):
    return tuple((n for n, v in zip(arg_names, arg_vals) if v is not None))


def import_required(mod_name, error_msg):
    """Attempt to import a required dependency.
    Raises a RuntimeError if the requested module is not available.
    """
    try:
        return import_module(mod_name)
    except ImportError:
        raise RuntimeError(error_msg)


class combomethod(object):
    def __init__(self, method):
        self.method = method

    def __get__(self, obj=None, objtype=None):
        @wraps(self.method)
        def _wrapper(*args, **kwargs):
            if obj is not None:
                return self.method(obj, *args, **kwargs)
            else:
                return self.method(objtype, *args, **kwargs)
        return _wrapper


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
        return hash(tuple(self._mapping.items()))

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


class SingleSlotPickleMixin(object):
    """Mixin class to add the ability to pickle objects whose state is defined
    by a single __slots__ attribute. Only necessary under Python 2.
    """
    def __getstate__(self):
        return getattr(self, self.__slots__[0])

    def __setstate__(self, state):
        setattr(self, self.__slots__[0], state)


class Frozen(Mapping, SingleSlotPickleMixin):
    """Wrapper around an object implementing the mapping interface to make it
    immutable. If you really want to modify the mapping, the mutable version is
    saved under the `mapping` attribute.
    """
    __slots__ = ['mapping']

    def __init__(self, mapping):
        self.mapping = mapping

    def __getitem__(self, key):
        return self.mapping[key]

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __contains__(self, key):
        return key in self.mapping

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.mapping)


def FrozenOrderedDict(*args, **kwargs):
    return Frozen(OrderedDict(*args, **kwargs))


class SortedKeysDict(MutableMapping, SingleSlotPickleMixin):
    """An wrapper for dictionary-like objects that always iterates over its
    items in sorted order by key but is otherwise equivalent to the underlying
    mapping.
    """
    __slots__ = ['mapping']

    def __init__(self, mapping=None):
        self.mapping = {} if mapping is None else mapping

    def __getitem__(self, key):
        return self.mapping[key]

    def __setitem__(self, key, value):
        self.mapping[key] = value

    def __delitem__(self, key):
        del self.mapping[key]

    def __iter__(self):
        return iter(sorted(self.mapping))

    def __len__(self):
        return len(self.mapping)

    def __contains__(self, key):
        return key in self.mapping

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.mapping)

    def copy(self):
        return type(self)(self.mapping.copy())
