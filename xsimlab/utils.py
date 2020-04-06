"""
Internal utilties; not for external use.

"""
from collections import OrderedDict
from collections.abc import KeysView, ItemsView, ValuesView
from contextlib import suppress
from importlib import import_module
from typing import Iterator, Mapping, TypeVar

from attr import fields_dict


K = TypeVar("K")
V = TypeVar("V")


def variables_dict(process_cls):
    """Get all xsimlab variables declared in a process.

    Exclude attr.Attribute objects that are not xsimlab-specific.
    """
    return OrderedDict(
        (k, v) for k, v in fields_dict(process_cls).items() if "var_type" in v.metadata
    )


def has_method(obj_or_cls, meth):
    return callable(getattr(obj_or_cls, meth, False))


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


def normalize_encoding(encoding, extra_keys=None):
    used_keys = [
        "dtype",
        "compressor",
        "fill_value",
        "order",
        "filters",
        "object_codec",
    ]

    if extra_keys is not None:
        used_keys += extra_keys

    if encoding is None:
        return {}
    else:
        return {k: v for k, v in encoding.items() if k in used_keys}


def get_batch_size(xr_dataset, batch_dim):
    if batch_dim is not None:
        if batch_dim not in xr_dataset.dims:
            raise KeyError(f"Batch dimension {batch_dim} missing in input dataset")

        return xr_dataset.dims[batch_dim]

    else:
        return -1


class AttrMapping:
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
        "D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None."
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
        if name != "__setstate__":
            # this avoids an infinite loop when pickle looks for the
            # __setstate__ attribute before the object is initialized
            with suppress(KeyError):
                return self._mapping[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __setattr__(self, name, value):
        if self._initialized and name in self._mapping:
            raise AttributeError(
                f"cannot override attribute {name!r} of "
                f"this {type(self).__name__!r} object"
            )
        object.__setattr__(self, name, value)

    def __dir__(self):
        """Provide method name lookup and completion. Only provide 'public'
        methods.
        """
        extra_attrs = list(self._mapping)
        return sorted(set(dir(type(self)) + extra_attrs))


class Frozen(Mapping[K, V]):
    """Wrapper around an object implementing the mapping interface to make it
    immutable. If you really want to modify the mapping, the mutable version is
    saved under the `mapping` attribute.
    """

    __slots__ = ("mapping",)

    def __init__(self, mapping: Mapping[K, V]):
        self.mapping = mapping

    def __getitem__(self, key: K) -> V:
        return self.mapping[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __contains__(self, key: object) -> bool:
        return key in self.mapping

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.mapping!r})"
