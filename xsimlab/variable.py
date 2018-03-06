# coding: utf-8
from enum import Enum

import attr
from attr._make import _CountingAttr as CountingAttr_


class AttrType(Enum):
    VARIABLE = 'variable'
    DERIVED = 'derived'
    FOREIGN = 'foreign'
    GROUP = 'group'


class _CountingAttr(CountingAttr_):
    """A hack to add a custom 'compute' decorator for on-request computation
    of derived variables.
    """

    def compute(self, method):
        self.metadata['compute'] = method

        return method


def variable(dims=(), intent='input', group=None, attrs=None, description=''):
    metadata = {'attr_type': AttrType.VARIABLE,
                'dims': dims,
                'intent': intent,
                'group': group,
                'attrs': attrs,
                'description': description}

    return attr.attrib(metadata=metadata, init=False, cmp=False, repr=False)


def derived(dims=(), group=None, attrs=None, description=''):
    """A derived variable that is computed on request.

    Instead of being computed systematically at every step of a simulation
    or at initialization, its value is only computed (or re-computed)
    each time when it is needed.

    Defined in a process class, such variable requires a method
    to compute its value, which must be defined in the same class and
    decorated (e.g., using `@myvar.compute` if the name of the variable
    is `myvar`).

    A derived variable never accepts an input value (i.e., intent='output'),
    and should never be set/updated.

    Its computation usually involves other variables (hence the
    term `derived`), although this is not required.

    These variables may be useful, e.g., for model diagnostics.

    """
    metadata = {'attr_type': AttrType.DERIVED,
                'dims': dims,
                'intent': 'output',
                'group': group,
                'attrs': attrs,
                'description': description}

    return _CountingAttr(
        default=attr.NOTHING,
        validator=None,
        repr=False,
        cmp=False,
        hash=None,
        init=False,
        converter=None,
        metadata=metadata,
        type=None,
    )


def foreign(other_process_cls, var_name, intent='input'):
    metadata = {'attr_type': AttrType.FOREIGN,
                'other_process_cls': other_process_cls,
                'var_name': var_name,
                'intent': intent}

    return attr.attrib(metadata=metadata, init=False, cmp=False, repr=False)


def group(name):
    metadata = {'attr_type': AttrType.GROUP,
                'group': group}

    return attr.attrib(metadata=metadata, init=False, cmp=False, repr=False)
