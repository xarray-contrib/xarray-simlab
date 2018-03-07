# coding: utf-8
from enum import Enum

import attr
from attr._make import _CountingAttr as CountingAttr_


class AttrType(Enum):
    VARIABLE = 'variable'
    ON_DEMAND = 'on_demand'
    FOREIGN = 'foreign'
    GROUP = 'group'


class _CountingAttr(CountingAttr_):
    """A hack to add a custom 'compute' decorator for on-request computation
    of on_demand variables.
    """

    def compute(self, method):
        self.metadata['compute'] = method

        return method


def variable(dims=(), intent='input', group=None, default=attr.NOTHING,
             validator=None, description='', attrs=None):
    """Create a variable.

    Variables store useful metadata such as dimension labels, a short
    description, a default value, validators or custom,
    user-provided metadata.

    Variables are the primitives of the modeling framework, they
    define the interface of each process in a model.

    Variables should be declared exclusively as class attributes in
    process classes (i.e., classes decorated with `:func:process`).

    Parameters
    ----------
    dims : str or tuple or list, optional
        Dimension label(s) of the variable. An empty tuple
        corresponds to a scalar variable (default), a string or a 1-length
        tuple corresponds to a 1-d variable and a n-length tuple corresponds to
        a n-d variable. A list of str or tuple items may also be provided if
        the variable accepts different numbers of dimensions.
        This should not include a time dimension, which may always be added.
    intent : {'input', 'output'}, optional
        Defines whether the variable is an input (i.e., a value is needed for
        the process computation) or an output (i.e., the process provides a
        value for that variable).
        (default: 'input').
    group : str, optional
        Variable group.
    default : any, optional
        Single default value for the variable (default: None). It
        will be automatically broadcasted to all of its dimensions.
        Ignored when ``intent='output'``.
    validator : callable or list of callable, optional
        Function that is called at simulation initialization (and possibly at
        other times too) to check the value given for the variable.
        The function must accept three arguments:

        - the process instance (access other variables)
        - the variable object (access metadata)
        - a passed value (check input).

        The function is expected to throw an exception in case of invalid
        value.
        If a ``list`` is passed, its items are treated as validators and must
        all pass.
        The validator can also be set using decorator notation.
    description : str, optional
        Short description of the variable.
    attrs : dict, optional
        Dictionnary of additional metadata (e.g., standard_name,
        units, math_symbol...).

    """
    metadata = {'attr_type': AttrType.VARIABLE,
                'dims': dims,
                'intent': intent,
                'group': group,
                'attrs': attrs,
                'description': description}

    return attr.attrib(metadata=metadata, init=False, cmp=False, repr=False)


def on_demand(dims=(), group=None, description='', attrs=None):
    """Create a variable that is computed on demand.

    Instead of being computed systematically at every step of a simulation
    or at initialization, its value is only computed (or re-computed)
    each time when it is needed.

    Like other variables, such variable should be declared in a
    process class. Additionally, it requires its own method to compute
    its value, which must be defined in the same class and decorated
    (e.g., using `@myvar.compute` if the name of the variable is
    `myvar`).

    An on-demand variable never accepts an input value
    (i.e., intent='output'), and should never be set/updated (read-only).

    Its computation usually involves other variables, although this is
    not required.

    These variables may be useful, e.g., for model diagnostics.

    Parameters
    ----------
    dims : str or tuple or list, optional
        Dimension label(s) of the variable. An empty tuple
        corresponds to a scalar variable (default), a string or a 1-length
        tuple corresponds to a 1-d variable and a n-length tuple corresponds to
        a n-d variable. A list of str or tuple items may also be provided if
        the variable accepts different numbers of dimensions.
        This should not include a time dimension, which may always be added.
    group : str, optional
    description : str, optional
        Short description of the variable.
    attrs : dict, optional
        Dictionnary of additional metadata (e.g., standard_name,
        units, math_symbol...).

    See Also
    --------
    :func:`variable`

    """
    metadata = {'attr_type': AttrType.ON_DEMAND,
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
    """Create a reference to a variable that is defined in another
    process class.

    Parameters
    ----------
    other_process_cls : class
        Class in which the variable is defined.
    var_name : str
        Name of the corresponding variable declared in `other_process_cls`.
    intent : {'input', 'output'}, optional
        Defines whether the variable is an input (i.e., a value is needed for
        the process computation) or an output (i.e., the process provides a
        value for that variable).
        (default: 'input').

    See Also
    --------
    :func:`variable`

    """
    metadata = {'attr_type': AttrType.FOREIGN,
                'other_process_cls': other_process_cls,
                'var_name': var_name,
                'intent': intent}

    return attr.attrib(metadata=metadata, init=False, cmp=False, repr=False)


def group(name):
    """Create a special variable which value returns an iterable of values of
    variables in a model that all belong to the same group.

    Access to these variable values is read-only (i.e., intent='input').

    Good examples of using group variables are processes that
    aggregate (e.g., sum, product, mean) the values of variables that
    are defined in various other processes in a model.

    Parameters
    ----------
    group : str
        Name of the group.

    See Also
    --------
    :func:`variable`

    """
    metadata = {'attr_type': AttrType.GROUP,
                'group': group,
                'intent': 'input'}

    return attr.attrib(metadata=metadata, init=False, cmp=False, repr=False)
