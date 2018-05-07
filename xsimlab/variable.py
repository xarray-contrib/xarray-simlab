from enum import Enum
import itertools

import attr
from attr._make import _CountingAttr as CountingAttr_


class VarType(Enum):
    VARIABLE = 'variable'
    ON_DEMAND = 'on_demand'
    FOREIGN = 'foreign'
    GROUP = 'group'


class VarIntent(Enum):
    IN = 'in'
    OUT = 'out'
    INOUT = 'inout'


class _CountingAttr(CountingAttr_):
    """A hack to add a custom 'compute' decorator for on-request computation
    of on_demand variables.
    """

    def compute(self, method):
        self.metadata['compute'] = method

        return method


def _as_dim_tuple(dims):
    """Return a tuple from one or more combination(s) of dimension labels
    given in `dims` either as a tuple, str or list.

    Also ensure that the number of dimensions for each item in the
    sequence is unique, e.g., dims=[('x', 'y'), ('y', 'x')] is
    ambiguous and thus not allowed.

    """
    if not len(dims):
        dims = [()]
    elif isinstance(dims, str):
        dims = [(dims,)]
    elif isinstance(dims, list):
        dims = [tuple([d]) if isinstance(d, str) else tuple(d)
                for d in dims]
    else:
        dims = [dims]

    # check ndim uniqueness could be simpler but provides detailed error msg
    fget_ndim = lambda dims: len(dims)
    dims_sorted = sorted(dims, key=fget_ndim)
    ndim_groups = [list(g)
                   for _, g in itertools.groupby(dims_sorted, fget_ndim)]

    if len(ndim_groups) != len(dims):
        invalid_dims = [g for g in ndim_groups if len(g) > 1]
        invalid_msg = ' and '.join(
            ', '.join(str(d) for d in group) for group in invalid_dims
        )
        raise ValueError("the following combinations of dimension labels "
                         "are ambiguous for a variable: {}"
                         .format(invalid_msg))

    return tuple(dims)


def variable(dims=(), intent='in', group=None, default=attr.NOTHING,
             validator=None, description='', attrs=None):
    """Create a variable.

    Variables store useful metadata such as dimension labels, a short
    description, a default value, validators or custom,
    user-provided metadata.

    Variables are the primitives of the modeling framework, they
    define the interface of each process in a model.

    Variables should be declared exclusively as class attributes in
    process classes (i.e., classes decorated with :func:`process`).

    Parameters
    ----------
    dims : str or tuple or list, optional
        Dimension label(s) of the variable. An empty tuple
        corresponds to a scalar variable (default), a string or a 1-length
        tuple corresponds to a 1-d variable and a n-length tuple corresponds to
        a n-d variable. A list of str or tuple items may also be provided if
        the variable accepts different numbers of dimensions.
        This should not include a time dimension, which may always be added.
    intent : {'in', 'out', 'inout'}, optional
        Defines whether the variable is an input (i.e., the process needs the
        variable's value for its computation), an output (i.e., the process
        computes a value for the variable) or both an input/output (i.e., the
        process may update the value of the variable).
        (default: input).
    group : str, optional
        Variable group.
    default : any, optional
        Single default value for the variable, ignored when ``intent='out'``
        (default: NOTHING). A default value may also be set using a decorator.
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
    metadata = {'var_type': VarType.VARIABLE,
                'dims': _as_dim_tuple(dims),
                'intent': VarIntent(intent),
                'group': group,
                'attrs': attrs or {},
                'description': description}

    return attr.attrib(metadata=metadata, default=default, validator=validator,
                       init=False, cmp=False, repr=False)


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

    An on-demand variable is always an output variable (i.e., intent='out').

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
    metadata = {'var_type': VarType.ON_DEMAND,
                'dims': _as_dim_tuple(dims),
                'intent': VarIntent.OUT,
                'group': group,
                'attrs': attrs or {},
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


def foreign(other_process_cls, var_name, intent='in'):
    """Create a reference to a variable that is defined in another
    process class.

    Parameters
    ----------
    other_process_cls : class
        Class in which the variable is defined.
    var_name : str
        Name of the corresponding variable declared in `other_process_cls`.
    intent : {'in', 'out'}, optional
        Defines whether the foreign variable is an input (i.e., the process
        needs the variable's value for its computation), an output (i.e., the
        process computes a value for the variable).
        (default: input).

    See Also
    --------
    :func:`variable`

    Notes
    -----
    Unlike for :func:`variable`, ``intent='inout'`` is not supported
    here (i.e., the process may not update the value of a foreign
    variable) as it would result in ambiguous process ordering in a
    model.

    """
    if intent == 'inout':
        raise ValueError("intent='inout' is not supported for "
                         "foreign variables")

    description = ("Reference to variable {!r} "
                   "defined in class {!r}"
                   .format(var_name, other_process_cls.__name__))

    metadata = {'var_type': VarType.FOREIGN,
                'other_process_cls': other_process_cls,
                'var_name': var_name,
                'intent': VarIntent(intent),
                'description': description}

    return attr.attrib(metadata=metadata, init=False, cmp=False, repr=False)


def group(name):
    """Create a special variable which value returns an iterable of values of
    variables in a model that all belong to the same group.

    Access to the variable values is read-only (i.e., intent='in').

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
    description = ("Iterable of all variables that "
                   "belong to group {!r}".format(name))

    metadata = {'var_type': VarType.GROUP,
                'group': name,
                'intent': VarIntent.IN,
                'description': description}

    return attr.attrib(metadata=metadata, init=False, cmp=False, repr=False)
