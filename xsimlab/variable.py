from enum import Enum
import itertools
import warnings

import attr
from attr._make import _CountingAttr

from .utils import normalize_encoding


class VarType(Enum):
    VARIABLE = "variable"
    INDEX = "index"
    ON_DEMAND = "on_demand"
    OBJECT = "object"
    FOREIGN = "foreign"
    GROUP = "group"


class VarIntent(Enum):
    IN = "in"
    OUT = "out"
    INOUT = "inout"


def compute(self, method):
    """A decorator that, when applied to an on-demand variable, returns a
    value for that variable.

    """
    self.metadata["compute"] = method

    return method


# monkey patch, waiting for cleaner solution:
# https://github.com/python-attrs/attrs/issues/340
_CountingAttr.compute = compute


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
        dims = [tuple([d]) if isinstance(d, str) else tuple(d) for d in dims]
    else:
        dims = [dims]

    # check ndim uniqueness could be simpler but provides detailed error msg
    def fget_ndim(dims):
        return len(dims)

    dims_sorted = sorted(dims, key=fget_ndim)
    ndim_groups = [list(g) for _, g in itertools.groupby(dims_sorted, fget_ndim)]

    if len(ndim_groups) != len(dims):
        invalid_dims = [g for g in ndim_groups if len(g) > 1]
        invalid_msg = " and ".join(
            ", ".join(str(d) for d in group) for group in invalid_dims
        )
        raise ValueError(
            "the following combinations of dimension labels "
            f"are ambiguous for a variable: {invalid_msg}"
        )

    return tuple(dims)


def _as_group_tuple(groups, group):
    if groups is None:
        groups = []
    elif isinstance(groups, str):
        groups = [groups]
    else:
        groups = list(groups)

    if group is not None:
        warnings.warn(
            "Setting variable group using `group` is depreciated; use `groups`.",
            FutureWarning,
            stacklevel=2,
        )
        if group not in groups:
            groups.append(group)

    return tuple(groups)


def variable(
    dims=(),
    intent="in",
    group=None,
    groups=None,
    default=attr.NOTHING,
    validator=None,
    converter=None,
    static=False,
    description="",
    attrs=None,
    encoding=None,
):
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
    intent : {'in', 'out', 'inout'}, optional
        Defines whether the variable is an input (i.e., the process needs the
        variable's value for its computation), an output (i.e., the process
        computes a value for the variable) or both an input/output (i.e., the
        process may update the value of the variable).
        (default: input).
    group : str, optional
        Variable group (depreciated, use ``groups`` instead).
    groups : str or list, optional
        Variable group(s).
    default : any, optional
        Single default value for the variable, ignored when ``intent='out'``
        (default: NOTHING). A default value may also be set using a decorator.
    validator : callable or list of callable, optional
        Function that could be called before or during a simulation (or when
        creating a new process instance) to check the value given
        for the variable.
        The function must accept three arguments:

        - the process instance (useful for accessing the value of other
          variables in that process)
        - the variable object (useful for accessing the variable metadata)
        - the value to be validated.

        The function should throw an exception in case where an invalid value
        is given.
        If a ``list`` is passed, its items are all are treated as validators.
        The validator can also be set using decorator notation.
    converter : callable, optional
        Callable that is called when setting a value for this variable, and that
        converts the value to the desired format. The callable must accept
        one argument and return one value. The value is converted before being
        passed to the validator, if any.
    static : bool, optional
        If True, the value of the (input) variable must be set once
        before the simulation starts and cannot be further updated
        externally (default: False). Note that it doesn't prevent updating
        the value internally, i.e., from within the process class in which
        the variable is declared if ``intent`` is set to 'out' or 'inout',
        or from another process class (foreign variable).
    description : str, optional
        Short description of the variable.
    attrs : dict, optional
        Dictionnary of additional metadata (e.g., standard_name,
        units, math_symbol...).
    encoding : dict, optional
        Dictionary specifying how to encode this variable's data into a
        serialized format (i.e., as a zarr dataset). Currently used keys
        include 'dtype', 'compressor', 'fill_value', 'order', 'filters'
        and 'object_codec'. See :func:`zarr.creation.create` for details
        about these options. Other keys are ignored.

    See Also
    --------
    :func:`attr.ib`
    :mod:`attr.validators`

    """
    metadata = {
        "var_type": VarType.VARIABLE,
        "dims": _as_dim_tuple(dims),
        "intent": VarIntent(intent),
        "groups": _as_group_tuple(groups, group),
        "static": static,
        "attrs": attrs or {},
        "description": description,
        "encoding": normalize_encoding(encoding),
    }

    if VarIntent(intent) == VarIntent.OUT:
        _init = False
        _repr = False
    else:
        _init = True
        _repr = True

    return attr.attrib(
        metadata=metadata,
        default=default,
        validator=validator,
        converter=converter,
        init=_init,
        repr=_repr,
        kw_only=True,
    )


def index(dims, groups=None, description="", attrs=None, encoding=None):
    """Create a variable aimed at indexing data.

    The process class in which this variable is declared should set its value
    (i.e., intent='out') with an index object or an index-compatible object. For
    example, xarray may accept 1-d arrays, :class:`pandas.Index`,
    :class:`pandas.MultiIndex`, etc.

    As a simple example, index variable(s) should be used for setting coordinate
    labels along the dimension(s) of a cartesian grid.

    Parameters
    ----------
    dims : str or tuple or list, optional
        Dimension label(s) of the variable. A string or a 1-length
        tuple corresponds to a 1-d variable and a n-length tuple corresponds to
        a n-d variable. A list of str or tuple items may also be provided if
        the variable accepts different numbers of dimensions. Note that an index
        variable does not accept scalar values.
    groups : str or list, optional
        Variable group(s).
    description : str, optional
        Short description of the variable.
    attrs : dict, optional
        Dictionnary of additional metadata (e.g., standard_name,
        units, math_symbol...).
    encoding : dict, optional
        Dictionary specifying how to encode this variable's data into a
        serialized format (i.e., as a zarr dataset). Currently used keys
        include 'dtype', 'compressor', 'fill_value', 'order', 'filters'
        and 'object_codec'. See :func:`zarr.creation.create` for details
        about these options. Other keys are ignored.

    See Also
    --------
    :func:`variable`

    """
    dims = _as_dim_tuple(dims)

    if tuple() in dims:
        raise ValueError("An index variable does not accept scalar values")

    metadata = {
        "var_type": VarType.INDEX,
        "dims": dims,
        "intent": VarIntent.OUT,
        "groups": _as_group_tuple(groups, None),
        "attrs": attrs or {},
        "description": description,
        "encoding": normalize_encoding(encoding),
    }

    return attr.attrib(metadata=metadata, init=False, repr=False)


def on_demand(
    dims=(), group=None, groups=None, description="", attrs=None, encoding=None
):
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
    group : str, optional
        Variable group (depreciated, use ``groups`` instead).
    groups : str or list, optional
        Variable group(s).
    description : str, optional
        Short description of the variable.
    attrs : dict, optional
        Dictionnary of additional metadata (e.g., standard_name,
        units, math_symbol...).
    encoding : dict, optional
        Dictionary specifying how to encode this variable's data into a
        serialized format (i.e., as a zarr dataset). Currently used keys
        include 'dtype', 'compressor', 'fill_value', 'order', 'filters'
        and 'object_codec'. See :func:`zarr.creation.create` for details
        about these options. Other keys are ignored.

    See Also
    --------
    :func:`variable`

    """
    metadata = {
        "var_type": VarType.ON_DEMAND,
        "dims": _as_dim_tuple(dims),
        "intent": VarIntent.OUT,
        "groups": _as_group_tuple(groups, group),
        "attrs": attrs or {},
        "description": description,
        "encoding": normalize_encoding(encoding),
    }

    return attr.attrib(metadata=metadata, init=False, repr=False)


def any_object(groups=None, description="", attrs=None):
    """Create a variable used to hold any arbitrary object that needs to be shared
    with other process classes.

    Use this instead of :func:`~xsimlab.variable` if you need to pass anything
    other than scalar/array values to other processes (e.g., a callable or an
    instance of a custom class that have no array-like interface).

    Unlike regular variables, 'object' variables are not intended to be used as
    model inputs or outputs. Additionally, a value must be set in the class
    within which this variable is declared (i.e., intent='out').

    Parameters
    ----------
    groups : str or list, optional
        Variable group(s).
    description : str, optional
        Short description of the variable.
    attrs : dict, optional
        Dictionnary of additional metadata.

    """
    metadata = {
        "var_type": VarType.OBJECT,
        "intent": VarIntent.OUT,
        "groups": _as_group_tuple(groups, None),
        "attrs": attrs or {},
        "description": description,
    }

    return attr.attrib(metadata=metadata, init=False, repr=False)


def foreign(other_process_cls, var_name, intent="in"):
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
    if intent == "inout":
        raise ValueError("intent='inout' is not supported for foreign variables")

    ref_var = attr.fields_dict(other_process_cls)[var_name]

    metadata = {
        "var_type": VarType.FOREIGN,
        "other_process_cls": other_process_cls,
        "var_name": var_name,
        "intent": VarIntent(intent),
    }

    for meta_key in ["description", "dims", "attrs", "encoding"]:
        ref_value = ref_var.metadata.get(meta_key)
        if ref_value is not None:
            metadata[meta_key] = ref_value

    if VarIntent(intent) == VarIntent.OUT:
        _init = False
        _repr = False
    else:
        _init = True
        _repr = True

    return attr.attrib(metadata=metadata, init=_init, repr=_repr, kw_only=True)


def group(name):
    """Create a special variable which value returns an iterable of values of
    variables in a model that all belong to the same group.

    Access to the variable values is read-only (i.e., intent='in').

    Good examples of using group variables are processes that
    aggregate (e.g., sum, product, mean) the values of variables that
    are defined in various other processes in a model.

    Parameters
    ----------
    name : str
        Name of the group.

    See Also
    --------
    :func:`variable`

    """
    description = f"Iterable of all variables that belong to group {name!r}"

    metadata = {
        "var_type": VarType.GROUP,
        "group": name,
        "intent": VarIntent.IN,
        "description": description,
    }

    return attr.attrib(
        metadata=metadata, init=True, repr=True, default=tuple(), kw_only=True
    )
