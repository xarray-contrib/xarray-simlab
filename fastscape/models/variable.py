# coding: utf-8
import itertools

from xarray.core.variable import as_variable


EMPTY_VALUES = (None, '', [], (), {})


class AbstractVariable(object):
    """Abstract class for all variables.

    This class aims at providing a common parent class
    for all regular, diagnostic, foreign and undefined variables.

    """
    def __init__(self, provided=False, description='', attrs=None):
        self.provided = provided
        self.description = description
        self.attrs = attrs or {}

    def __repr__(self):
        return "<fastscape.models.%s>" % type(self).__name__


class Variable(AbstractVariable):
    """Base class that represents a variable in a process or a model.

    `Variable` objects store useful metadata such as dimension labels,
    a short description, a default value or other user-provided metadata.

    Variables allow to convert any given value to a `xarray.Variable` object
    after having perfomed some sanity checks.

    In processes, variables are instantiated as class attributes. They
    represent fundamental elements of a process interface (see
    :class:`Process`) and by extension a model interface.
    Some attributes such as `provided` and `optional` also contribute to
    the definition of the interface.

    """
    default_validators = []  # Default set of validators

    def __init__(self, allowed_dims, provided=False, optional=False,
                 default_value=None, validators=(), description='',
                 attrs=None):
        """
        Parameters
        ----------
        allowed_dims : str or tuple or list
            Dimension label(s) allowed for the variable. An empty tuple
            corresponds to a scalar variable, a string or a 1-length tuple
            corresponds to a 1-d variable and a n-length tuple corresponds to a
            n-d variable. A list of str or tuple items may also be provided if
            the variable accepts different numbers of dimensions.
            This should not include a time dimension, which is always allowed.
        provided : bool, optional
            Defines whether a value for the variable is required (False)
            or provided (True) by the process in which it is defined
            (default: False).
            If `provided=True`, then the variable in a process/model won't
            be considered as an input of that process/model.
        optional : bool, optional
            If True, a value may not be required for the variable
            (default: False). Ignored when `provided` is True.
        default_value : any, optional
            Single default value for the variable (default: None). It
            will be automatically broadcasted to all of its dimensions.
            Ignored when `provided` is True.
        validators : tuple or list, optional
            A list of callables that take an `xarray.Variable` object and
            raises a `ValidationError` if it doesn’t meet some criteria.
            It may be useful for custom, advanced validation that
            can be reused for different variables.
        description : str, optional
            Short description of the variable (ideally one-line).
        attrs : dict, optional
            Dictionnary of additional metadata (e.g., standard_name,
            units, math_symbol...).

        """
        super(Variable, self).__init__(
            provided=provided, description=description, attrs=attrs
        )

        if not len(allowed_dims):
            allowed_dims = [tuple()]
        if isinstance(allowed_dims, str):
            allowed_dims = [(allowed_dims,)]
        elif isinstance(allowed_dims, list):
            allowed_dims = [tuple([d]) if isinstance(d, str) else tuple(d)
                            for d in allowed_dims]
        self.allowed_dims = tuple(allowed_dims)

        self.optional = optional
        self.default_value = default_value
        self._validators = list(validators)
        self._state = None
        self._rate = None
        self._change = None

    @property
    def validators(self):
        return list(itertools.chain(self.default_validators, self._validators))

    def run_validators(self, xarray_variable):
        for vfunc in self.validators:
            vfunc(xarray_variable)

    def validate(self, xarray_variable):
        pass

    def to_xarray_variable(self, value):
        """Convert the input value to an `xarray.Variable` object and
        perform some validation.

        Parameters
        ----------
        value : any
            The input value can be in the form of a single value,
            an array-like, a ``(dims, data[, attrs])`` tuple, another
            `xarray.Variable` object or a `xarray.DataArray` object.

        Returns
        -------
        variable : `xarray.Variable`
            The value converted to an xarray variable if it passed all
            validation tests.

        Raises
        ------
        ValueError
            If the value doesn't pass the validation tests.

        """
        if value in EMPTY_VALUES:
            value = self.default_value

        try:
            xarray_variable = as_variable(value)
        except TypeError:
            # handle allowed case of a coordinate
            # dimension has to be further renamed by the name of the variable
            # in a model.
            dim = 'this_variable_name'
            xarray_variable = as_variable((dim, value))

        self.run_validators(xarray_variable)
        self.validate(xarray_variable)

        xarray_variable.attrs.update(self.attrs)
        xarray_variable.attrs['description'] = self.description

        return xarray_variable

    @property
    def state(self):
        """State value of the variable, i.e., the instant value at a given
        time or simply the value if the variable is not time dependent.
        """
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    value = state

    @property
    def rate(self):
        """Rate value of the variable, i.e., the rate of change in time
        (time derivative).
        """
        return self._rate

    @rate.setter
    def rate(self, value):
        self._rate = value

    @property
    def change(self):
        """Change value of the variable, i.e., the rate of change in time
        (time derivative) integrated over the time step.
        """
        return self._change

    @change.setter
    def change(self, value):
        self._change = value

    def __repr__(self):
        dims_str = ', '.join(['(%s)' % ', '.join(['%r' % d for d in dims])
                              for dims in self.allowed_dims])
        return ("<fastscape.models.%s %s>" % (type(self).__name__, dims_str))


class ForeignVariable(AbstractVariable):
    """Reference to a variable that is defined in another `Process` class.

    """
    def __init__(self, other_process, var_name, provided=False):
        """
        Parameters
        ----------
        other_process : str or class
            Class or class name in which the variable is defined.
        var_name : str
            Name of the corresponding class attribute in `other_process`.
            The value of this class attribute must be a `Variable` object.
        provided : bool, optional
            Defines whether a value for the variable is required (False) or
            provided (True) by the process in which this reference is
            defined (default: False).

        """
        super(ForeignVariable, self).__init__(provided=provided)

        self._other_process_cls = other_process
        self._other_process_obj = None
        self.var_name = var_name

    @property
    def ref_process(self):
        """The process where the original variable is defined.

        Returns either the Process class or a Process instance attached to
        a model.
        """
        if self._other_process_obj is None:
            return self._other_process_cls

        return self._other_process_obj

    @property
    def ref_var(self):
        """The original variable object."""
        return self.ref_process.variables[self.var_name]

    @property
    def state(self):
        return self.ref_var._state

    @state.setter
    def state(self, value):
        self.ref_var.state = value

    value = state

    @property
    def rate(self):
        return self.ref_var._rate

    @rate.setter
    def rate(self, value):
        self.ref_var.rate = value

    @property
    def change(self):
        return self.ref_var._change

    @change.setter
    def change(self, value):
        self.ref_var._change = value

    def __repr__(self):
        ref_str = "%s.%s" % (self.ref_process.name, self.var_name)

        return "<fastscape.models.%s (%s)>" % (type(self).__name__, ref_str)


ForeignVariable.state.__doc__ = Variable.state.__doc__
ForeignVariable.rate.__doc__ = Variable.rate.__doc__
ForeignVariable.change.__doc__ = Variable.change.__doc__


class DiagnosticVariable(AbstractVariable):
    """Variable for model diagnostic purpose only.

    The value of a diagnostic variable is computed on the fly during a
    model run (there is no initialization nor update of any state).

    A diagnostic variable is defined inside a `Process` subclass, but
    it shouldn't be created directly as a class attribute.
    Instead it should be defined by applying the `@diagnostic` decorator
    on a method of that class.

    Diagnostic variables declared in a process should never be referenced
    in other processes as foreign variable.

    The diagnostic variables declared in a process are computed after the
    execution of all processes in a model at the end of a time step.

    """
    def __init__(self, func, description='', attrs=None):
        super(DiagnosticVariable, self).__init__(
            provided=True, description=description, attrs=attrs
        )

        self._func = func
        self._process_obj = None

    def assign_process_obj(self, process_obj):
        self._process_obj = process_obj

    @property
    def state(self):
        """State value of this variable (read-only), i.e., the instant value
        at a given time or simply the value if the variable is time
        independent.
        """
        return self._func(self._process_obj)

    value = state

    def __call__(self):
        return self.state


def diagnostic(attrs_or_function=None, attrs=None):
    """Applied to a method of a `Process` subclass, this decorator
    allows registering that method as a diagnostic variable.

    The method's docstring is used as a description of the
    variable (it should be short, one-line).

    Parameters
    ----------
    attrs : dict (optional)
        Variable metadata (e.g., standard_name, units, math_symbol...).

    Examples
    --------
    @diagnostic
    def slope(self):
        '''topographic slope'''
        return self._compute_slope()

    @diagnostic({'units': '1/m'})
    def curvature(self):
        '''terrain curvature'''
        return self._compute_curvature()

    """
    func = None
    if callable(attrs_or_function):
        func = attrs_or_function
    elif isinstance(attrs_or_function, dict):
        attrs = attrs_or_function

    def _add_diagnostic_attrs(function):
        function._diagnostic = True
        function._diagnostic_attrs = attrs
        return function

    if func is not None:
        return _add_diagnostic_attrs(func)
    else:
        return _add_diagnostic_attrs


class UndefinedVariable(AbstractVariable):
    """Represent variable(s) that has to be defined later, i.e.,
    when creating a new `Process` object.

    Undefined variables are useful in cases when we want to reuse
    the same process in different contexts without having to re-write
    other `Process` subclasses. Good examples are processes that
    aggregate (e.g., sum, product, mean) variables provided by
    other processes.

    """
    def __init__(self):
        super(UndefinedVariable, self).__init__(provided=False)


class VariableList(tuple):
    """A tuple of only `Variable` or `ForeignVariable` objects."""
    def __new__(cls, variables):
        var_list = [var for var in variables
                    if isinstance(var, (Variable, ForeignVariable))]

        if len(var_list) != len(variables):
            raise ValueError("found variables mixed with objects of other "
                             "types in %s" % variables)

        return tuple.__new__(cls, var_list)
