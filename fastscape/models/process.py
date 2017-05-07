import sys
import inspect
import copy

from .variable import AbstractVariable, DiagnosticVariable, UndefinedVariable
from ..core.formatting import process_info
from ..core.utils import AttrMapping, combomethod


_process_meta_default = {
    'time_dependent': True
}


def _extract_variables(mapping):
    # type: Dict[str, Any] -> Dict[str, Union[AbstractVariable, tuple]]
    """extract from the input mapping all `AbstractVariable`
    objects and/or tuples/lists of `AbstractVariable` objects.
    """
    var_dict = {}

    for key, value in mapping.items():
        if isinstance(value, AbstractVariable):
            var_dict[key] = value

        elif getattr(value, '_diagnostic', False):
            var = DiagnosticVariable(value, description=value.__doc__,
                                     attrs=value._diagnostic_attrs)
            var_dict[key] = var

        elif isinstance(value, (tuple, list)):
            is_obj_var = [isinstance(obj, AbstractVariable) for obj in value]
            if all(is_obj_var):
                var_dict[key] = tuple(value)
            elif any(is_obj_var):
                raise ValueError("found variables mixed with other objects\n"
                                 "%s = %s" % (key, value))

    return var_dict


class ProcessBase(type):
    """Metaclass for all processes."""

    def __new__(cls, name, bases, attrs):
        parents = [b for b in bases if isinstance(b, ProcessBase)]
        if not parents:
            # Skip customization for the `Process` class
            # (only applied to its subclasses)
            return super().__new__(cls, name, bases, attrs)
        for p in parents:
            mro = [c for c in inspect.getmro(p)
                   if isinstance(c, ProcessBase)]
            if len(mro) > 1:
                # Currently not supported to create a class that
                # inherits from a subclass of Process
                raise TypeError("subclassing a subclass of Process "
                                "is not supported")

        # Create the class with new attributes
        new_attrs = {'__module__': attrs.pop('__module__')}
        classcell = attrs.pop('__classcell__', None)
        if classcell is not None:
            new_attrs['__classcell__'] = classcell
        new_class = super().__new__(cls, name, bases, new_attrs)

        # check and add metadata
        meta_cls = attrs.pop('Meta', None)
        meta_dict = _process_meta_default.copy()

        if meta_cls is not None:
            meta_attrs = {k: v for k, v in meta_cls.__dict__.items()
                          if not k.startswith('__')}
            invalid_attrs = set(meta_attrs) - set(meta_dict)
            if invalid_attrs:
                keys = ", ".join(["%r" % k for k in invalid_attrs])
                raise AttributeError(
                    "invalid attribute(s) %s set in class %s.Meta"
                    % (keys, new_class.__name__)
                )
            meta_dict.update(meta_attrs)

        new_class._meta = meta_dict

        # add all variables and diagnostics defined in the class
        new_class._variables = _extract_variables(attrs)

        return new_class

    @property
    def variables(cls):
        """Process variables."""
        return cls._variables

    @property
    def meta(cls):
        """Process metadata."""
        return cls._meta


class Process(AttrMapping, metaclass=ProcessBase):
    """Base class that represents a logical unit in a computational model.

    A subclass of `Process` usually implements:

    - A process interface as a set of `Variable`, `ForeignVariable`
      or `UndefinedVariable` objects (all defined as class attributes).

    - Three `.initialize()`, `.run_step()` and `.finalize()` methods,
      which use or compute values of the variables defined in the
      interface.

    - Additional methods decorated with `@diagnostic` that compute
      the values of diagnostic variables.

    Once created, a `Process` object provides both dict-like and
    attribute-like access for all its variables, including diagnostic
    variables if any.

    """
    def __init__(self, **variables):
        """
        Parameters
        ----------
        **variables : key, `Variable` pairs
            Variables that still need to be defined for this process
            (i.e., undefined at the class level).

        """
        # prevent modifying variables at the class level. also prevent
        # using the same variable objects in two distinct instances
        self._variables = copy.deepcopy(self._variables)

        undefined_var_names = [k for k, v in self._variables.items()
                               if isinstance(v, UndefinedVariable)]
        if undefined_var_names and not variables:
            raise ValueError("missing external variables ")
            # TODO: more informative error msg (like var names)
            # TODO: more logic, e.g., when required variables are missing
        if variables:
            # TODO: check for non valid variable names.
            self._variables.update(
                copy.deepcopy(_extract_variables(variables)))

        super(Process, self).__init__(self._variables)

        for var in self._variables.values():
            if isinstance(var, DiagnosticVariable):
                var.assign_process_obj(self)

        self._name = None
        self._initialized = True

    @property
    def variables(self):
        """Process variables."""
        return self._variables

    @property
    def meta(self):
        """Process metadata."""
        return self._meta

    @property
    def name(self):
        """Process name (None if not used in any Model object)."""
        return self._name

    def initialize(self):
        pass

    def run_step(self):
        raise NotImplementedError(
            "class %s has no method 'run_step' implemented"
            % type(self).__name__
        )

    def finalize(self):
        pass

    @combomethod
    def info(cls_or_self, buf=None):
        """
        Concise summary of Process variables and metadata.

        Parameters
        ----------
        buf : writable buffer (default: sys.stdout).

        """
        if buf is None:  # pragma: no cover
            buf = sys.stdout

        buf.write(process_info(cls_or_self))

    def __repr__(self):
        cls = "'%s.%s'" % (self.__module__, type(self).__name__)
        header = "<fastscape.models.Process %s>\n" % cls

        return header + process_info(self)
