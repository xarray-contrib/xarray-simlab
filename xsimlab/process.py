import sys
import inspect
import copy

from .variable.base import (AbstractVariable, DiagnosticVariable,
                            VariableList, VariableGroup)
from .formatting import process_info
from .utils import AttrMapping, combomethod


_process_meta_default = {
    'time_dependent': True
}


def _extract_variables(mapping):
    # type: Dict[str, Any] -> Tuple[
    #     Dict[str, Union[AbstractVariable, Variablelist, VariableGroup]],
    #     Dict[str, Any]]

    var_dict = {}

    for key, value in mapping.items():
        if isinstance(value, (AbstractVariable, VariableList, VariableGroup)):
            var_dict[key] = value

        elif getattr(value, '_diagnostic', False):
            var = DiagnosticVariable(value, description=value.__doc__,
                                     attrs=value._diagnostic_attrs)
            var_dict[key] = var

    no_var_dict = {k: v for k, v in mapping.items() if k not in var_dict}

    return var_dict, no_var_dict


class ProcessBase(type):
    """Metaclass for all processes."""

    def __new__(cls, name, bases, attrs):
        parents = [b for b in bases if isinstance(b, ProcessBase)]
        if not parents:
            # Skip customization for the `Process` class
            # (only applied to its subclasses)
            new_attrs = attrs.copy()
            new_attrs.update({'_variables': {}, '_meta': {}})
            return super().__new__(cls, name, bases, new_attrs)
        for p in parents:
            mro = [c for c in inspect.getmro(p)
                   if isinstance(c, ProcessBase)]
            if len(mro) > 1:
                # Currently not supported to create a class that
                # inherits from a subclass of Process
                raise TypeError("subclassing a subclass of Process "
                                "is not supported")

        # start with new attributes
        new_attrs = {'__module__': attrs.pop('__module__')}
        classcell = attrs.pop('__classcell__', None)
        if classcell is not None:
            new_attrs['__classcell__'] = classcell

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
                    % (keys, cls.__name__)
                )
            meta_dict.update(meta_attrs)

        new_attrs['_meta'] = meta_dict

        # add variables and diagnostics separately from the rest of
        # the attributes and methods defined in the class
        vars, novars = _extract_variables(attrs)
        new_attrs['_variables'] = vars
        for k, v in novars.items():
            new_attrs[k] = v

        new_class = super().__new__(cls, name, bases, new_attrs)

        return new_class


class Process(AttrMapping, metaclass=ProcessBase):
    """Base class that represents a logical unit in a computational model.

    A subclass of `Process` usually implements:

    - A process interface as a set of `Variable`, `ForeignVariable`,
      `VariableGroup` or `VariableList` objects, all defined as class
      attributes.

    - Some of the five `.validate()`, `.initialize()`, `.run_step()`,
      `.finalize_step()` and `.finalize()` methods, which use or compute
      values of the variables defined in the interface during a model run.

    - Additional methods decorated with `@diagnostic` that compute
      the values of diagnostic variables during a model run.

    Once created, a `Process` object provides both dict-like and
    attribute-like access for all its variables, including diagnostic
    variables if any.

    """
    def __init__(self):
        # prevent modifying variables at the class level. also prevent
        # using the same variable objects in two distinct instances
        self._variables = copy.deepcopy(self._variables)

        super(Process, self).__init__(self._variables)

        for var in self._variables.values():
            if isinstance(var, DiagnosticVariable):
                var.assign_process_obj(self)

        self._name = None
        self._initialized = True

    def clone(self):
        """Clone the process.

        This is equivalent to a deep copy, except that variable data
        (i.e., `state`, `value`, `change` or `rate` properties) are not copied.
        """
        obj = type(self)()
        obj._name = self._name
        return obj

    @property
    def variables(self):
        """A dictionary of Process variables."""
        return self._variables

    @property
    def meta(self):
        """A dictionary of Process metadata (i.e., Meta attributes)."""
        return self._meta

    @property
    def name(self):
        """Process name.

        Returns the name of the Process subclass if it is not attached to
        any Model object.

        """
        if self._name is None:
            return type(self).__name__

        return self._name

    def validate(self):
        """Validate and/or update the process variables values.

        Implementation is optional (by default it does nothing).

        An implementation of this method should be provided if the process
        has variables that are optional and/or that depend on other
        variables defined in this process.

        To validate values of variables taken independently, it is
        prefered to use Variable validators.

        See Also
        --------
        Variable.validators

        """
        pass

    def initialize(self):
        """This method will be called once at the beginning of a model run.

        Implementation is optional (by default it does nothing).
        """
        pass

    def run_step(self, *args):
        """This method will be called at every time step of a model run.

        It should accepts one argument that corresponds to the time step
        duration.

        This must be implemented for all time dependent processes.
        """
        raise NotImplementedError(
            "class %s has no method 'run_step' implemented"
            % type(self).__name__
        )

    def finalize_step(self):
        """This method will be called at the end of every time step, i.e,
        after `run_step` has been executed for all processes in a model.

        Implementation is optional (by default it does nothing).
        """
        pass

    def finalize(self):
        """This method will be called once at the end of a model run.

        Implementation is optional (by default does nothing).
        """
        pass

    @combomethod
    def info(cls_or_self, buf=None):
        """info(buf=None)

        Concise summary of Process variables and metadata.

        Parameters
        ----------
        buf : object, optional
            Writable buffer (default: sys.stdout).

        """
        if buf is None:  # pragma: no cover
            buf = sys.stdout

        buf.write(process_info(cls_or_self))

    def __repr__(self):
        cls = "'%s.%s'" % (self.__module__, type(self).__name__)
        header = "<xsimlab.Process %s>\n" % cls

        return header + process_info(self)
