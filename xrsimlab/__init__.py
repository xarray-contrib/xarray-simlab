"""
xarray-simlab.

"""
from __future__ import absolute_import

from .xr_accessor import SimLabAccessor
from .variable.base import (Variable, ForeignVariable, VariableList,
                            VariableGroup, diagnostic, ValidationError)
from .variable.custom import NumberVariable, FloatVariable, IntegerVariable
from .process import Process
from .model import Model

from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
