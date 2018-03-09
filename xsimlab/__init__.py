"""
xarray-simlab.

"""
from .xr_accessor import SimlabAccessor, create_setup
from .variable import variable, on_demand, foreign, group
from .process import get_variables, process
from .model import Model
from .process import filter_variables, process

from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
