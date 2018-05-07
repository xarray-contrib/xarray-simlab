"""
xarray-simlab.

"""
from .xr_accessor import SimlabAccessor, create_setup
from .variable import variable, on_demand, foreign, group
from .process import filter_variables, process, process_info, variable_info
from .model import Model

from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
