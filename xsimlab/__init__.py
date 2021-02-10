"""
xarray-simlab.

"""
# flake8: noqa

from .hook import RuntimeHook, runtime_hook
from .model import Model
from .process import (
    filter_variables,
    process,
    process_info,
    runtime,
    RuntimeSignal,
    variable_info,
)
from .variable import (
    any_object,
    variable,
    index,
    on_demand,
    foreign,
    global_ref,
    group,
    group_dict,
)
from .utils import MAIN_CLOCK
from .xr_accessor import SimlabAccessor, create_setup
from . import monitoring

from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
