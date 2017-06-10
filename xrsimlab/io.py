import re

import yaml

import numpy as np
from xarray import Dataset, Variable

from .xr_accessor import SimLabAccessor


# fix parsing scientific notation
# see http://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


KEY_FVAL = {'range': lambda v: np.arange(*v),
            'arange': lambda v: np.arange(*v),
            'linspace': lambda v: np.linspace(*v),
            'logspace': lambda v: np.logspace(*v),
            'value': lambda v: np.asarray(v),
            'array': lambda v: np.asarray(v)}


def _extract_value_and_attrs(input_val_or_dict):
    """Extract a value from an input entry as a numpy array and return it
    separately from other key/val pairs if any.
    """
    if not isinstance(input_val_or_dict, dict):
        return np.asarray(input_val_or_dict), {}

    else:
        input_dict = input_val_or_dict.copy()
        count_keys = len([k for k in input_dict if k in KEY_FVAL])
        if count_keys > 1:
            raise ValueError('ambiguous value set in input\n%s\n'
                             'multiple options given from %s'
                             % (input_dict, tuple(KEY_FVAL)))
        elif count_keys == 0:
            raise ValueError('no value set in input\n%s\n'
                             'set one option among %s'
                             % (input_dict, tuple(KEY_FVAL)))

        for k, f in KEY_FVAL.items():
            if k in input_dict:
                value = f(input_dict.pop(k))
                return value, input_dict


def create_model_setup(filename_or_dict):
    """Create a model setup.

    Parameters
    ----------
    filename_or_dict: str or dict-like
        Either the name/path to an input file (YAML format) or a dict-like
        object containing all input values needed to setup one or multiple
        models.

    Returns
    -------
    xarray.Dataset

    """
    if isinstance(filename_or_dict, str):
        with open(filename_or_dict) as f:
            inputs = yaml.load(f, Loader=loader)
    else:
        inputs = filename_or_dict

    ds = Dataset()
    # TODO: feed Dataset

    return ds
