import numpy as np


def _expand_value(value, nrepeats):
    if value is None:
        return None
    elif np.isscalar(value):
        return np.repeat(value, nrepeats)
    else:
        return np.asarray(value)