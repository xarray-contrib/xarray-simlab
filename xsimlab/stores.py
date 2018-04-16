from collections import defaultdict
from copy import copy

import numpy as np


class InMemoryOutputStore(object):
    """A simple, in-memory store for model outputs.

    It basically consists of a Python dictionary with lists as values,
    which are converted to numpy arrays on store read-access.

    """
    def __init__(self):
        self._store = defaultdict(list)

    def append(self, key, value):
        self._store[key].append(copy(value))

    def __getitem__(self, key):
        return np.array(self._store[key])
