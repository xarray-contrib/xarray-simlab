import numpy as np
from numpy.testing import assert_array_equal

from xsimlab.stores import _get_output_steps_by_clock


def test_get_output_steps_by_clock(in_dataset):
    expected = {
        "clock": np.array([True, True, True, True, True]),
        "out": np.array([True, False, True, False, True]),
        None: np.array([False, False, False, False, True]),
    }

    actual = _get_output_steps_by_clock(in_dataset)

    assert actual.keys() == expected.keys()
    for k in expected:
        assert_array_equal(actual[k], expected[k])
