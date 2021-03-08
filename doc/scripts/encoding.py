import xsimlab as xs
import numpy as np


@xs.process
class Foo:
    v_bool_nan = xs.variable(dims="x", intent="out")
    # suppress nan values by setting an explicit fill value:
    v_bool = xs.variable(dims="x", intent="out", encoding={"fill_value": None})

    def initialize(self):
        self.v_bool_nan = [True, False]
        self.v_bool = [True, False]
