import xsimlab as xs
import numpy as np


@xs.process
class Foo:
    a = xs.variable(intent="out", dims=(xs.MAIN_CLOCK))
    b = xs.variable(intent="out", dims=[(), "a"])
    # c = xs.variable(intent="in", dims=["a", ("clock", xs.MAIN_CLOCK)])

    @xs.runtime(args="main_clock_values")
    def initialize(self, clock_values):
        self.b = 3
        self.a = clock_values

    @xs.runtime(args="step")
    def run_step(self, step):
        self.a[step] += 1 * self.b


model = xs.Model({"foo": Foo})
ds_in = xs.create_setup(
    model=model,
    clocks={"clock": range(5), "iclock": [2, 4]},
    main_clock="clock",
    input_vars={},  # "foo__c": 5},
    output_vars={"foo__a": None, "foo__b": "iclock"},
)
print(ds_in.xsimlab.run(model=model).foo__a.data == [3, 4, 5, 6, 4])
