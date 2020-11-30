import numpy as np
import pytest

import xsimlab as xs
from xsimlab.process import SimulationStage


@pytest.mark.parametrize(
    "trigger,signal,expected",
    [
        ("pre", xs.RuntimeSignal.SKIP, [1.0, 2.0, 4.0, 5.0]),
        ("post", xs.RuntimeSignal.SKIP, [1.0, 3.0, 5.0, 6.0]),
        ("pre", xs.RuntimeSignal.CONTINUE, [1.0, 2.0, 3.0, 4.0]),
        ("post", xs.RuntimeSignal.CONTINUE, [1.0, 3.0, 4.0, 5.0]),
        ("pre", xs.RuntimeSignal.BREAK, [1.0, 2.0, np.nan, np.nan]),
        ("post", xs.RuntimeSignal.BREAK, [1.0, 3.0, np.nan, np.nan]),
    ]
)
def test_signal_model_level(trigger, signal, expected):
    @xs.process
    class Foo:
        v = xs.variable(intent="out")
        vv = xs.variable(intent="out")

        def initialize(self):
            self.v = 0.0
            self.vv = 10.0

        def run_step(self):
            self.v += 1.0

        def finalize_step(self):
            self.v += 1.0

    @xs.runtime_hook("run_step", level="model", trigger=trigger)
    def hook_func(model, context, state):
        if context["step"] == 1:
            return signal

    model = xs.Model({'foo': Foo})
    ds_in = xs.create_setup(
        model=model,
        clocks={'clock': range(4)},
        output_vars={"foo__v": "clock", "foo__vv": None},
    )
    ds_out = ds_in.xsimlab.run(model=model, hooks=[hook_func])

    np.testing.assert_equal(ds_out.foo__v.values, expected)

    # ensure that clock-independent output variables are properly
    # saved even when the simulation stops early
    assert ds_out.foo__vv == 10.0


@pytest.fixture(params=[True, False])
def parallel(request):
    return request.param


@pytest.mark.parametrize(
    "step,trigger,signal,break_bar,expected_v1,expected_v2",
    [
        # Both Foo.run_step and Bar.run_step are run
        (0, "pre", xs.RuntimeSignal.SKIP, False, 1, 2),
        # None of Foo.run_step and Bar.run_step are run
        (1, "pre", xs.RuntimeSignal.SKIP, False, 0, 0),
        # BREAK signal returned in Bar.run_step prevails
        (1, "post", xs.RuntimeSignal.SKIP, True, 0, 2),
        # BREAK signal returned by Bar.run_step only
        (0, "post", xs.RuntimeSignal.BREAK, True, 0, 2)
    ]
)
def test_signal_process_level(step, trigger, signal, break_bar, expected_v1, expected_v2, parallel):
    @xs.process
    class Foo:
        v1 = xs.variable(intent="out")
        v2 = xs.variable()

        def initialize(self):
            self.v1 = 0

        def run_step(self):
            self.v1 = 1

    @xs.process
    class Bar:
        v2 = xs.foreign(Foo, "v2", intent="out")

        def initialize(self):
            self.v2 = 0

        def run_step(self):
            self.v2 = 2

            if break_bar:
                return xs.RuntimeSignal.BREAK

    def hook_func(model, context, state):
        if context["step"] == 1:
            return signal

    hook_dict = {SimulationStage.RUN_STEP: {"process": {trigger: [hook_func]}}}

    model = xs.Model({"foo": Foo, "bar": Bar})
    model.execute("initialize", {})
    model.execute("run_step", {"step": step}, hooks=hook_dict, parallel=parallel)

    assert model.state[("foo", "v1")] == expected_v1
    assert model.state[("foo", "v2")] == expected_v2
