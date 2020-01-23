import pytest

import xsimlab as xs
from xsimlab import runtime_hook, RuntimeHook


@xs.process
class P:
    u = xs.variable(intent="inout", default=0)

    def run_step(self):
        self.u += 1


@xs.process
class Dummy:
    def run_step(self):
        pass


@pytest.fixture
def model():
    return xs.Model({"dummy": Dummy, "p": P})


def test_runtime_hook():
    with pytest.raises(ValueError, match="level argument.*"):

        @runtime_hook("run_step", "invalid", "pre")
        def func1():
            pass

    with pytest.raises(ValueError, match="trigger argument.*"):

        @runtime_hook("run_step", "model", "invalid")
        def func2():
            pass


@pytest.mark.parametrize(
    "event,expected_ncalls,expected_u",
    [
        (("run_step", "model", "pre"), 1, 0),
        (("run_step", "model", "post"), 1, 1),
        (("run_step", "process", "pre"), 2, None),
        (("run_step", "process", "post"), 2, None),
    ],
)
def test_runtime_hook_calls(model, event, expected_ncalls, expected_u):
    inc = [0]

    @runtime_hook(*event)
    def test_hook(_model, context, state):
        inc[0] += 1

        assert "step" in context
        assert _model is model

        # TODO: get name of current process executed in context?
        if expected_u is not None:
            assert state[("p", "u")] == expected_u

    in_ds = xs.create_setup(model=model, clocks={"c": [0, 1]})
    # safe mode disabled so that we can assert _model is model above
    in_ds.xsimlab.run(model=model, hooks=[test_hook], safe_mode=False)

    assert inc[0] == expected_ncalls


def test_runtime_hook_call_frozen(model):
    in_ds = xs.create_setup(model=model, clocks={"c": [0, 1]})

    @runtime_hook("run_step", "model", "pre")
    def change_context(model, context, state):
        context["step"] = 0

    with pytest.raises(TypeError, match=".*not support item assignment"):
        in_ds.xsimlab.run(model=model, hooks=[change_context])

    @runtime_hook("run_step", "model", "pre")
    def change_state(model, context, state):
        state[("p", "u")] = 0

    with pytest.raises(TypeError, match=".*not support item assignment"):
        in_ds.xsimlab.run(model=model, hooks=[change_state])


@pytest.mark.parametrize("given_as", ["argument", "context", "register"])
def test_runtime_hook_instance(model, given_as):
    flag = [False]

    @runtime_hook("run_step", "model", "pre")
    def test_hook(_model, context, state):
        flag[0] = True

    rd = RuntimeHook(test_hook)

    in_ds = xs.create_setup(model=model, clocks={"c": [0, 1]})

    if given_as == "argument":
        in_ds.xsimlab.run(model=model, hooks=[rd])

    elif given_as == "context":
        with rd:
            in_ds.xsimlab.run(model=model)

    elif given_as == "register":
        rd.register()
        in_ds.xsimlab.run(model=model)

    assert flag[0] is True

    if given_as == "register":
        flag[0] = False
        rd.unregister()
        in_ds.xsimlab.run(model=model)
        assert flag[0] is False


def test_runtime_hook_init():
    def not_a_decorated_hook(model, context, state):
        pass

    with pytest.raises(TypeError, match=".*only runtime_hook decorated.*"):
        RuntimeHook(not_a_decorated_hook)


def test_runtime_hook_subclass(model):
    flag = [False]

    class TestHook(RuntimeHook):
        @runtime_hook("run_step", "model", "pre")
        def test_hook(self, _model, context, state):
            flag[0] = True

    in_ds = xs.create_setup(model=model, clocks={"c": [0, 1]})

    with TestHook():
        in_ds.xsimlab.run(model=model)

    assert flag[0] is True


def test_hook_arg_type(model):
    in_ds = xs.create_setup(model=model, clocks={"c": [0, 1]})

    with pytest.raises(TypeError, match=".*not a RuntimeHook.*"):
        in_ds.xsimlab.run(model=model, hooks=[1])
