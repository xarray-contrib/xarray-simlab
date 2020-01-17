from .process import SimulationStage


class RuntimeHook:

    def __init__(self, func, stage, level, mode):

        if not callable(func):
            raise TypeError(f"{func!r} is not callable")

        if level not in ("model", "process"):
            raise ValueError("level argument must be either 'model' or 'process'")

        if mode not in ("pre", "post"):
            raise ValueError("side argument must be either 'pre' or 'post'")

        self._func = func
        self._stage = SimulationStage(stage)
        self._level = level
        self._mode = mode

    @property
    def trigger_at(self):
        return (self._stage.value, self._level, self._mode)

    def __call__(self, runtime_context, store):
        self._func(runtime_context, store)

    def __repr__(self):
        return f"<{type(self).__name__} {self.trigger_at!r}>"


def runtime_hook(stage, level='model', mode='post'):
    """Decorator that allows a function or a method be called
    at one or more specific times during a simulation.

    The decorated function must have the following signature:
    ``func(runtime_context, store)``.

    Parameters
    ----------
    stage : {'initialize', 'run_step', 'finalize_step', 'finalize'}
        The simulation stage at which the function is called.
    level : {'model', 'process'}
        Sets whether the function is called just once for a simulation
        stage ('model', default) or for each process in the model
        ('process').
    side : {'pre', 'post'}
        Sets whether the function is called before ('pre') or after
        ('post', default) the execution of the model or process
        simulation stage.

    """
    def wrap(func):
        return RuntimeHook(func, stage, level, mode)

    return wrap
