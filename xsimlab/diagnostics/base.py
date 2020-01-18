import inspect

from ..process import SimulationStage


def runtime_hook(stage, level="model", trigger="post"):
    """Decorator that allows a function or a method be called
    at one or more specific times during a simulation.

    The decorated function must have the following signature:
    ``func(runtime_context, store)``.

    Parameters
    ----------
    stage : {'initialize', 'run_step', 'finalize_step', 'finalize'}
        The simulation stage at which the function is called.
    level : {'model', 'process'}
        Sets whether the simulation stage is treated model-wise ('model')
        or process-wise ('process'). For the model-wise case (default), the
        function is called only once during the execution of the simulation
        stage. For the process-wise case, the function is executed as many
        times as there are processes in the model that provide an
        implementation of that simulation stage.
    trigger : {'pre', 'post'}
        Sets when exactly to trigger the function call, i.e., just before
        ('pre') or just after ('post') the execution of the model's or
        process' simulation stage (default: after).

    """
    stage = SimulationStage(stage)

    if level not in ("model", "process"):
        raise ValueError("level argument must be either 'model' or 'process'")

    if trigger not in ("pre", "post"):
        raise ValueError("trigger argument must be either 'pre' or 'post'")

    def wrap(func):
        func.__xsimlab_hook__ = (stage, level, trigger)
        return func

    return wrap


def _is_hook(func):
    return hasattr(func, "__xsimlab_hook__")


class RuntimeDiagnostic:

    def __init__(self, hooks=None):
        if hooks is None:
            hooks = []

        if not all(_is_hook(h) for h in hooks):
            raise TypeError("'hooks' must be an iterable of runtime_hook decorated function")

        self._hooks = hooks

    def _get_hooks(self):
        hook_methods = [m for _, m in inspect.getmembers(self, predicate=_is_hook)]

        return getattr(self, '_hooks', []) + hook_methods
