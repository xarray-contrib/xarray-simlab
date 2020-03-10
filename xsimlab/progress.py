from xsimlab.hook import RuntimeHook, runtime_hook


class ProgressBar(RuntimeHook):
    """
    Progress bar implementation using the tqdm package.

    Parameters
    ----------
    frontend : {"auto", "console", "gui", "notebook"}, optional
        Allows control over Python environment.
    **kwargs : dict, optional
        Arbitrary keyword arguments for progress bar customization.

    Examples
    --------
    :class:`ProgressBar` takes full advantage of :class:`RuntimeHook`.

    Call it as part of :func:`run`:
    >>> out_ds = in_ds.xsimlab.run(model=model, hooks=[xs.progress.ProgressBar()])

    In a context manager using the `with` statement`:
    >>> with xs.progress.ProgressBar():
    ...    out_ds = in_ds.xsimlab.run(model=model)

    Globally with `register` method:
    >>> pbar = xs.progress.ProgressBar()
    >>> pbar.register()
    >>> out_ds = in_ds.xsimlab.run(model=model)
    >>> pbar.unregister()

    For additional customization, see: https://tqdm.github.io/docs/tqdm/
    """

    env_list = ["auto", "console", "gui", "notebook"]

    def __init__(self, frontend="auto", **kwargs):
        if frontend == "auto":
            from tqdm.auto import tqdm
        elif frontend == "console":
            from tqdm import tqdm
        elif frontend == "gui":
            from tqdm.gui import tqdm
        elif frontend == "notebook":
            from tqdm.notebook import tqdm
        else:
            raise ValueError(
                f"Frontend argument {self.frontend!r} not supported. Please select one of the following: {', '.join(self.env_list)}"
            )

        self.tqdm = tqdm
        self.pbar_dict = {"bar_format": "{bar} {desc} {percentage:3.0f}%"}
        self.pbar_dict.update(kwargs)

    @runtime_hook("initialize", trigger="pre")
    def init_bar(self, model, context, state):
        self.pbar_dict.update(total=context["nsteps"] + 2, desc="initialize")
        self.pbar_model = self.tqdm(**self.pbar_dict)

    @runtime_hook("initialize", trigger="post")
    def update_init(self, mode, context, state):
        self.pbar_model.update(1)

    @runtime_hook("run_step", trigger="post")
    def update_runstep(self, mode, context, state):
        self.pbar_model.set_description_str(
            f"run step {context['step']}/{context['nsteps']}"
        )
        self.pbar_model.update(1)

    @runtime_hook("finalize", trigger="pre")
    def update_finalize(self, model, context, state):
        self.pbar_model.set_description_str("finalize")

    @runtime_hook("finalize", trigger="post")
    def close_bar(self, model, context, state):
        self.pbar_model.update(1)
        self.pbar_model.close()
