from xsimlab.hook import RuntimeHook, runtime_hook


__all__ = ("ProgressBar",)


class ProgressBar(RuntimeHook):
    """
    Progress bar implementation using the tqdm package.

    Examples
    --------
    ProgressBar takes full advantage of :class:`RuntimeHook`.

    Call it as part of :meth:`xarray.Dataset.xsimlab.run`:

    >>> from xsimlab.monitoring import ProgressBar
    >>> out_ds = in_ds.xsimlab.run(model=model, hooks=[ProgressBar()])

    In a context manager using the ``with`` statement:

    >>> with ProgressBar():
    ...    out_ds = in_ds.xsimlab.run(model=model)

    Globally with ``register`` method:

    >>> pbar = ProgressBar()
    >>> pbar.register()
    >>> out_ds = in_ds.xsimlab.run(model=model)
    >>> pbar.unregister()

    """

    def __init__(self, frontend="auto", **kwargs):
        """
        Parameters
        ----------
        frontend : {"auto", "console", "gui", "notebook"}, optional
            Selects a frontend for displaying the progress bar. By default ("auto"),
            the frontend is chosen by guessing in which environment the simulation
            is run. The "console" frontend displays an ascii progress bar, while the
            "gui" frontend is based on matplotlib and the "notebook" frontend is based
            on ipywidgets.
        **kwargs : dict, optional
            Arbitrary keyword arguments for progress bar customization.
            See https://tqdm.github.io/docs/tqdm/.

        """
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
                f"Frontend argument {frontend!r} not supported. "
                "Please select one of the following: "
                ", ".join(["auto", "console", "gui", "notebook"])
            )

        self.custom_description = False
        if "desc" in kwargs.keys():
            self.custom_description = True

        self.tqdm = tqdm
        self.tqdm_kwargs = {"bar_format": "{bar} {percentage:3.0f}% | {desc} "}
        self.tqdm_kwargs.update(kwargs)

    @runtime_hook("initialize", trigger="pre")
    def init_bar(self, model, context, state):
        if self.custom_description:
            self.tqdm_kwargs.update(total=context["nsteps"] + 2)
        else:
            self.tqdm_kwargs.update(total=context["nsteps"] + 2, desc="initialize")
        self.pbar_model = self.tqdm(**self.tqdm_kwargs)

    @runtime_hook("initialize", trigger="post")
    def update_init(self, mode, context, state):
        self.pbar_model.update(1)

    @runtime_hook("run_step", trigger="post")
    def update_run_step(self, model, context, state):
        if not self.custom_description:
            self.pbar_model.set_description_str(
                f"run step {context['step']}/{context['nsteps']}"
            )
        self.pbar_model.update(1)

    @runtime_hook("finalize", trigger="pre")
    def update_finalize(self, model, context, state):
        if not self.custom_description:
            self.pbar_model.set_description_str("finalize")

    @runtime_hook("finalize", trigger="post")
    def close_bar(self, model, context, state):
        self.pbar_model.update(1)
        elapsed_time = self.tqdm.format_interval(self.pbar_model.format_dict["elapsed"])
        self.pbar_model.set_description_str(f"Simulation finished in {elapsed_time}")
        self.pbar_model.close()
